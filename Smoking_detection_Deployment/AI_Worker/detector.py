import cv2
import math
import numpy as np
import os
import sys
import contextlib
import torch
import threading
from pathlib import Path
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

MODEL_PATH_ENV = os.environ.get("MODEL_PATH")
if MODEL_PATH_ENV:
    MODEL_PATH = Path(MODEL_PATH_ENV)
else:
    # Docker layout: Model/weights is bind-mounted directly under this
    # service's WORKDIR (see docker-compose.yaml), i.e. <app>/Model/weights.
    # Local (non-Docker) dev layout: this file lives in AI_Worker/ next to
    # the project root's own Model/ folder, one level up. Auto-detect which
    # one actually exists instead of hardcoding either — set the MODEL_PATH
    # env var to override explicitly if neither guess is right.
    _docker_path = Path(__file__).parent / "Model" / "weights" / "best.pt"
    _local_path = Path(__file__).parent.parent / "Model" / "weights" / "best.pt"
    MODEL_PATH = _docker_path if _docker_path.exists() else _local_path
CONF_THRESHOLD = 0.25
IMG_SIZE = 1280

# Video/tracking path uses a lower YOLO-side conf gate than single-image
# detection. ByteTrack itself is designed with two confidence tiers (see
# bytetrack_custom.yaml: track_high_thresh/new_track_thresh for starting a
# track, track_low_thresh for continuing one that's already confirmed) —
# but that only works if weak (0.1-0.25 conf) detections are actually
# handed to it in the first place. Passing CONF_THRESHOLD (0.25) straight
# into model.track()'s own conf= filter discarded every sub-0.25 box
# *before* ByteTrack ever saw them, which silently made track_low_thresh
# dead code and caused the tracker to "lose" and re-ID a cigarette the
# moment its confidence dipped (e.g. hand rotating it, motion blur) rather
# than keeping the same ID alive. Passing this lower threshold instead lets
# ByteTrack's own two-tier logic do its job: a brand new track still needs
# new_track_thresh (0.3) to spawn, but an existing one can be kept alive
# down to track_low_thresh (0.1).
TRACK_CONF_THRESHOLD = 0.1


@contextlib.contextmanager
def _suppress_native_stderr():
    """Silences noisy native (C++) stderr logging from MediaPipe's
    underlying TensorFlow Lite / glog machinery (e.g. "Created TensorFlow
    Lite XNNPACK delegate", "Feedback manager requires a model with a
    single signature inference", "Using NORM_RECT without
    IMAGE_DIMENSIONS..."). These come from abseil/glog logging that's
    already initialized by the time Python runs, so the usual
    GLOG_minloglevel / TF_CPP_MIN_LOG_LEVEL environment variables don't
    reliably suppress them on Windows — temporarily redirecting the raw
    stderr file descriptor is the only approach that reliably works.
    """
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(devnull_fd)
        os.close(saved_fd)

# NMS IoU threshold passed to both predict() and track(). Ultralytics'
# default (0.7) is too loose for this model: measured on real footage, the
# same physical cigarette sometimes produces two overlapping boxes with IoU
# ~0.64-0.67 (slightly different confidence/extent), which survive the
# default NMS and get treated as two separate objects by the tracker. 0.5
# collapses these duplicate boxes while leaving genuinely distinct,
# simultaneously-visible cigarettes (which don't overlap at all) untouched.
NMS_IOU = 0.5

TRACKER_CONFIG = str(Path(__file__).parent / "trackers" / "bytetrack_custom.yaml")

# Max pixel distance, between where one track was last seen and where a
# later track first appears, for _stitch_track_ids() to consider them the
# same physical object reacquired after a gap. Set just above the largest
# same-object jump measured on validation footage (~62px after a ~1.3s gap)
# and well below the separation between two genuinely distinct,
# simultaneously-visible cigarettes (~270px), so distinct objects are never
# merged.
STITCH_MAX_DIST_PX = 65

# When pose tracking (require_human) is active, we have a much stronger
# "same object" signal than raw pixel distance: whether the same tracked
# *person* was holding both track fragments. If so, allow a far looser
# distance ceiling (a hand/cigarette can swing a lot further than 65px
# across an occlusion without it being a different cigarette) as long as
# the gap between fragments is still short enough to plausibly be the same
# continuous hold.
STITCH_HOLDER_MAX_DIST_PX = 400
STITCH_HOLDER_MAX_GAP_FRAMES = 150

# --- Optional "require human" filter (pose-based, not finger-based) ---
#
# We deliberately use MediaPipe *Pose* (body skeleton: wrist/mouth joints),
# not MediaPipe *Hands* (finger/skin-texture landmarks). Hands landmarks
# failed completely on gloved hands in testing (0 detections on a
# work-glove-wearing smoker in validation footage) because that model
# relies on visible finger/skin texture. Pose estimates joints from overall
# body silhouette/shape instead, so it keeps working through gloves, heavy
# clothing, etc — much closer to how a Kinect-style skeleton tracker works.
#
# A cigarette detection only counts as a "human-held" violation if its
# center falls inside one of that person's *hold regions* — small boxes
# built around the mouth and each hand (wrist + nearby finger-base
# landmarks) only, NOT a box spanning the whole body. A single whole-body
# box let a cigarette sitting anywhere near a person (e.g. on a table at
# hip height) count as "held", which was too permissive. Each region box
# is expanded by HOLDER_BBOX_MARGIN and floored to a minimum size derived
# from the person's shoulder width (so the box scales with how close they
# are to the camera, instead of a fixed pixel size that's too small up
# close / too large far away). Using a small region box (instead of one
# exact point) still tolerates a single landmark flickering in and out of
# visibility for a frame or two, since the mouth/hand each have multiple
# contributing landmarks.
POSE_MODEL_PATH = Path(__file__).parent / "mediapipe_models" / "pose_landmarker_lite.task"
HOLDER_BBOX_MARGIN = 2.2
_HOLDER_MIN_BOX_PX = 70
# MediaPipe Pose landmark indices: 9/10 = mouth corners; 15/17/19/21 =
# left wrist/pinky/index/thumb; 16/18/20/22 = right wrist/pinky/index/thumb.
_HOLDER_REGION_LANDMARKS = {
    "mouth": (9, 10),
    "left_hand": (15, 17, 19, 21),
    "right_hand": (16, 18, 20, 22),
}
_SHOULDER_LANDMARKS = (11, 12)
_POSE_VISIBILITY_THRESHOLD = 0.3

# Reduced "Kinect-style" skeleton for drawing only: mouth + both arms (down
# to the hand) + both legs (down to the foot) — deliberately not the full
# 33-point/35-connection mesh (face mesh, fingers, etc), which is visually
# noisy and not what matters for this app (we only care about cigarette
# proximity to a hand or mouth).
_POSE_KEY_LANDMARKS = (9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)
_POSE_DRAW_CONNECTIONS = [
    (9, 10),                      # mouth
    (11, 13), (13, 15),           # left arm -> left hand
    (12, 14), (14, 16),           # right arm -> right hand
    (11, 12), (23, 24),           # shoulders, hips
    (11, 23), (12, 24),           # torso sides (links arms to legs visually)
    (23, 25), (25, 27),           # left leg -> left foot
    (24, 26), (26, 28),           # right leg -> right foot
]
_POSE_SKELETON_COLOR = (255, 160, 0)   # bones, BGR
_POSE_JOINT_COLOR = (0, 215, 255)      # joints, BGR

_image_pose_landmarker = None
_pose_lock = threading.Lock()


def _get_image_pose_landmarker():
    """Singleton IMAGE-mode landmarker for single-image requests (run_on_image).
    Safe to reuse across unrelated images since IMAGE mode treats every
    call independently (no temporal state)."""
    global _image_pose_landmarker
    if _image_pose_landmarker is None:
        with _suppress_native_stderr():
            options = mp_vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_poses=4,
                min_pose_detection_confidence=0.3,
            )
            _image_pose_landmarker = mp_vision.PoseLandmarker.create_from_options(options)
    return _image_pose_landmarker


def _new_video_pose_landmarker():
    """Creates a fresh VIDEO-mode landmarker for one run_on_video() call.
    VIDEO mode (vs IMAGE mode) lets MediaPipe use the previous frame's
    detection region/motion as a prior for the next frame instead of
    re-detecting from scratch every time, which is what actually fixes most
    of the frame-to-frame flicker/jitter (a new instance is used per video,
    not shared, because VIDEO mode requires strictly-increasing timestamps
    within one logical stream — reusing one across unrelated videos would
    violate that)."""
    with _suppress_native_stderr():
        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=4,
            min_pose_detection_confidence=0.3,
        )
        return mp_vision.PoseLandmarker.create_from_options(options)


def _detect_poses_image(frame_bgr):
    """Runs MediaPipe Pose (IMAGE mode) on a single frame/image and returns
    the raw per-person landmark lists (33 points each, normalized 0-1)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with _pose_lock, _suppress_native_stderr():
        result = _get_image_pose_landmarker().detect(mp_img)
    return result.pose_landmarks


def _detect_poses_video(landmarker, frame_bgr, timestamp_ms: int):
    """Runs MediaPipe Pose (VIDEO mode) on one video frame using the given
    landmarker instance + monotonically increasing timestamp."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with _pose_lock, _suppress_native_stderr():
        result = landmarker.detect_for_video(mp_img, timestamp_ms)
    return result.pose_landmarks


class _PoseTracker:
    """Smooths MediaPipe Pose output across video frames and assigns a
    stable per-person id to each detected body.

    This does two things that plain per-frame pose detection doesn't:
    1. Flicker reduction: MediaPipe estimates each pose mostly
       independently per frame, so a wrist/mouth landmark can briefly drop
       below the visibility threshold for a frame or two even though
       nothing actually changed (occlusion flicker, motion blur, etc).
       Instead of treating that as "gone", we hold the landmark's last
       known position for a short window (HOLD_TTL_FRAMES) before treating
       it as genuinely lost.
    2. Person identity: gives cigarette-track stitching a way to know two
       track fragments were held by the *same* physical person (see
       _stitch_track_ids), which is a much stronger "same object" signal
       than raw pixel distance alone — e.g. when a hand (and the cigarette
       in it) is lost by YOLO mid-occlusion and reappears with a new
       track id, but it's still visibly the same person's same hand.

       The centroid-match distance used to re-identify a person on the
       next frame grows the longer that person went undetected (gap-scaled
       tolerance, capped at MATCH_GROWTH_CAP_FRAMES): a fixed radius was
       too tight whenever MediaPipe missed the *whole pose* (not just one
       landmark) for several consecutive frames — e.g. challenging
       lighting/occlusion — since a moving person can easily drift further
       than MATCH_MAX_DIST_PX during that gap. Without this, the same
       physical person would spuriously get a brand-new person_id every
       time pose detection briefly dropped out entirely, which silently
       broke holder-based cigarette track stitching (fragments "held" by
       different person_ids don't get merged).
    """

    MATCH_MAX_DIST_PX = 150     # max person-centroid jump between frames to still count as the same person
    MATCH_GROWTH_PX_PER_FRAME = 6   # extra tolerance per frame the person was undetected, up to MATCH_GROWTH_CAP_FRAMES
    MATCH_GROWTH_CAP_FRAMES = 60     # stop growing tolerance after this many frames unseen (~2s at 30fps)
    STALE_FRAMES = 200          # drop a person's identity after this many frames unseen (~6-7s at 30fps)
    HOLD_TTL_FRAMES = 15        # how long a dropped landmark's last position is reused (~0.5s at 30fps)

    def __init__(self):
        self._people: dict[int, dict] = {}
        self._next_id = 0

    @staticmethod
    def _centroid(landmarks):
        pts = [(landmarks[i].x, landmarks[i].y) for i in (11, 12, 23, 24)
               if landmarks[i].visibility >= _POSE_VISIBILITY_THRESHOLD]
        if not pts:
            pts = [(landmarks[0].x, landmarks[0].y)]
        return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))

    def update(self, raw_poses, frame_idx: int, w: int, h: int) -> list[tuple[int, list]]:
        for pid in [pid for pid, p in self._people.items() if frame_idx - p["last_frame"] > self.STALE_FRAMES]:
            del self._people[pid]

        unmatched = set(self._people.keys())
        output = []

        for raw in raw_poses:
            cx, cy = self._centroid(raw)

            best_id, best_dist = None, None
            for pid in unmatched:
                pcx, pcy = self._centroid(self._people[pid]["landmarks"])
                dist = math.dist((cx * w, cy * h), (pcx * w, pcy * h))
                gap = frame_idx - self._people[pid]["last_frame"]
                max_dist = self.MATCH_MAX_DIST_PX + min(gap, self.MATCH_GROWTH_CAP_FRAMES) * self.MATCH_GROWTH_PX_PER_FRAME
                if dist <= max_dist and (best_dist is None or dist < best_dist):
                    best_id, best_dist = pid, dist

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                self._people[best_id] = {
                    "landmarks": list(raw),
                    "ages": [frame_idx] * len(raw),
                    "last_frame": frame_idx,
                }
            else:
                unmatched.discard(best_id)
                person = self._people[best_id]
                merged = []
                for i, lm in enumerate(raw):
                    if lm.visibility >= _POSE_VISIBILITY_THRESHOLD:
                        merged.append(lm)
                        person["ages"][i] = frame_idx
                    elif frame_idx - person["ages"][i] <= self.HOLD_TTL_FRAMES:
                        merged.append(person["landmarks"][i])  # hold last known position through brief flicker
                    else:
                        merged.append(lm)  # stale too long — use raw (invisible) landmark
                person["landmarks"] = merged
                person["last_frame"] = frame_idx

            output.append((best_id, self._people[best_id]["landmarks"]))

        return output


def _region_box(landmarks, indices, w: int, h: int, min_half_px: float):
    """Builds a bounding box tightly around the given landmark indices
    (only the ones currently visible), expands it by HOLDER_BBOX_MARGIN,
    and floors each half-dimension to min_half_px. Returns None if none of
    the given landmarks are currently visible."""
    xs = [landmarks[i].x * w for i in indices if landmarks[i].visibility >= _POSE_VISIBILITY_THRESHOLD]
    ys = [landmarks[i].y * h for i in indices if landmarks[i].visibility >= _POSE_VISIBILITY_THRESHOLD]
    if not xs:
        return None
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half_w = max((x2 - x1) / 2 * HOLDER_BBOX_MARGIN, min_half_px)
    half_h = max((y2 - y1) / 2 * HOLDER_BBOX_MARGIN, min_half_px)
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def _person_hold_boxes(landmarks, w: int, h: int):
    """Builds small "hold region" boxes around the mouth and each hand
    only (not the whole body) for one tracked person. Region size floors
    to a minimum derived from shoulder width, so it scales naturally with
    how close the person is to the camera. Returns a list of 0-3 boxes
    (one per region that currently has a visible landmark)."""
    l_sh, r_sh = landmarks[_SHOULDER_LANDMARKS[0]], landmarks[_SHOULDER_LANDMARKS[1]]
    if l_sh.visibility >= _POSE_VISIBILITY_THRESHOLD and r_sh.visibility >= _POSE_VISIBILITY_THRESHOLD:
        shoulder_w_px = math.hypot((l_sh.x - r_sh.x) * w, (l_sh.y - r_sh.y) * h)
        min_half_px = max(shoulder_w_px * 0.55, _HOLDER_MIN_BOX_PX / 2)
    else:
        min_half_px = _HOLDER_MIN_BOX_PX / 2

    boxes = []
    for indices in _HOLDER_REGION_LANDMARKS.values():
        box = _region_box(landmarks, indices, w, h, min_half_px)
        if box is not None:
            boxes.append(box)
    return boxes


def _pose_person_boxes(tracked_poses, w: int, h: int) -> list[dict]:
    """Returns the mouth/hand hold-region boxes (+ owning person_id) for
    every tracked pose, for the human-held containment check and for
    identity-aware track stitching. A person can contribute up to 3 boxes
    (mouth, left hand, right hand)."""
    boxes = []
    for person_id, landmarks in tracked_poses:
        for bbox in _person_hold_boxes(landmarks, w, h):
            boxes.append({"bbox": bbox, "person_id": person_id})
    return boxes


def _draw_pose_skeletons(frame, tracked_poses) -> None:
    """Draws a reduced Kinect-style skeleton (mouth + both arms/hands +
    both legs/feet only) on frame for every tracked pose."""
    h, w = frame.shape[:2]
    for _person_id, landmarks in tracked_poses:
        pts = {}
        for i in _POSE_KEY_LANDMARKS:
            lm = landmarks[i]
            if lm.visibility >= _POSE_VISIBILITY_THRESHOLD:
                pts[i] = (int(lm.x * w), int(lm.y * h))
        for start, end in _POSE_DRAW_CONNECTIONS:
            if start in pts and end in pts:
                cv2.line(frame, pts[start], pts[end], _POSE_SKELETON_COLOR, 2, cv2.LINE_AA)
        for pt in pts.values():
            cv2.circle(frame, pt, 4, _POSE_JOINT_COLOR, -1, cv2.LINE_AA)


def _bbox_contains_point(bbox, x: float, y: float) -> bool:
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _is_human_held(bbox, person_boxes: list[dict]) -> bool:
    """True if the center of bbox falls inside any person's mouth/hand
    hold-region box."""
    if not person_boxes:
        return False
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    return any(_bbox_contains_point(p["bbox"], cx, cy) for p in person_boxes)


def _nearest_holder_id(bbox, person_boxes: list[dict]):
    """Returns the person_id whose mouth/hand hold-region box contains
    bbox's center — i.e. "whose mouth/hand is this cigarette in" — or None
    if it's not inside anybody's box. If it falls inside more than one
    (overlapping people, or both a hand box and a mouth box from the same
    person), picks whichever box's center is closest. Used to let track
    stitching recognize that two fragmented tracks belong to the same
    physical person, not just a pixel-distance coincidence."""
    if not person_boxes:
        return None
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    containing = [p for p in person_boxes if _bbox_contains_point(p["bbox"], cx, cy)]
    if not containing:
        return None
    if len(containing) == 1:
        return containing[0]["person_id"]

    def box_center(p):
        bx1, by1, bx2, by2 = p["bbox"]
        return ((bx1 + bx2) / 2, (by1 + by2) / 2)

    return min(containing, key=lambda p: math.dist((cx, cy), box_center(p)))["person_id"]


# FP16 inference only helps on CUDA GPUs; it's a no-op (or slower) on CPU.
# Passed as quantize=16 (not the older half=True) since ultralytics now
# treats 'half' as a deprecated alias for the unified 'quantize' kwarg and
# logs a warning every time it's used — quantize=16 is the same FP16
# behavior with no warning. Gated behind CUDA availability so CPU-only
# machines don't request it at all.
_USE_HALF = torch.cuda.is_available()
_QUANTIZE = 16 if _USE_HALF else None

_model = None

# model.track(persist=True) stores tracker state (track IDs, Kalman filters,
# etc.) directly on the shared model/predictor object. Since app.py runs
# each video job in its own background thread but all jobs reuse the same
# get_model() singleton, two videos processed back-to-back or concurrently
# could otherwise corrupt or leak track IDs between unrelated videos. This
# lock serializes video tracking runs so only one video's tracker state is
# live on the model at a time.
_track_lock = threading.Lock()

def get_model():
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def run_on_image(image_path: str, output_path: str, require_human: bool = False) -> dict:
    model = get_model()
    results = model.predict(
        source=image_path,
        conf=CONF_THRESHOLD,
        imgsz=IMG_SIZE,
        iou=NMS_IOU,
        save=False,
        verbose=False,
    )[0]

    img = cv2.imread(image_path)
    # A single image has no frames to track identity across, so each raw
    # detected pose just gets its own positional id (0, 1, 2, ...) to match
    # the (person_id, landmarks) shape the shared pose helpers expect.
    poses = list(enumerate(_detect_poses_image(img))) if require_human else []
    if require_human:
        _draw_pose_skeletons(img, poses)
    person_boxes = _pose_person_boxes(poses, img.shape[1], img.shape[0]) if require_human else None
    detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if require_human and not _is_human_held((x1, y1, x2, y2), person_boxes):
            continue

        detections.append({"label": label, "confidence": round(conf, 3), "bbox": [x1, y1, x2, y2]})
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 60, 255), 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 60, 255), -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(output_path, img)
    return {"detections": detections, "count": len(detections)}


def _stitch_track_ids(detections_by_frame: list[list[dict]]) -> dict[int, int]:
    """Merges ByteTrack IDs that likely belong to the same physical object
    reacquired after a long gap (occlusion, hand rotating the cigarette out
    of view, briefly leaving frame, etc).

    ByteTrack is purely motion-based (Kalman filter + IoU matching): once a
    track goes unmatched for longer than track_buffer frames it's deleted,
    and when the object reappears it gets a brand new ID with no memory of
    the old one. For a mostly-static camera we can recover a lot of these
    splits cheaply: if a new track's first position is near where a
    recently-ended track was last seen, it's almost certainly the same
    object.

    When require_human pose tracking supplied a "holder_id" (the pose-
    tracked person nearest each detection — see _nearest_holder_id), a
    candidate held by the same tracked person gets a much looser
    distance/gap allowance (STITCH_HOLDER_MAX_DIST_PX/GAP_FRAMES) than a
    plain pixel-distance-only candidate (STITCH_MAX_DIST_PX) — same-person
    identity is a much stronger "same object" signal than raw pixel
    distance alone.

    Candidate selection picks the chronologically EARLIEST qualifying
    successor (not the spatially closest one across all future fragments).
    Picking by raw distance alone can "leapfrog" over the true immediate
    continuation — e.g. fragment C occurs right after A ends, and D occurs
    later but happens to land a few pixels closer to A's last position than
    C did — merging A directly into D and stranding C as its own separate,
    never-reconnected chain (this was observed causing 1 real held
    cigarette to fragment into multiple final "unique instances": A->D
    leaves C isolated, so C's own forward search then merges with whatever
    comes after it, producing two disconnected chains for what was really
    one continuous object). Choosing the earliest-starting qualifying
    candidate instead preserves the natural reacquisition order and keeps
    the full chain connected.

    We never consider a candidate that's simultaneously visible alongside
    another live track sharing its merge group, so two genuinely distinct,
    co-occurring cigarettes are never collapsed into one ID.

    Returns a dict mapping every original track_id to its merged (root) id.
    """
    by_id: dict[int, list[tuple[int, float, float, object]]] = {}
    for frame_idx, dets in enumerate(detections_by_frame):
        for d in dets:
            tid = d["track_id"]
            if tid is None:
                continue
            by_id.setdefault(tid, []).append((frame_idx, d["cx"], d["cy"], d.get("holder_id")))

    tracks = []
    for tid, pts in by_id.items():
        pts.sort()
        tracks.append({
            "id": tid, "start": pts[0][0], "end": pts[-1][0],
            "start_pos": pts[0][1:3], "end_pos": pts[-1][1:3],
            "start_holder": pts[0][3], "end_holder": pts[-1][3],
        })
    tracks.sort(key=lambda t: t["start"])

    remap = {t["id"]: t["id"] for t in tracks}

    def find(x):
        while remap[x] != x:
            x = remap[x]
        return x

    used_as_successor = set()
    for a in tracks:
        candidates = [
            b for b in tracks
            if b["start"] > a["end"]
            and b["id"] not in used_as_successor
            and find(b["id"]) != find(a["id"])
        ]

        qualifying = []
        for b in candidates:
            dist = math.dist(a["end_pos"], b["start_pos"])
            is_same_holder = (
                a["end_holder"] is not None
                and a["end_holder"] == b["start_holder"]
                and b["start"] - a["end"] <= STITCH_HOLDER_MAX_GAP_FRAMES
            )
            threshold = STITCH_HOLDER_MAX_DIST_PX if is_same_holder else STITCH_MAX_DIST_PX
            if dist <= threshold:
                qualifying.append(b)

        if qualifying:
            b = min(qualifying, key=lambda b: (b["start"], math.dist(a["end_pos"], b["start_pos"])))
            remap[find(b["id"])] = find(a["id"])
            used_as_successor.add(b["id"])

    return {tid: find(tid) for tid in by_id}


def run_on_video(video_path: str, output_path: str, on_progress=None, require_human: bool = False) -> dict:
    """Runs detection + ByteTrack tracking over every frame of a video.

    Using model.track(..., persist=True) instead of per-frame model.predict()
    assigns a stable track ID to each detected object across frames. That
    lets us count unique smoking instances (e.g. "3 distinct cigarettes
    seen") instead of just "frames containing a detection", which wildly
    over-counts a single sustained violation as soon as you measure it
    frame-by-frame. Tracking needs consecutive frames to stay stable, so
    the old frame `stride` skip-and-reuse approach is dropped — every frame
    is now run through inference (affordable since GPU inference here is
    ~0.14s/frame vs ~0.6s/frame on CPU).

    This runs in two passes. Pass 1 does the actual (expensive, GPU) model
    inference + tracking and caches only lightweight per-frame box data
    (coordinates/confidence/track id) — not full frames. Pass 2 then
    resolves any ID fragmentation via _stitch_track_ids() and re-reads the
    raw video (cheap, no GPU) to draw final boxes with the corrected IDs.
    This avoids paying for inference twice just to fix up track IDs.

    If require_human is True, a cigarette detection only counts if it's
    near a detected person's wrist/mouth (see _is_human_held) — this filters
    out false positives on cigarette-shaped objects with nobody holding
    them (e.g. a pipe lying down), at the cost of extra per-frame pose
    estimation work. It also feeds pose-tracked person identity into
    _stitch_track_ids so fragmented tracks held by the same person merge
    correctly even when they reappear too far apart for pure pixel-distance
    stitching to catch.

    on_progress, if given, is called as on_progress(percent, frame_idx, total_frames)
    after every processed frame, where percent (0-100) is the true overall
    progress across both passes (not a raw frame count), so callers can
    report real progress (e.g. to a polling status endpoint) instead of
    guessing.
    """
    model = get_model()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_detections = 0
    frames_with_smoking = 0
    frame_idx = 0
    detections_by_frame: list[list[dict]] = []
    poses_by_frame: list[list] = []

    # A fresh VIDEO-mode landmarker + tracker per run: VIDEO mode (unlike
    # IMAGE mode, used for single-image requests) lets MediaPipe use the
    # previous frame's region as a prior for the next one instead of
    # re-detecting from scratch every frame, which is what actually cuts
    # down most of the frame-to-frame flicker/jitter — on top of that,
    # _PoseTracker smooths brief visibility dropouts and assigns stable
    # person ids across frames (see _PoseTracker docstring).
    video_landmarker = _new_video_pose_landmarker() if require_human else None
    pose_tracker = _PoseTracker() if require_human else None
    last_ts_ms = -1

    try:
        with _track_lock:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # persist=False on the first frame forces ultralytics to build a
                # brand-new tracker for this run instead of reusing state left
                # over from a previous video on the shared model singleton.
                # conf=TRACK_CONF_THRESHOLD (not CONF_THRESHOLD) deliberately
                # lets weak (0.1-0.25) detections through to ByteTrack so its
                # own track_low_thresh can keep an existing track alive
                # through a confidence dip — see TRACK_CONF_THRESHOLD comment.
                results = model.track(
                    source=frame,
                    conf=TRACK_CONF_THRESHOLD,
                    imgsz=IMG_SIZE,
                    iou=NMS_IOU,
                    persist=frame_idx > 0,
                    tracker=TRACKER_CONFIG,
                    save=False,
                    verbose=False,
                    quantize=_QUANTIZE,
                )[0]

                boxes = results.boxes

                if require_human:
                    ts_ms = int(frame_idx * 1000 / fps)
                    if ts_ms <= last_ts_ms:
                        ts_ms = last_ts_ms + 1
                    last_ts_ms = ts_ms
                    raw_poses = _detect_poses_video(video_landmarker, frame, ts_ms)
                    poses = pose_tracker.update(raw_poses, frame_idx, w, h)
                    person_boxes = _pose_person_boxes(poses, w, h)
                else:
                    poses = []
                    person_boxes = None
                poses_by_frame.append(poses)

                frame_dets = []
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = model.names[cls_id]
                    track_id = int(box.id[0]) if box.id is not None else None

                    # Lowering conf= above (TRACK_CONF_THRESHOLD) intentionally
                    # lets ByteTrack see weak boxes so it can keep an existing
                    # track alive through a confidence dip — but a box that
                    # never got assigned to any track (too weak to start a
                    # new one, and not matched to an existing one) is just
                    # noise and shouldn't be drawn/counted.
                    if track_id is None:
                        continue

                    if require_human and not _is_human_held((x1, y1, x2, y2), person_boxes):
                        continue

                    frame_dets.append({
                        "bbox": (x1, y1, x2, y2),
                        "conf": conf,
                        "label": label,
                        "track_id": track_id,
                        "cx": (x1 + x2) / 2,
                        "cy": (y1 + y2) / 2,
                        "holder_id": _nearest_holder_id((x1, y1, x2, y2), person_boxes) if require_human else None,
                    })

                if frame_dets:
                    frames_with_smoking += 1
                    total_detections += len(frame_dets)
                detections_by_frame.append(frame_dets)

                frame_idx += 1
                if on_progress is not None:
                    # Pass 1 is the expensive (GPU) half of the work; report it
                    # as the first 85% of true overall progress, leaving room
                    # for pass 2 below.
                    on_progress(frame_idx / total_frames * 85 if total_frames else 0, frame_idx, total_frames)
    finally:
        if video_landmarker is not None:
            video_landmarker.close()

    cap.release()

    id_remap = _stitch_track_ids(detections_by_frame)
    unique_smokers = len(set(id_remap.values()))

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for frame_idx, frame_dets in enumerate(detections_by_frame):
        ret, frame = cap.read()
        if not ret:
            break

        if require_human:
            _draw_pose_skeletons(frame, poses_by_frame[frame_idx])

        for det in frame_dets:
            x1, y1, x2, y2 = det["bbox"]
            track_id = det["track_id"]
            merged_id = id_remap.get(track_id, track_id) if track_id is not None else None

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 60, 255), 2)
            id_part = f"#{merged_id} " if merged_id is not None else ""
            text = f"{id_part}{det['label']} {det['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 60, 255), -1)
            cv2.putText(frame, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        if on_progress is not None:
            # Pass 2 is the cheap (no-GPU) remaining 15% of true overall progress.
            pct = 85 + (frame_idx + 1) / max(total_frames, 1) * 15
            on_progress(pct, frame_idx + 1, total_frames)

    cap.release()
    out.release()

    return {
        "total_frames": total_frames,
        "frames_with_smoking": frames_with_smoking,
        "total_detections": total_detections,
        "unique_smokers": unique_smokers,
    }
