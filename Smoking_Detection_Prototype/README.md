# Smoking Detection — Prototype

Single-process Flask app running the full detection pipeline in-process — the development/experimentation variant. See the [root README](../README.md) for the project and the [Deployment docs](../Smoking_detection_Deployment/README.md) for the containerized variant.

## Prototype Purpose

One Flask process serves the UI, receives uploads, and calls the detection/tracking/pose code directly in Python — no network hop, no worker service. Edit [`model/detector.py`](Flask_UI/model/detector.py) and restart to see changes immediately.

## AI Pipeline

![YOLO26s detection, ByteTrack ID, and MediaPipe Pose skeleton overlay on the same frame](../Assets/Adding_tracker_and_mediapipe.png)

_Raw YOLO26s detection (left) → ByteTrack ID + optional MediaPipe Pose skeleton (right)._

Stages, in order (see [`model/detector.py`](Flask_UI/model/detector.py)):

1. **Upload** — `app.py`'s `/api/detect` receives an image/video.
2. **Detect** — YOLO26s `predict()` (image) or `track()` (video), `imgsz=1280`, `iou=0.5`.
3. **Track** (video only) — ByteTrack assigns IDs; a custom `_stitch_track_ids()` pass merges IDs split by an occlusion gap.
4. **Pose gate** (optional) — if `require_human`, keep only detections near a tracked person's mouth/hand; draw a reduced skeleton.
5. **Draw** boxes (and skeleton, if enabled).
6. **Video only** — ffmpeg re-encodes to a browser-playable `.mp4`.
7. **Output** — saved to `results/`, stats returned as JSON.

Images skip ByteTrack entirely — there's nothing to track across a single frame.

## Model Information

|                          |                                                                                                                                                                       |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Architecture             | YOLO26s (Ultralytics), fine-tuned, single class `cigarette`                                                                                                           |
| Weight file              | [`Model/weights/best.pt`](../Smoking_Detection_Prototype/Model/weights/best.pt) (already included in the repo, relative to the `Smoking_Detection_Prototype/` folder) |
| Path resolution          | Hardcoded in `detector.py`: `<this file>/../../../Model/weights/best.pt`, i.e. `Smoking_Detection_Prototype/Model/weights/best.pt`                                    |
| Training resolution      | 640×640 (see the [training notebook](../Yolo26s%20Training%20%28kaggle%29.ipynb))                                                                                     |
| **Inference resolution** | **1280×1280** (`IMG_SIZE` in `detector.py`) — higher than the training resolution, to give the small `cigarette` class more pixels                                    |
| Confidence threshold     | `0.25` for single images (`CONF_THRESHOLD`); `0.1` for video frames fed into ByteTrack (`TRACK_CONF_THRESHOLD`, see [below](#bytetrack-and-mediapipe))                |
| NMS IoU threshold        | `0.5` (`NMS_IOU`) — tightened from Ultralytics' default `0.7` to collapse duplicate overlapping boxes observed on real footage                                        |
| Precision                | FP16 (`quantize=16`) automatically when CUDA is available, FP32 otherwise                                                                                             |
| Classes                  | Single class: `cigarette` (`nc: 1` in the training `data.yaml`)                                                                                                       |
| Loading                  | `ultralytics.YOLO(MODEL_PATH)`, lazily instantiated once as a module-level singleton (`get_model()`)                                                                  |

## ByteTrack and MediaPipe

**ByteTrack** (video only) uses a custom config, [`model/trackers/bytetrack_custom.yaml`](Flask_UI/model/trackers/bytetrack_custom.yaml), tuned for a single small, fast-moving object that frequently disappears behind fingers:

| Parameter           | Value             | Why                                                                                       |
| ------------------- | ----------------- | ----------------------------------------------------------------------------------------- |
| `track_high_thresh` | 0.3               | Min confidence for first-pass association / starting a track                              |
| `track_low_thresh`  | 0.1               | Min confidence to keep an already-confirmed track alive through a confidence dip          |
| `new_track_thresh`  | 0.3               | Min confidence to spawn a brand-new track ID                                              |
| `track_buffer`      | 150 (~6s @ 25fps) | Keeps a lost track alive through long hand-occlusion/rotation gaps                        |
| `match_thresh`      | 0.9               | Tightened from the default 0.8; measured to cut spurious ID switches on a validation clip |

A custom `_stitch_track_ids()` pass also merges IDs likely belonging to the same cigarette after an occlusion gap, using position and (if pose is on) shared tracked-person identity.

**MediaPipe** is optional, **Pose only** (`pose_landmarker_lite.task`), toggled via the UI's **"Only count cigarettes held by a person"** checkbox (`require_human`):

- **On:** keeps only detections inside a tracked person's mouth/hand region; draws a reduced skeleton.
- **Off** (default): plain detection/tracking, no pose estimation.
- MediaPipe Hands was tried and dropped — it failed on gloved hands, so Pose (body silhouette) is used instead.
- `mediapipe_models/hand_landmarker.task` ships in the repo but **isn't used by any current code path** — leftover from that abandoned approach.

## Dataset and Training

Trained entirely in [`Yolo26s Training (kaggle).ipynb`](../Yolo26s%20Training%20%28kaggle%29.ipynb) on Kaggle (not in this folder).

![Dataset overview: single-class cigarette dataset, train/val/test split](../Assets/Dataset_overview.png)

- **Dataset:** [cigarette detection](https://platform.ultralytics.com/ahmed-aboubakr/datasets/cigarette-detection1) (Ultralytics Platform), single class `cigarette`, 14,309 images — 12,259 train / 1,152 val / 898 test.
- **Training:** `yolo26s.pt` base, up to 60 epochs @ 640×640, batch 32, 2 GPUs. Early-stopped at **epoch 37**.
- **Validation metrics:** Precision 0.822, Recall 0.794, mAP@50 0.838, mAP@50–95 0.495 — see the [root README](../README.md#model-training--evaluation) for the full breakdown and caveats.

## Installation and Execution

```bash
cd Smoking_Detection_Prototype/Flask_UI
pip install -r requirements.txt       # flask, ultralytics, opencv-python, numpy, mediapipe
python app.py
```

Open http://localhost:5000.

- **Python:** not pinned here; 3.11 recommended (matches Deployment's Docker images).
- **Weights:** `Model/weights/best.pt` already included — no download needed.
- **ffmpeg:** `bin/ffmpeg.exe` is **not in the repo** (size). On Windows, `app.py` falls back to a conda env's ffmpeg, then `PATH`; if none found, video still works but serves the raw, non-browser-optimized file.
- **Pose gating:** toggle the "Only count cigarettes held by a person" checkbox in the UI — no CLI flag.
- **Upload limit:** 500 MB (`MAX_CONTENT_LENGTH`).

## Configuration Reference

Python constants in [`model/detector.py`](Flask_UI/model/detector.py) — no env var/CLI override; edit source + restart.

| Constant                                                     | Value     | Purpose                                                                                               |
| ------------------------------------------------------------ | --------- | ----------------------------------------------------------------------------------------------------- |
| `CONF_THRESHOLD`                                             | 0.25      | Confidence threshold for single-image detection                                                       |
| `TRACK_CONF_THRESHOLD`                                       | 0.1       | Confidence threshold fed into ByteTrack for video (lower, so ByteTrack's own two-tier logic can work) |
| `IMG_SIZE`                                                   | 1280      | Inference resolution for both images and video                                                        |
| `NMS_IOU`                                                    | 0.5       | NMS IoU threshold for both `predict()` and `track()`                                                  |
| `STITCH_MAX_DIST_PX`                                         | 65        | Max pixel distance to merge two track fragments without pose data                                     |
| `STITCH_HOLDER_MAX_DIST_PX` / `STITCH_HOLDER_MAX_GAP_FRAMES` | 400 / 150 | Looser merge distance/gap allowed when both fragments were held by the same pose-tracked person       |
| `HOLDER_BBOX_MARGIN` / `_HOLDER_MIN_BOX_PX`                  | 2.2 / 70  | Expansion margin and minimum size (px) of the mouth/hand "hold region" boxes used by `require_human`  |

## Outputs and Troubleshooting

- **Image:** `results/<uid>_result.<ext>` + JSON `{"detections": [...], "count": N}`.
- **Video:** `results/<uid>_result.mp4` (raw), plus `<uid>_web.mp4` if the ffmpeg re-encode succeeds. JSON: `{"total_frames", "frames_with_smoking", "total_detections", "unique_smokers"}`.
  - `unique_smokers` = distinct **stitched ByteTrack tracks**, not an independently verified count of people.
- **Model file not found** — check `Model/weights/best.pt` exists.
- **Video won't play / no sound** — ffmpeg wasn't found; raw `mp4v` was served instead.
- **Pose gating misses a smoker** — needs a reasonably visible mouth/hand; heavy occlusion or odd angles can cause misses.

## Development and Testing

No automated test suite (no `pytest`/CI). To validate changes manually: run the app, upload samples from [`../Testing_Samples/Images`](../Testing_Samples/Images) / [`../Testing_Samples/Vidoes`](../Testing_Samples/Vidoes), and visually check detections with and without "require human" on.

## Limitations and Next Steps

- Small, easily-occluded target — quality drops with distance, poor lighting, occlusion.
- False positives possible on other small elongated objects (pens, lighters, tools).
- No temporal-consistency gate — a single confirmed track counts, however brief.
- Not evaluated on real petroleum/industrial CCTV footage.
- Higher-resolution **training** (e.g. 960px) remains a candidate improvement, separate from the 1280px inference already in use.
