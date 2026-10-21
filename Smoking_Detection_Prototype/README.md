# Smoking Detection — Prototype

Single-process Flask application implementing the full detection pipeline in-process. This is the development/experimentation variant of the project — see the [root README](../README.md) for the overall project and the [Deployment docs](../Smoking_detection_Deployment/README.md) for the containerized variant.

## Prototype Purpose

This directory is a self-contained reference implementation: one Flask process that serves the web UI, receives uploads, and calls the detection/tracking/pose code directly in Python — no network hop, no separate worker service. It's the fastest way to iterate on the model pipeline itself: edit [`model/detector.py`](Flask_UI/model/detector.py) and restart the Flask dev server to see the effect immediately.

## AI Pipeline

![YOLO26s detection, ByteTrack ID, and MediaPipe Pose skeleton overlay on the same frame](../Assets/Adding_tracker_and_mediapipe.png)

_Left: raw YOLO26s detection. Right: the same detection with a persistent ByteTrack ID and the optional MediaPipe Pose skeleton overlay drawn when `require_human` is enabled._

Processing stages, in order (see [`model/detector.py`](Flask_UI/model/detector.py)):

1. **Upload** — `app.py`'s `/api/detect` receives an image or video file.
2. **Detection** — `model/detector.py` runs YOLO26s (`predict()` for images, `track()` for video) at `imgsz=1280`, `iou=0.5`.
3. **Tracking** (video only) — ByteTrack assigns persistent IDs across frames, followed by a custom `_stitch_track_ids()` pass that merges IDs likely belonging to the same physical cigarette after an occlusion gap.
4. **Optional MediaPipe Pose gating** — if `require_human` is enabled, only detections falling inside a tracked person's mouth/hand region are kept, and a reduced pose skeleton is drawn.
5. **Visualization** — bounding boxes (and the pose skeleton, if enabled) are drawn on the frame(s).
6. **Video only: ffmpeg re-encode** — the raw OpenCV output is re-encoded to a browser-playable `.mp4` (falls back to serving the raw file if ffmpeg is unavailable).
7. **Output** — the annotated image/video is saved to `results/` and detection stats are returned as JSON.

Single images never go through ByteTrack (there's nothing to track across frames); only video jobs run the tracking, ID-stitching, and ffmpeg steps.

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

On top of ByteTrack itself, `detector.py` runs a custom `_stitch_track_ids()` post-processing pass that merges track IDs likely belonging to the same physical cigarette after a tracking gap (e.g. reacquired after being hidden by a rotating hand), using positional proximity and — when pose tracking is enabled — shared tracked-person identity as a stronger merge signal.

**MediaPipe** is optional and is **Pose only** (`pose_landmarker_lite.task`), toggled per-request via the `require_human` flag (exposed in the UI as the **"Only count cigarettes held by a person"** checkbox):

- When **on**: MediaPipe Pose estimates body keypoints; a cigarette detection only counts if its center falls inside a small region around a tracked person's mouth or either hand. A reduced "Kinect-style" skeleton is also drawn for visualization.
- When **off** (default): plain YOLO26s detection/tracking, with no pose estimation at all.
- Code comments explain the Pose-over-Hands choice explicitly: MediaPipe Hands was tried and **failed completely on gloved hands** (0 detections in testing), so Pose (which infers from body silhouette) is used instead.
- **Note:** `mediapipe_models/hand_landmarker.task` ships in this repository but is **not referenced by any current code path** — it appears to be left over from that earlier, abandoned approach.

## Dataset and Training

Training happened entirely in [`Yolo26s Training (kaggle).ipynb`](../Yolo26s%20Training%20%28kaggle%29.ipynb) (run on Kaggle, not in this folder).

![Dataset overview: single-class cigarette dataset, train/val/test split](../Assets/Dataset_overview.png)

Summary:

- **Dataset:** single class `cigarette`, 12,259 training images (120 background/negative images included) and 1,152 validation images (88 backgrounds), sourced from a Kaggle dataset (`cigarette-detection`, owner `omarmdiab`) and copied to local SSD storage for faster I/O during training.
- **Base checkpoint:** `yolo26s.pt`, fine-tuned for up to 60 epochs at 640×640, batch size 32, on 2 GPUs. Training early-stopped at **epoch 37** (no improvement for 15 epochs).
- **Reported validation metrics:** Precision 0.822, Recall 0.794, mAP@50 0.838, mAP@50–95 0.495 (consistent with the project presentation's rounded 82.5% / 79.0% / 83.8% / 49.5%). Inference speed measured at 5.0 ms/image (model forward pass only) in that same Kaggle notebook environment — see the [root README](../README.md#model-training--evaluation) for the full breakdown and caveats.

Do not add dataset images, label files, or model weights directly as documentation examples — refer to the dataset by name/owner and to the weight file by path instead, as done above.

## Installation and Execution

```bash
cd Smoking_Detection_Prototype/Flask_UI
pip install -r requirements.txt       # flask, ultralytics, opencv-python, numpy, mediapipe
python app.py
```

Then open http://localhost:5000.

- **Python version:** not independently pinned in this variant; 3.11 is recommended for parity with the Deployment variant's Docker images.
- **Model weights:** `../Model/weights/best.pt` is already included in the repository — no download step needed.
- **ffmpeg:** `bin/ffmpeg.exe` is **not included in the repository** (excluded for its size). This only affects Windows: `app.py` looks for `bin/ffmpeg.exe` first, then a conda environment's `ffmpeg.exe`, then `ffmpeg` on `PATH`. If none is found, video results are still produced — just served as the raw OpenCV-encoded `.mp4` instead of a browser-optimized re-encode (which may not play in all browsers).
- **Enabling pose gating:** there's no separate CLI flag — check the "Only count cigarettes held by a person" box in the web UI before running a scan (sent as a `require_human` form field).
- **Upload limit:** 500 MB (`MAX_CONTENT_LENGTH` in `app.py`).
- This repo's `.vscode/settings.json` defaults to a conda environment manager if you prefer conda over `venv`/`pip` directly — no `environment.yml` is provided, so `requirements.txt` is still the source of truth for dependencies either way.

## Configuration Reference

These are Python constants in [`model/detector.py`](Flask_UI/model/detector.py) — there is no environment-variable or CLI override in the Prototype variant; change the source and restart to adjust them.

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

- **Image result:** `results/<uid>_result.<ext>` plus a JSON response `{"detections": [...], "count": N}`, each detection shaped as `{"label", "confidence", "bbox": [x1, y1, x2, y2]}`.
- **Video result:** `results/<uid>_result.mp4` (raw) and, if the ffmpeg re-encode succeeds, `results/<uid>_web.mp4` (browser-playable) — the UI serves whichever one succeeded. JSON response: `{"total_frames", "frames_with_smoking", "total_detections", "unique_smokers"}`.
  - **`unique_smokers` is the number of distinct stitched ByteTrack cigarette tracks**, not an independently verified count of unique people — for example, two people who are never tracked simultaneously but whose cigarettes get merged by the ID-stitching heuristic would undercount.
- **Common issues:**
  - _Model file not found_ — ensure `Model/weights/best.pt` exists relative to the project root (already included in this repo).
  - _Video plays with no sound / won't play in-browser_ — ffmpeg wasn't found; the raw `mp4v` file was served instead (see [Installation](#installation-and-execution)).
  - _Pose gating misses a smoker_ — pose estimation needs a reasonably visible mouth/hand; heavy occlusion, extreme angles, or multiple overlapping people can cause misses.

## Development and Testing

There is no automated test suite in this repository (no `pytest`/`unittest` files, no CI workflow). To validate changes manually:

1. Run the app locally (`python app.py`).
2. Upload the sample media from [`../Testing_Samples/Images`](../Testing_Samples/Images) and [`../Testing_Samples/Vidoes`](../Testing_Samples/Vidoes) through the UI.
3. Visually inspect the annotated output and the returned detection stats, with and without the "require human" toggle.

Adding an automated test suite is a manual, not yet implemented, suggestion — there is currently no tooling for it in this repo.

## Limitations and Next Steps

- Cigarettes remain a small, easily-occluded target; detection quality degrades with distance, poor lighting, and partial occlusion by hands/smoke.
- False positives on other small, elongated objects (pens, lighters, tools) are possible — the `require_human` toggle reduces but does not eliminate this.
- No temporal-consistency/minimum-duration gate exists before a tracked cigarette counts in `unique_smokers` — a track that ByteTrack confirms still counts even if briefly seen.
- The model has not been evaluated on real petroleum/industrial-site CCTV footage.
- Higher-resolution **training** (e.g. 960px, vs. the current 640px) is a candidate improvement, distinct from the 1280px inference resolution already in use.
