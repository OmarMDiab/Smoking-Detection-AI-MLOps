# Smoking Detection — Prototype

Single-process Flask app running the full pipeline in-process — the dev/experimentation variant. See the [root](../README.md) and [Deployment](../Smoking_detection_Deployment/README.md) docs.

## Pipeline

![YOLO26s + ByteTrack + MediaPipe Pose pipeline](../Assets/Adding_tracker_and_mediapipe.png)

1. **Upload** — `app.py`'s `/api/detect`.
2. **Detect** — YOLO26s `predict()`/`track()`, `imgsz=1280`, `iou=0.5`.
3. **Track** (video) — ByteTrack + a custom `_stitch_track_ids()` pass for occlusion gaps.
4. **Pose gate** (optional) — `require_human` keeps only detections near a person's mouth/hand.
5. **Draw**, ffmpeg re-encode (video), save to `results/`.

## Model

|                   |                                                  |
| ----------------- | ------------------------------------------------ |
| Weights           | [`Model/weights/best.pt`](Model/weights/best.pt) |
| Train / infer res | 640×640 / **1280×1280**                          |
| Confidence        | 0.25 image, 0.1 video→ByteTrack                  |
| NMS IoU           | 0.5                                              |
| Classes           | 1 — `cigarette`                                  |

## ByteTrack & MediaPipe

Custom [`bytetrack_custom.yaml`](Flask_UI/model/trackers/bytetrack_custom.yaml): `track_high_thresh=0.3`, `track_low_thresh=0.1`, `new_track_thresh=0.3`, `track_buffer=150` (~6s), `match_thresh=0.9` — tuned for a small object that disappears behind fingers for a while.

MediaPipe **Pose only** (`pose_landmarker_lite.task`), toggled by the UI's "Only count cigarettes held by a person" checkbox. Pose replaced Hands (which failed on gloved hands); `hand_landmarker.task` ships but is unused.

## Dataset and Training

![Dataset overview](../Assets/Dataset_overview.png)

[cigarette detection](https://platform.ultralytics.com/ahmed-aboubakr/datasets/cigarette-detection1) — 14,309 images, single class `cigarette`: 12,259 train / 1,152 val / 898 test. `yolo26s.pt` fine-tuned 60 epochs @ 640×640, early-stopped at epoch 37. Metrics: see the [root README](../README.md#model-training--evaluation).

## Run It

```bash
cd Smoking_Detection_Prototype/Flask_UI
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. Weights included; `bin/ffmpeg.exe` isn't (falls back to `PATH`). 500MB upload limit.

## Outputs

- Image: `results/<uid>_result.<ext>` + JSON `{detections, count}`.
- Video: `results/<uid>_result.mp4` (+ `_web.mp4` if ffmpeg succeeds) + JSON `{total_frames, frames_with_smoking, total_detections, unique_smokers}`.
- `unique_smokers` = stitched ByteTrack track count, not verified unique people.

## Limitations

- Small, occlusion-prone target; false positives possible on pens/lighters/tools.
- No temporal-consistency gate — a single confirmed track counts.
- Not evaluated on real petroleum/industrial footage; no automated tests.
