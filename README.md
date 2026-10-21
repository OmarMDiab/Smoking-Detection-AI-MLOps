# Smoking Detection — AI MLOps Project

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Proof%20of%20Concept-orange)

A computer-vision proof of concept for **detecting cigarettes in CCTV footage from petroleum sites and other industrial environments**, built around a custom-trained **YOLO26s** detector, **ByteTrack** multi-object tracking, and optional **MediaPipe Pose** gating.

## Overview

Smoking near flammable materials, fuel, or process equipment is a serious safety hazard on petroleum and industrial sites. Manually reviewing hours of CCTV footage for smoking incidents doesn't scale. This project explores whether a lightweight, real-time object detector can flag cigarettes in camera footage automatically, to help safety engineers prioritize which footage to review.

**This is a proof of concept, not a certified or production-ready safety system.** It detects a _cigarette-shaped object in a frame_ — optionally narrowed to objects held by a detected person — it does not independently verify that a complete "smoking incident" took place, and it has not been validated on real petroleum-site CCTV footage (see [Model Training & Evaluation](#model-training--evaluation)).

The repository contains two parallel implementations of the same detection pipeline, built for different purposes:

|               | Prototype                                            | Deployment                                                                           |
| ------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------ |
| Purpose       | AI pipeline development & experimentation            | Running the system as a service                                                      |
| Process model | Single Flask process; model code imported in-process | Two containers: a thin `Flask_UI` web app + an `AI_Worker` FastAPI inference service |
| Run with      | `python app.py`                                      | `docker compose up --build`                                                          |

## Repository Documentation

- **[Prototype Documentation](Smoking_Detection_Prototype/README.md)** — the AI/model pipeline: detection, tracking, optional pose gating, training, and how to run everything locally for development.
- **[Deployment Documentation](Smoking_detection_Deployment/README.md)** — the containerized architecture: services, Docker Compose configuration, ports, volumes, and how to operate the deployed app.

## System Architecture

Both variants implement the same conceptual pipeline; they differ only in _where_ the code runs (see the linked docs for the exact process/container diagrams).

![YOLO26s detection, ByteTrack ID, and MediaPipe Pose skeleton overlay on the same frame](Assets/Adding_tracker_and_mediapipe.png)

_Left: raw YOLO26s detection. Right: the same detection with a persistent ByteTrack ID and the optional MediaPipe Pose skeleton overlay used for "require human" gating._

Flow: an image or video (upload / CCTV clip) is detected by YOLO26s → tracked across frames by ByteTrack (video only) → optionally filtered to human-held detections by MediaPipe Pose → returned as annotated output + detection stats through the Flask web UI.

No database, message queue, or external API is used by either variant — detections are returned as JSON/annotated media, and in the Deployment variant, job progress lives in-process (see its [limitations](Smoking_detection_Deployment/README.md#current-limitations-and-future-deployment-improvements)).

## Why These Technologies?

| Technology                                                     | Role in this project                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **[YOLO26s](https://docs.ultralytics.com/models/yolo26/)**     | Custom-trained, single-class (`cigarette`) real-time object detector. Cigarettes are a small-object detection problem (see [below](#the-small-object-detection-challenge)); YOLO26 specifically targets this with a Small-Target-Aware Label Assignment (STAL) training strategy. The "s" (small) scale was chosen for its accuracy/latency balance on the official Ultralytics COCO benchmark — not because it's universally better than every other detector or scale.                            |
| **[ByteTrack](https://github.com/FoundationVision/ByteTrack)** | Associates per-frame cigarette detections into persistent tracks across video, so the app can report "N distinct cigarettes" instead of "N frames with a detection." Tracking alone doesn't eliminate false positives — it only keeps identities consistent across frames.                                                                                                                                                                                                                          |
| **MediaPipe (Pose)**                                           | _Optional_ — when the "require a human" toggle is on, [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) estimates body keypoints, and a detection only counts if it falls inside a hold region around a tracked person's mouth/hands. A `hand_landmarker.task` model also ships in the repo but is **not used by any current code path** — code comments indicate MediaPipe Hands was tried and dropped because it failed on gloved hands. |
| **Flask**                                                      | Serves the web UI and the upload/results API in both variants. In the Deployment variant, `Flask_UI` is a thin proxy only — it contains no model code.                                                                                                                                                                                                                                                                                                                                              |
| **FastAPI** _(Deployment only)_                                | Powers `AI_Worker`'s internal inference API (`/infer/image`, `/infer/video`, `/infer/jobs/{id}`, `/health`), reachable only from `Flask_UI` over the Docker-internal network.                                                                                                                                                                                                                                                                                                                       |
| **OpenCV**                                                     | Frame/video I/O, drawing boxes and pose skeletons, and video encode/decode (`cv2.VideoCapture` / `VideoWriter`).                                                                                                                                                                                                                                                                                                                                                                                    |
| **PyTorch & Ultralytics**                                      | Underlying training/inference runtime for YOLO26s (the `ultralytics` package); FP16 inference is used automatically when a CUDA GPU is available.                                                                                                                                                                                                                                                                                                                                                   |
| **ffmpeg**                                                     | Re-encodes the raw OpenCV video output to a browser-playable H.264/AAC `.mp4`; if it's unavailable, the app falls back to serving the raw file directly instead of failing.                                                                                                                                                                                                                                                                                                                         |
| **Docker & Docker Compose** _(Deployment only)_                | Packages `Flask_UI` and `AI_Worker` as separate services with their own dependencies, a shared data volume, and an isolated internal network.                                                                                                                                                                                                                                                                                                                                                       |

![Official Ultralytics YOLO26 family benchmark: COCO mAP50-95 vs T4 TensorRT FP16 latency, compared against prior YOLO generations and other real-time detectors](Assets/mAP%20comparison%20%28yolo%20models%29.jpg)

_Ultralytics' own published COCO benchmark (not project-specific) — shown here only as context for picking the YOLO26s scale, not as evidence it outperforms every detector on every task._

## Model Training & Evaluation

The model was fine-tuned from the `yolo26s.pt` checkpoint on a single-class (`cigarette`) dataset in [`Yolo26s Training (kaggle).ipynb`](Yolo26s%20Training%20%28kaggle%29.ipynb), run on Kaggle (dual-GPU).

|                     |                                                                                                                                       |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset             | ~12,259 training images / 1,152 validation images, single class `cigarette` (Kaggle dataset `cigarette-detection`, owner `omarmdiab`) |
| Training resolution | 640 × 640                                                                                                                             |
| Epochs              | Up to 60, early-stopped at **epoch 37** (no improvement for 15 epochs)                                                                |
| Fused model         | ~9.5M parameters, 20.8 GFLOPs                                                                                                         |

**Reported validation-set results** (from the training notebook, consistent with the project presentation's rounded figures):

| Metric    | Notebook (precise) | Presentation (rounded) |
| --------- | ------------------ | ---------------------- |
| Precision | 0.822              | 82.5%                  |
| Recall    | 0.794              | 79.0%                  |
| mAP@50    | 0.838              | 83.8%                  |
| mAP@50–95 | 0.495              | 49.5%                  |

- **Precision** is the share of predicted cigarette boxes that were correct; **recall** is the share of actual cigarettes that were found.
- **mAP@50** scores a prediction as correct at a loose 50%-box-overlap threshold; **mAP@50–95** averages that score over stricter thresholds (50–95%) and is a harder, more location-sensitive metric.
- **Model inference latency** was **5.0 ms/image** (plus 0.5 ms preprocess + 0.7 ms postprocess), measured _on the validation set, in the Kaggle training environment_. This is the detector's forward-pass time only — **not** the end-to-end time to process a video, which also includes tracking, optional pose estimation, drawing, and the ffmpeg re-encode step — and it has not been independently re-measured on the deployment target hardware.
- These are validation-set results, not an independently verified production or real-world-footage benchmark. The notebook also runs a secondary evaluation against a folder it labels `images/test` (different results: mAP@50 ≈ 0.78), but an earlier note in the same notebook flags that this split had known data-quality issues when `data.yaml` was authored — treat it as informational rather than a clean held-out test score.

## The Small-Object Detection Challenge

Cigarettes are a difficult target for object detection:

- They occupy very few pixels in typical CCTV framing — the project presentation notes many annotated cigarette boxes fall under roughly 70px. As one corroborating data point, the pose-gating code independently floors its own "hold region" boxes to a 70px minimum for similar reasons (see [`detector.py`](Smoking_Detection_Prototype/Flask_UI/model/detector.py)).
- Hands, fingers, and smoke can partially obscure them.
- Lighting, camera distance, and video/image quality all affect detectability.

**Implemented mitigation:** both app variants run inference at **1280×1280**, higher than the 640×640 training resolution, to give small objects more pixels at inference time.

**Not yet implemented:** re-training the model itself at a higher resolution (e.g. 960) remains future work (see [Limitations](#limitations--future-work)).

## Repository Structure

```
.
├── Assets/                                # Presentation slides, diagrams, and demo media
├── Smoking_Detection_Prototype/           # Single-process Flask app (dev/experimentation)
│   ├── Flask_UI/
│   │   ├── app.py
│   │   ├── model/                         # detector.py, bytetrack_custom.yaml, mediapipe_models/
│   │   ├── static/, templates/
│   │   └── requirements.txt
│   └── Model/weights/best.pt
├── Smoking_detection_Deployment/          # Dockerized Flask_UI + AI_Worker services
│   ├── docker-compose.yaml
│   ├── AI_Worker/                         # FastAPI inference service
│   ├── Flask_UI/                          # Thin web UI / API proxy
│   └── Model/weights/best.pt
├── Testing_Samples/                       # Sample images/videos for manual testing
│   ├── Images/, Vidoes/
├── Yolo26s Training (kaggle).ipynb        # Training notebook
├── LICENSE                                # MIT
└── .gitignore
```

## Quick Start

Both variants already include the trained weights (`Model/weights/best.pt`), so no separate model download is required.

**Prototype** (local, single process):

```bash
cd Smoking_Detection_Prototype/Flask_UI
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000.

**Deployment** (Docker Compose, two services):

```bash
cd Smoking_detection_Deployment
docker compose up --build
```

Open http://localhost:5000. GPU acceleration requires an NVIDIA GPU with the NVIDIA Container Toolkit configured — see the [Deployment docs](Smoking_detection_Deployment/README.md#prerequisites).

Full setup, configuration, and troubleshooting details are in the linked [Prototype](Smoking_Detection_Prototype/README.md) and [Deployment](Smoking_detection_Deployment/README.md) docs.

## Limitations & Future Work

**Implemented today:** single-class cigarette detection, ByteTrack-based tracking with custom ID-stitching across occlusions, optional pose-based "held by a human" filtering, 1280px inference resolution.

**Planned / not yet implemented:**

- Evaluate larger model variants (YOLO26m, YOLO26l) for an accuracy/latency trade-off better suited to the deployment target.
- Investigate higher-resolution **training** (e.g. 960px) — only inference currently runs above the training resolution.
- Improve precision and reduce false-positive alerts (e.g. on cigarette-shaped objects like pipes, pens, or lighters).
- Investigate temporal-consistency requirements (e.g. minimum sustained frames) before raising an alert — the current pipeline counts a tracked instance as soon as ByteTrack confirms it.
- Evaluate performance on representative industrial/petroleum-site CCTV footage — training/validation data has not been shown to be representative of real deployment cameras.
- Expose `AI_Worker` to other external services, as already noted as a future direction on the deployment architecture diagram.

## License

[MIT License](LICENSE), Copyright (c) 2026 Omar Diab.

## Acknowledgments & References

This project fine-tunes and integrates several third-party models and libraries; it does not claim to have developed them:

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) ([paper](https://arxiv.org/abs/2606.03748)) — base detector architecture and training framework ([ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)).
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) ([paper](https://arxiv.org/abs/2110.06864)) — multi-object tracking algorithm.
- [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) — body pose estimation.
- [PyTorch](https://pytorch.org/), [Flask](https://flask.palletsprojects.com/), [FastAPI](https://fastapi.tiangolo.com/), [OpenCV](https://opencv.org/), [Docker Compose](https://docs.docker.com/compose/).
- Training dataset: a single-class `cigarette-detection` dataset (Kaggle, owner `omarmdiab`), referenced by its Kaggle input path in the training notebook.
