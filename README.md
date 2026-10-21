# Smoking Detection — AI MLOps Project

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Proof%20of%20Concept-orange)

A computer-vision proof of concept for **detecting cigarettes in CCTV footage from petroleum sites and other industrial environments**, built around a custom-trained **YOLO26s** detector, **ByteTrack** multi-object tracking, and optional **MediaPipe Pose** gating.

## Overview

Smoking near fuel or process equipment is a safety hazard on industrial sites, and manually reviewing CCTV footage doesn't scale. This is a proof of concept for flagging cigarettes in footage automatically, to help safety engineers prioritize what to review.

**Not a certified or production-ready safety system.** It detects a _cigarette-shaped object in a frame_ (optionally only when held by a detected person) — it does not confirm a full "smoking incident," and hasn't been validated on real petroleum-site footage (see [Model Training & Evaluation](#model-training--evaluation)).

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

Same pipeline in both variants — only _where_ the code runs differs (see the linked docs for process/container diagrams).

![YOLO26s detection, ByteTrack ID, and MediaPipe Pose skeleton overlay on the same frame](Assets/Adding_tracker_and_mediapipe.png)

_Raw YOLO26s detection (left) → ByteTrack ID + optional MediaPipe Pose skeleton (right)._

Image/video in → YOLO26s detects → ByteTrack tracks (video only) → optional MediaPipe Pose filters to human-held detections → annotated output + stats out via the Flask web UI. No database, queue, or external API; the Deployment variant keeps job progress in-process (see its [limitations](Smoking_detection_Deployment/README.md#current-limitations-and-future-deployment-improvements)).

## Why These Technologies?

| Technology                                                     | Role in this project                                                                                                                                                                                                                                                                                                           |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **[YOLO26s](https://docs.ultralytics.com/models/yolo26/)**     | Single-class `cigarette` detector. The "s" scale balances accuracy/latency on Ultralytics' COCO benchmark — not a claim it beats every detector (see [small-object challenge](#the-small-object-detection-challenge)).                                                                                                         |
| **[ByteTrack](https://github.com/FoundationVision/ByteTrack)** | Tracks cigarettes across video frames so one isn't double-counted as "N frames with a detection." Doesn't eliminate false positives by itself — only keeps identities consistent.                                                                                                                                              |
| **MediaPipe (Pose)**                                           | _Optional_ "require human" gate — [Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) keeps only detections near a tracked person's mouth/hands. A `hand_landmarker.task` ships too but is **unused by any code path** (Hands was tried and dropped — it failed on gloved hands). |
| **Flask**                                                      | Web UI + upload/results API in both variants. In Deployment, `Flask_UI` is a thin proxy with no model code.                                                                                                                                                                                                                    |
| **FastAPI** _(Deployment only)_                                | Powers `AI_Worker`'s internal inference API, reachable only from `Flask_UI` over the Docker-internal network.                                                                                                                                                                                                                  |
| **OpenCV**                                                     | Frame/video I/O, drawing, and encode/decode.                                                                                                                                                                                                                                                                                   |
| **PyTorch & Ultralytics**                                      | Training/inference runtime for YOLO26s; FP16 automatically on CUDA.                                                                                                                                                                                                                                                            |
| **ffmpeg**                                                     | Re-encodes output to a browser-playable `.mp4`; falls back to the raw file if unavailable.                                                                                                                                                                                                                                     |
| **Docker & Docker Compose** _(Deployment only)_                | Splits the web UI and inference into separate, independently-scaled services.                                                                                                                                                                                                                                                  |

![Official Ultralytics YOLO26 family benchmark: COCO mAP50-95 vs T4 TensorRT FP16 latency](Assets/mAP%20comparison%20%28yolo%20models%29.jpg)

_Ultralytics' official COCO benchmark — context for the YOLO26s pick, not a project-specific result._

## Model Training & Evaluation

Fine-tuned from `yolo26s.pt` in [`Yolo26s Training (kaggle).ipynb`](Yolo26s%20Training%20%28kaggle%29.ipynb), on Kaggle (2 GPUs).

|                     |                                                                                                                                                                                                            |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset             | [cigarette detection](https://platform.ultralytics.com/ahmed-aboubakr/datasets/cigarette-detection1) (Ultralytics Platform) — 14,309 images, single class `cigarette`: 12,259 train / 1,152 val / 898 test |
| Training resolution | 640 × 640                                                                                                                                                                                                  |
| Epochs              | Up to 60, early-stopped at **epoch 37** (no improvement for 15 epochs)                                                                                                                                     |
| Fused model         | ~9.5M parameters, 20.8 GFLOPs                                                                                                                                                                              |

**Reported validation-set results** (from the training notebook, consistent with the project presentation's rounded figures):

| Metric    | Notebook (precise) | Presentation (rounded) |
| --------- | ------------------ | ---------------------- |
| Precision | 0.822              | 82.5%                  |
| Recall    | 0.794              | 79.0%                  |
| mAP@50    | 0.838              | 83.8%                  |
| mAP@50–95 | 0.495              | 49.5%                  |

- Precision = share of predicted boxes that were correct; recall = share of real cigarettes found.
- mAP@50 (loose 50% overlap) vs. mAP@50–95 (stricter, harder, more location-sensitive).
- Inference latency: **5.0 ms/image** (model forward pass only, measured in the Kaggle notebook) — not the same as end-to-end video processing, which adds tracking, optional pose, drawing, and the ffmpeg re-encode.
- Validation-set results, not an independent production benchmark. The notebook also scores a `images/test` folder (mAP@50 ≈ 0.78) but flags that split as having had data-quality issues — informational only.

## The Small-Object Detection Challenge

- Cigarettes occupy very few pixels in CCTV framing — the presentation notes many boxes fall under ~70px (the pose-gating code independently floors its own hold-region boxes at 70px for a similar reason; see [`detector.py`](Smoking_Detection_Prototype/Flask_UI/model/detector.py)).
- Hands, fingers, and smoke often obscure them; lighting, distance, and quality compound it.
- **Implemented:** inference runs at **1280×1280**, above the 640×640 training resolution, for more pixels on small objects.
- **Not implemented:** training itself at a higher resolution (e.g. 960) — see [Limitations](#limitations--future-work).

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

- Evaluate larger model variants (YOLO26m, YOLO26l).
- Train at a higher resolution (e.g. 960px) — only inference runs above 640px today.
- Reduce false positives on cigarette-shaped objects (pipes, pens, lighters).
- Require temporal consistency (e.g. minimum sustained frames) before alerting, not just a confirmed track.
- Validate on representative petroleum/industrial CCTV footage.
- Expose `AI_Worker` to other external services (already flagged on the deployment diagram).

## References

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) ([paper](https://arxiv.org/abs/2606.03748)) — base detector architecture and training framework ([ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)).
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) ([paper](https://arxiv.org/abs/2110.06864)) — multi-object tracking algorithm.
- [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) — body pose estimation.
- [PyTorch](https://pytorch.org/), [Flask](https://flask.palletsprojects.com/), [FastAPI](https://fastapi.tiangolo.com/), [OpenCV](https://opencv.org/), [Docker Compose](https://docs.docker.com/compose/).
