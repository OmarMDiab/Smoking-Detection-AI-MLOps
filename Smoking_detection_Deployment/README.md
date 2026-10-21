# Smoking Detection — Deployment

Containerized version of the pipeline, split into two Docker Compose services. See the [root README](../README.md) for the project and the [Prototype docs](../Smoking_Detection_Prototype/README.md) for the single-process dev variant this was built from.

## Deployment Overview

Two cooperating containers instead of one process:

- **`flask_ui`** — thin Flask web app + API proxy, no model code.
- **`ai_worker`** — FastAPI service owning YOLO26s/ByteTrack/MediaPipe inference + the ffmpeg re-encode.

This lets the small web tier and the heavy, GPU-hungry inference tier be built, scaled, and resourced independently.

## Deployment Architecture

![Docker Compose deployment architecture: browser to flask_ui to ai_worker over an internal network, shared data volume, read-only model mount, GPU passthrough](../Assets/Linux%20Docker%20Compose%20AI%20Architecture.png)

- `flask_ui`: `python:3.11-slim`, port `5000` published to the host.
- `ai_worker`: `python:3.11-slim` + PyTorch/Ultralytics/MediaPipe; port `8001` is internal only (`expose`, not published).
- Shared bind mount `./data:/data` — both containers see `/data/uploads`/`/data/results`, so `flask_ui` hands `ai_worker` a file path, not raw bytes.
- `./Model/weights:/app/Model/weights:ro` — read-only into `ai_worker` only, not baked into the image.
- `ai_worker` reserves an NVIDIA GPU via `deploy.resources.reservations.devices`.
- `flask_ui` won't start until `ai_worker`'s healthcheck passes (`depends_on: condition: service_healthy`).

## Prerequisites

- Docker + Docker Compose (`docker compose` CLI plugin).
- `Model/weights/best.pt` at `Smoking_detection_Deployment/Model/weights/best.pt` (already included).
- **GPU (expected by the compose file):** NVIDIA GPU + driver + NVIDIA Container Toolkit. The GPU reservation is unconditional — there's no CPU-only fallback, so on a host without the toolkit, `ai_worker` may fail to start unless you remove that `deploy` block. Installing PyTorch alone doesn't guarantee GPU acceleration.
- Disk space: per the architecture diagram, `ai_worker`'s image (~6–9GB) is much larger than `flask_ui`'s (~150MB).

## Configuration

| Variable / setting | Where              | Default                                                                                                       | Purpose                                          |
| ------------------ | ------------------ | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `AI_WORKER_URL`    | `flask_ui` env     | `http://ai_worker:8001` (set in `docker-compose.yaml`); code falls back to `http://localhost:8001` if unset   | Where `flask_ui` sends inference requests        |
| `DATA_DIR`         | both services' env | `/data` (set in `docker-compose.yaml`); code falls back to a local `data/` folder next to `app.py` if unset   | Root of the shared uploads/results volume        |
| `MODEL_PATH`       | `ai_worker` env    | unset — auto-detects `<AI_Worker>/Model/weights/best.pt` (Docker layout) or the local dev layout one level up | Override if the weight file lives somewhere else |
| Port `5000`        | `flask_ui`         | published to host                                                                                             | Web UI / API                                     |
| Port `8001`        | `ai_worker`        | internal only (`expose`, not published)                                                                       | Inference API, reachable only from `flask_ui`    |

No secrets are used today. If any are added, pass them via Compose env vars or an untracked `.env` (already in [`.gitignore`](../.gitignore)) — never commit them.

## Build and Run

All commands run from `Smoking_detection_Deployment/`, using its `docker-compose.yaml`:

```bash
# Build and start both services in the foreground
docker compose up --build

# Or detached:
docker compose up --build -d

# Stop:
docker compose down

# Rebuild after code changes:
docker compose up --build

# Tail logs for one service:
docker compose logs -f ai_worker
docker compose logs -f flask_ui
```

## Application Usage

Open http://localhost:5000. Drop or select an image/video, optionally enable **"Only count cigarettes held by a person"**, then run the scan. Results show annotated output plus detection stats, with a download link for the processed file.

## Health Checks, Logs, and Troubleshooting

- `ai_worker`'s real `GET /health` backs both the Compose healthcheck (10s interval, 5 retries) and `flask_ui`'s startup gate.
- `GET /infer/jobs/{job_id}` reports video progress; `flask_ui`'s `/api/jobs/<job_id>` proxies it to the UI.
- **"AI worker unavailable"** — `flask_ui` can't reach `ai_worker`; check `docker compose logs ai_worker` / `ps`.
- **`ai_worker` won't start / GPU errors** — usually a missing or misconfigured NVIDIA Container Toolkit.
- **Video has no sound / won't play** — ffmpeg re-encode failed; check `ai_worker` logs for `[ffmpeg] re-encode failed`.

## Model Weights and Data

- `best.pt` must exist at `Model/weights/best.pt` on the host — bind-mounted read-only, not baked into the image.
- Uploads/results live under `./data` on the host, auto-created, persisted across restarts (bind mount, not cloud storage).
- Job progress is **in-memory, single `ai_worker` process** (`--workers 1` deliberately). Restarting `ai_worker` loses in-flight job status; scaling to multiple workers needs a shared store (e.g. Redis).

## Security and Operational Considerations

- **No authentication** on `flask_ui` or `ai_worker` — anyone reaching port `5000` can submit files. _(Not implemented.)_
- **`flask_ui` runs with Flask `debug=True`** — Werkzeug's debugger shouldn't be exposed on a network-reachable deployment. _(Not addressed.)_
- **Upload validation is extension-only**, 500MB cap, no malware scanning. _(Not implemented.)_
- `ai_worker`'s port `8001` isn't published to the host — limits, but (without auth) doesn't eliminate, exposure.
- **No HTTPS/TLS** — add a reverse proxy (nginx/Caddy/Traefik) for non-local use. _(Not implemented.)_
- **No concurrency limit** on video jobs — each upload gets its own thread, uncapped. _(Not implemented.)_

## Updating and Maintenance

- **Weights:** replace `Model/weights/best.pt` on the host, then `docker compose restart ai_worker` — no rebuild needed (bind mount, not an image layer).
- **Code:** `docker compose up --build` rebuilds the affected image(s).

## Current Limitations and Future Deployment Improvements

- Auth, HTTPS/TLS, and a reverse proxy in front of `flask_ui`.
- Disable Flask `debug=True` for any non-local use.
- A shared job-state store (e.g. Redis) for more than one `ai_worker` replica.
- A CPU-only fallback path in `docker-compose.yaml` for hosts without an NVIDIA GPU.
- Exposing `ai_worker` to other external services (already flagged on the architecture diagram).
- Model-side improvements in the [root README](../README.md#limitations--future-work): larger variants, higher-res training, fewer false positives, temporal-consistency gating, real-footage validation.
