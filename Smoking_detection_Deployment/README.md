# Smoking Detection — Deployment

Containerized version of the same detection pipeline, split into two Docker Compose services. See the [root README](../README.md) for the overall project and the [Prototype docs](../Smoking_Detection_Prototype/README.md) for the single-process development variant this was built from.

## Deployment Overview

This directory packages the system as two cooperating containers instead of one monolithic process:

- **`flask_ui`** — a thin Flask web app and API proxy. Contains no model code at all.
- **`ai_worker`** — a FastAPI service that owns all YOLO26s/ByteTrack/MediaPipe inference and the ffmpeg re-encode step.

This separation lets the small, CPU-only-capable web tier and the heavy, GPU-hungry inference tier be built, scaled, and resourced independently.

## Deployment Architecture

![Docker Compose deployment architecture: browser to flask_ui to ai_worker over an internal network, shared data volume, read-only model mount, GPU passthrough](../Assets/Linux%20Docker%20Compose%20AI%20Architecture.png)

- `flask_ui`: image `python:3.11-slim`, port `5000` published to the host.
- `ai_worker`: image `python:3.11-slim` with PyTorch/Ultralytics/MediaPipe installed; port `8001` is only `expose`d (internal to the Compose network, not published to the host).
- Shared bind-mounted volume `./data:/data` — both containers see `/data/uploads` and `/data/results` at the same paths, so `flask_ui` can hand `ai_worker` a file _path_ instead of re-uploading bytes.
- `./Model/weights:/app/Model/weights:ro` — model weights are bind-mounted **read-only** into `ai_worker` only; they are not baked into the image.
- `ai_worker` requests an NVIDIA GPU via a `deploy.resources.reservations.devices` block in `docker-compose.yaml`.
- `flask_ui`'s `depends_on: ai_worker: condition: service_healthy` means Compose won't start the `flask_ui` container until `ai_worker`'s healthcheck passes.

## Prerequisites

- Docker and Docker Compose (the `docker compose` CLI plugin).
- `Model/weights/best.pt` present on the host at `Smoking_detection_Deployment/Model/weights/best.pt` (already included in this repository).
- **GPU (optional, but expected by the current Compose file):** an NVIDIA GPU, its driver, and the NVIDIA Container Toolkit installed on the Docker host. `docker-compose.yaml` reserves a GPU unconditionally for `ai_worker` (`driver: nvidia`, `count: 1`, `capabilities: ["gpu"]`) — on a host without a working NVIDIA Container Toolkit setup, this reservation can prevent `ai_worker` from starting at all. There is no automatic CPU-only fallback in the compose file; running on CPU-only hardware requires manually removing or commenting out that `deploy` block. PyTorch being installed (`requirements.txt`) does not by itself guarantee GPU acceleration — the toolkit and device reservation above are what actually make the GPU available inside the container.
- Enough disk space for both images — per the architecture diagram, `ai_worker`'s image (~6–9GB, PyTorch/Ultralytics/MediaPipe) is substantially larger than `flask_ui`'s (~150MB).

## Configuration

| Variable / setting | Where              | Default                                                                                                       | Purpose                                          |
| ------------------ | ------------------ | ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| `AI_WORKER_URL`    | `flask_ui` env     | `http://ai_worker:8001` (set in `docker-compose.yaml`); code falls back to `http://localhost:8001` if unset   | Where `flask_ui` sends inference requests        |
| `DATA_DIR`         | both services' env | `/data` (set in `docker-compose.yaml`); code falls back to a local `data/` folder next to `app.py` if unset   | Root of the shared uploads/results volume        |
| `MODEL_PATH`       | `ai_worker` env    | unset — auto-detects `<AI_Worker>/Model/weights/best.pt` (Docker layout) or the local dev layout one level up | Override if the weight file lives somewhere else |
| Port `5000`        | `flask_ui`         | published to host                                                                                             | Web UI / API                                     |
| Port `8001`        | `ai_worker`        | internal only (`expose`, not published)                                                                       | Inference API, reachable only from `flask_ui`    |

No secrets (API keys, passwords, tokens) are used by this stack today. If any are added in the future, supply them via Docker Compose environment variables or an untracked `.env` file (already excluded by [`../.gitignore`](../.gitignore)) — never commit them.

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

- `ai_worker` exposes a real `GET /health` endpoint (`worker_app.py`), used both by `docker-compose.yaml`'s healthcheck (`curl -f http://localhost:8001/health`, every 10s, 5s timeout, 5 retries, 30s start period) and indirectly by `flask_ui`'s `depends_on` start ordering.
- `GET /infer/jobs/{job_id}` on `ai_worker` reports per-video progress; `flask_ui`'s `/api/jobs/<job_id>` proxies it for the UI's progress bar.
- **"AI worker unavailable" error in the UI** — `flask_ui` couldn't reach `ai_worker` over HTTP; check `docker compose logs ai_worker` and `docker compose ps`.
- **`ai_worker` fails to start / GPU errors** — usually a missing or misconfigured NVIDIA Container Toolkit; see [Prerequisites](#prerequisites).
- **Video has no sound or won't play in-browser** — the ffmpeg re-encode subprocess failed or wasn't found; `ai_worker`'s Docker image installs `ffmpeg` via `apt-get`, so this should only happen if that install step was skipped or modified — check `ai_worker` logs for a `[ffmpeg] re-encode failed` message.

## Model Weights and Data

- `best.pt` must exist at `Smoking_detection_Deployment/Model/weights/best.pt` on the host (already included in this repo) — it's bind-mounted read-only into `ai_worker`, not copied into the image.
- Uploads and results live under `./data` on the host (`./data/uploads`, `./data/results`), auto-created by Docker on first run and persisted across container restarts since it's a bind mount, not stored in any cloud/object storage.
- Per-video job progress is kept **in-memory, in a single `ai_worker` process** (`worker_app.py` runs Uvicorn with `--workers 1` deliberately — job state lives in a plain Python dict). Restarting the `ai_worker` container loses in-flight job status, and this design would need a shared store (e.g. Redis) before it could scale to multiple worker replicas.

## Security and Operational Considerations

- **No authentication or authorization** on either the `flask_ui` web app/API or the `ai_worker` internal API — anyone who can reach port `5000` can submit files for processing. _(Not implemented; recommended before any network-exposed deployment.)_
- **`flask_ui` runs with Flask's `debug=True`** (`app.py`) — Werkzeug's interactive debugger should never be enabled on a network-reachable deployment, since it can allow arbitrary code execution if triggered. _(Not yet addressed — disable debug mode or front the app with a production WSGI server for anything beyond local testing.)_
- **Upload validation is extension-based only** (`IMAGE_EXTS`/`VIDEO_EXTS` allow-lists) with a 500 MB size cap; there is no content/malware scanning of uploaded files. _(Not implemented.)_
- `ai_worker`'s port `8001` is not published to the host, so it's only reachable from within the Compose network — this limits, but (without the auth note above) does not eliminate, exposure.
- No HTTPS/TLS termination is configured in `docker-compose.yaml` — add a reverse proxy (e.g. nginx/Caddy/Traefik) in front of `flask_ui` for any non-local deployment. _(Not implemented.)_
- No request queue or concurrency limit on video jobs — each upload spawns its own background thread with no cap. _(Not implemented.)_

## Updating and Maintenance

- **Updating model weights:** replace `Model/weights/best.pt` on the host and restart the `ai_worker` container (`docker compose restart ai_worker`, or `docker compose up -d` to recreate it) — no image rebuild is needed since the weights are a bind mount, not a layer in the image.
- **Updating code:** `docker compose up --build` rebuilds the affected image(s) from the current source.

## Current Limitations and Future Deployment Improvements

**Implemented today:** two-service split with a health-gated startup order, shared data volume, read-only weight mount, GPU device reservation, internal-only worker port.

**Not yet implemented / recommended:**

- Authentication, HTTPS/TLS, and a reverse proxy in front of `flask_ui`.
- Disabling Flask `debug=True` for any non-local use.
- A shared job-state store (e.g. Redis) to allow more than one `ai_worker` replica.
- A CPU-only fallback path in `docker-compose.yaml` for hosts without an NVIDIA GPU/Container Toolkit.
- Exposing `ai_worker` to other external services — already called out as future work on the architecture diagram itself.
- The same model-side improvements noted in the [root README](../README.md#limitations--future-work): larger model variants, higher-resolution training, reduced false positives, temporal-consistency gating, and evaluation on representative industrial CCTV footage.
