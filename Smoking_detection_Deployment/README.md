# Smoking Detection — Deployment

Containerized version of the pipeline: two Docker Compose services. See the [root](../README.md) and [Prototype](../Smoking_Detection_Prototype/README.md) docs.

- **`flask_ui`** — thin Flask web app + API proxy, no model code.
- **`ai_worker`** — FastAPI service running YOLO26s/ByteTrack/MediaPipe + the ffmpeg re-encode.

## Architecture

![Deployment architecture](../Assets/Linux%20Docker%20Compose%20AI%20Architecture.png)

`flask_ui` (port `5000`, published) → `ai_worker` (port `8001`, internal only) over the Compose network. Shared bind mount `./data:/data`; `./Model/weights` mounted read-only into `ai_worker`. `ai_worker` reserves an NVIDIA GPU; `flask_ui` waits for its healthcheck.

## Prerequisites

- Docker + Docker Compose.
- `Model/weights/best.pt` (already included).
- NVIDIA GPU + driver + NVIDIA Container Toolkit — the compose file reserves a GPU unconditionally (remove the `deploy` block to run CPU-only).

## Configuration

| Var             | Default                 | Purpose                         |
| --------------- | ----------------------- | ------------------------------- |
| `AI_WORKER_URL` | `http://ai_worker:8001` | Where `flask_ui` sends requests |
| `DATA_DIR`      | `/data`                 | Shared uploads/results root     |
| `MODEL_PATH`    | auto-detected           | Override weight file location   |

No secrets used today; add via Compose env vars or an untracked `.env` if needed.

## Run It

```bash
cd Smoking_detection_Deployment
docker compose up --build         # start
docker compose down                # stop
docker compose logs -f ai_worker   # logs
```

Open http://localhost:5000, upload an image/video, optionally enable "Only count cigarettes held by a person".

## Health & Troubleshooting

- `ai_worker`'s `GET /health` backs the Compose healthcheck and `flask_ui`'s startup gate.
- "AI worker unavailable" → `ai_worker` unreachable, check `docker compose logs ai_worker`.
- GPU errors → missing/misconfigured NVIDIA Container Toolkit.
- No sound/won't play → ffmpeg re-encode failed, check `ai_worker` logs.

## Data & Security

- `best.pt` and `./data` are bind mounts, not baked into images — persisted on the host.
- Job progress is in-memory in a single `ai_worker` process (`--workers 1`); scaling needs a shared store (e.g. Redis).
- No auth, no HTTPS/TLS, `flask_ui` runs with `debug=True`, extension-only upload validation, no concurrency cap — add these before any network-exposed deployment.

## Updating

Replace `Model/weights/best.pt` + `docker compose restart ai_worker` (no rebuild needed). Code changes: `docker compose up --build`.

## Limitations

Same as the [root README](../README.md#limitations--future-work), plus: no auth/TLS, single-process job store, no CPU-only compose fallback.
