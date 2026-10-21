FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# ffmpeg: re-encodes the raw tracked video into a browser-playable H.264
# file. libglib2.0-0/libsm6/libxext6/libxrender1: opencv-python-headless's
# runtime shared-lib deps. curl: used by the docker-compose healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libglib2.0-0 libsm6 libxext6 libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001
# --workers 1: job progress state lives in this process's memory (see the
# _jobs dict in worker_app.py) — multiple worker processes would each have
# their own copy and polling would see stale/missing jobs. A real
# multi-worker deployment would need a shared store (e.g. Redis) instead.
CMD ["uvicorn", "worker_app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
