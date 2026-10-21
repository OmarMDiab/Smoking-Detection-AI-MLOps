"""AI_Worker's internal HTTP API.

This is NOT the public-facing web app (that's Flask_UI). It's a small
FastAPI service, reachable only from Flask_UI over the Docker-internal
network, that owns everything related to actually running the model:
YOLO detection/tracking, MediaPipe pose, and the ffmpeg re-encode of the
raw tracked video into a browser-playable file.

Flask_UI and AI_Worker share a single media volume (mounted at /data in
both containers — see docker-compose.yaml). Flask_UI writes the uploaded
file into that shared volume and sends AI_Worker the *path*, not the raw
bytes, since both containers see the same filesystem location.
"""
import os
import subprocess
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from detector import run_on_image, run_on_video

app = FastAPI(title="Smoking Detection AI Worker")

# Locate ffmpeg: in the Docker image it's installed system-wide via
# apt-get (see ai_worker.dockerfile), so plain "ffmpeg" resolves via PATH.
# The bundled bin/ffmpeg.exe fallback is only relevant if someone runs this
# service directly on Windows (outside Docker) without ffmpeg on PATH.
_BUNDLED_FFMPEG = Path(__file__).parent / "bin" / "ffmpeg.exe"
FFMPEG = str(_BUNDLED_FFMPEG) if _BUNDLED_FFMPEG.exists() and os.name == "nt" else "ffmpeg"

# In-memory job store for background video processing, so Flask_UI can poll
# real progress (frames processed so far) instead of a fake timer. Fine for
# a single-process worker (see --workers 1 in ai_worker.dockerfile); a real
# multi-worker deployment would need a shared store (e.g. Redis) instead.
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


class ImageRequest(BaseModel):
    input_path: str
    output_path: str
    require_human: bool = False


class VideoRequest(BaseModel):
    job_id: str
    input_path: str
    raw_output_path: str
    web_output_path: str
    require_human: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/infer/image")
def infer_image(req: ImageRequest):
    try:
        stats = run_on_image(req.input_path, req.output_path, require_human=req.require_human)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return stats


def _process_video_job(job_id: str, input_path: str, raw_output: str, web_output: str, require_human: bool):
    def on_progress(percent, frame_idx, total_frames):
        with _jobs_lock:
            _jobs[job_id]["percent"] = percent
            _jobs[job_id]["frame_idx"] = frame_idx
            _jobs[job_id]["total_frames"] = total_frames

    try:
        stats = run_on_video(input_path, raw_output, on_progress=on_progress, require_human=require_human)

        with _jobs_lock:
            # Frame-by-frame work is done; the only thing left is the ffmpeg
            # re-encode below, which has no granular progress of its own
            # (it's one subprocess call) — park the bar here until it finishes.
            _jobs[job_id]["percent"] = 97

        used_web_output = False
        try:
            result = subprocess.run(
                [FFMPEG, "-y", "-i", raw_output, "-vcodec", "libx264", "-acodec", "aac", web_output],
                capture_output=True,
            )
            if result.returncode == 0 and Path(web_output).exists():
                used_web_output = True
            else:
                print(f"[ffmpeg] re-encode failed (exit {result.returncode}) using {FFMPEG}:\n"
                      f"{result.stderr.decode(errors='replace')}")
        except FileNotFoundError:
            print(f"[ffmpeg] binary not found at {FFMPEG} — serving raw mp4v output directly")

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = {
                "type": "video",
                "output_path": web_output if used_web_output else raw_output,
                **stats,
            }
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)


@app.post("/infer/video")
def infer_video(req: VideoRequest):
    with _jobs_lock:
        _jobs[req.job_id] = {"status": "processing", "percent": 0, "frame_idx": 0, "total_frames": 0}
    thread = threading.Thread(
        target=_process_video_job,
        args=(req.job_id, req.input_path, req.raw_output_path, req.web_output_path, req.require_human),
        daemon=True,
    )
    thread.start()
    return {"job_id": req.job_id}


@app.get("/infer/jobs/{job_id}")
def job_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return dict(job)
