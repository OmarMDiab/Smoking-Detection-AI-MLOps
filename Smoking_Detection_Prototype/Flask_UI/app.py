import os
import sys
import uuid
import threading
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template

# Locate ffmpeg: prefer the bundled static binary shipped with this project
# (bin/ffmpeg.exe), which has no external DLL dependencies and therefore
# can't collide with other tools' DLLs on PATH (unlike the conda-forge
# build, which is dynamically linked against ~80 shared libraries and can
# fail on Windows with STATUS_ENTRYPOINT_NOT_FOUND if an incompatible
# same-named DLL from another conda env / MSYS2 / etc. gets resolved
# instead). Falls back to the conda env's ffmpeg, then system PATH.
_BUNDLED_FFMPEG = Path(__file__).parent / "bin" / "ffmpeg.exe"
_CONDA_FFMPEG = Path(sys.executable).parent.parent / "Library" / "bin" / "ffmpeg.exe"
if _BUNDLED_FFMPEG.exists():
    FFMPEG = str(_BUNDLED_FFMPEG)
elif _CONDA_FFMPEG.exists():
    FFMPEG = str(_CONDA_FFMPEG)
else:
    FFMPEG = "ffmpeg"

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpeg", ".mpg", ".3gp"}

MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# In-memory job store for background video processing, so the client can
# poll real progress (frames processed so far) instead of a fake timer.
# Fine for a single-process dev server; a real deployment with multiple
# workers would need a shared store (e.g. Redis) instead.
_jobs = {}
_jobs_lock = threading.Lock()


def _ext(filename: str) -> str:
    return Path(filename).suffix.lower()


def _process_video_job(job_id: str, input_path: str, raw_output: str, uid: str, require_human: bool):
    from model.detector import run_on_video

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

        output_filename = f"{uid}_result.mp4"
        browser_output = str(RESULT_DIR / f"{uid}_web.mp4")
        try:
            result = subprocess.run(
                [FFMPEG, "-y", "-i", raw_output, "-vcodec", "libx264", "-acodec", "aac", browser_output],
                capture_output=True,
            )
            if result.returncode == 0 and Path(browser_output).exists():
                output_filename = f"{uid}_web.mp4"
            else:
                print(f"[ffmpeg] re-encode failed (exit {result.returncode}) using {FFMPEG}:\n"
                      f"{result.stderr.decode(errors='replace')}")
        except FileNotFoundError:
            print(f"[ffmpeg] binary not found at {FFMPEG} — serving raw mp4v output directly")

        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = {
                "type": "video",
                "result_url": f"/results/{output_filename}",
                **stats,
            }
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = _ext(file.filename)
    if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    uid = uuid.uuid4().hex
    input_path = str(UPLOAD_DIR / f"{uid}{ext}")
    file.save(input_path)

    # "require_human" toggle (sent as a form field, e.g. "true"/"1") — when
    # enabled, a cigarette detection only counts if it's near a detected
    # person's wrist/mouth (pose-based, works through gloves), filtering out
    # false positives on cigarette-shaped objects nobody is holding.
    require_human = request.form.get("require_human", "false").lower() in ("1", "true", "on", "yes")

    is_video = ext in VIDEO_EXTS

    if is_video:
        raw_output = str(RESULT_DIR / f"{uid}_result.mp4")
        job_id = uid
        with _jobs_lock:
            _jobs[job_id] = {"status": "processing", "percent": 0, "frame_idx": 0, "total_frames": 0}
        thread = threading.Thread(
            target=_process_video_job, args=(job_id, input_path, raw_output, uid, require_human), daemon=True
        )
        thread.start()
        return jsonify({"type": "video_job", "job_id": job_id})

    output_filename = f"{uid}_result{ext}"
    raw_output = str(RESULT_DIR / output_filename)
    try:
        from model.detector import run_on_image
        stats = run_on_image(input_path, raw_output, require_human=require_human)
        return jsonify({"type": "image", "result_url": f"/results/{output_filename}", **stats})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Unknown job"}), 404
        # Return a shallow copy so the response is consistent even if the
        # background thread mutates the dict right after we read it.
        return jsonify(dict(job))


@app.route("/results/<filename>")
def serve_result(filename):
    return send_from_directory(RESULT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
