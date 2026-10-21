import os
import uuid
import threading
from pathlib import Path

import requests
from flask import Flask, request, jsonify, send_from_directory, render_template

# AI_Worker is a separate service/container exposing a small internal
# FastAPI — see AI_Worker/worker_app.py. This Flask app never imports or
# runs any model code itself; it only proxies requests to AI_Worker over
# HTTP and serves the static UI + result files from the shared volume.
AI_WORKER_URL = os.environ.get("AI_WORKER_URL", "http://localhost:8001")

app = Flask(__name__)

# DATA_DIR is the shared volume mounted in both this container and
# AI_Worker's (see docker-compose.yaml) — both services see the exact same
# files at the exact same paths, so Flask_UI can hand AI_Worker a plain
# path instead of re-uploading the file bytes over HTTP.
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
RESULT_DIR = DATA_DIR / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif", ".gif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".mpeg", ".mpg", ".3gp"}

MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH


def _ext(filename: str) -> str:
    return Path(filename).suffix.lower()


def _result_filename(output_path: str) -> str:
    """AI_Worker returns an absolute path inside the shared volume
    (/data/results/...); we only ever need the filename to build the
    /results/<filename> download URL this Flask app itself serves."""
    return Path(output_path).name


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
    # person's mouth/hand (pose-based, works through gloves), filtering out
    # false positives on cigarette-shaped objects nobody is holding.
    require_human = request.form.get("require_human", "false").lower() in ("1", "true", "on", "yes")

    is_video = ext in VIDEO_EXTS

    if is_video:
        job_id = uid
        raw_output_path = str(RESULT_DIR / f"{uid}_result.mp4")
        web_output_path = str(RESULT_DIR / f"{uid}_web.mp4")
        try:
            resp = requests.post(
                f"{AI_WORKER_URL}/infer/video",
                json={
                    "job_id": job_id,
                    "input_path": input_path,
                    "raw_output_path": raw_output_path,
                    "web_output_path": web_output_path,
                    "require_human": require_human,
                },
                timeout=10,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            return jsonify({"error": f"AI worker unavailable: {e}"}), 503
        return jsonify({"type": "video_job", "job_id": job_id})

    output_path = str(RESULT_DIR / f"{uid}_result{ext}")
    try:
        resp = requests.post(
            f"{AI_WORKER_URL}/infer/image",
            json={"input_path": input_path, "output_path": output_path, "require_human": require_human},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return jsonify({"error": f"AI worker unavailable: {e}"}), 503
    stats = resp.json()
    return jsonify({"type": "image", "result_url": f"/results/{_result_filename(output_path)}", **stats})


@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    try:
        resp = requests.get(f"{AI_WORKER_URL}/infer/jobs/{job_id}", timeout=10)
    except requests.RequestException as e:
        return jsonify({"error": f"AI worker unavailable: {e}"}), 503

    if resp.status_code == 404:
        return jsonify({"error": "Unknown job"}), 404
    resp.raise_for_status()
    job = resp.json()

    # AI_Worker's result payload carries an absolute shared-volume path
    # (output_path); translate it to the public /results/<filename> URL
    # this Flask app serves, same shape the UI already expects.
    if job.get("status") == "done" and "result" in job:
        result = dict(job["result"])
        output_path = result.pop("output_path", None)
        if output_path:
            result["result_url"] = f"/results/{_result_filename(output_path)}"
        job = {**job, "result": result}
    return jsonify(job)


@app.route("/results/<filename>")
def serve_result(filename):
    return send_from_directory(RESULT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

