"""
server.py — FastAPI web server for the ADAS pipeline.

Run with:
    python app/server.py

Then open http://localhost:8000 in your browser.
"""

import os
import queue
import shutil
import sys
import threading
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Request, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Make sure adas_pipeline root is on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
from app.xml_exporter import generate_xml

app = FastAPI(title="ADAS Pipeline")

# Serve static files (index.html etc.)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─── Job state ────────────────────────────────────────────────────────────────

jobs: dict = {}   # job_id → { status, log_queue, result }


def _run_pipeline(job_id: str, video_path: str):
    """Run the full pipeline in a background thread, pushing log lines to a queue."""
    import logging

    log_q = jobs[job_id]["log_queue"]

    class QueueHandler(logging.Handler):
        def emit(self, record):
            msg = self.format(record)
            log_q.put(msg)

    # Attach queue handler to root logger AND PipelineRunner logger.
    # PipelineRunner sets propagate=False so its [START]/[DONE] messages
    # would otherwise never reach the root logger and thus never reach the UI.
    handler = QueueHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("PipelineRunner").addHandler(handler)
    logging.getLogger("PipelineRunner").setLevel(logging.INFO)

    try:
        from pipeline_runner import PipelineRunner
        import modules.input_handler as input_handler
        import modules.frame_extractor as frame_extractor
        import modules.frame_cleaner as frame_cleaner
        import modules.detector as detector
        import modules.tracker as tracker
        import modules.pose_estimator as pose_estimator
        import modules.behavior_analyzer as behavior_analyzer
        import modules.intent_predictor as intent_predictor
        import modules.tagger as tagger
        import exporter as exporter_mod
        import visualizer

        stages = [
            ("input_validation",  input_handler.run),
            ("frame_extraction",  frame_extractor.run),
            ("frame_cleaning",    frame_cleaner.run),
            ("detection",         detector.run),
            ("tracking",          tracker.run),
            ("pose_estimation",   pose_estimator.run),
            ("behavior_analysis", behavior_analyzer.run),
            ("intent_prediction", intent_predictor.run),
            ("tagging",           tagger.run),
            ("export",            exporter_mod.run),
        ]

        runner = PipelineRunner(stages, resume=False)
        context = {"mode": "video", "video_path": video_path}
        runner.run(context)

        # Render annotated video — capture actual output path (may be .avi if ffmpeg unavailable)
        log_q.put("INFO: Rendering annotated video...")
        video_out = visualizer.render_video(
            dataset_path=os.path.join(config.FINAL_DIR, config.OUTPUT_JSON_NAME),
            output_path=os.path.join(config.FINAL_DIR, "annotated_video.mp4"),
            fps=30.0,
        )
        log_q.put("INFO: Video saved")

        # Generate XML
        log_q.put("INFO: Generating XML annotations...")
        xml_out = os.path.join(config.FINAL_DIR, "annotations.xml")
        generate_xml(
            dataset_json_path=os.path.join(config.FINAL_DIR, config.OUTPUT_JSON_NAME),
            output_xml_path=xml_out,
            video_id=Path(video_path).stem,
        )
        log_q.put(f"INFO: XML written: {xml_out}")

        # Copy outputs to a job-specific directory so multiple jobs don't overwrite each other
        job_out_dir = os.path.join(config.OUTPUT_DIR, "jobs", job_id)
        os.makedirs(job_out_dir, exist_ok=True)
        job_video = os.path.join(job_out_dir, os.path.basename(video_out))
        job_json  = os.path.join(job_out_dir, "dataset.json")
        job_csv   = os.path.join(job_out_dir, "dataset.csv")
        job_xml   = os.path.join(job_out_dir, "annotations.xml")
        for src, dst in [
            (video_out, job_video),
            (os.path.join(config.FINAL_DIR, config.OUTPUT_JSON_NAME), job_json),
            (os.path.join(config.FINAL_DIR, config.OUTPUT_CSV_NAME),  job_csv),
            (xml_out, job_xml),
        ]:
            if os.path.exists(src):
                shutil.copy2(src, dst)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = {
            "video": job_video,
            "json":  job_json,
            "csv":   job_csv,
            "xml":   job_xml,
        }
        log_q.put("DONE")

    except Exception as e:
        import traceback
        log_q.put(f"ERROR: {e}\n{traceback.format_exc()}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        log_q.put("DONE")
    finally:
        logging.getLogger().removeHandler(handler)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Accept a video file, save it, and start the pipeline."""
    # Validate extension
    allowed = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    job_id = str(uuid.uuid4())[:8]

    # Save uploaded file
    os.makedirs(config.INPUT_DIR, exist_ok=True)
    save_path = os.path.join(config.INPUT_DIR, f"{job_id}{ext}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Initialize job
    jobs[job_id] = {
        "status": "running",
        "log_queue": queue.Queue(),
        "result": None,
        "filename": file.filename,
    }

    # Run pipeline in background thread
    thread = threading.Thread(target=_run_pipeline, args=(job_id, save_path), daemon=True)
    thread.start()

    return {"job_id": job_id, "filename": file.filename}


@app.get("/progress/{job_id}")
async def progress(job_id: str):
    """Server-Sent Events stream of pipeline log lines."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    def event_stream():
        log_q = jobs[job_id]["log_queue"]
        while True:
            try:
                msg = log_q.get(timeout=30)
                yield f"data: {msg}\n\n"
                if msg == "DONE":
                    break
            except queue.Empty:
                yield "data: [heartbeat]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    return {
        "status": job["status"],
        "has_result": job["result"] is not None,
    }


@app.get("/stream/{job_id}/video")
async def stream_video(job_id: str, request: Request):
    """Stream the annotated video with HTTP range support for browser playback."""

    if job_id not in jobs or jobs[job_id]["status"] != "done":
        raise HTTPException(404, "Not ready")

    file_path = jobs[job_id]["result"]["video"]
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, "Video file not found")

    file_size = os.path.getsize(file_path)
    mime = "video/mp4" if file_path.endswith(".mp4") else "video/x-msvideo"
    range_header = request.headers.get("range")

    if range_header:
        start, end = range_header.replace("bytes=", "").split("-")
        start = int(start)
        end = int(end) if end else file_size - 1
        chunk_size = end - start + 1

        def iter_file():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = chunk_size
                while remaining:
                    data = f.read(min(65536, remaining))
                    if not data:
                        break
                    remaining -= len(data)
                    yield data

        headers = {
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_size),
            "Content-Type": mime,
        }
        return StreamingResponse(iter_file(), status_code=206, headers=headers)

    def iter_full():
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                yield chunk

    return StreamingResponse(
        iter_full(),
        media_type=mime,
        headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
    )


@app.get("/download/{job_id}/{file_type}")
async def download(job_id: str, file_type: str):
    """Download a result file. file_type: video | json | csv | xml"""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    if job["status"] != "done" or not job["result"]:
        raise HTTPException(400, "Job not complete")

    type_map = {
        "video": ("annotated_video.mp4", "video/mp4"),
        "json":  ("dataset.json",        "application/json"),
        "csv":   ("dataset.csv",         "text/csv"),
        "xml":   ("annotations.xml",     "application/xml"),
    }

    if file_type not in type_map:
        raise HTTPException(400, f"Unknown file type: {file_type}")

    filename, media_type = type_map[file_type]
    file_path = job["result"].get(file_type)

    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, f"Output file not found: {file_path}")

    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=media_type,
    )


if __name__ == "__main__":
    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=False)
