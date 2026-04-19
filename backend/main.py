from __future__ import annotations

import glob
import os
import shutil
import sqlite3
import sys
import threading
from typing import Any, TypeAlias, Union

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Path Setup ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

CHATBOT_DIR = os.path.join(BASE_DIR, "chatbot")
if CHATBOT_DIR not in sys.path:
    sys.path.insert(0, CHATBOT_DIR)

TOOLS_DIR = os.path.join(BASE_DIR, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

# ── Directories ───────────────────────────────────────────
DB_DIR      = os.path.join(BASE_DIR, "database")
INPUT_DIR   = os.path.join(BASE_DIR, "input_video")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output_video")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "ui"))

for d in [DB_DIR, INPUT_DIR, OUTPUT_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI."""
    global bot
    try:
        from chatbot.framework import DatabaseChatbot
        bot = DatabaseChatbot(db_directory=DB_DIR)
        print("✅ Chatbot initialized successfully")
    except Exception as e:
        print(f"⚠️  Chatbot init failed (will retry on first /chat call): {e}")
    yield

# ── FastAPI App ───────────────────────────────────────────
app = FastAPI(
    title="AquaVision API",
    description="Marine Object Detection & AI Chatbot Backend",
    lifespan=lifespan,
    version="1.0.0",
)


# ── CORS (allow all origins for frontend access) ──────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static file serving for output videos ─────────────────
app.mount("/output_video", StaticFiles(directory=OUTPUT_DIR), name="output_video")

# ── Serve frontend UI at /ui/* and redirect root to it ────
if os.path.exists(FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    # Redirect legacy paths
    @app.get("/frontend/ui/index.html")
    def redirect_legacy():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
    @app.get("/ui/index.html")
    def redirect_ui_root():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

# ── Detection job state ───────────────────────────────────
JSONDict: TypeAlias = dict[str, Any]
ObjectList: TypeAlias = list[str]

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_EXTENSIONS = ALLOWED_VIDEO_EXTENSIONS | ALLOWED_IMAGE_EXTENSIONS

detection_status: JSONDict = {"status": "idle", "message": "No detection running"}
current_video_name: str | None = None  # tracks the uploaded video filename
current_image_name: str | None = None  # tracks the uploaded image filename

# ── Chatbot singleton ────────────────────────────────────
bot: Any | None = None

# ── Pydantic Models ───────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class PredictRequest(BaseModel):
    video_name: str | None = None
    objects: ObjectList | None = None


def get_upload_filename(file: UploadFile) -> str:
    """Return a validated upload filename."""
    filename = file.filename
    if not filename:
        raise HTTPException(400, "Uploaded file is missing a filename")
    return filename


# ══════════════════════════════════════════════════════════
#  ROOT — Serve frontend HTML directly
# ══════════════════════════════════════════════════════════
@app.get("/", response_model=None)
def root() -> Union[FileResponse, JSONDict]:
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"status": "online", "message": "AquaVision API is ready 🌊"}

@app.get("/api")
def api_status() -> JSONDict:
    return {"status": "online", "message": "AquaVision API is ready 🌊"}

@app.get("/health")
def health_check() -> JSONDict:
    return {"status": "healthy"}


# ══════════════════════════════════════════════════════════
#  1. POST /upload — Upload video to input_video/
# ══════════════════════════════════════════════════════════
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> JSONDict:
    """Upload a video file to input_video/ directory."""
    global current_video_name
    try:
        filename = get_upload_filename(file)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(400, f"File type '{ext}' not allowed. Use: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}")

        # Save with original filename
        save_path = os.path.join(INPUT_DIR, filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        current_video_name = filename
        size_mb = os.path.getsize(save_path) / 1e6
        return {
            "status": "uploaded",
            "filename": filename,
            "size_mb": round(size_mb, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════
#  1b. POST /upload-image — Upload image to input_video/
# ══════════════════════════════════════════════════════════
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)) -> JSONDict:
    """Upload an image file for object detection."""
    global current_image_name
    try:
        filename = get_upload_filename(file)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(400, f"File type '{ext}' not allowed. Use: {', '.join(ALLOWED_IMAGE_EXTENSIONS)}")

        save_path = os.path.join(INPUT_DIR, filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        current_image_name = filename
        size_mb = os.path.getsize(save_path) / 1e6
        return {
            "status": "uploaded",
            "filename": filename,
            "file_type": "image",
            "size_mb": round(size_mb, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════
#  1c. POST /upload-media — Upload image OR video (auto-detect)
# ══════════════════════════════════════════════════════════
@app.post("/upload-media")
async def upload_media(file: UploadFile = File(...)) -> JSONDict:
    """Upload an image or video file. Auto-detects the type."""
    global current_video_name, current_image_name
    try:
        filename = get_upload_filename(file)
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, f"File type '{ext}' not allowed. Use: {', '.join(ALLOWED_EXTENSIONS)}")

        save_path = os.path.join(INPUT_DIR, filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        file_type = "image" if ext in ALLOWED_IMAGE_EXTENSIONS else "video"
        if file_type == "image":
            current_image_name = filename
        else:
            current_video_name = filename

        size_mb = os.path.getsize(save_path) / 1e6
        return {
            "status": "uploaded",
            "filename": filename,
            "file_type": file_type,
            "size_mb": round(size_mb, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════
#  2. POST /detect — Start background detection
# ══════════════════════════════════════════════════════════
@app.post("/detect")
async def start_detection() -> JSONDict:
    """Start YOLO object detection in a background thread."""
    global detection_status, current_video_name

    if detection_status["status"] == "running":
        return {"status": "already_running", "message": "Detection already in progress"}

    # Find a video to process
    video_name = current_video_name
    if not video_name:
        # Fallback: pick the first video in input_video/
        video_files = [f for f in os.listdir(INPUT_DIR)
                       if f.lower().endswith(tuple(ALLOWED_VIDEO_EXTENSIONS))]
        if not video_files:
            raise HTTPException(400, "No video found — upload a video first!")
        video_name = video_files[0]

    video_path = os.path.join(INPUT_DIR, video_name)
    if not os.path.exists(video_path):
        raise HTTPException(400, f"Video '{video_name}' not found in input_video/")

    detection_status = {"status": "running", "message": f"Detection started for {video_name}..."}

    def run_in_background():
        global detection_status
        try:
            from tools.detect_objects import run_detection
            result = run_detection(video_name, db_directory=DB_DIR)
            detection_status = {"status": "done", "message": result}

            # Also run environment prediction
            try:
                from tools.predict_environment import run_prediction_for_video
                env = run_prediction_for_video(video_name)
                detection_status["message"] += f" | Environment: {env}"
            except Exception as env_e:
                print(f"⚠️  Environment prediction failed: {env_e}")

        except Exception as e:
            detection_status = {"status": "error", "message": str(e)}

    threading.Thread(target=run_in_background, daemon=True).start()
    return {"status": "started", "message": "Detection running in background"}


# ══════════════════════════════════════════════════════════
#  3. GET /detect/status — Poll detection status
# ══════════════════════════════════════════════════════════
@app.get("/detect/status")
def get_detection_status() -> JSONDict:
    return detection_status


# ══════════════════════════════════════════════════════════
#  4. POST /detect-video — Upload + detect in one call
#     (Combined endpoint per user spec)
# ══════════════════════════════════════════════════════════
@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...)) -> JSONDict:
    """Accept a video file, run YOLO detection, save output, return video URL."""
    global detection_status, current_video_name

    if detection_status["status"] == "running":
        raise HTTPException(409, "Detection already in progress")

    # Validate file type
    filename = get_upload_filename(file)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(400, f"File type '{ext}' not allowed")

    # Save video
    save_path = os.path.join(INPUT_DIR, filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    current_video_name = filename
    detection_status = {"status": "running", "message": f"Processing {filename}..."}

    def run_in_background():
        global detection_status
        try:
            from tools.detect_objects import run_detection
            result = run_detection(filename, db_directory=DB_DIR)

            # Run environment prediction too
            try:
                from tools.predict_environment import run_prediction_for_video
                env = run_prediction_for_video(filename)
                result += f" | Environment: {env}"
            except Exception:
                pass

            output_url = f"/output_video/detected_{filename}"
            detection_status = {
                "status": "done",
                "message": result,
                "output_video_url": output_url,
            }
        except Exception as e:
            detection_status = {"status": "error", "message": str(e)}

    threading.Thread(target=run_in_background, daemon=True).start()
    return {
        "status": "started",
        "message": f"Detection running for {filename}",
        "poll_url": "/detect/status",
    }


# ══════════════════════════════════════════════════════════
#  4b. POST /detect-image — Run YOLO on a single image
# ══════════════════════════════════════════════════════════
@app.post("/detect-image")
async def detect_image(file: UploadFile | None = File(None)) -> JSONDict:
    """Run YOLO detection on an uploaded image. 
    If no file is provided, uses the last uploaded image."""
    global current_image_name

    try:
        if file:
            filename = get_upload_filename(file)
            ext = os.path.splitext(filename)[1].lower()
            if ext not in ALLOWED_IMAGE_EXTENSIONS:
                raise HTTPException(400, f"File type '{ext}' not allowed for image detection")
            save_path = os.path.join(INPUT_DIR, filename)
            with open(save_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            current_image_name = filename
        else:
            filename = current_image_name
            if not filename:
                raise HTTPException(400, "No image uploaded. Upload an image first.")

        image_path = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(image_path):
            raise HTTPException(400, f"Image '{filename}' not found")

        # Run YOLO detection on the image
        from tools.detect_objects import run_image_detection
        result = run_image_detection(filename, db_directory=DB_DIR)

        # Run environment prediction
        try:
            from tools.predict_environment import run_prediction_for_video
            env = run_prediction_for_video(filename)
            result["environment"] = env
        except Exception:
            pass

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════
#  5. POST /predict-environment — Predict environment
# ══════════════════════════════════════════════════════════
@app.post("/predict-environment")
async def predict_environment_endpoint(request: PredictRequest) -> JSONDict:
    """Predict the environment from detected objects.
    
    - If 'video_name' is provided, look up objects from the DB
    - If 'objects' list is provided, predict directly from that list
    """
    try:
        from tools.predict_environment import (
            run_prediction_for_video,
            predict_environment,
            get_unique_objects,
        )

        if request.video_name:
            env = run_prediction_for_video(request.video_name)
            objects = get_unique_objects(request.video_name)
            return {
                "video_name": request.video_name,
                "objects_detected": objects,
                "predicted_environment": env,
            }
        elif request.objects:
            env = predict_environment(request.objects)
            return {
                "objects_provided": request.objects,
                "predicted_environment": env,
            }
        else:
            # Try the most recent video
            db_path = os.path.join(DB_DIR, "detections.db")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("SELECT DISTINCT video_name FROM detections LIMIT 1")
                row = cur.fetchone()
                conn.close()
                if row:
                    latest_video = str(row[0])
                    env = run_prediction_for_video(latest_video)
                    return {"video_name": latest_video, "predicted_environment": env}

            raise HTTPException(400, "Provide 'video_name' or 'objects' list")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════
#  6. POST /chat — AI Chatbot
# ══════════════════════════════════════════════════════════
@app.post("/chat")
async def chat(request: ChatRequest) -> JSONDict:
    """Send a message to AquaBot and get a response."""
    global bot

    # Lazy init if startup failed
    if not bot:
        try:
            from chatbot.framework import DatabaseChatbot
            bot = DatabaseChatbot(db_directory=DB_DIR)
        except Exception as e:
            raise HTTPException(500, f"Chatbot initialization failed: {e}")

    try:
        response = await bot.chat(request.message, request.session_id)
        return {"response": response}
    except Exception as e:
        raise HTTPException(500, str(e))


# ══════════════════════════════════════════════════════════
#  7. GET /output-video — Serve detected output video
# ══════════════════════════════════════════════════════════
@app.get("/output-video")
def get_output_video() -> FileResponse:
    """Serve the most recent detected output video."""
    # Find any detected_* file in output_video/
    output_files = glob.glob(os.path.join(OUTPUT_DIR, "detected_*"))
    if not output_files:
        raise HTTPException(404, "No output video found — run detection first")

    # Return the most recent one
    latest = max(output_files, key=os.path.getmtime)
    return FileResponse(
        latest,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"},
    )


# ══════════════════════════════════════════════════════════
#  8. GET /stats — Dashboard statistics from DB
# ══════════════════════════════════════════════════════════
@app.get("/stats")
def get_stats() -> JSONDict:
    """Return detection statistics for the dashboard."""
    db_path = os.path.join(DB_DIR, "detections.db")
    if not os.path.exists(db_path):
        return {
            "total_detections": 0,
            "unique_objects": [],
            "top_objects": [],
            "activities": [],
            "environment": None,
        }
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Unique object classes (e.g. "diver", "jellyfish")
        cur.execute("SELECT DISTINCT object_name FROM detections ORDER BY object_name")
        unique_classes: ObjectList = [str(r[0]) for r in cur.fetchall()]

        # Unique TRACKED objects (e.g. "diver_1", "diver_2" = 2 unique divers)
        cur.execute("SELECT COUNT(DISTINCT object_label) FROM detections")
        tracked_row = cur.fetchone()
        unique_tracked_count = int(tracked_row[0]) if tracked_row else 0

        # Per-class: count of unique tracked objects (NOT frame detections)
        cur.execute(
            "SELECT object_name, COUNT(DISTINCT object_label) as cnt FROM detections "
            "GROUP BY object_name ORDER BY cnt DESC LIMIT 10"
        )
        top: list[JSONDict] = [{"label": str(r[0]), "count": int(r[1])} for r in cur.fetchall()]

        cur.execute(
            "SELECT activity, COUNT(*) as cnt FROM detections "
            "GROUP BY activity ORDER BY cnt DESC"
        )
        activities: list[JSONDict] = [{"label": str(r[0]), "count": int(r[1])} for r in cur.fetchall()]

        # Environment prediction
        env = None
        try:
            cur.execute("SELECT predicted_environment FROM video_environments LIMIT 1")
            row = cur.fetchone()
            if row:
                env = str(row[0])
        except Exception:
            pass

        conn.close()
        return {
            "total_detections": unique_tracked_count,
            "unique_objects": unique_classes,
            "top_objects": top,
            "activities": activities,
            "environment": env,
        }
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════
#  9. GET /download/{filename} — Download a specific output
# ══════════════════════════════════════════════════════════
@app.get("/download/{filename}")
def download_output(filename: str) -> FileResponse:
    """Download a specific output video file."""
    if ".." in filename or filename.startswith(("/", "\\")):
        raise HTTPException(400, "Invalid filename")
    filepath = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(404, f"File '{filename}' not found")
    return FileResponse(
        filepath,
        media_type="video/mp4",
        filename=filename,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ══════════════════════════════════════════════════════════
#  RUN (direct execution)
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    print("=" * 55)
    print("🌊 AquaVision API — Starting on http://localhost:8000")
    print("=" * 55)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
