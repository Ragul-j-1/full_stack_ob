import cv2
import sqlite3
import math
import os
import sys
from ultralytics import YOLO
from collections import Counter
import imageio

def log(msg):
    """Log to stderr so it doesn't interfere with MCP stdout."""
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()

def run_detection(video_name: str = "video.mp4", db_directory: str = None) -> str:
    # -----------------------------
    # Configuration & Paths
    # -----------------------------
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    MODEL_PATH = os.path.join(ROOT_DIR, "models", "yolov8n.pt")
    VIDEO_PATH = os.path.join(ROOT_DIR, "input_video", video_name)
    
    if db_directory is None:
        db_directory = os.path.join(ROOT_DIR, "database")
        
    DB_PATH = os.path.join(db_directory, "detections.db")
    output_dir = os.path.join(ROOT_DIR, "output_video")
    
    os.makedirs(db_directory, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    OUTPUT_VIDEO_PATH = os.path.join(output_dir, f"detected_{video_name}")

    # -----------------------------
    # Load YOLO model
    # -----------------------------
    if not os.path.exists(MODEL_PATH):
        fallback_model = "yolov8n.pt"
        log(f"Warning: Model not found at {MODEL_PATH}. Trying {fallback_model}...")
        model = YOLO(fallback_model)
    else:
        model = YOLO(MODEL_PATH)

    # -----------------------------
    # Database Setup
    # -----------------------------
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Core Detections Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_name TEXT,
        object_name TEXT,
        object_label TEXT,
        activity TEXT,
        speed REAL,
        frame_number INTEGER
    )
    """)

    # [NEW] Video Metadata Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS video_metadata (
        video_name TEXT PRIMARY KEY,
        fps REAL,
        total_frames INTEGER,
        duration_seconds REAL,
        width INTEGER,
        height INTEGER,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # [NEW] Video Events Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS video_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_name TEXT,
        object_label TEXT,
        object_name TEXT,
        start_frame INTEGER,
        end_frame INTEGER,
        dominant_activity TEXT,
        avg_speed REAL
    )
    """)
    conn.commit()

    # -----------------------------
    # Open Video
    # -----------------------------
    if not os.path.exists(VIDEO_PATH):
        conn.close()
        return f"Error: Video not found at {VIDEO_PATH}"

    cap = cv2.VideoCapture(VIDEO_PATH)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Save metadata
    cursor.execute("""
    INSERT OR REPLACE INTO video_metadata (video_name, fps, total_frames, duration_seconds, width, height)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (video_name, fps, total_frames, duration, width, height))
    conn.commit()

    # -----------------------------
    # Output Video Setup (H.264 for Browser Compatibility)
    # -----------------------------
    try:
        # imageio-ffmpeg bundles a static ffmpeg binary, ensuring libx264 works
        out = imageio.get_writer(OUTPUT_VIDEO_PATH, fps=fps, codec='libx264', quality=7)
        using_imageio = True
        log(f"Streaming output to {OUTPUT_VIDEO_PATH} using H.264 (imageio)")
    except Exception as e:
        log(f"Warning: imageio codec failed ({e}). Falling back to OpenCV mp4v.")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
        using_imageio = False

    # -----------------------------
    # Tracking & Processing
    # -----------------------------
    frame_id = 0
    previous_positions = {}
    recognized_objects = {}
    object_counter = {}
    MATCH_DISTANCE = 50

    # For Event Tracking
    event_data = {} # {label_id: {'name': name, 'frames': [], 'activities': [], 'speeds': []}}

    log(f"Processing video: {VIDEO_PATH} ({width}x{height} @ {fps} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use YOLO's built-in tracker
        results = model.track(frame, persist=True, verbose=False, iou=0.5, conf=0.25)

        for r in results:
            if r.boxes is None or r.boxes.id is None:
                continue

            boxes = r.boxes
            ids = boxes.id.tolist()
            cls = boxes.cls.tolist()
            xyxy = boxes.xyxy.tolist()

            for i, track_id in enumerate(ids):
                class_id = int(cls[i])
                object_name = model.names[class_id]
                x1, y1, x2, y2 = map(int, xyxy[i])
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)

                label_id = f"{object_name}_{int(track_id)}"

                # Speed & Activity calculation
                activity = "unknown"
                speed = 0
                if track_id in previous_positions:
                    px, py = previous_positions[track_id]
                    distance = math.sqrt((cx-px)**2 + (cy-py)**2)
                    speed = round(distance, 2)
                    if distance > 25: activity = "fast movement"
                    elif distance > 5: activity = "normal movement"
                    else: activity = "stationary"

                previous_positions[track_id] = (cx, cy)

                # Collect for Event Summary
                if label_id not in event_data:
                    event_data[label_id] = {'name': object_name, 'frames': [], 'activities': [], 'speeds': []}
                
                event_data[label_id]['frames'].append(frame_id)
                event_data[label_id]['activities'].append(activity)
                event_data[label_id]['speeds'].append(speed)

                # Only save to DB if object has persisted for N frames (reduces noise/flicker)
                if len(event_data[label_id]['frames']) >= 5:
                    # Visuals
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                    cv2.putText(frame, label_id, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                    # DB Save
                    cursor.execute("""
                    INSERT INTO detections (video_name, object_name, object_label, activity, speed, frame_number)
                    VALUES (?,?,?,?,?,?)
                    """, (video_name, object_name, label_id, activity, speed, frame_id))

        if frame_id % 50 == 0:
            conn.commit()
            log(f"Processed frame {frame_id}/{total_frames}...")

        if using_imageio:
            # OpenCV uses BGR, imageio expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.append_data(frame_rgb)
        else:
            out.write(frame)
        frame_id += 1

    # -----------------------------
    # Event Summarization
    # -----------------------------
    log("Summarizing behavioral events...")
    for label_id, data in event_data.items():
        start_f = min(data['frames'])
        end_f = max(data['frames'])
        
        # Most common activity
        activity_counts = Counter(data['activities'])
        dominant_activity = activity_counts.most_common(1)[0][0] if activity_counts else "unknown"
        
        # Average speed
        avg_speed = sum(data['speeds']) / len(data['speeds']) if data['speeds'] else 0
        
        cursor.execute("""
        INSERT INTO video_events (video_name, object_label, object_name, start_frame, end_frame, dominant_activity, avg_speed)
        VALUES (?,?,?,?,?,?,?)
        """, (video_name, label_id, data['name'], start_f, end_f, dominant_activity, avg_speed))

    # -----------------------------
    # Cleanup
    # -----------------------------
    cap.release()
    if using_imageio:
        out.close()
    else:
        out.release()
    conn.commit()
    conn.close()

    result_msg = f"Detection completed for {video_name}. Metadata & Events recorded at {DB_PATH}. Output: {OUTPUT_VIDEO_PATH}."
    log(result_msg)
    return result_msg

if __name__ == "__main__":
    run_detection()
