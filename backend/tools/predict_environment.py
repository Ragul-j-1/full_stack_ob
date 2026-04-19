import sqlite3
import os
import sys
from typing import Sequence
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

def log(msg: str) -> None:
    """Log to stderr so it doesn't interfere with MCP stdout."""
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()

# Updated Paths for New Root Structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))
DB_PATH = os.path.join(ROOT_DIR, "database", "detections.db")

MODEL_NAME = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def clear_dead_local_proxy() -> None:
    """Ignore known-bad local proxy values that break outbound API calls."""
    dead_proxy_values = {
        "http://127.0.0.1:9",
        "http://localhost:9",
    }
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        value = os.environ.get(key)
        if value in dead_proxy_values:
            os.environ.pop(key, None)


def format_llm_error(exc: Exception) -> str:
    """Convert provider exceptions into concise status text."""
    error_text = str(exc)
    lowered = error_text.lower()

    if "credit balance is too low" in lowered or "plans & billing" in lowered:
        return "Billing Required"

    if "rate limit" in lowered or "429" in lowered:
        return "Rate Limited"

    if "api key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
        return "Invalid API Key"

    return "Prediction Failed"

def setup_database() -> None:
    """Create the video_environments table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS video_environments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_name TEXT UNIQUE,
        predicted_environment TEXT
    )
    """)
    conn.commit()
    conn.close()

def get_unique_objects(video_name: str) -> list[str]:
    """Retrieve a list of unique objects detected in a specific video."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT object_name FROM detections WHERE video_name = ?", (video_name,))
    objects = [row[0] for row in cursor.fetchall()]
    conn.close()
    return objects

def predict_environment(objects: Sequence[str]) -> str:
    """Use the LLM to infer the environment based on detected objects."""
    if not objects:
        return "Unknown (No objects detected)"

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("GROQ_API_KEY is missing from backend/.env")

    clear_dead_local_proxy()
    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=groq_api_key,
        temperature=0,
    )
    
    prompt = f"""
    Based on the following list of objects detected in a video, what is the most likely environment or place where this video was recorded?
    Provide only a short, concise answer (1-3 words) like 'Street', 'Office', 'Kitchen', 'Highway', etc.
    
    Detected objects: {', '.join(objects)}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        log(f"Error predicting environment: {e}")
        return format_llm_error(e)

def save_environment(video_name: str, environment: str) -> None:
    """Save or update the predicted environment in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO video_environments (video_name, predicted_environment)
    VALUES (?, ?)
    """, (video_name, environment))
    conn.commit()
    conn.close()

def run_prediction_for_video(video_name: str) -> str:
    log(f"Analyzing environment for video: {video_name}")
    setup_database()
    objects = get_unique_objects(video_name)
    if not objects:
        log(f"No detected objects found for {video_name}.")
        return "Unknown"
        
    env = predict_environment(objects)
    log(f"Predicted Environment: {env}")
    save_environment(video_name, env)
    return env

def main() -> None:
    log("=== Environment Prediction ===")
    if not os.path.exists(DB_PATH):
        log(f"Error: Database not found at {DB_PATH}")
        return

    setup_database()

    # Get all unique videos that have detections
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT video_name FROM detections")
    videos = [row[0] for row in cursor.fetchall()]
    conn.close()

    if not videos:
        log("No videos found in the detection database.")
        return

    for video in videos:
        run_prediction_for_video(video)

    log("\nEnvironment prediction complete!")

if __name__ == "__main__":
    main()
