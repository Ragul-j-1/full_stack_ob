import sqlite3
import os
import sys
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

def log(msg):
    """Log to stderr so it doesn't interfere with MCP stdout."""
    sys.stderr.write(f"{msg}\n")
    sys.stderr.flush()

# Updated Paths for New Root Structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))
DB_PATH = os.path.join(ROOT_DIR, "database", "detections.db")

MODEL_NAME = "gpt-oss:120b-cloud"

def setup_database():
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

def get_unique_objects(video_name: str) -> list:
    """Retrieve a list of unique objects detected in a specific video."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT object_name FROM detections WHERE video_name = ?", (video_name,))
    objects = [row[0] for row in cursor.fetchall()]
    conn.close()
    return objects

def predict_environment(objects: list) -> str:
    """Use the LLM to infer the environment based on detected objects."""
    if not objects:
        return "Unknown (No objects detected)"

    llm = ChatOllama(
        model=MODEL_NAME, 
        base_url="http://127.0.0.1:11434",
        client_kwargs={'headers': {'Authorization': f'Bearer {os.environ.get("OLLAMA_API_KEY", "")}'}}
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
        return "Prediction Failed"

def save_environment(video_name: str, environment: str):
    """Save or update the predicted environment in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT OR REPLACE INTO video_environments (video_name, predicted_environment)
    VALUES (?, ?)
    """, (video_name, environment))
    conn.commit()
    conn.close()

def run_prediction_for_video(video_name: str):
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

def main():
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
