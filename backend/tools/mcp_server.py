import sqlite3
import os
import glob
import sys
from mcp.server.fastmcp import FastMCP

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detect_objects import run_detection
from predict_environment import run_prediction_for_video

# Setup logging
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log")

def log(msg):
    with open(LOG_FILE, "a") as f:
        f.write(str(msg) + "\n")

mcp = FastMCP('maritime_database_server')

# Directory containing the databases
DB_DIRECTORY = os.environ.get("DB_DIRECTORY", ".")

def get_db_path(db_name: str) -> str:
    if ".." in db_name or db_name.startswith("/") or db_name.startswith("\\"):
        raise ValueError("Invalid database name")
    return os.path.join(DB_DIRECTORY, db_name)

@mcp.tool()
def list_databases() -> list:
    """List all SQLite databases (.db or .sqlite files) in the configured directory."""
    try:
        db_files = glob.glob(os.path.join(DB_DIRECTORY, "*.db")) + glob.glob(os.path.join(DB_DIRECTORY, "*.sqlite"))
        return [os.path.basename(f) for f in db_files]
    except Exception as e:
        log(f"Error in list_databases: {e}")
        return [{"error": str(e)}]

@mcp.tool()
def list_tables(database_name: str) -> list:
    """List all tables available in the specified database along with their column names."""
    try:
        db_path = get_db_path(database_name)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        tables = [row[0] for row in cur.fetchall()]

        result = []
        for table in tables:
            cur.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cur.fetchall()]
            result.append({"table": table, "columns": columns})

        conn.close()
        return result
    except Exception as e:
        log(f"Error in list_tables for {database_name}: {e}")
        return [{"error": str(e)}]

@mcp.tool()
def run_sql(database_name: str, query: str):
    """Execute a DQL (SELECT) query on the specified database."""
    try:
        if not query.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed."
            
        db_path = get_db_path(database_name)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(query)
        result = cur.fetchall()
        conn.close()
        log(f"SQL Result: {result}")
        return result
    except Exception as e:
        log(f"Error in run_sql on {database_name}: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
def detect_objects_in_video(video_filename: str) -> str:
    """Detect and track objects in a video file, then predict the environment.
    This also populates 'video_metadata' and 'video_events' for high-level reasoning.
    """
    try:
        result = run_detection(video_filename, DB_DIRECTORY)
        try:
            env = run_prediction_for_video(video_filename)
            result += f" Environment predicted as: {env}."
        except Exception as env_e:
            log(f"Env prediction error: {env_e}")
        return result
    except Exception as e:
        log(f"Error in detect_objects_in_video: {e}")
        return f"Error: {str(e)}"

@mcp.tool()
def get_video_report(database_name: str, video_filename: str) -> dict:
    """Get a high-level behavioral and technical report for a specific video.
    Returns technical metadata (FPS, duration) and an event timeline of objects.
    """
    log(f"Tool called: get_video_report for {video_filename} in {database_name}")
    try:
        db_path = get_db_path(database_name)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Get metadata
        cur.execute("SELECT * FROM video_metadata WHERE video_name = ?", (video_filename,))
        metadata_row = cur.fetchone()
        
        # Get events
        cur.execute("SELECT * FROM video_events WHERE video_name = ?", (video_filename,))
        events_rows = cur.fetchall()
        
        # Get environment (if exists)
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='video_environments'")
        env_table_exists = cur.fetchone()
        env = "Unknown"
        if env_table_exists:
            cur.execute("SELECT predicted_environment FROM video_environments WHERE video_name = ?", (video_filename,))
            env_row = cur.fetchone()
            if env_row: env = env_row[0]

        conn.close()

        report = {
            "video": video_filename,
            "environment": env,
            "metadata": metadata_row if metadata_row else "No metadata found",
            "events_count": len(events_rows),
            "events_summary": [
                {
                    "object": row[2], # object_label
                    "class": row[3],  # object_name
                    "start_frame": row[4],
                    "end_frame": row[5],
                    "activity": row[6],
                    "avg_speed": row[7]
                } for row in events_rows
            ]
        }
        return report
    except Exception as e:
        log(f"Error in get_video_report: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    mcp.run()
