"""
RAG-based Chatbot for AquaVision.

Retrieves detection data from SQLite databases and feeds it as context
to the LLM so the model can answer questions about detected objects,
activities, environments, and video metadata — no MCP required.
"""

from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os
import sys
import glob
import sqlite3

# ── Path Setup ────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))


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
    """Convert provider exceptions into short user-facing messages."""
    error_text = str(exc)
    lowered = error_text.lower()

    if "credit balance is too low" in lowered or "plans & billing" in lowered:
        return (
            "Groq API is connected, but your account balance is too low to process this request. "
            "Open the Groq console billing page, add credits if needed, and try again."
        )

    if "rate limit" in lowered or "429" in lowered:
        return (
            "Groq API rate limit reached. Please wait a moment and try again."
        )

    if "api key" in lowered or "authentication" in lowered or "unauthorized" in lowered:
        return (
            "Groq API authentication failed. Please check `GROQ_API_KEY` in `backend/.env`."
        )

    return "I’m having trouble connecting to Groq right now. Please try again in a moment."


class RAGRetriever:
    """Retrieves relevant detection data from the SQLite database
    and formats it as context documents for the LLM."""

    def __init__(self, db_directory: str):
        self.db_directory = os.path.abspath(db_directory)
        self.db_path = os.path.join(self.db_directory, "detections.db")

    def _connect(self):
        """Return a connection to the detection DB, or None if it doesn't exist."""
        if not os.path.exists(self.db_path):
            return None
        return sqlite3.connect(self.db_path)

    def get_all_context(self, user_query: str = "") -> str:
        """Retrieve all relevant detection context from the database.
        Returns a formatted string that can be injected into the LLM prompt."""

        conn = self._connect()
        if not conn:
            return "[No detection database found. Tell the user to upload and detect a video/image first.]"

        sections = []
        cur = conn.cursor()

        try:
            # 1. Video Metadata
            sections.append(self._get_video_metadata(cur))

            # 2. Detection Summary (unique objects with counts)
            sections.append(self._get_detection_summary(cur))

            # 3. Activity Breakdown
            sections.append(self._get_activity_breakdown(cur))

            # 4. Video Events Timeline
            sections.append(self._get_video_events(cur))

            # 5. Environment Prediction
            sections.append(self._get_environment(cur))

            # 6. Raw sample detections (limited, for specific queries)
            if any(kw in user_query.lower() for kw in [
                "frame", "speed", "when", "time", "second", "appear",
                "first", "last", "specific", "detail"
            ]):
                sections.append(self._get_sample_detections(cur))

        except Exception as e:
            sections.append(f"[Error retrieving data: {e}]")
        finally:
            conn.close()

        context = "\n\n".join(s for s in sections if s)
        return context if context.strip() else "[Database exists but contains no detection data yet.]"

    def _get_video_metadata(self, cur) -> str:
        """Get technical metadata for all processed videos."""
        try:
            cur.execute(
                "SELECT video_name, fps, total_frames, duration_seconds, width, height, processed_at "
                "FROM video_metadata ORDER BY processed_at DESC"
            )
            rows = cur.fetchall()
            if not rows:
                return ""

            lines = ["=== VIDEO METADATA ==="]
            for r in rows:
                name, fps, frames, duration, w, h, ts = r
                lines.append(
                    f"• Video: {name} | Resolution: {w}x{h} | FPS: {fps} | "
                    f"Frames: {frames} | Duration: {duration:.1f}s | Processed: {ts}"
                )
            return "\n".join(lines)
        except Exception:
            return ""

    def _get_detection_summary(self, cur) -> str:
        """Get unique object classes with their tracked counts."""
        try:
            cur.execute(
                "SELECT object_name, COUNT(DISTINCT object_label) as unique_count "
                "FROM detections GROUP BY object_name ORDER BY unique_count DESC"
            )
            rows = cur.fetchall()
            if not rows:
                return ""

            total_unique = sum(r[1] for r in rows)
            lines = [f"=== DETECTION SUMMARY ({total_unique} unique tracked objects) ==="]
            for name, count in rows:
                lines.append(f"• {name}: {count} unique instance(s)")
            return "\n".join(lines)
        except Exception:
            return ""

    def _get_activity_breakdown(self, cur) -> str:
        """Get activity distribution across all detections."""
        try:
            cur.execute(
                "SELECT activity, COUNT(*) as cnt FROM detections "
                "GROUP BY activity ORDER BY cnt DESC"
            )
            rows = cur.fetchall()
            if not rows:
                return ""

            total = sum(r[1] for r in rows)
            lines = ["=== ACTIVITY BREAKDOWN ==="]
            for activity, count in rows:
                pct = (count / total * 100) if total > 0 else 0
                lines.append(f"• {activity}: {count} detections ({pct:.1f}%)")
            return "\n".join(lines)
        except Exception:
            return ""

    def _get_video_events(self, cur) -> str:
        """Get behavioral event timeline for objects."""
        try:
            cur.execute(
                "SELECT video_name, object_label, object_name, start_frame, end_frame, "
                "dominant_activity, avg_speed FROM video_events ORDER BY start_frame"
            )
            rows = cur.fetchall()
            if not rows:
                return ""

            # Also get FPS for time conversion
            fps_map = {}
            try:
                cur.execute("SELECT video_name, fps FROM video_metadata")
                for vn, fps in cur.fetchall():
                    fps_map[vn] = fps
            except Exception:
                pass

            lines = ["=== OBJECT EVENT TIMELINE ==="]
            for video, label, name, sf, ef, activity, speed in rows:
                fps = fps_map.get(video, 30)
                start_sec = sf / fps if fps > 0 else 0
                end_sec = ef / fps if fps > 0 else 0
                lines.append(
                    f"• [{video}] {label} ({name}) — appeared frame {sf}-{ef} "
                    f"({start_sec:.1f}s - {end_sec:.1f}s) | activity: {activity} | avg speed: {speed:.1f}px/frame"
                )
            return "\n".join(lines)
        except Exception:
            return ""

    def _get_environment(self, cur) -> str:
        """Get predicted environment for videos."""
        try:
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='video_environments'")
            if not cur.fetchone():
                return ""

            cur.execute("SELECT video_name, predicted_environment FROM video_environments")
            rows = cur.fetchall()
            if not rows:
                return ""

            lines = ["=== PREDICTED ENVIRONMENT ==="]
            for video, env in rows:
                lines.append(f"• {video}: {env}")
            return "\n".join(lines)
        except Exception:
            return ""

    def _get_sample_detections(self, cur, limit: int = 50) -> str:
        """Get a sample of raw detections for frame-level queries."""
        try:
            cur.execute(
                "SELECT video_name, object_name, object_label, activity, speed, frame_number "
                "FROM detections ORDER BY frame_number LIMIT ?", (limit,)
            )
            rows = cur.fetchall()
            if not rows:
                return ""

            lines = [f"=== SAMPLE RAW DETECTIONS (first {len(rows)} rows) ==="]
            for video, name, label, activity, speed, frame in rows:
                lines.append(
                    f"  frame {frame}: {label} | activity: {activity} | speed: {speed:.1f}"
                )
            return "\n".join(lines)
        except Exception:
            return ""


class DatabaseChatbot:
    """RAG-based chatbot that answers questions about detected objects
    by retrieving context from the SQLite database and passing it to the LLM.
    
    Name kept as DatabaseChatbot for backward compatibility with main.py.
    """

    def __init__(self, db_directory: str, model_name: str | None = None):
        self.db_directory = os.path.abspath(db_directory)
        self.history_db = f"sqlite:///{os.path.join(self.db_directory, 'chat_history.db')}"
        self.retriever = RAGRetriever(db_directory)
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing from backend/.env")

        clear_dead_local_proxy()
        groq_model = model_name or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

        self.llm = ChatGroq(
            model=groq_model,
            api_key=groq_api_key,
            temperature=0,
        )

        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return (
            "You are AquaBot, a Senior Maritime Video Analysis Agent specialized in "
            "human-like reasoning over video/image detection data.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. You will receive DETECTION DATA as context below. Use ONLY this data to answer questions.\n"
            "2. If the data shows specific counts, report them accurately.\n"
            "3. Convert frame numbers to seconds using: seconds = frame_number / FPS (FPS is in the metadata).\n"
            "4. When asked about 'how many', count UNIQUE object instances (e.g., person_1, person_2 = 2 persons).\n"
            "5. Be conversational, clear, and concise. Use emojis sparingly for friendliness.\n"
            "6. If no detection data is available, tell the user to upload and process a video/image first.\n"
            "7. Never make up data that isn't in the provided context.\n"
        )

    def get_history(self, session_id: str):
        return SQLChatMessageHistory(session_id=session_id, connection=self.history_db)

    async def chat(self, user_query: str, session_id: str) -> str:
        """Process a user query using RAG: retrieve context → build prompt → get LLM response."""

        # 1. Retrieve relevant detection data from the database
        context = self.retriever.get_all_context(user_query)

        # 2. Load chat history
        history = self.get_history(session_id)
        past_messages = list(history.messages)

        # 3. Build the message chain
        messages = [
            SystemMessage(content=self.system_prompt),
            SystemMessage(content=f"--- DETECTION DATA (Retrieved from Database) ---\n{context}\n--- END OF DETECTION DATA ---"),
        ]

        # Add recent chat history (last 10 exchanges to keep context window manageable)
        recent_history = past_messages[-20:]  # 20 messages = ~10 exchanges
        messages.extend(recent_history)

        # Add the current user query
        messages.append(HumanMessage(content=user_query))

        # 4. Call the LLM
        try:
            response = await self.llm.ainvoke(messages)
            ai_response = response.content.strip()
        except Exception as e:
            ai_response = format_llm_error(e)

        # 5. Save to chat history
        history.add_user_message(user_query)
        history.add_ai_message(ai_response)

        return ai_response
