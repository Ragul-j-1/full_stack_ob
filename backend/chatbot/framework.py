from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.tools import Tool, StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import os
import sys
import glob
import asyncio

# Add paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
load_dotenv(os.path.join(ROOT_DIR, ".env"))
TOOLS_DIR = os.path.join(ROOT_DIR, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.append(TOOLS_DIR)

from tools.mcp_client import run_query_sync, run_list_tables_sync, run_list_databases_sync, run_detection_sync, run_report_sync

class DatabaseChatbot:
    def __init__(self, db_directory: str, model_name: str = "gpt-oss:120b-cloud"):
        self.db_directory = os.path.abspath(db_directory)
        self.history_db = f"sqlite:///{os.path.join(self.db_directory, 'chat_history.db')}"
        
        self.llm = ChatOllama(
            model=model_name, 
            base_url="http://127.0.0.1:11434",
            client_kwargs={'headers': {'Authorization': f'Bearer {os.environ.get("OLLAMA_API_KEY", "")}'}}
        )
        
        # Discover databases directly from filesystem (more robust than spawning MCP for discovery)
        self.databases = []
        try:
            db_files = glob.glob(os.path.join(self.db_directory, "*.db")) + glob.glob(os.path.join(self.db_directory, "*.sqlite"))
            self.databases = [os.path.basename(f) for f in db_files]
        except Exception as e:
            print(f"⚠️  Database discovery failed: {e}")

        self.schema_info = {}
        # We'll skip pre-populating schema_info with sycn calls to avoid loop issues.
        # The agent can use ListTablesTool if it needs schema info.
        
        self.system_prompt = self._build_system_prompt()
        self.agent = create_react_agent(self.llm, self._get_tools(), prompt=self.system_prompt)

    def _build_system_prompt(self) -> str:
        prompt = (
            "You are a Senior Maritime Video Analysis Agent. You are specialized in human-like reasoning over video detection data.\n\n"
            "CRITICAL KNOWLEDGE:\n"
            "1. To answer 'what happened' or get a 'summary', use the 'GetVideoReportTool'. It provides technical metadata (FPS, duration) and a behavioral timeline of distinct objects.\n"
            "2. To answer specific questions about individual objects (e.g., 'how many divers'), use 'SELECT COUNT(DISTINCT object_label)' in the 'detections' table.\n"
            "3. The 'video_metadata' table contains FPS and Total Frames. User seconds = frame_number / FPS.\n"
            "4. The 'video_events' table summarizes behavior: it shows when an object first appeared, when it left, and its dominant activity (e.g., 'fast movement').\n\n"
            f"Available Databases: {', '.join(self.databases) if self.databases else 'None discovered yet'}\n"
        )
        
        if self.schema_info:
            prompt += "\nInternal Schema Knowledge:\n"
            for db, tables in self.schema_info.items():
                prompt += f"- Database: {db}\n"
                if isinstance(tables, list):
                    for table_info in tables:
                        if isinstance(table_info, dict):
                            prompt += f"  * Table: {table_info.get('table')} (Cols: {', '.join(table_info.get('columns', []))})\n"
        
        prompt += "\nInstructions:\n"
        prompt += "1. Always check 'video_metadata' first for duration and FPS.\n"
        prompt += "2. Use GetVideoReportTool for overall behavioral summaries.\n"
        prompt += "3. Only run SELECT queries via DatabaseTool.\n"
        return prompt

    def _get_tools(self):
        from langchain_core.tools import tool
        db_dir = self.db_directory

        @tool
        async def sql_tool(database_name: str, query: str) -> str:
            """Execute a SQL SELECT query."""
            from tools.mcp_client import run_query
            result = await run_query(database_name, query, db_dir)
            return str(result)

        @tool
        async def list_tables_tool(database_name: str) -> str:
            """List tables and columns in a database."""
            from tools.mcp_client import run_list_tables
            result = await run_list_tables(database_name, db_dir)
            return str(result)

        @tool
        async def detect_tool(video_filename: str) -> str:
            """Detect objects, predict environment, and generate behavioral events/metadata."""
            from tools.mcp_client import run_detection
            result = await run_detection(video_filename, db_dir)
            return str(result)

        @tool
        async def report_tool(database_name: str, video_filename: str) -> str:
            """Get a summarized behavioral and technical report for a specific video.
            Args:
                database_name: The name of the database file (e.g., 'detections.db').
                video_filename: The name of the video file (e.g., 'video.mp4').
            """
            from tools.mcp_client import run_report
            result = await run_report(database_name, video_filename, db_dir)
            return str(result)

        return [list_tables_tool, sql_tool, detect_tool, report_tool]

    def get_history(self, session_id: str):
        return SQLChatMessageHistory(session_id=session_id, connection=self.history_db)

    async def chat(self, user_query: str, session_id: str) -> str:
        history = self.get_history(session_id)
        messages = [msg for msg in history.messages]
        
        # Inject most recent video context
        context_msg = ""
        try:
            import sqlite3
            db_path = os.path.join(self.db_directory, "detections.db")
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("SELECT video_name FROM video_metadata ORDER BY processed_at DESC LIMIT 1")
                row = cur.fetchone()
                if row:
                    context_msg = f"[System Note: The most recently processed video is '{row[0]}'. If the user asks about 'the video' without specifying a name, assume they mean this one.]\n\n"
                conn.close()
        except:
            pass
            
        augmented_query = f"{context_msg}{user_query}" if context_msg else user_query
        messages.append(HumanMessage(content=augmented_query))
        
        result = await self.agent.ainvoke({"messages": messages})
        ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
        response = ai_messages[-1].content if ai_messages else "Analysis failed."
        
        history.add_user_message(user_query) # Keep original query in history for clean UX
        history.add_ai_message(response)
        return response
