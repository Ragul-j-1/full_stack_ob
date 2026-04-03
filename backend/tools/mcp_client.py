import asyncio
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
import os

def _server_params(db_directory: str = "."):
    # Updated to point to tools/mcp_server.py
    server_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "mcp_server.py"))
    
    # Merge environment variables to include the database directory
    env = os.environ.copy()
    env["DB_DIRECTORY"] = os.path.abspath(db_directory)
    
    return StdioServerParameters(
        command="python",
        args=[server_path],
        env=env
    )

async def call_mcp_tool(tool_name, arguments, db_directory: str = ".", timeout: float = 15.0):
    """Generic helper to call MCP tools with a timeout to prevent hangs."""
    try:
        async with stdio_client(_server_params(db_directory)) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=10.0)
                
                result = await asyncio.wait_for(
                    session.call_tool(tool_name, arguments), 
                    timeout=timeout
                )
                return result.content
    except asyncio.TimeoutError:
        return f"Error: MCP session timed out while calling {tool_name}"
    except Exception as e:
        return f"Error: {str(e)}"

async def run_query(database_name, query, db_directory):
    return await call_mcp_tool("run_sql", {"database_name": database_name, "query": query}, db_directory)

async def run_list_tables(database_name, db_directory):
    return await call_mcp_tool("list_tables", {"database_name": database_name}, db_directory)

async def run_list_databases(db_directory):
    return await call_mcp_tool("list_databases", {}, db_directory)

async def run_detection(video_filename, db_directory):
    # Detection takes longer, increase the timeout significantly (e.g., 5 minutes = 300 seconds)
    return await call_mcp_tool("detect_objects_in_video", {"video_filename": video_filename}, db_directory, timeout=300.0)

async def run_report(database_name, video_filename, db_directory):
    return await call_mcp_tool("get_video_report", {"database_name": database_name, "video_filename": video_filename}, db_directory)

def run_query_sync(database_name, query, db_directory):
    try:
        return asyncio.run(run_query(database_name, query, db_directory))
    except Exception as e:
        return f"Sync Error: {str(e)}"

def run_list_tables_sync(database_name, db_directory):
    try:
        return asyncio.run(run_list_tables(database_name, db_directory))
    except Exception as e:
        return f"Sync Error: {str(e)}"

def run_list_databases_sync(db_directory):
    try:
        return asyncio.run(run_list_databases(db_directory))
    except Exception as e:
        return f"Sync Error: {str(e)}"

def run_detection_sync(video_filename, db_directory):
    try:
        return asyncio.run(run_detection(video_filename, db_directory))
    except Exception as e:
        return f"Sync Error: {str(e)}"

def run_report_sync(database_name, video_filename, db_directory):
    try:
        return asyncio.run(run_report(database_name, video_filename, db_directory))
    except Exception as e:
        return f"Sync Error: {str(e)}"
