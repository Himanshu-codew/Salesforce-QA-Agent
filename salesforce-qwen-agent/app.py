"""
Salesforce Qwen Agent — FastAPI Application Entry Point

Serves the chat Web UI and handles WebSocket connections
for real-time agent interactions.
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import uuid

# ── Load environment variables ──
load_dotenv(override=True)

# ── Configure logging ──
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("salesforce-agent")

# ── Import application modules ──
from llm.qwen import QwenLLM
from mcp.client import SalesforceMCPClient
from mcp.registry import ToolRegistry
from mcp.executor import ToolExecutor
from agent.agent import SalesforceAgent
from utils.file_parser import parse_uploaded_file

# ── Upload directory setup ──
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Global instances ──
mcp_client: SalesforceMCPClient | None = None
tool_registry: ToolRegistry | None = None
tool_executor: ToolExecutor | None = None
llm: QwenLLM | None = None
agent: SalesforceAgent | None = None


def create_mcp_client() -> SalesforceMCPClient:
    """Create and configure the Salesforce MCP Client."""
    return SalesforceMCPClient(
        mcp_url=os.getenv("SALESFORCE_MCP_URL", ""),
        instance_url=os.getenv("SALESFORCE_INSTANCE_URL", ""),
        client_id=os.getenv("SALESFORCE_CLIENT_ID", ""),
        client_secret=os.getenv("SALESFORCE_CLIENT_SECRET", ""),
        username=os.getenv("SALESFORCE_USERNAME", ""),
        password=os.getenv("SALESFORCE_PASSWORD", ""),
        security_token=os.getenv("SALESFORCE_SECURITY_TOKEN", ""),
        domain=os.getenv("SALESFORCE_DOMAIN", "login"),
        access_token=os.getenv("SALESFORCE_ACCESS_TOKEN"),
        refresh_token=os.getenv("SALESFORCE_REFRESH_TOKEN"),
    )


def create_llm() -> QwenLLM:
    """Create and configure the Qwen LLM client."""
    api_key = os.getenv("QWEN_API_KEY", "")
    base_url = os.getenv("QWEN_BASE_URL", "http://localhost:11434/v1")
    model = os.getenv(
        "QWEN_MODEL",
        "hf.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:UD-Q4_K_XL"
    )

    if not api_key:
        if "localhost" in base_url or "127.0.0.1" in base_url or "ollama" in base_url or "ngrok" in base_url:
            api_key = "ollama"
        else:
            logger.warning(
                "QWEN_API_KEY not set! Setting fallback key for OpenAI client."
            )
            api_key = "dummy"

    return QwenLLM(
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


# ── Application Lifecycle ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle for the FastAPI app."""
    global mcp_client, tool_registry, tool_executor, llm, agent

    logger.info("=" * 60)
    logger.info("Starting Salesforce Qwen Agent...")
    logger.info("=" * 60)

    # 1. Initialize MCP Client
    mcp_client = create_mcp_client()
    try:
        await mcp_client.connect()
        logger.info("✅ Salesforce MCP Client connected.")
    except Exception as e:
        logger.warning(f"⚠️ MCP connection issue: {e}. Will use REST API fallback.")

    # 2. Initialize Tool Registry
    tool_registry = ToolRegistry()
    await tool_registry.initialize(mcp_client)
    logger.info(f"✅ Tool Registry: {len(tool_registry)} tools registered.")

    # 3. Initialize LLM
    llm = create_llm()
    logger.info("✅ Qwen3 LLM client initialized.")

    # 4. Initialize Tool Executor
    tool_executor = ToolExecutor(mcp_client, tool_registry)

    # 5. Create Agent
    agent = SalesforceAgent(
        llm=llm,
        executor=tool_executor,
        max_iterations=int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "10")),
        max_history=int(os.getenv("MAX_CONVERSATION_HISTORY", "20")),
    )
    logger.info("✅ Salesforce Agent ready! Server running on port 8000.")
    logger.info("=" * 60)

    yield  # App is running

    # Shutdown
    logger.info("Shutting down Salesforce Qwen Agent...")
    if mcp_client:
        await mcp_client.disconnect()
    if llm:
        await llm.close()
    logger.info("Goodbye!")


# ── FastAPI App ──
app = FastAPI(
    title="Salesforce Qwen Agent",
    description="AI-powered Salesforce assistant using Qwen3 Instruct and MCP",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ═══════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════

@app.get("/")
async def serve_ui():
    """Serve the chat UI."""
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "mcp_connected": mcp_client.is_connected if mcp_client else False,
        "tools_registered": len(tool_registry) if tool_registry else 0,
    }


@app.post("/upload")
async def upload_file_endpoint(file: UploadFile = File(...)):
    """
    Upload and parse a file (CSV, Excel, PDF, JSON, TXT, Image).
    Returns parsed summary, tabular preview, and file metadata.
    """
    try:
        file_id = str(uuid.uuid4())[:8]
        clean_filename = os.path.basename(file.filename or "upload")
        saved_filename = f"{file_id}_{clean_filename}"
        file_path = os.path.join(UPLOAD_DIR, saved_filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Also keep a clean copy with exact filename for attachment lookup
        clean_copy_path = os.path.join(UPLOAD_DIR, clean_filename)
        shutil.copyfile(file_path, clean_copy_path)

        parsed_info = parse_uploaded_file(file_path, clean_filename)
        parsed_info["file_id"] = file_id
        parsed_info["saved_path"] = file_path

        # Bulletproof JSON serialization guarantee (handles all pandas/numpy types & NaNs)
        json_safe_info = json.loads(json.dumps(parsed_info, default=str))

        logger.info(f"📁 File uploaded & parsed: {clean_filename} ({parsed_info.get('file_type')})")
        return JSONResponse(status_code=200, content=json_safe_info)
    except Exception as e:
        logger.error(f"File upload error for {file.filename}: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Failed to parse file: {str(e)}"})


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    file_info: dict | None = None


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    HTTP-based chat endpoint (alternative to WebSocket).
    Returns the full response after all tool calls complete.
    """
    if not agent:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent not initialized"},
        )

    user_message = request.message.strip()
    if request.file_info:
        summary = request.file_info.get("summary", "")
        preview = request.file_info.get("content_preview", "")
        filename = request.file_info.get("filename", "uploaded_file")
        if user_message:
            user_message = f"[Attached File: {filename} ({summary})]\n{preview}\n\nUser Message: {user_message}"
        else:
            user_message = f"[Attached File: {filename} ({summary})]\n{preview}\n\nPlease analyze this file and proceed as requested."

    events = []
    async for event in agent.process_message(
        user_message, request.session_id
    ):
        events.append(event)

    # Extract the final response
    response_text = ""
    tool_calls = []

    for event in events:
        if event["type"] == "response":
            response_text = event["data"]
        elif event["type"] == "confirmation":
            response_text = event["data"]
        elif event["type"] == "error":
            response_text = event["data"]
        elif event["type"] == "tool_call":
            tool_calls.append(event["data"])

    return {
        "response": response_text,
        "tool_calls": tool_calls,
        "session_id": request.session_id,
    }


# ═══════════════════════════════════════════
# WebSocket Chat
# ═══════════════════════════════════════════

@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat.
    Streams agent events (thinking, tool_call, tool_result, response)
    as they happen for a responsive UI experience.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: session={session_id}")

    try:
        while True:
            # Receive message from client
            raw = await websocket.receive_text()
            data = json.loads(raw)

            if data.get("type") == "clear":
                if agent:
                    agent.clear_session(session_id)
                continue

            if data.get("type") == "message":
                user_message = data.get("content", "").strip()
                file_info = data.get("file_info")

                if file_info:
                    summary = file_info.get("summary", "")
                    preview = file_info.get("content_preview", "")
                    filename = file_info.get("filename", "uploaded_file")
                    if user_message:
                        user_message = f"[Attached File: {filename} ({summary})]\n{preview}\n\nUser Message: {user_message}"
                    else:
                        user_message = f"[Attached File: {filename} ({summary})]\n{preview}\n\nPlease analyze this file and proceed as requested."

                if not user_message:
                    continue

                if not agent:
                    await websocket.send_json({
                        "type": "error",
                        "data": "Agent not initialized. Check server logs.",
                    })
                    continue

                # Stream agent events to the client
                try:
                    async for event in agent.process_message(
                        user_message, session_id
                    ):
                        try:
                            await websocket.send_json(event)
                        except (RuntimeError, WebSocketDisconnect) as send_err:
                            logger.warning(f"Client disconnected during event delivery ({session_id}): {send_err}")
                            break
                except Exception as e:
                    logger.error(f"Agent error: {e}", exc_info=True)
                    try:
                        await websocket.send_json({
                            "type": "error",
                            "data": f"An error occurred: {str(e)}",
                        })
                    except Exception:
                        pass

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: session={session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.info(f"WebSocket closed: session={session_id}")


# ═══════════════════════════════════════════
# Run
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import webbrowser
    from threading import Timer

    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))

    # Auto-open browser after a short delay so server is ready
    def open_browser():
        webbrowser.open(f"http://localhost:{port}")

    Timer(1.5, open_browser).start()

    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
