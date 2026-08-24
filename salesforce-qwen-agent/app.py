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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import shutil
import uuid
import httpx

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

    # 6. Initialize User Session Manager
    from mcp.session_manager import session_manager
    session_manager.initialize_defaults(
        default_mcp_client=mcp_client,
        default_tool_registry=tool_registry,
        default_executor=tool_executor,
        default_agent=agent,
        llm=llm,
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
# OAuth 2.0 Multi-User Authentication Routes
# ═══════════════════════════════════════════
from mcp.session_manager import session_manager


@app.get("/api/auth/login")
async def oauth_login(
    session_id: str = Query("default"),
    domain: str = Query("login"),
):
    """
    Redirect user to Salesforce OAuth 2.0 authorization URL.
    Supports login.salesforce.com (production/dev) or test.salesforce.com (sandbox).
    """
    client_id = os.getenv("SALESFORCE_CLIENT_ID", "")
    redirect_uri = os.getenv("SALESFORCE_REDIRECT_URI", "http://localhost:8000/api/auth/callback")
    
    oauth_url = (
        f"https://{domain}.salesforce.com/services/oauth2/authorize"
        f"?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&state={session_id}"
    )
    logger.info(f"🔗 Initiating Salesforce OAuth login for session '{session_id}' -> {oauth_url}")
    return RedirectResponse(oauth_url)


@app.get("/api/auth/callback")
async def oauth_callback(
    code: str = Query(...),
    state: str = Query("default"),
):
    """
    Callback endpoint for Salesforce OAuth redirect.
    Exchanges authorization code for access token & user identity details.
    """
    session_id = state
    client_id = os.getenv("SALESFORCE_CLIENT_ID", "")
    client_secret = os.getenv("SALESFORCE_CLIENT_SECRET", "")
    redirect_uri = os.getenv("SALESFORCE_REDIRECT_URI", "http://localhost:8000/api/auth/callback")
    domain = os.getenv("SALESFORCE_DOMAIN", "login")

    token_url = f"https://{domain}.salesforce.com/services/oauth2/token"
    token_params = {
        "grant_type": "authorization_code",
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "code": code,
    }

    try:
        async with httpx.AsyncClient(timeout=15.0) as http:
            res = await http.post(token_url, data=token_params)
            res.raise_for_status()
            token_data = res.json()

            access_token = token_data.get("access_token")
            refresh_token = token_data.get("refresh_token", "")
            instance_url = token_data.get("instance_url")
            id_url = token_data.get("id")

            # Fetch user identity
            id_res = await http.get(id_url, headers={"Authorization": f"Bearer {access_token}"})
            id_res.raise_for_status()
            id_data = id_res.json()

            display_name = id_data.get("display_name") or id_data.get("username", "Salesforce User")
            email = id_data.get("email", "")
            username = id_data.get("username", "")
            org_id = id_data.get("organization_id", "")

            user_info = {
                "display_name": display_name,
                "email": email,
                "username": username,
                "org_id": org_id,
                "org_name": f"Org ({org_id[:8]})" if org_id else "Salesforce Org",
                "authenticated": True,
            }

            await session_manager.register_oauth_session(
                session_id=session_id,
                access_token=access_token,
                refresh_token=refresh_token,
                instance_url=instance_url,
                user_info=user_info,
            )

            # Return popup close HTML
            html_content = """
            <!DOCTYPE html>
            <html>
            <head><title>Salesforce Connected</title></head>
            <body style="font-family: sans-serif; text-align: center; padding: 40px; background: #0f172a; color: white;">
                <h2>✅ Connected to Salesforce!</h2>
                <p>Closing window and returning to Chat UI...</p>
                <script>
                    if (window.opener) {
                        window.opener.postMessage({ type: 'oauth_success', session_id: '""" + session_id + """' }, '*');
                        window.close();
                    } else {
                        window.location.href = '/';
                    }
                </script>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)

    except Exception as e:
        logger.error(f"OAuth callback error: {e}", exc_info=True)
        return HTMLResponse(content=f"<h2>❌ Salesforce OAuth Failed</h2><p>{str(e)}</p>", status_code=500)


@app.get("/api/auth/me")
async def get_user_me(session_id: str = Query("default")):
    """Get current user connection profile for session."""
    return session_manager.get_user_info(session_id)


class DirectConnectRequest(BaseModel):
    session_id: str = "default"
    mode: str = "password"  # "password" or "token"
    username: str | None = None
    password: str | None = None
    security_token: str | None = None
    instance_url: str | None = None
    access_token: str | None = None
    domain: str = "login"


@app.post("/api/auth/connect_direct")
async def connect_direct_endpoint(req: DirectConnectRequest):
    """
    Connect any user's Salesforce Org using Username + Password + Security Token
    OR Access Token + Instance URL.
    """
    try:
        access_token = ""
        instance_url = ""
        display_name = ""
        email = ""
        username = req.username or ""

        if req.mode == "password":
            if not req.username or not req.password:
                return JSONResponse(status_code=400, content={"error": "Username and Password are required."})
            
            domain = req.domain or "login"
            if "." in domain or "http" in domain:
                domain_clean = domain.replace("https://", "").replace("http://", "").rstrip("/")
                soap_url = f"https://{domain_clean}/services/Soap/u/58.0"
                domain_prefix = domain_clean
            else:
                soap_url = f"https://{domain}.salesforce.com/services/Soap/u/58.0"
                domain_prefix = f"{domain}.salesforce.com"

            sec_token = req.security_token or ""
            soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
            <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:urn="urn:partner.soap.sforce.com">
              <soapenv:Body>
                <urn:login>
                  <urn:username>{req.username}</urn:username>
                  <urn:password>{req.password}{sec_token}</urn:password>
                </urn:login>
              </soapenv:Body>
            </soapenv:Envelope>"""

            async with httpx.AsyncClient(timeout=15.0) as http:
                # Attempt 1: SOAP Partner Login
                res = await http.post(soap_url, data=soap_body, headers={"Content-Type": "text/xml", "SOAPAction": "login"})
                
                import xml.etree.ElementTree as ET
                import urllib.parse
                
                if res.status_code == 200:
                    try:
                        root = ET.fromstring(res.text)
                        ns = {"soap": "http://schemas.xmlsoap.org/soap/envelope/", "urn": "urn:partner.soap.sforce.com"}
                        session_id_elem = root.find(".//urn:sessionId", ns)
                        server_url_elem = root.find(".//urn:serverUrl", ns)
                        user_full_name_elem = root.find(".//urn:userFullName", ns)
                        user_email_elem = root.find(".//urn:userEmail", ns)

                        if session_id_elem is not None and session_id_elem.text:
                            access_token = session_id_elem.text
                            if server_url_elem is not None and server_url_elem.text:
                                parsed = urllib.parse.urlparse(server_url_elem.text)
                                instance_url = f"{parsed.scheme}://{parsed.netloc}"
                            if user_full_name_elem is not None and user_full_name_elem.text:
                                display_name = user_full_name_elem.text
                            if user_email_elem is not None and user_email_elem.text:
                                email = user_email_elem.text
                    except Exception:
                        pass

                # Attempt 2: REST OAuth Password Grant Fallback
                if not access_token:
                    client_id = os.getenv("SALESFORCE_CLIENT_ID", "")
                    client_secret = os.getenv("SALESFORCE_CLIENT_SECRET", "")
                    if client_id and client_secret:
                        token_url = f"https://{domain_prefix}/services/oauth2/token"
                        token_data = {
                            "grant_type": "password",
                            "client_id": client_id,
                            "client_secret": client_secret,
                            "username": req.username,
                            "password": f"{req.password}{sec_token}",
                        }
                        try:
                            rest_res = await http.post(token_url, data=token_data)
                            if rest_res.status_code == 200:
                                rest_json = rest_res.json()
                                access_token = rest_json.get("access_token", "")
                                instance_url = rest_json.get("instance_url", "")
                                id_url = rest_json.get("id", "")
                                if id_url:
                                    id_res = await http.get(id_url, headers={"Authorization": f"Bearer {access_token}"})
                                    if id_res.status_code == 200:
                                        id_data = id_res.json()
                                        display_name = id_data.get("display_name") or id_data.get("username", "")
                                        email = id_data.get("email", "")
                        except Exception:
                            pass

                if not access_token:
                    return JSONResponse(status_code=401, content={"error": "Invalid Salesforce Username or Password. Please check your credentials."})

        elif req.mode == "token":
            if not req.access_token or not req.instance_url:
                return JSONResponse(status_code=400, content={"error": "Access Token and Instance URL are required."})
            access_token = req.access_token.strip()
            instance_url = req.instance_url.strip().rstrip("/")

        if not access_token or not instance_url:
            return JSONResponse(status_code=400, content={"error": "Failed to authenticate with Salesforce credentials."})

        # Register session
        user_info = {
            "display_name": display_name or (username.split("@")[0].title() if username else "Salesforce User"),
            "email": email or username,
            "username": username,
            "org_name": f"Org ({instance_url.replace('https://', '')[:16]})",
            "authenticated": True,
        }

        await session_manager.register_oauth_session(
            session_id=req.session_id,
            access_token=access_token,
            refresh_token="",
            instance_url=instance_url,
            user_info=user_info,
        )

        logger.info(f"✅ User connected via direct credentials for session '{req.session_id}': {user_info['display_name']} ({instance_url})")
        return {"success": True, "session_id": req.session_id, "user": user_info}

    except Exception as e:
        logger.error(f"Direct Salesforce connection error: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Authentication failed: {str(e)}"})


@app.post("/api/auth/logout")
async def oauth_logout(session_id: str = Query("default")):
    """Logout and clear user session."""
    success = await session_manager.logout_session(session_id)
    return {"success": success, "session_id": session_id}


# ═══════════════════════════════════════════
# Core Routes & App Endpoints
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

        clean_copy_path = os.path.join(UPLOAD_DIR, clean_filename)
        shutil.copyfile(file_path, clean_copy_path)

        parsed_info = parse_uploaded_file(file_path, clean_filename)
        parsed_info["file_id"] = file_id
        parsed_info["saved_path"] = file_path

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


# ── Session active uploaded files mapping ──
session_files: dict[str, dict] = {}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    HTTP-based chat endpoint (alternative to WebSocket).
    Returns the full response after all tool calls complete.
    """
    target_agent = await session_manager.get_or_create_agent(request.session_id)
    if not target_agent:
        return JSONResponse(
            status_code=503,
            content={"error": "Agent not initialized"},
        )

    if request.file_info:
        session_files[request.session_id] = request.file_info
    elif request.session_id in session_files:
        request.file_info = session_files[request.session_id]

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
    async for event in target_agent.process_message(
        user_message, request.session_id
    ):
        events.append(event)

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
            raw = await websocket.receive_text()
            data = json.loads(raw)

            target_agent = await session_manager.get_or_create_agent(session_id)

            if data.get("type") == "clear":
                session_files.pop(session_id, None)
                if target_agent:
                    target_agent.clear_session(session_id)
                continue

            if data.get("type") == "message":
                user_message = data.get("content", "").strip()
                file_info = data.get("file_info")

                if file_info:
                    session_files[session_id] = file_info
                elif session_id in session_files:
                    file_info = session_files[session_id]

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

                if not target_agent:
                    await websocket.send_json({
                        "type": "error",
                        "data": "Agent not initialized. Check server logs.",
                    })
                    continue

                # Stream agent events to the client
                try:
                    async for event in target_agent.process_message(
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
        reload_excludes=["uploads/*", "*.csv", "*.html", "*.xlsx", "*.log", ".pytest_cache/*", "__pycache__/*"],
        log_level="info",
    )
