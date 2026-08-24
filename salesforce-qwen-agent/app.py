"""
Salesforce Qwen Agent — FastAPI Application Entry Point

Serves the chat Web UI and handles WebSocket connections
for real-time agent interactions.
"""

import asyncio
import json
import logging
import os
import secrets
import shutil
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from xml.sax.saxutils import escape as xml_escape

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

    # Validate the Connected App consumer key in the background so OAuth
    # breakage is reported immediately (see /health) rather than mid-login.
    asyncio.create_task(_validate_connected_app())

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

# Pending OAuth flows awaiting callback: state -> flow metadata.
# Binds each authorization code to the exact auth host + Connected App
# credentials that started the flow (Salesforce requires the token exchange
# to hit the same environment that issued the code).
_oauth_pending_flows: dict[str, dict] = {}
_OAUTH_STATE_TTL_SECONDS = 600

# SOAP Partner API version used for direct credential logins.
SOAP_API_VERSION = "58.0"


def _prune_expired_states() -> None:
    """Evict stale pending OAuth states to bound memory usage."""
    cutoff = time.time() - _OAUTH_STATE_TTL_SECONDS
    expired = [s for s, meta in _oauth_pending_flows.items() if meta["created_at"] < cutoff]
    for s in expired:
        _oauth_pending_flows.pop(s, None)


def _resolve_auth_host(domain: str | None) -> str:
    """
    Map a user-selected environment to its Salesforce auth host:
      - Production / Developer Org  -> login.salesforce.com
      - Sandbox                     -> test.salesforce.com
      - Custom My Domain            -> e.g. mycompany.my.salesforce.com
    """
    d = (domain or "").strip().lower().replace("https://", "").replace("http://", "").rstrip("/")
    if d in ("", "login", "production", "prod", "developer", "dev"):
        return "login.salesforce.com"
    if d == "test":
        return "test.salesforce.com"
    return d


def _default_redirect_uri() -> str:
    port = os.getenv("APP_PORT", "8000")
    return os.getenv("SALESFORCE_REDIRECT_URI", f"http://localhost:{port}/api/auth/callback")


def _resolve_redirect_uri(request: Request) -> str:
    """
    Callback URI for this login attempt.
    An explicit SALESFORCE_REDIRECT_URI always wins; otherwise the callback
    mirrors the address the caller actually used (localhost, LAN IP, tunnel),
    so remote users aren't redirected back to their own machine's localhost.
    NOTE: every address variant must be whitelisted in the Connected App.
    """
    explicit = os.getenv("SALESFORCE_REDIRECT_URI", "").strip()
    if explicit:
        return explicit
    base = str(request.base_url).rstrip("/")
    return f"{base}/api/auth/callback"


# Runtime health of the configured Connected App consumer key. Probed once at
# startup so a dead/invalid key is reported immediately instead of failing
# silently inside the OAuth popup.
connected_app_status: dict = {"valid": None, "checked_at": 0.0, "detail": ""}


async def _validate_connected_app() -> None:
    """Probe Salesforce to confirm SALESFORCE_CLIENT_ID resolves to a live app."""
    client_id = os.getenv("SALESFORCE_CLIENT_ID", "").strip()
    if not client_id:
        connected_app_status.update(
            valid=False, checked_at=time.time(), detail="No Consumer Key configured."
        )
        return

    probe_url = (
        "https://login.salesforce.com/services/oauth2/authorize?"
        + urllib.parse.urlencode(
            {
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": "http://localhost/probe",
            }
        )
    )
    try:
        async with httpx.AsyncClient(timeout=10.0) as http:
            res = await http.get(probe_url, follow_redirects=False)
        if res.status_code in (200, 302):
            connected_app_status.update(valid=True, checked_at=time.time(), detail="")
            logger.info("✅ Connected App Consumer Key validated against Salesforce.")
        elif res.status_code == 400:
            connected_app_status.update(
                valid=False,
                checked_at=time.time(),
                detail=(
                    "Consumer Key rejected by Salesforce (invalid_client_id). The Connected App "
                    "was deleted or belongs to an expired org. Create a new one: Setup → App Manager "
                    "→ New Connected App → Enable OAuth (callback + scopes api/refresh_token/id), "
                    "then update SALESFORCE_CLIENT_ID and SALESFORCE_CLIENT_SECRET in .env and restart."
                ),
            )
            logger.warning("⚠️ %s", connected_app_status["detail"])
        else:
            connected_app_status.update(
                valid=False,
                checked_at=time.time(),
                detail=f"Unexpected Salesforce response validating Consumer Key: HTTP {res.status_code}",
            )
            logger.warning("⚠️ %s", connected_app_status["detail"])
    except Exception as e:
        connected_app_status.update(
            valid=None, checked_at=time.time(), detail=f"Could not reach Salesforce to validate Consumer Key: {e}"
        )


def _popup_html(title: str, body_html: str, success: bool) -> str:
    """Standalone result page rendered inside the OAuth popup window."""
    return f"""<!DOCTYPE html>
<html>
<head><title>{title}</title></head>
<body style="font-family: sans-serif; text-align: center; padding: 40px; background: #0f172a; color: white;">
    <h2>{'✅' if success else '❌'} {title}</h2>
    {body_html}
</body>
</html>"""


@app.get("/api/auth/login")
async def oauth_login(
    request: Request,
    session_id: str = Query("default"),
    domain: str = Query("login"),
    client_id: str | None = Query(None),
    client_secret: str | None = Query(None),
):
    """
    Redirect user to Salesforce OAuth 2.0 authorization URL.

    Multi-tenant aware:
      - The authorize host follows the caller-selected environment
        (login.salesforce.com, test.salesforce.com, or a custom My Domain).
      - Callers may supply their own org's Connected App consumer key/secret
        (BYO credentials); otherwise the server-wide app from .env is used.
        Salesforce Connected Apps are org-local, so external users from other
        orgs MUST authorize with a consumer key registered in THEIR org.
      - The callback URI mirrors the address the caller used (unless
        SALESFORCE_REDIRECT_URI is set), so remote users work too.
    """
    auth_host = _resolve_auth_host(domain)
    redirect_uri = _resolve_redirect_uri(request)
    effective_client_id = (client_id or os.getenv("SALESFORCE_CLIENT_ID", "")).strip()
    effective_client_secret = (client_secret or os.getenv("SALESFORCE_CLIENT_SECRET", "")).strip()

    if not effective_client_id:
        return HTMLResponse(
            content=_popup_html(
                "Missing Consumer Key",
                f"<p>No Connected App consumer key is configured. Create one under "
                f"<b>Setup → App Manager → New Connected App</b> with callback URL "
                f"<code>{xml_escape(redirect_uri)}</code>, then paste its Consumer Key &amp; Secret "
                f"in the Connect dialog.</p>",
                success=False,
            ),
            status_code=400,
        )

    # Bind this flow's parameters to an opaque state value so the callback
    # exchanges the code against the SAME host & credentials.
    state = secrets.token_urlsafe(24)
    _oauth_pending_flows[state] = {
        "session_id": session_id,
        "auth_host": auth_host,
        "client_id": effective_client_id,
        "client_secret": effective_client_secret,
        "redirect_uri": redirect_uri,
        "created_at": time.time(),
    }
    _prune_expired_states()

    oauth_url = (
        f"https://{auth_host}/services/oauth2/authorize?"
        + urllib.parse.urlencode(
            {
                "response_type": "code",
                "client_id": effective_client_id,
                "redirect_uri": redirect_uri,
                "state": state,
                "scope": os.getenv("SALESFORCE_OAUTH_SCOPE", "api refresh_token id"),
                "prompt": "consent",
            }
        )
    )
    logger.info(f"🔗 Initiating Salesforce OAuth login for session '{session_id}' -> https://{auth_host}/services/oauth2/authorize")
    return RedirectResponse(oauth_url)


@app.get("/api/auth/callback")
async def oauth_callback(request: Request):
    """
    Callback endpoint for the Salesforce OAuth redirect.

    Exchanges the authorization code for access tokens using the same auth host
    and Connected App credentials recorded at flow start (via the state param).
    Also gracefully renders Salesforce's ?error=... redirects (e.g.
    error=invalid_client_id) instead of failing with a validation error.
    """
    params = dict(request.query_params)
    state = params.get("state", "")
    flow = _oauth_pending_flows.pop(state, None) if state else None

    # ── Salesforce redirected back with an OAuth error ──
    oauth_error = params.get("error")
    if oauth_error:
        error_desc = params.get("error_description", "")
        callback_hint = xml_escape(flow["redirect_uri"]) if flow and flow.get("redirect_uri") else xml_escape(_default_redirect_uri())
        hint = ""
        if oauth_error == "invalid_client_id":
            hint = (
                "<p><b>Why does this happen?</b> Salesforce Connected Apps are org-local — a Consumer Key created in "
                "one org is not recognized by any other org.</p>"
                "<p><b>Fix:</b> In YOUR org go to <b>Setup → App Manager → New Connected App</b>, enable OAuth with "
                f"callback <code>{callback_hint}</code> and scopes "
                "<code>api, refresh_token, id</code>, then paste that app's Consumer Key &amp; Secret under "
                "<b>“Use my own Connected App”</b> in the Connect dialog and retry.</p>"
            )
        logger.error(f"OAuth error returned by Salesforce: {oauth_error} — {error_desc}")
        return HTMLResponse(
            content=_popup_html(
                "Salesforce Authorization Failed",
                f"<p><code>{xml_escape(oauth_error)}: {xml_escape(error_desc)}</code></p>{hint}",
                success=False,
            ),
            status_code=400,
        )

    code = params.get("code")
    if not code or not flow:
        return HTMLResponse(
            content=_popup_html(
                "Salesforce OAuth Failed",
                "<p>The authorization response was invalid or the login flow expired. Please restart the login.</p>",
                success=False,
            ),
            status_code=400,
        )

    session_id = flow["session_id"]
    token_url = f"https://{flow['auth_host']}/services/oauth2/token"
    token_params = {
        "grant_type": "authorization_code",
        "client_id": flow["client_id"],
        "client_secret": flow["client_secret"],
        "redirect_uri": flow["redirect_uri"],
        "code": code,
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            res = await http.post(token_url, data=token_params)

            if res.status_code != 200:
                try:
                    err_json = res.json()
                    err_text = f"{err_json.get('error', '')}: {err_json.get('error_description', '')}".strip(": ")
                except Exception:
                    err_text = res.text[:300]
                raise RuntimeError(f"Token exchange failed ({res.status_code}): {err_text}")

            token_data = res.json()
            access_token = token_data.get("access_token")
            refresh_token = token_data.get("refresh_token", "")
            instance_url = token_data.get("instance_url", "")

            # Fetch user identity (best-effort; must not block a valid token).
            display_name, email, username, org_id = "", "", "", ""
            id_url = token_data.get("id")
            if access_token and id_url:
                try:
                    id_res = await http.get(id_url, headers={"Authorization": f"Bearer {access_token}"})
                    if id_res.status_code == 200:
                        id_data = id_res.json()
                        display_name = id_data.get("display_name") or ""
                        email = id_data.get("email", "")
                        username = id_data.get("username", "")
                        org_id = id_data.get("organization_id", "")
                except Exception as ident_err:
                    logger.warning(f"Identity lookup failed during OAuth callback: {ident_err}")

            user_info = {
                "display_name": display_name or username or "Salesforce User",
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

            script_session_id = json.dumps(session_id)
            body_html = (
                "<p>Closing window and returning to Chat UI…</p>"
                "<script>"
                f"if (window.opener) {{ window.opener.postMessage({{ type: 'oauth_success', session_id: {script_session_id} }}, '*'); window.close(); }}"
                "else { window.location.href = '/'; }"
                "</script>"
            )
            return HTMLResponse(content=_popup_html("Connected to Salesforce!", body_html, success=True))

    except Exception as e:
        logger.error(f"OAuth callback error: {e}", exc_info=True)
        return HTMLResponse(
            content=_popup_html(
                "Salesforce OAuth Failed",
                f"<p>{xml_escape(str(e))}</p>",
                success=False,
            ),
            status_code=500,
        )


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


def _extract_soap_fault(response_text: str) -> tuple[str, str]:
    """Return (faultcode, faultstring) from a Salesforce SOAP fault envelope."""
    try:
        root = ET.fromstring(response_text)
    except ET.ParseError:
        return "", response_text.strip()[:300]
    fault = root.find(".//soapenv:Fault", {"soapenv": "http://schemas.xmlsoap.org/soap/envelope/"})
    if fault is None:
        return "", ""
    fault_code = (fault.findtext(".//faultcode", "") or "").strip()
    fault_string = (fault.findtext(".//faultstring", "") or "").strip()
    return fault_code, fault_string


def _friendly_direct_login_error(fault_code: str, fault_string: str) -> str:
    """Translate raw Salesforce SOAP faults into actionable user guidance."""
    blob = f"{fault_code} {fault_string}".upper()

    if "LOGIN_MUST_USE_SECURITY_TOKEN" in blob or ("INVALID_LOGIN" in blob and "TOKEN" in blob):
        return (
            "Salesforce rejected this login because it originates from an untrusted IP. "
            "Paste your Security Token in the 'Security Token' field "
            "(get it: Setup → My Personal Information → Reset Security Token), or add your "
            "IP under Setup → Security → Network Access."
        )
    if "INVALID_LOGIN" in blob or "INVALID_USERNAME" in blob or "INVALID_PASSWORD" in blob:
        return (
            "Invalid username, password, or security token. Note: when logging in from an "
            "untrusted IP/network you must append your Security Token to the password "
            "(enter it in the Security Token field)."
        )
    if "IP_RESTRICTED" in blob or "LOGIN_ADDRESS" in blob or "RESTRICTED_IP" in blob:
        return (
            "Your IP address is blocked by the org's login restrictions. Ask an admin to add it "
            "under Setup → Security → Network Access (Trusted IP Ranges)."
        )
    if "API_DISABLED" in blob or "UNSUPPORTED_CLIENT" in blob or "API_CURRENTLY_DISABLED" in blob:
        return (
            "API access is disabled for this org/user. Enable the 'API Enabled' permission "
            "(requires Enterprise/Unlimited/Developer edition) and retry."
        )
    if "ORG_LOCKED" in blob or "LOCKED_OUT" in blob or "PASSWORD_LOCKOUT" in blob:
        return "This user account is locked out. Wait a few minutes or ask an admin to unlock it under Setup → Users."
    if "SERVER_UNAVAILABLE" in blob or "REQUEST_LIMIT_EXCEEDED" in blob:
        return "Salesforce is temporarily unavailable or the org hit its API request limit. Please try again shortly."
    return f"Salesforce rejected the login: {fault_string or fault_code or 'unknown error'}"


@app.post("/api/auth/connect_direct")
async def connect_direct_endpoint(req: DirectConnectRequest):
    """
    Connect any user's Salesforce Org using Username + Password + Security Token
    OR Access Token + Instance URL.

    Direct password logins use the SOAP Partner API; from untrusted IPs
    Salesforce requires the Security Token appended to the password.
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

            auth_host = _resolve_auth_host(req.domain)
            sec_token = (req.security_token or "").strip()
            full_password = req.password + sec_token

            # SOAP Partner API login. XML-escape all user input to keep the
            # envelope well-formed (& < > are common in passwords).
            soap_body = (
                '<?xml version="1.0" encoding="utf-8"?>'
                '<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" '
                'xmlns:urn="urn:partner.soap.sforce.com">'
                "<soapenv:Body>"
                "<urn:login>"
                f"<urn:username>{xml_escape(req.username)}</urn:username>"
                f"<urn:password>{xml_escape(full_password)}</urn:password>"
                "</urn:login>"
                "</soapenv:Body>"
                "</soapenv:Envelope>"
            )

            fault_code, fault_string = "", ""
            rest_error = ""

            try:
                async with httpx.AsyncClient(timeout=20.0) as http:
                    # ── Attempt 1: SOAP Partner Login ──
                    try:
                        res = await http.post(
                            f"https://{auth_host}/services/Soap/u/{SOAP_API_VERSION}",
                            data=soap_body.encode("utf-8"),
                            headers={
                                "Content-Type": "text/xml; charset=utf-8",
                                "SOAPAction": "login",
                            },
                        )

                        if res.status_code == 200:
                            try:
                                root = ET.fromstring(res.text)
                                ns = {"urn": "urn:partner.soap.sforce.com"}
                                session_id_elem = root.findtext(".//urn:sessionId", "", ns) or ""
                                server_url = root.findtext(".//urn:serverUrl", "", ns) or ""

                                if session_id_elem:
                                    access_token = session_id_elem
                                    parsed = urllib.parse.urlparse(server_url)
                                    instance_url = (
                                        f"{parsed.scheme}://{parsed.netloc}"
                                        if parsed.netloc else f"https://{auth_host}"
                                    )
                                    display_name = root.findtext(".//urn:userFullName", "", ns) or ""
                                    email = root.findtext(".//urn:userEmail", "", ns) or ""
                            except ET.ParseError as parse_err:
                                logger.warning(f"Unexpected SOAP success payload: {parse_err}")
                        else:
                            fault_code, fault_string = _extract_soap_fault(res.text)
                            logger.warning(
                                f"SOAP login failed ({res.status_code}) via {auth_host}: "
                                f"{fault_code} {fault_string}"
                            )
                    except httpx.RequestError as req_err:
                        logger.warning(f"SOAP login unreachable at {auth_host}: {req_err}")

                    # ── Attempt 2: REST OAuth password grant fallback ──
                    if not access_token:
                        client_id = os.getenv("SALESFORCE_CLIENT_ID", "")
                        client_secret = os.getenv("SALESFORCE_CLIENT_SECRET", "")
                        if client_id and client_secret:
                            try:
                                rest_res = await http.post(
                                    f"https://{auth_host}/services/oauth2/token",
                                    data={
                                        "grant_type": "password",
                                        "client_id": client_id,
                                        "client_secret": client_secret,
                                        "username": req.username,
                                        "password": full_password,
                                    },
                                )
                                if rest_res.status_code == 200:
                                    rest_json = rest_res.json()
                                    access_token = rest_json.get("access_token", "")
                                    instance_url = rest_json.get("instance_url", "")
                                    id_url = rest_json.get("id", "")
                                    if id_url:
                                        id_res = await http.get(
                                            id_url, headers={"Authorization": f"Bearer {access_token}"}
                                        )
                                        if id_res.status_code == 200:
                                            id_data = id_res.json()
                                            display_name = id_data.get("display_name") or id_data.get("username", "")
                                            email = id_data.get("email", "")
                                elif not (fault_code or fault_string):
                                    try:
                                        err_json = rest_res.json()
                                        rest_error = (
                                            f"{err_json.get('error', '')}: {err_json.get('error_description', '')}"
                                        ).strip(": ")
                                    except Exception:
                                        rest_error = rest_res.text[:200]
                            except httpx.RequestError as rest_req_err:
                                rest_error = str(rest_req_err)
            except Exception as net_exc:
                logger.error(f"Direct connect network failure: {net_exc}", exc_info=True)
                return JSONResponse(
                    status_code=502,
                    content={"error": f"Could not reach Salesforce at https://{auth_host}. Check the selected Environment Domain."},
                )

            if not access_token:
                if fault_code or fault_string:
                    error_message = _friendly_direct_login_error(fault_code, fault_string)
                elif rest_error:
                    error_message = f"Salesforce rejected the login: {rest_error}"
                else:
                    error_message = (
                        "Invalid Salesforce Username or Password. If this org is a Sandbox, "
                        "switch Environment Domain to Sandbox; from untrusted networks also supply your Security Token."
                    )
                return JSONResponse(status_code=401, content={"error": error_message})

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
    checked_ago = (
        round(time.time() - connected_app_status["checked_at"], 1)
        if connected_app_status["checked_at"] else None
    )
    return {
        "status": "healthy",
        "mcp_connected": mcp_client.is_connected if mcp_client else False,
        "tools_registered": len(tool_registry) if tool_registry else 0,
        "connected_app": {
            "valid": connected_app_status["valid"],
            "detail": connected_app_status["detail"],
            "checked_seconds_ago": checked_ago,
            "oauth_login_available": connected_app_status["valid"] is not False,
        },
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
