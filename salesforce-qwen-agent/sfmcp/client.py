"""
Salesforce MCP Client — connects to Salesforce's hosted MCP Server
via the official mcp SDK (Streamable HTTP transport) with OAuth Bearer
token authentication, envelope-encrypted token storage and auto-refresh.
"""

import json
import logging
import os
import time
from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from sfmcp.crypto.envelope import TokenVault
from tools.salesforce import is_mutating, is_destructive

logger = logging.getLogger(__name__)


def _extract_sobject(arguments: dict[str, Any]) -> str:
    for key in ["sobject-name", "sobject_name", "sobject", "object", "sobjectName", "objectName"]:
        if key in arguments and arguments[key]:
            return str(arguments[key]).strip()
    return ""


def _extract_body(arguments: dict[str, Any]) -> dict[str, Any]:
    if "body" in arguments and isinstance(arguments["body"], dict):
        return dict(arguments["body"])
    if "fields" in arguments and isinstance(arguments["fields"], dict):
        return dict(arguments["fields"])
    if "record" in arguments and isinstance(arguments["record"], dict):
        return dict(arguments["record"])
    ignore_keys = {"sobject-name", "sobject_name", "sobject", "object", "sobjectName", "objectName", "id", "record_id", "relationship-path", "relationship_path"}
    return {k: v for k, v in arguments.items() if k not in ignore_keys}


# Picklist values attached to a required/enum field are capped so the
# deterministic required-fields ask stays compact. Values are the ACTIVE
# picklist entries in Describe order (deterministic, mirroring Salesforce).
_PICKLIST_VALUES_CAP = 12


def _simplify_describe_fields(raw: dict) -> list[dict]:
    """Flatten a raw /sobjects/X/describe payload into a minimal field list.

    A field is REQUIRED for create when it is createable, non-nillable and NOT
    defaulted on create (Salesforce supplies the value otherwise — e.g. Task
    Status). Never guessed here; this mirrors Salesforce's own metadata.

    The minimal record also keeps the LIVE metadata the deterministic
    required-fields ask needs (A8): active picklist values for enum fields and
    the referenced object names for lookup/reference fields — labels and hints
    are always derived from Describe, never from a hard-coded per-object list.
    """
    simplified = []
    for f in raw.get("fields", []):
        req = not f.get("nillable") and not f.get("defaultedOnCreate") and bool(f.get("createable"))
        info = {
            "name": f.get("name"),
            "label": f.get("label"),
            "type": f.get("type"),
        }
        if req:
            info["required"] = True
        if f.get("type") == "picklist" and f.get("picklistValues"):
            active_vals = [p.get("value") for p in f.get("picklistValues", []) if p.get("active")]
            if active_vals:
                info["picklist_values"] = active_vals[:_PICKLIST_VALUES_CAP]
        if f.get("type") == "reference" and f.get("referenceTo"):
            refs = [r for r in f.get("referenceTo", []) if str(r).strip()]
            if refs:
                info["reference_to"] = refs
        simplified.append(info)
    return simplified


def _extract_required_from_describe(fields: list[dict]) -> list[tuple[str, str]]:
    """Return [(api_name, human_label)] for the required fields in a simplified
    field list (already marked 'required' by _simplify_describe_fields)."""
    return [
        (f.get("name", ""), f.get("label", "") or f.get("name", ""))
        for f in fields
        if f.get("required") and f.get("name")
    ]


def _extract_required_field_options(fields: list[dict]) -> dict[str, list[str]]:
    """Return {api_name: [choice_hints]} for the REQUIRED fields in a simplified
    field list.

    - picklist   -> the active picklist values (a user must choose one)
    - reference  -> the referenced sObject names (the user must supply a
                    well-formed record ID of one of these objects)

    Only required fields are keyed, the order is the Describe field order, and
    every value derives from live metadata (never a hard-coded list).
    """
    options: dict[str, list[str]] = {}
    for f in fields:
        if not f.get("required") or not f.get("name"):
            continue
        if f.get("type") == "picklist" and f.get("picklist_values"):
            options[str(f["name"])] = [str(v) for v in f["picklist_values"]]
        elif f.get("type") == "reference" and f.get("reference_to"):
            options[str(f["name"])] = [str(o) for o in f["reference_to"]]
    return options


class SalesforceMCPClient:
    """
    Manages connection to the Salesforce MCP Server.

    Handles:
    - OAuth 2.0 Authorization Code + PKCE token management (refresh, vault)
    - Envelope-encrypted token persistence via TokenVault
    - Streamable HTTP transport via the official mcp SDK
    - Session lifecycle (initialize -> use -> close)
    - Auto-reauthentication on 401/expired token
    - REST API fallback when the MCP session is unavailable
    """

    def __init__(
        self,
        mcp_url: str,
        instance_url: str,
        client_id: str,
        client_secret: str,
        username: str,
        password: str,
        security_token: str,
        domain: str = "login",
        access_token: str | None = None,
        refresh_token: str | None = None,
        expires_at: float = 0.0,
        oauth_scope: str | None = None,
        auth_host: str | None = None,
        token_vault: TokenVault | None = None,
        session_id: str | None = None,
    ):
        self.mcp_url = mcp_url
        self.instance_url = instance_url
        self.client_id = client_id
        self.client_secret = client_secret
        self.username = username
        self.password = password
        self.security_token = security_token
        self.domain = domain

        self._access_token = access_token
        self._refresh_token = refresh_token
        self._expires_at = expires_at
        self.token_scopes: str | None = None
        self.oauth_scope = oauth_scope
        self.auth_host = auth_host
        self.token_vault = token_vault
        self.session_id = session_id

        self._session = None
        self._mcp_ctx = None
        self._mcp_read = None
        self._mcp_write = None
        self._connected = False
        self._name_map: dict[str, str] = {}
        self._schema_cache: dict[str, Any] = {}
        self._schema_cache_order: list[str] = []  # FIFO eviction order
        self._schema_cache_max = 30  # Max cached schemas to bound memory
        # Tracks whether MCP has been successfully initialized at least once.
        # Unlike `_connected` (which is cleared when the idle session is closed
        # after tool discovery), this stays True so /health can accurately
        # report that live call_tool() reconnects and executes via MCP.
        self.mcp_transport = "REST"

        self._http_client = httpx.AsyncClient(timeout=60.0)
        self._session_vault_id: str | None = None

        # When true, MCP is the only acceptable transport: any MCP failure is
        # reported loudly and the REST/local fallback is NOT used as a silent
        # substitute. Controlled by SALESFORCE_MCP_REQUIRED=true.
        self.mcp_required = os.getenv("SALESFORCE_MCP_REQUIRED", "false").lower() in (
            "true", "1", "yes", "on",
        )
        # One-time noise reduction: when the hosted MCP server rejects us with
        # 401 (token lacks MCP scope), emit a single actionable warning and keep
        # subsequent boots/retries quiet so the REST-fallback state is signaled
        # but not noisy. Clear on a successful MCP init so a real fix is always
        # reported.
        self._mcp_401_warned = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def access_token(self) -> str | None:
        return self._access_token

    # ──────────────────────────────────────────────────────────
    # OAuth Authentication (Authorization Code + PKCE only)
    # ──────────────────────────────────────────────────────────

    async def authenticate(self) -> str:
        """
        Authenticate with Salesforce.

        Only OAuth 2.0 Authorization Code + PKCE is supported. Interactive user
        sessions obtain tokens via the browser popup flow (/api/auth/login).
        The server-side default client waits for a user to authenticate before
        it can connect to MCP.
        """
        # Try a refresh_token grant first if we have one (token lifecycle).
        if self._refresh_token:
            if await self._try_oauth_refresh():
                return self._access_token or ""

        # Try loading an existing PKCE token from the encrypted vault.
        if self._load_mcp_scoped_token_from_vault():
            return self._access_token or ""

        logger.warning(
            "No PKCE token available. Please authenticate via the browser login flow "
            "(click 'Connect to Salesforce' in the UI) before MCP can connect. "
            "If you just authenticated, your session may have expired — please login again."
        )
        return ""

    async def refresh_access_token(self) -> str:
        """Refresh the access token via OAuth refresh_token grant."""
        if await self._try_oauth_refresh():
            return self._access_token or ""
        return await self.authenticate()

    # ──────────────────────────────────────────────────────────
    # MCP Connection
    # ──────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """
        Establish connection to the Salesforce MCP Server.
        Ensures a fresh OAuth token (auto-refresh), then opens a
        Streamable HTTP session via the official mcp SDK.
        """
        try:
            await self._ensure_fresh_token()
        except Exception as e:
            logger.warning(f"Initial token refresh warning ({e}). Will retry on tool execution.")

        logger.info(f"Connecting to Salesforce MCP Server: {self.mcp_url}")
        await self._ensure_connected()
        if self._session is not None:
            logger.info("MCP Client ready (Streamable HTTP transport).")
        elif self._mcp_401_warned:
            logger.debug("MCP session not established (401); REST fallback active.")
        else:
            logger.warning("MCP session not established; falls back to REST API.")

    # ──────────────────────────────────────────────────────────
    # OAuth token lifecycle (auto-refresh)
    # ──────────────────────────────────────────────────────────

    def _token_url(self) -> str:
        auth_host = (self.auth_host or self.domain or "login").strip().rstrip("/")
        if auth_host.startswith(("https://", "http://")):
            return f"{auth_host}/services/oauth2/token"
        if "." in auth_host:
            return f"https://{auth_host}/services/oauth2/token"
        return f"https://{auth_host}.salesforce.com/services/oauth2/token"

    async def _try_oauth_refresh(self) -> bool:
        """Refresh via OAuth refresh_token grant. Returns True on success."""
        if not self._refresh_token:
            return False
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        # Preserve MCP scope on refresh so a refreshed token stays MCP-capable.
        if self.oauth_scope:
            payload["scope"] = self.oauth_scope
        try:
            response = await self._http_client.post(
                self._token_url(),
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            data = response.json()
            new_token = data.get("access_token")
            if not new_token:
                logger.warning("OAuth refresh_token grant returned no access_token.")
                return False
            self._access_token = new_token
            self.token_scopes = data.get("scope") or getattr(self, "token_scopes", None)
            self._refresh_token = data.get("refresh_token", self._refresh_token)
            self._expires_at = time.time() + int(data.get("expires_in", 3600))
            if data.get("instance_url"):
                self.instance_url = data["instance_url"]
            self._persist_tokens()
            logger.info("Access token refreshed via OAuth refresh_token grant.")
            return True
        except Exception as e:
            # On 400 errors (invalid_grant, token_revoked, etc.), the refresh token
            # is stale/rotated/expired. Remove it from the vault to prevent repeated
            # failed attempts and force re-authentication.
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code == 400 and self.token_vault and self.session_id:
                logger.warning(
                    f"OAuth refresh_token grant failed with 400 (stale/rotated token). "
                    f"Removing stale vault entry for session '{self.session_id}'."
                )
                try:
                    self.token_vault.delete(self.session_id)
                except Exception as delete_err:
                    logger.warning(f"Failed to delete stale vault entry: {delete_err}")
                # Clear the stale refresh token so we don't retry it
                self._refresh_token = None
            else:
                logger.warning(f"OAuth refresh_token grant failed: {e}")
            return False

    async def _ensure_fresh_token(self) -> None:
        """Refresh proactively if the current token is missing/expired. Never raises."""
        if not self._needs_mcp_token():
            return
        # Try loading a live PKCE token from the vault first.
        self._load_mcp_scoped_token_from_vault()
        token_unusable = (
            not self._access_token
            or not self._expires_at
            or time.time() > self._expires_at - 30
        )
        if self._refresh_token and token_unusable:
            if await self._try_oauth_refresh():
                return
        if not self._access_token or (
            self._expires_at and time.time() > self._expires_at - 30
        ):
            try:
                await self.authenticate()
            except Exception as e:
                logger.warning(f"Token acquisition failed: {e}")

    def _needs_mcp_token(self) -> bool:
        """True when the current token is missing/expired and would need replacement."""
        return (
            not self._access_token
            or not self._expires_at
            or time.time() > self._expires_at - 30
        )

    @staticmethod
    def _scope_has_mcp_capability(scope: str) -> bool:
        """Return True if *scope* carries any MCP-authorizing OAuth scope string."""
        s = scope.lower()
        if "sfap:mcp" in s:
            return True
        if "sfap_api" in s or "mcp_api" in s:
            return True
        return False

    def _load_mcp_scoped_token_from_vault(self) -> bool:
        """
        Load a live MCP-capable OAuth token from the token vault.

        The interactive /api/auth/login flow stores a token carrying the
        scopes required by the hosted MCP server (sfap_api, mcp_api, or the
        legacy sfap:mcp:* names). When present and not expired, this token is
        used for MCP tool calls.

        SECURITY: Only adopt tokens for the current session_id to prevent
        cross-session token leakage.
        """
        if not self.token_vault:
            return False
        
        # SECURITY: Only look up our own session_id, not all vault sessions.
        # This prevents cross-session token leakage when multiple users are authenticated.
        sid = self.session_id
        if not sid:
            return False
            
        try:
            rec = self.token_vault.get(sid)
            if not rec:
                return False
                
            scope = rec.get("oauth_scope") or ""
            if not self._scope_has_mcp_capability(scope):
                return False
                
            token = rec.get("access_token") or ""
            if not token:
                return False
                
            # Only adopt a scoped token that has a concrete, future expiry
            # (register_oauth_session stores now + expires_in). Tokens with
            # an unknown/zero or black-expired expiry are stale and would
            # break REST fallback too, so skip them.
            expires_at = float(rec.get("expires_at") or 0.0)
            if expires_at <= 0 or time.time() > expires_at - 30:
                # The MCP-capable access token is expired, but this record is
                # still a valid OAuth session: it carries a long-lived refresh
                # token + client credentials + the MCP scope. Load those so the
                # caller's _try_oauth_refresh() can refresh MCP-capably (and
                # _persist_tokens() handles refresh-token rotation). We must NOT
                # adopt the stale access_token here (that would break REST
                # fallback), hence we only populate the refresh path.
                self._session_vault_id = sid
                refresh_tok = rec.get("refresh_token")
                if refresh_tok:
                    self._refresh_token = refresh_tok
                self.oauth_scope = self.oauth_scope or scope
                if rec.get("instance_url"):
                    self.instance_url = rec["instance_url"]
                self.auth_host = rec.get("auth_host") or self.auth_host
                if rec.get("client_id"):
                    self.client_id = rec["client_id"]
                if rec.get("client_secret"):
                    self.client_secret = rec["client_secret"]
                logger.info(
                    f"Vault session '{sid}' MCP token expired (expires_at={expires_at:.0f}) "
                    f"but has a refresh token; queued for OAuth refresh."
                )
                return False
                
            self._access_token = token
            self._refresh_token = rec.get("refresh_token") or self._refresh_token
            if rec.get("instance_url"):
                self.instance_url = rec["instance_url"]
            self._expires_at = expires_at
            self._session_vault_id = sid
            logger.info(
                f"Using MCP-capable OAuth token from vault session '{sid}' (scope={scope!r})."
            )
            return True
        except Exception as e:
            logger.warning(f"Could not load scoped token from vault: {e}")
        return False

    def _persist_tokens(self) -> None:
        """Encrypt the current token state back into the vault (no-op without a vault)."""
        if not self.token_vault or not self.session_id:
            return
        try:
            self.token_vault.update(
                self.session_id,
                access_token=self._access_token,
                refresh_token=self._refresh_token,
                instance_url=self.instance_url,
                expires_at=self._expires_at or time.time() + 3600,
                client_id=self.client_id,
                client_secret=self.client_secret,
                oauth_scope=self.oauth_scope,
                auth_host=self.auth_host or self.domain,
            )
        except Exception as e:
            logger.error(f"Failed to persist tokens to vault: {e}")

    # ──────────────────────────────────────────────────────────
    # MCP SDK session (Streamable HTTP)
    # ──────────────────────────────────────────────────────────

    async def _ensure_connected(self) -> None:
        """Lazily open the mcp SDK session if it is not already active."""
        if self._session is not None:
            return
        await self._ensure_fresh_token()
        if not self._access_token:
            logger.warning("No access token available; MCP session cannot be opened.")
            return
        try:
            logger.info(f"[MCP] Connecting... → {self.mcp_url}")
            import httpx2
            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            }
            # The mcp SDK (2.x) is built on httpx2; auth must ride on a
            # pre-configured http_client (httpx2.HTTPStatusError is not httpx's).
            self._mcp_ctx = streamable_http_client(
                self.mcp_url,
                http_client=httpx2.AsyncClient(timeout=120.0, headers=headers),
            )
            transport_streams = await self._mcp_ctx.__aenter__()
            self._mcp_read, self._mcp_write = transport_streams
            self._session = ClientSession(self._mcp_read, self._mcp_write)
            await self._session.__aenter__()
            await self._session.initialize()
            self._connected = True
            self.mcp_transport = "MCP"
            self._name_map = {}
            self._mcp_401_warned = False
            logger.info("[MCP] MCP SDK session initialized; is_connected=True. Transport: MCP")
        except Exception as e:
            # Noise reduction: a persistent 401 (token lacks MCP scope) is
            # reported ONCE with actionable guidance; subsequent boots/retries
            # stay quiet with a DEBUG line. Non-auth failures are still logged at
            # WARNING every time because they can be transient or genuine breakage.
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            is_auth_rejection = status_code == 401 or "Unauthorized" in str(e)
            if is_auth_rejection:
                if not self._mcp_401_warned:
                    logger.warning(
                        "[MCP] Session init rejected (401 Unauthorized). The token lacks the "
                        "MCP OAuth scope the hosted server requires; using REST fallback. Verify "
                        "the Connected App allows the MCP scopes (mcp_api / sfap_api) and that the "
                        "login is OAuth-scoped, then restart."
                    )
                    self._mcp_401_warned = True
                else:
                    logger.debug("[MCP] Session init rejected (401); REST fallback active.")
            else:
                logger.warning(f"[MCP] Session init failed: {e}.")
            self._connected = False
            await self._close_mcp_session()
            if self.mcp_required:
                raise RuntimeError(
                    "MCP is required (SALESFORCE_MCP_REQUIRED=true) but the MCP "
                    f"session could not be established: {e}. Not falling back to REST."
                ) from e

    async def _close_mcp_session(self) -> None:
        """Tear down the mcp SDK session and its streamable HTTP context."""
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:
                pass
            self._session = None
        if self._mcp_ctx is not None:
            try:
                await self._mcp_ctx.__aexit__(None, None, None)
            except Exception:
                pass
            self._mcp_ctx = None
        self._connected = False

    @staticmethod
    def _format_mcp_result(result: Any) -> Any:
        """Convert an mcp SDK CallToolResult into a JSON-safe value."""
        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            return structured
        text_parts = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                text_parts.append(text)
        if text_parts:
            joined = "\n".join(text_parts).strip()
            if not joined:
                return None
            try:
                return json.loads(joined)
            except Exception:
                return joined
        return str(result)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """
        Execute a tool call against Salesforce.
        Primary: mcp SDK session (Streamable HTTP). Fallback: direct REST API.
        Auto-reauthenticates on 401/expired token.
        """
        await self._ensure_fresh_token()

        plain_name = tool_name.rsplit(":", 1)[-1]

        # A MUTATING/DESTRUCTIVE tool (create/update/delete/upload) writes to
        # Salesforce. Its outcome is UNKNOWN whenever the MCP call fails without
        # a definitive signal (connection drop, timeout, transport error): the
        # server may or may not have already executed it. Such an uncertain
        # mutation must NEVER be re-invoked automatically — not by an MCP
        # reconnect-retry and not by the REST fallback — or one user submission
        # would create two Leads. Only AUTH rejections (401: request refused
        # before execution) and DEFINITIVE tool errors (isError: execution
        # failed) are safe to retry/fall back for writes. Read-only tools keep
        # the full reconnect-then-fallback behavior so a dropped idle MCP
        # session never spuriously routes a normal query to REST.
        is_write = is_mutating(plain_name) or is_destructive(plain_name)

        # MCP is the primary path. On an auth/session/transient failure we give
        # MCP ONE clean reconnect before EVER falling back to REST, so a normal
        # Salesforce query does not spuriously route to REST just because the
        # idle Streamable HTTP session was dropped or a token lapsed. REST is
        # only used when MCP genuinely cannot complete the call.
        reconnect_retried = False
        definitive_mcp_error = False
        for attempt in range(2):
            await self._ensure_connected()
            if self._session is not None:
                server_name = self._name_map.get(tool_name) or self._name_map.get(plain_name, tool_name)
                try:
                    logger.info(f"[MCP] Executing tool {server_name} (requested as {tool_name}). Transport: MCP")
                    result = await self._session.call_tool(server_name, arguments)
                    if getattr(result, "isError", False):
                        # Definitive server-side failure: the tool reported an
                        # error and did not perform the operation, so a clean
                        # REST fallback is safe even for writes.
                        definitive_mcp_error = True
                        raise RuntimeError(
                            f"MCP tool {tool_name} returned an error: {self._format_mcp_result(result)}"
                        )
                    self.mcp_transport = "MCP"
                    return self._format_mcp_result(result)
                except Exception as e:
                    if definitive_mcp_error:
                        logger.warning(
                            f"MCP tool {tool_name} returned a definitive error; "
                            "falling back to REST API (safe — the tool did not execute)."
                        )
                        break
                    status_code = getattr(getattr(e, "response", None), "status_code", None)
                    is_auth = status_code == 401 or "Unauthorized" in str(e)
                    if is_auth and not reconnect_retried:
                        logger.warning("MCP session returned 401; refreshing token and retrying MCP once.")
                        await self._close_mcp_session()
                        if await self._try_oauth_refresh():
                            reconnect_retried = True
                            continue
                    if is_write and not is_auth:
                        # Uncertain mutation outcome: refuse to re-invoke and refuse
                        # the REST fallback so we cannot create a duplicate record.
                        logger.error(
                            f"[MUTATION-SAFETY] '{tool_name}' failed with an UNKNOWN outcome "
                            f"({e}); NOT retrying or falling back to REST to avoid a "
                            "duplicate record."
                        )
                        raise RuntimeError(
                            f"MCP connection for mutating tool '{tool_name}' failed with an "
                            f"unknown outcome ({e}). No automatic retry was performed to avoid "
                            "a duplicate record. Please check Salesforce for this record before "
                            "deciding to retry."
                        ) from e
                    if not is_auth and not reconnect_retried:
                        # Transient non-auth failure (dropped idle session, network
                        # blip, timeout): perform one clean MCP reconnect + retry
                        # before falling back to REST. Reads only — writes already
                        # raised above on an uncertain outcome.
                        logger.warning(f"MCP tool call failed ({e}); reconnecting MCP once before REST.")
                        await self._close_mcp_session()
                        self.mcp_transport = "MCP"
                        reconnect_retried = True
                        continue
                    if self.mcp_required:
                        raise RuntimeError(
                            f"MCP is required (SALESFORCE_MCP_REQUIRED=true) and MCP "
                            f"tool call for '{tool_name}' failed: {e}. Not falling back to REST."
                        ) from e
                    logger.warning(f"MCP tool call failed ({e}); falling back to REST API.")
                    break
            else:
                # _ensure_connected() left us without a live session (e.g. MCP init
                # rejected with 401 before initialize). Refresh the token and retry
                # MCP connection once before REST.
                if not reconnect_retried:
                    logger.warning("No live MCP session; refreshing token and retrying MCP connection once.")
                    await self._try_oauth_refresh()
                    await self._close_mcp_session()
                    reconnect_retried = True
                    continue
            break

        if self.mcp_required and self._session is None:
            raise RuntimeError(
                "MCP is required (SALESFORCE_MCP_REQUIRED=true) but no live MCP "
                f"session handled tool '{tool_name}'. Not falling back to REST."
            )

        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            return await self._fallback_rest_api(plain_name, arguments, headers)
        except httpx.HTTPStatusError as status_err:
            if status_err.response.status_code == 401 or "INVALID_SESSION_ID" in status_err.response.text:
                logger.warning(f"Session expired during {tool_name} (401). Re-authenticating...")
                try:
                    await self._try_oauth_refresh()
                    headers["Authorization"] = f"Bearer {self._access_token}"
                    return await self._fallback_rest_api(plain_name, arguments, headers)
                except Exception as auth_err:
                    logger.error(f"Re-authentication retry failed: {auth_err}")
                    raise RuntimeError(f"Salesforce API 401 Unauthorized: {status_err.response.text}") from auth_err
            raise

    async def _fallback_rest_api(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        headers: dict[str, str],
    ) -> Any:
        """
        Fallback: execute the tool via direct Salesforce REST API
        when MCP endpoint is unreachable.
        """
        self.mcp_transport = "REST"
        base = self.instance_url.rstrip("/")
        api_version = "v62.0"

        try:
            if tool_name == "soqlQuery":
                query_str = arguments.get("q") or arguments.get("query") or arguments.get("soql") or ""
                url = f"{base}/services/data/{api_version}/query"
                resp = await self._http_client.get(
                    url, params={"q": query_str}, headers=headers
                )
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "find":
                search_term = arguments.get("q") or arguments.get("query") or arguments.get("search") or arguments.get("find") or ""
                search_term = search_term.strip()
                if not search_term.startswith("FIND"):
                    # Format as valid SOSL query
                    clean_term = search_term.replace("{", "").replace("}", "").replace("'", "")
                    search_term = f"FIND {{{clean_term}*}} IN ALL FIELDS RETURNING Account(Id, Name), Contact(Id, Name, Email), Lead(Id, Name, Company, Email) LIMIT 20"

                url = f"{base}/services/data/{api_version}/search"
                resp = await self._http_client.get(
                    url, params={"q": search_term}, headers=headers
                )
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "getUserInfo":
                try:
                    url = f"{base}/services/oauth2/userinfo"
                    resp = await self._http_client.get(url, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
                except Exception:
                    # Fallback to querying User via SOQL for current logged-in username
                    url = f"{base}/services/data/{api_version}/query"
                    query = f"SELECT Id, Name, Username, Email, Profile.Name FROM User WHERE Username = '{self.username}' LIMIT 1"
                    resp = await self._http_client.get(url, params={"q": query}, headers=headers)
                    resp.raise_for_status()
                    return resp.json()

            elif tool_name == "getObjectSchema":
                objects = arguments.get("objects") or arguments.get("object") or arguments.get("sobject-name") or arguments.get("sobject")
                if objects:
                    # Get specific object schema
                    obj_list = [o.strip() for o in str(objects).split(",") if o.strip()]
                    results = {}
                    uncached = []
                    for obj in obj_list:
                        cache_key = f"schema:{obj.lower()}"
                        if cache_key in self._schema_cache:
                            results[obj] = self._schema_cache[cache_key]
                        else:
                            uncached.append(obj)

                    for obj in uncached:
                        fields = await self._get_simplified_field_list(obj, headers)
                        if fields is None:
                            raw = {}
                        else:
                            raw = {"name": obj, "label": obj}
                        obj_schema = {
                            "name": raw.get("name") or obj,
                            "label": raw.get("label") or obj,
                            "total_fields": len(fields) if fields else 0,
                            "fields": fields or [],
                        }
                        results[obj] = obj_schema

                    return results
                else:
                    # Get all queryable objects (cached & filtered)
                    if "all_sobjects" in self._schema_cache:
                        return self._schema_cache["all_sobjects"]

                    url = f"{base}/services/data/{api_version}/sobjects"
                    resp = await self._http_client.get(url, headers=headers)
                    resp.raise_for_status()
                    raw = resp.json()

                    # Filter sObjects: keep queryable standard/custom objects, skip internal metadata noise
                    filtered_sobjects = []
                    skip_suffixes = ("ChangeEvent", "Feed", "History", "Share", "Tag", "Permission", "Group", "Access")
                    for sobj in raw.get("sobjects", []):
                        name = sobj.get("name", "")
                        if not sobj.get("queryable"):
                            continue
                        if any(name.endswith(s) for s in skip_suffixes):
                            continue
                        filtered_sobjects.append({
                            "name": name,
                            "label": sobj.get("label"),
                            "custom": sobj.get("custom", False),
                        })

                    result_data = {
                        "total": len(filtered_sobjects),
                        "sobjects": filtered_sobjects,
                    }
                    self._schema_cache["all_sobjects"] = result_data
                    if "all_sobjects" not in self._schema_cache_order:
                        self._schema_cache_order.append("all_sobjects")
                    return result_data

            elif tool_name == "getRelatedRecords":
                sobject = _extract_sobject(arguments)
                record_id = arguments.get("id") or arguments.get("record_id") or ""
                rel_path = arguments.get("relationship-path") or arguments.get("relationship_path") or ""
                url = f"{base}/services/data/{api_version}/sobjects/{sobject}/{record_id}/{rel_path}"
                resp = await self._http_client.get(url, headers=headers)
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "listRecentSobjectRecords":
                sobject = _extract_sobject(arguments)
                if sobject:
                    try:
                        url = f"{base}/services/data/{api_version}/query"
                        query = f"SELECT Id, Name, CreatedDate, LastModifiedDate FROM {sobject} ORDER BY LastModifiedDate DESC LIMIT 10"
                        resp = await self._http_client.get(url, params={"q": query}, headers=headers)
                        if resp.status_code == 200:
                            return resp.json()
                    except Exception:
                        pass
                url = f"{base}/services/data/{api_version}/recent"
                resp = await self._http_client.get(url, headers=headers)
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "createSobjectRecord":
                sobject = _extract_sobject(arguments)
                body = _extract_body(arguments)

                # FAIL-CLOSED (defense-in-depth REST gate): a mutation that reaches
                # this REST path has ALREADY passed the executor's mandatory
                # mutation-validation gate. If the body is empty, REFUSE — never
                # guess or fabricate defaults (no "Unknown"/"Individual"/"New
                # Account"/"New Opportunity"/hard-coded StageName/CloseDate).
                if not sobject:
                    raise RuntimeError(
                        "Cannot create a record: the Salesforce object name is missing."
                    )
                if not body:
                    logger.error(
                        f"[MUTATION-VALIDATION] Fail-closed: createSobjectRecord for "
                        f"'{sobject}' reached REST with an EMPTY body; refusing to create."
                    )
                    return json.dumps({
                        "validation_error": True,
                        "retry_allowed": False,
                        "requires_user_input": True,
                        "tool": "createSobjectRecord",
                        "sobject_name": sobject,
                        "missing_fields": [],
                        "missing_fields_human": [],
                        "error": (
                            f"Cannot create {sobject}: no field values were provided. "
                            "Please provide the values to set on the new record."
                        ),
                        "suggestion": (
                            "Ask the user which fields to populate on the new record. "
                            "The request cannot proceed without field values."
                        ),
                    })

                # REQUIRED-FIELD CHECK (REST is the last gate): if live Describe
                # metadata says required fields are missing/blank, refuse the REST
                # call as well. Required fields are never auto-filled.
                missing = await self._missing_known_required(sobject, body, headers)
                if missing:
                    api_missing = [api for api, _ in missing]
                    labels = ", ".join(label for _, label in missing)
                    logger.error(
                        f"[MUTATION-VALIDATION] Fail-closed: createSobjectRecord for "
                        f"'{sobject}' missing required fields {api_missing}; refusing to create."
                    )
                    return json.dumps({
                        "validation_error": True,
                        "retry_allowed": False,
                        "requires_user_input": True,
                        "tool": "createSobjectRecord",
                        "sobject_name": sobject,
                        "missing_fields": api_missing,
                        "missing_fields_human": [label for _, label in missing],
                        "error": (
                            f"Cannot create {sobject}: required fields are missing or "
                            f"blank. Please provide: {labels}."
                        ),
                        "suggestion": (
                            "Provide the missing required fields in the request body "
                            "before retrying. No default values are ever fabricated."
                        ),
                    })

                url = f"{base}/services/data/{api_version}/sobjects/{sobject}"
                post_headers = {**headers, "Sforce-Duplicate-Rule-Header": "allowSave=true"}
                resp = await self._http_client.post(url, json=body, headers=post_headers)
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "updateSobjectRecord":
                sobject = _extract_sobject(arguments)
                record_id = arguments.get("id") or arguments.get("record_id") or ""
                body = _extract_body(arguments)
                url = f"{base}/services/data/{api_version}/sobjects/{sobject}/{record_id}"
                resp = await self._http_client.patch(url, json=body, headers=headers)
                if resp.status_code == 204:
                    return {"success": True, "id": record_id}
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "updateRelatedRecord":
                sobject = _extract_sobject(arguments)
                record_id = arguments.get("id") or arguments.get("record_id") or ""
                rel_path = arguments.get("relationship-path") or arguments.get("relationship_path") or ""
                body = _extract_body(arguments)
                url = f"{base}/services/data/{api_version}/sobjects/{sobject}/{record_id}/{rel_path}"
                resp = await self._http_client.patch(url, json=body, headers=headers)
                if resp.status_code == 204:
                    return {"success": True}
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "deleteSobjectRecord":
                sobject = _extract_sobject(arguments)
                record_id = arguments.get("id") or arguments.get("record_id") or ""
                url = f"{base}/services/data/{api_version}/sobjects/{sobject}/{record_id}"
                resp = await self._http_client.delete(url, headers=headers)
                if resp.status_code == 204:
                    return {"success": True, "deleted": record_id}
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "deleteRelatedRecord":
                sobject = _extract_sobject(arguments)
                record_id = arguments.get("id") or arguments.get("record_id") or ""
                rel_path = arguments.get("relationship-path") or arguments.get("relationship_path") or ""
                url = f"{base}/services/data/{api_version}/sobjects/{sobject}/{record_id}/{rel_path}"
                resp = await self._http_client.delete(url, headers=headers)
                if resp.status_code == 204:
                    return {"success": True}
                resp.raise_for_status()
                return resp.json()

            elif tool_name == "uploadRecordAttachment":
                import base64, os
                record_id = (arguments.get("record_id") or arguments.get("id") or "").strip()
                file_name = arguments.get("file_name") or arguments.get("filename") or "attachment"
                title = arguments.get("title") or os.path.splitext(file_name)[0]
                base64_data = arguments.get("file_content_base64") or arguments.get("base64") or ""

                if not record_id or record_id.lower() in ("001g500000ddq7saau", "001000000000000", "account_id", "record_id", "dummy"):
                    raise RuntimeError("A valid Salesforce Record ID is required to attach the file. Please provide the specific Record ID.")

                if not base64_data and os.path.exists(os.path.join("uploads", file_name)):
                    with open(os.path.join("uploads", file_name), "rb") as f:
                        base64_data = base64.b64encode(f.read()).decode("utf-8")

                if not base64_data:
                    raise RuntimeError("Missing file content (base64) for uploadRecordAttachment.")

                url = f"{base}/services/data/{api_version}/sobjects/ContentVersion"
                payload = {
                    "Title": title,
                    "PathOnClient": file_name,
                    "VersionData": base64_data,
                }
                if record_id:
                    payload["FirstPublishLocationId"] = record_id

                resp = await self._http_client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                result = resp.json()
                return {
                    "success": True,
                    "content_version_id": result.get("id"),
                    "linked_record_id": record_id,
                    "file_name": file_name,
                    "message": f"File '{file_name}' successfully attached to Salesforce record {record_id}."
                }

            else:
                raise RuntimeError(f"Unknown tool: {tool_name}")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401 or "INVALID_SESSION_ID" in e.response.text:
                raise  # Re-raise so call_tool catches 401 and auto-reauthenticates
            error_body = e.response.text
            logger.error(f"REST API fallback failed for {tool_name}: {error_body}")
            raise RuntimeError(f"Salesforce API error for {tool_name}: {error_body}")

    async def _get_simplified_field_list(
        self,
        sobject_name: str,
        headers: dict[str, str] | None = None,
    ) -> list[dict] | None:
        """Return the simplified Describe field list for an object, from the
        schema cache when available or a live /sobjects/<obj>/describe call
        (cached afterwards). Returns None when the schema cannot be established
        — the caller then falls back / fails closed. Everything that needs
        Describe metadata (required fields AND the required-field choice hints)
        shares this single loader so one describe populates both."""
        cache_key = f"schema:{sobject_name.lower()}"
        cached = self._schema_cache.get(cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("fields"), list):
            return cached["fields"]

        if headers is None:
            headers = {
                "Authorization": f"Bearer {self._access_token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        base = self.instance_url.rstrip("/")
        api_version = "v62.0"
        url = f"{base}/services/data/{api_version}/sobjects/{sobject_name}/describe"
        try:
            resp = await self._http_client.get(url, headers=headers)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as exc:  # noqa: BLE001 - resolver failure fails closed
            logger.error(
                f"[MUTATION-VALIDATION] describe failed for '{sobject_name}': {exc}"
            )
            return None

        simplified = _simplify_describe_fields(raw)
        self._schema_cache[cache_key] = {
            "name": raw.get("name"),
            "label": raw.get("label"),
            "total_fields": len(simplified),
            "fields": simplified,
        }
        if cache_key not in self._schema_cache_order:
            self._schema_cache_order.append(cache_key)
        # FIFO eviction: drop oldest entries when the cache exceeds its bound.
        while len(self._schema_cache) > self._schema_cache_max and self._schema_cache_order:
            old_key = self._schema_cache_order.pop(0)
            self._schema_cache.pop(old_key, None)
        return simplified

    async def describe_required_fields(
        self,
        sobject_name: str,
        headers: dict[str, str] | None = None,
    ) -> list[tuple[str, str]] | None:
        """
        Resolve the required fields (api_name, human_label) of an object from live
        Salesforce Describe metadata (read-only schema path). Used by the
        mutation-validation gate for BOTH standard and custom objects, so per-org
        custom required fields are honored and fields Salesforce defaults on create
        are never required. Returns None when the schema cannot be established —
        the caller then fails closed. An authoritative EMPTY list (the describe
        resolved and the object genuinely has no create-required fields, e.g. a
        custom object whose only mandatory fields default on create) is a
        VALID resolution, distinct from None (unknown / Describe failure).
        """
        fields = await self._get_simplified_field_list(sobject_name, headers)
        if fields is None:
            return None
        return _extract_required_from_describe(fields)

    async def describe_required_field_options(
        self,
        sobject_name: str,
        headers: dict[str, str] | None = None,
    ) -> dict[str, list[str]] | None:
        """Resolve {api_name -> [choice hints]} for the REQUIRED fields of an
        object from live Describe metadata: active picklist values for enum
        fields and referenced object names for lookup fields. Used to render a
        deterministic, metadata-accurate required-fields ask. Returns None when
        the schema cannot be established. Reuses the same schema cache as
        ``describe_required_fields`` (no duplicate describe call)."""
        fields = await self._get_simplified_field_list(sobject_name, headers)
        if fields is None:
            return None
        return _extract_required_field_options(fields)

    async def _missing_known_required(
        self,
        sobject_name: str,
        body: dict[str, Any],
        headers: dict[str, str],
    ) -> list[tuple[str, str]]:
        """Return the (api_name, human_label) required fields of `sobject_name`
        that are missing/blank in `body`, using live Describe metadata. This is the
        LAST-GATE defense behind the executor: required fields are never auto-filled.

        If Describe cannot be resolved here, an empty list is returned — the
        executor gate already handled the authoritative decision (static fallback /
        fail-closed) BEFORE the mutation reached this REST path."""
        from agent.mutation_validation import _is_present

        try:
            required = await self.describe_required_fields(sobject_name, headers)
        except Exception as exc:  # noqa: BLE001 - resolver failure fails closed
            logger.error(
                f"[MUTATION-VALIDATION] _missing_known_required describe failed for "
                f"'{sobject_name}': {exc}"
            )
            required = None
        if not required:
            return []
        return [(api, label) for (api, label) in required if not _is_present(body.get(api))]

    async def list_tools(self) -> list[dict[str, Any]]:
        """
        List available tools from the MCP Server via the mcp SDK.
        Falls back to local definitions if MCP is unreachable/unauthorized.
        """
        await self._ensure_fresh_token()
        await self._ensure_connected()

        if self._session is not None:
            try:
                result = await self._session.list_tools()
                tools = []
                for tool in result.tools:
                    name = tool.name or ""
                    # The installed MCP SDK (mcp_types) exposes the schema on the
                    # snake_case attribute `input_schema`. Accept the camelCase
                    # alias too for forward-compatibility across SDK versions.
                    raw_schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
                    input_schema = dict(raw_schema) or {"type": "object", "properties": {}}
                    tools.append({
                        "name": name,
                        "description": tool.description or "",
                        "input_schema": input_schema,
                    })
                    plain = name.rsplit(":", 1)[-1] or name
                    self._name_map[plain] = name
                logger.info(f"[MCP] Tools discovered: {len(tools)} tools from MCP Server.")
                return tools
            except Exception as e:
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code == 401:
                    logger.warning("MCP list_tools returned 401. Token/scope issue; using local definitions.")
                else:
                    logger.warning(f"Could not list tools from MCP: {e}. Using local definitions.")
            await self._close_mcp_session()
            if self.mcp_required:
                raise RuntimeError(
                    "MCP is required (SALESFORCE_MCP_REQUIRED=true) but tools could "
                    "not be discovered from the MCP server. Not falling back to local definitions."
                )

        # Fallback to local tool definitions
        from tools.salesforce import get_tool_definitions
        return get_tool_definitions()

    async def disconnect(self) -> None:
        """Close the MCP session/context and clean up resources."""
        logger.info("MCP Client disconnected.")
        await self._close_mcp_session()
        try:
            await self._http_client.aclose()
        except Exception as e:
            logger.warning(f"Error closing HTTP client: {e}")
