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


class SalesforceMCPClient:
    """
    Manages connection to the Salesforce MCP Server.

    Handles:
    - OAuth token management (password flow + OAuth refresh_token grant)
    - Envelope-encrypted token persistence via TokenVault
    - Streamable HTTP transport via the official mcp SDK
    - Session lifecycle (initialize → use → close)
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

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def access_token(self) -> str | None:
        return self._access_token

    # ──────────────────────────────────────────────────────────
    # OAuth Authentication
    # ──────────────────────────────────────────────────────────

    async def _soap_authenticate(self) -> str:
        """SOAP partner login fallback using username + password + security_token."""
        login_url = f"https://{self.domain}.salesforce.com/services/Soap/u/58.0"
        soap_body = f"""<?xml version="1.0" encoding="utf-8"?>
        <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/" xmlns:urn="urn:partner.soap.sforce.com">
          <soapenv:Body>
            <urn:login>
              <urn:username>{self.username}</urn:username>
              <urn:password>{self.password}{self.security_token}</urn:password>
            </urn:login>
          </soapenv:Body>
        </soapenv:Envelope>"""

        headers = {
            "Content-Type": "text/xml",
            "SOAPAction": "login",
        }

        try:
            response = await self._http_client.post(
                login_url,
                data=soap_body,
                headers=headers,
            )
            response.raise_for_status()

            import xml.etree.ElementTree as ET
            import urllib.parse
            root = ET.fromstring(response.text)
            ns = {"soap": "http://schemas.xmlsoap.org/soap/envelope/", "urn": "urn:partner.soap.sforce.com"}
            session_id_elem = root.find(".//urn:sessionId", ns)
            server_url_elem = root.find(".//urn:serverUrl", ns)

            if session_id_elem is not None and session_id_elem.text:
                self._access_token = session_id_elem.text
                self._expires_at = time.time() + 7200
                if server_url_elem is not None and server_url_elem.text:
                    parsed = urllib.parse.urlparse(server_url_elem.text)
                    self.instance_url = f"{parsed.scheme}://{parsed.netloc}"
                logger.info(f"Authenticated with Salesforce via SOAP partner login. Instance: {self.instance_url}")
                return self._access_token
            else:
                raise RuntimeError("SOAP login response missing sessionId.")
        except Exception as err:
            logger.error(f"Salesforce SOAP auth error: {err}")
            raise RuntimeError(f"Salesforce authentication failed: {err}")

    async def authenticate(self) -> str:
        """
        Authenticate with Salesforce using fast SOAP partner login,
        falling back to OAuth if needed.
        """
        if self.username and self.password and self.security_token:
            try:
                return await self._soap_authenticate()
            except Exception as soap_err:
                logger.warning(f"SOAP auth failed ({soap_err}), trying OAuth fallback...")

        token_url = f"https://{self.domain}.salesforce.com/services/oauth2/token"

        payload = {
            "grant_type": "password",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "username": self.username,
            "password": f"{self.password}{self.security_token}",
        }

        # The hosted MCP server requires OAuth tokens carrying sfap:mcp:* scope.
        # Pass the configured scope through on the initial grant so the returned
        # token is MCP-capable (REST-only session tokens are rejected with 401).
        if self.oauth_scope:
            payload["scope"] = self.oauth_scope

        try:
            response = await self._http_client.post(
                token_url,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            data = response.json()

            self._access_token = data["access_token"]
            self.instance_url = data.get("instance_url", self.instance_url)
            self._expires_at = time.time() + int(data.get("expires_in", 3600))

            logger.info(
                f"Authenticated with Salesforce via OAuth. Instance: {self.instance_url}"
            )
            return self._access_token

        except Exception as e:
            logger.warning(f"OAuth authentication failed ({e}). Falling back to SOAP Login...")
            return await self._soap_authenticate()

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
        else:
            logger.warning("MCP session not established; falls back to REST API.")

    # ──────────────────────────────────────────────────────────
    # OAuth token lifecycle (auto-refresh)
    # ──────────────────────────────────────────────────────────

    def _token_url(self) -> str:
        auth_host = self.auth_host or self.domain
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
            self._refresh_token = data.get("refresh_token", self._refresh_token)
            self._expires_at = time.time() + int(data.get("expires_in", 3600))
            if data.get("instance_url"):
                self.instance_url = data["instance_url"]
            self._persist_tokens()
            logger.info("Access token refreshed via OAuth refresh_token grant.")
            return True
        except Exception as e:
            logger.warning(f"OAuth refresh_token grant failed: {e}")
            return False

    async def _ensure_fresh_token(self) -> None:
        """Refresh proactively if the current token is missing/expired. Never raises."""
        # Prefer a live MCP-capable OAuth token already stored in the vault
        # (produced by the interactive /api/auth/login flow). Only fall back to
        # SOAP/password auth when no valid scoped token exists, because SOAP
        # session tokens lack the MCP scopes the hosted server requires.
        if self._needs_mcp_token():
            self._load_mcp_scoped_token_from_vault()
        if self._refresh_token and (
            not self._access_token or not self._expires_at or time.time() > self._expires_at - 30
        ):
            if await self._try_oauth_refresh():
                return
        if not self._access_token or (self._expires_at and time.time() > self._expires_at - 30):
            try:
                await self.authenticate()
            except Exception as e:
                logger.warning(f"Token acquisition failed: {e}")

    def _needs_mcp_token(self) -> bool:
        """True when the current token is missing/expired and would need replacement."""
        return not self._access_token or not self._expires_at or time.time() > self._expires_at - 30

    @staticmethod
    def _scope_has_mcp_capability(scope: str) -> bool:
        """Return True if *scope* carries any MCP-authorizing OAuth scope string."""
        s = scope.lower()
        # Legacy format (e.g. "sfap:mcp:all sfap:mcp:remote api ...")
        if "sfap:mcp" in s:
            return True
        # Current Salesforce format (e.g. "api sfap_api mcp_api ...")
        if "sfap_api" in s or "mcp_api" in s:
            return True
        return False

    def _load_mcp_scoped_token_from_vault(self) -> bool:
        """
        Load a live MCP-capable OAuth token from the token vault.

        The interactive /api/auth/login flow stores a token carrying the
        scopes required by the hosted MCP server (sfap_api, mcp_api, or the
        legacy sfap:mcp:* names). When present and not expired, this token is
        used for MCP instead of a SOAP session token (which lacks the MCP
        scopes and is rejected with 401).
        """
        if not self.token_vault:
            return False
        try:
            for sid in self.token_vault.sessions():
                rec = self.token_vault.get(sid)
                if not rec:
                    continue
                scope = rec.get("oauth_scope") or ""
                if not self._scope_has_mcp_capability(scope):
                    continue
                token = rec.get("access_token") or ""
                if not token:
                    continue
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
                    if self._session_vault_id is None:
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
                    continue
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
            logger.info("[MCP] MCP SDK session initialized; is_connected=True. Transport: MCP")
        except Exception as e:
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

        # MCP is the primary path. On an auth/session/transient failure we give
        # MCP ONE clean reconnect before EVER falling back to REST, so a normal
        # Salesforce query does not spuriously route to REST just because the
        # idle Streamable HTTP session was dropped or a token lapsed. REST is
        # only used when MCP genuinely cannot complete the call.
        reconnect_retried = False
        for attempt in range(2):
            await self._ensure_connected()
            if self._session is not None:
                server_name = self._name_map.get(tool_name) or self._name_map.get(plain_name, tool_name)
                try:
                    logger.info(f"[MCP] Executing tool {server_name} (requested as {tool_name}). Transport: MCP")
                    result = await self._session.call_tool(server_name, arguments)
                    if getattr(result, "isError", False):
                        raise RuntimeError(
                            f"MCP tool {tool_name} returned an error: {self._format_mcp_result(result)}"
                        )
                    self.mcp_transport = "MCP"
                    return self._format_mcp_result(result)
                except Exception as e:
                    status_code = getattr(getattr(e, "response", None), "status_code", None)
                    is_auth = status_code == 401 or "Unauthorized" in str(e)
                    if is_auth and not reconnect_retried:
                        logger.warning("MCP session returned 401; refreshing token and retrying MCP once.")
                        await self._close_mcp_session()
                        if await self._try_oauth_refresh():
                            reconnect_retried = True
                            continue
                    if not is_auth and not reconnect_retried:
                        # Transient non-auth failure (dropped idle session, network
                        # blip, timeout): perform one clean MCP reconnect + retry
                        # before falling back to REST.
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
                        url = f"{base}/services/data/{api_version}/sobjects/{obj}/describe"
                        resp = await self._http_client.get(url, headers=headers)
                        resp.raise_for_status()
                        raw = resp.json()

                        simplified_fields = []
                        for f in raw.get("fields", []):
                            req = not f.get("nillable") and not f.get("defaultedOnCreate") and f.get("createable")
                            field_info = {
                                "name": f.get("name"),
                                "label": f.get("label"),
                                "type": f.get("type"),
                            }
                            if req:
                                field_info["required"] = True
                            if f.get("type") == "picklist" and f.get("picklistValues"):
                                active_vals = [p.get("value") for p in f.get("picklistValues", []) if p.get("active")]
                                if active_vals:
                                    field_info["picklist_values"] = active_vals[:15]
                            simplified_fields.append(field_info)

                        obj_schema = {
                            "name": raw.get("name"),
                            "label": raw.get("label"),
                            "total_fields": len(simplified_fields),
                            "fields": simplified_fields,
                        }
                        cache_key = f"schema:{obj.lower()}"
                        self._schema_cache[cache_key] = obj_schema
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
                sobject = _extract_sobject(arguments) or "Lead"
                body = _extract_body(arguments)
                
                # Auto-fix standard mandatory fields if user provided minimal input
                s_lower = sobject.lower()
                if s_lower == "lead":
                    if "LastName" not in body:
                        raw_name = body.pop("Name", None) or body.pop("FirstName", None) or arguments.get("name")
                        if raw_name:
                            name_parts = str(raw_name).strip().split(None, 1)
                            if len(name_parts) == 2:
                                body["FirstName"] = name_parts[0]
                                body["LastName"] = name_parts[1]
                            else:
                                body["LastName"] = name_parts[0]
                                body.pop("FirstName", None)
                        else:
                            body["LastName"] = "Unknown"
                    if "Company" not in body:
                        body["Company"] = "Individual"

                elif s_lower == "contact":
                    if "LastName" not in body:
                        raw_name = body.pop("Name", None) or body.pop("FirstName", None) or arguments.get("name")
                        if raw_name:
                            name_parts = str(raw_name).strip().split(None, 1)
                            if len(name_parts) == 2:
                                body["FirstName"] = name_parts[0]
                                body["LastName"] = name_parts[1]
                            else:
                                body["LastName"] = name_parts[0]
                                body.pop("FirstName", None)
                        else:
                            body["LastName"] = "Unknown"

                elif s_lower == "account":
                    if "Name" not in body and "LastName" in body:
                        body["Name"] = body.pop("LastName")
                    if "Name" not in body:
                        body["Name"] = "New Account"

                elif s_lower == "opportunity":
                    if "Name" not in body:
                        body["Name"] = "New Opportunity"
                    if "StageName" not in body:
                        body["StageName"] = "Prospecting"
                    if "CloseDate" not in body:
                        body["CloseDate"] = "2026-12-31"

                elif s_lower == "case":
                    if "Subject" not in body and "Name" in body:
                        body["Subject"] = body.pop("Name")
                    if "Subject" not in body:
                        body["Subject"] = "New Customer Inquiry"
                    if "Status" not in body:
                        body["Status"] = "New"

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
