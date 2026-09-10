"""
User Session Manager — handles per-session Salesforce authentication,
user profile isolation, and session-specific Agent & MCP client routing.
"""

import logging
import os
import time
from typing import Any, Dict, Optional

from sfmcp.client import SalesforceMCPClient
from sfmcp.crypto.envelope import token_vault
from sfmcp.registry import ToolRegistry
from sfmcp.executor import ToolExecutor
from agent.agent import SalesforceAgent
from llm.base import BaseLLM

logger = logging.getLogger(__name__)

# ── Memory-bounds for session state ──
# Hard ceiling on concurrent authenticated sessions.  On Render (512 MB) each
# session costs ~3-6 MB (MCPClient httpx pool + ConversationMemory + agent
# objects).  20 sessions = ~60-120 MB — comfortably below budget with the
# ~120 MB RAG baseline.  Excess sessions are evicted LRU.
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "20"))

# Idle timeout: sessions with no activity for this many seconds are evicted.
SESSION_IDLE_TIMEOUT = float(os.getenv("SESSION_IDLE_TIMEOUT", "1800"))  # 30 min


class UserSessionManager:
    """
    Manages isolated per-user / per-session Salesforce connections and agents.

    Each session maintains:
    - User authentication state (OAuth tokens, identity, org details)
    - Session-specific SalesforceMCPClient
    - Session-specific ToolExecutor & SalesforceAgent
    """

    def __init__(self):
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._session_activity: Dict[str, float] = {}  # session_id -> last activity timestamp
        self._default_mcp_client: Optional[SalesforceMCPClient] = None
        self._default_tool_registry: Optional[ToolRegistry] = None
        self._default_executor: Optional[ToolExecutor] = None
        self._default_agent: Optional[SalesforceAgent] = None
        self._llm: Optional[BaseLLM] = None

    def initialize_defaults(
        self,
        default_mcp_client: SalesforceMCPClient,
        default_tool_registry: ToolRegistry,
        default_executor: ToolExecutor,
        default_agent: SalesforceAgent,
        llm: BaseLLM,
    ) -> None:
        """Set default server instances used as fallbacks."""
        self._default_mcp_client = default_mcp_client
        self._default_tool_registry = default_tool_registry
        self._default_executor = default_executor
        self._default_agent = default_agent
        self._llm = llm
        logger.info("✅ UserSessionManager initialized with default server instances.")

    def touch_session(self, session_id: str) -> None:
        """Update last activity timestamp for a session (called on each request)."""
        self._session_activity[session_id] = time.time()
        # Opportunistically evict idle/overflow sessions
        self._evict_idle_sessions()
        self._evict_lru_if_needed()

    def _evict_idle_sessions(self) -> None:
        """Evict sessions idle longer than SESSION_IDLE_TIMEOUT."""
        now = time.time()
        expired = [
            sid for sid, ts in list(self._session_activity.items())
            if now - ts > SESSION_IDLE_TIMEOUT
        ]
        for sid in expired:
            if sid in self._sessions:
                logger.info(f"[SESSION-EVICT] Evicting idle session '{sid}' (idle {now - self._session_activity.get(sid, 0):.0f}s)")
                self._remove_session_sync(sid)

    def _evict_lru_if_needed(self) -> None:
        """If we have more than MAX_SESSIONS, evict the least-recently-used."""
        while len(self._sessions) > MAX_SESSIONS:
            # Find the session with the oldest activity
            if not self._session_activity:
                break
            oldest_sid = min(self._session_activity, key=self._session_activity.get)
            if oldest_sid in self._sessions:
                logger.info(f"[SESSION-EVICT] Evicting LRU session '{oldest_sid}' (max {MAX_SESSIONS} reached)")
                self._remove_session_sync(oldest_sid)
            else:
                self._session_activity.pop(oldest_sid, None)

    def _remove_session_sync(self, session_id: str) -> None:
        """Synchronously remove a session and release its resources.

        For async cleanup (MCP disconnect), schedule it via the event loop.
        This method clears the in-memory references so the GC can reclaim them.
        """
        session = self._sessions.pop(session_id, None)
        self._session_activity.pop(session_id, None)
        if not session:
            return
        # Clear agent's conversation memory to free tool result buffers
        agent = session.get("agent")
        if agent and hasattr(agent, "clear_session"):
            try:
                agent.clear_session(session_id)
            except Exception:
                pass
        # Clear the agent's _memories dict entirely for this session
        if agent and hasattr(agent, "_memories"):
            agent._memories.pop(session_id, None)
        # Disconnect MCP client asynchronously if event loop is running
        client = session.get("mcp_client")
        if client:
            try:
                import asyncio
                loop = asyncio.get_running_loop()
                loop.create_task(client.disconnect())
            except (RuntimeError, ImportError):
                # No event loop running; close the httpx client directly
                try:
                    import httpx
                    if hasattr(client, "_http_client") and isinstance(client._http_client, httpx.AsyncClient):
                        # Can't close async client synchronously; just drop the reference
                        pass
                except Exception:
                    pass
        logger.info(f"[SESSION-EVICT] Session '{session_id}' resources released.")

    def get_session(self, session_id: str = "default") -> Optional[Dict[str, Any]]:
        """Get raw session dictionary by session_id."""
        return self._sessions.get(session_id)

    async def get_or_create_agent(self, session_id: str = "default") -> SalesforceAgent:
        """
        Get or create the SalesforceAgent instance for a given session.
        If user has logged in via OAuth, returns their isolated Agent connected to THEIR org.
        Otherwise, returns the default Agent.
        """
        self.touch_session(session_id)
        if session_id in self._sessions and "agent" in self._sessions[session_id]:
            return self._sessions[session_id]["agent"]

        # Return default server agent if unauthenticated
        return self._default_agent

    def get_user_info(self, session_id: str = "default") -> Dict[str, Any]:
        """
        Get user profile & org connection details for UI presentation.
        """
        session = self._sessions.get(session_id)
        if session and session.get("authenticated"):
            return {
                "authenticated": True,
                "session_id": session_id,
                "user": session.get("user_info", {}),
                "instance_url": session.get("credentials", {}).get("instance_url", ""),
            }

        # Check default fallback info
        default_user = "Salesforce Admin"
        if self._default_mcp_client and self._default_mcp_client.username:
            default_user = self._default_mcp_client.username.split("@")[0].title()

        return {
            "authenticated": False,
            "session_id": session_id,
            "user": {
                "display_name": default_user,
                "email": os.getenv("SALESFORCE_USERNAME", "Default Org"),
                "username": os.getenv("SALESFORCE_USERNAME", "admin@salesforce.com"),
                "org_name": "Connected Org (.env Default)",
                "is_default": True,
            },
            "instance_url": os.getenv("SALESFORCE_INSTANCE_URL", ""),
        }

    async def register_oauth_session(
        self,
        session_id: str,
        access_token: str,
        refresh_token: str,
        instance_url: str,
        user_info: Dict[str, Any],
        expires_at: Optional[float] = 0.0,
        oauth_scope: Optional[str] = None,
    ) -> SalesforceAgent:
        """
        Register a newly authenticated Salesforce user session from OAuth flow.
        Creates an isolated MCP client, ToolExecutor, and SalesforceAgent for this user.
        Credentials are stored envelope-encrypted in the TokenVault (no plaintext in memory).
        """
        logger.info(f"🔑 Registering OAuth session '{session_id}' for user: {user_info.get('display_name')} ({instance_url})")

        auth_host = os.getenv("SALESFORCE_AUTH_HOST", "login").strip()
        client_id = os.getenv("SALESFORCE_CLIENT_ID", "")
        client_secret = os.getenv("SALESFORCE_CLIENT_SECRET", "")
        oauth_scope = oauth_scope or os.getenv("SALESFORCE_OAUTH_SCOPE", "")

        # 1. Persist credentials encrypted (memory + local file mirror)
        token_vault.put(
            session_id,
            access_token=access_token,
            refresh_token=refresh_token or None,
            instance_url=instance_url,
            expires_at=float(expires_at or 0.0),
            client_id=client_id or None,
            client_secret=client_secret or None,
            oauth_scope=oauth_scope or None,
            auth_host=auth_host,
        )

        # 2. Create session-specific MCP client with user's access token
        user_mcp_client = SalesforceMCPClient(
            mcp_url=os.getenv("SALESFORCE_MCP_URL", ""),
            instance_url=instance_url,
            client_id=client_id,
            client_secret=client_secret,
            username=user_info.get("username", ""),
            password="",
            security_token="",
            domain=auth_host,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at or 0.0,
            oauth_scope=oauth_scope,
            auth_host=auth_host,
            token_vault=token_vault,
            session_id=session_id,
        )

        try:
            await user_mcp_client.connect()
            logger.info(f"✅ User MCP Client connected for session '{session_id}'.")
        except Exception as e:
            logger.warning(f"⚠️ User MCP connection notice for session '{session_id}': {e}")

        # 3. Use existing registry or create tool executor
        registry = self._default_tool_registry or ToolRegistry()
        executor = ToolExecutor(user_mcp_client, registry)

        # 4. Create isolated agent for this user
        user_agent = SalesforceAgent(
            llm=self._llm,
            executor=executor,
            max_iterations=int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "10")),
            max_history=int(os.getenv("MAX_CONVERSATION_HISTORY", "20")),
        )

        # 5. Save session state (no plaintext tokens in memory)
        self._sessions[session_id] = {
            "session_id": session_id,
            "authenticated": True,
            "credentials": {
                "instance_url": instance_url,
                "vault_id": session_id,
            },
            "user_info": user_info,
            "mcp_client": user_mcp_client,
            "executor": executor,
            "agent": user_agent,
        }
        self.touch_session(session_id)

        return user_agent

    def get_mcp_status(self) -> Dict[str, Any]:
        """
        Aggregate the real MCP connection state across all authenticated user
        sessions. MCP is intentionally session-scoped: each authenticated user
        gets an isolated SalesforceMCPClient (see register_oauth_session). The
        server-level default client is a SEPARATE instance whose idle session is
        closed after tool discovery, so it does not reflect whether an actual
        user's SOQL/query path is live over MCP. This method reports that.
        """
        connected_sessions: list[str] = []
        mcp_sessions: list[str] = []
        using_rest_or_unconnected: list[str] = []
        for sid, session in self._sessions.items():
            if not session.get("authenticated"):
                continue
            client = session.get("mcp_client")
            if client is None:
                using_rest_or_unconnected.append(sid)
                continue
            transport = getattr(client, "mcp_transport", "REST")
            if transport == "MCP":
                mcp_sessions.append(sid)
            if getattr(client, "is_connected", False):
                connected_sessions.append(sid)
            else:
                using_rest_or_unconnected.append(sid)
        return {
            "any_connected": bool(connected_sessions),
            "any_mcp": bool(mcp_sessions),
            "any_live": bool(connected_sessions) or bool(mcp_sessions),
            "connected_sessions": connected_sessions,
            "mcp_sessions": mcp_sessions,
            "unconnected_sessions": using_rest_or_unconnected,
            "total_sessions": len(self._sessions),
        }

    async def logout_session(self, session_id: str = "default") -> bool:
        """Disconnect and clear a user session."""
        session = self._sessions.pop(session_id, None)
        if session:
            client = session.get("mcp_client")
            if client:
                try:
                    await client.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting MCP client on logout ({session_id}): {e}")
            try:
                token_vault.delete(session_id)
            except Exception as e:
                logger.warning(f"Error deleting vault record on logout ({session_id}): {e}")
            # F5: clear any pending destructive-action confirmation before logout so a
            # recycled session_id can never confirm a stale action. Uses whichever
            # planner instance the session's agent owns (Orchestrator or SalesforceAgent).
            agent = session.get("agent")
            planner = getattr(agent, "planner", None) or getattr(agent, "safety_planner", None)
            if planner is not None and hasattr(planner, "clear_pending"):
                try:
                    planner.clear_pending(session_id)
                    logger.info(f"[F5] Cleared pending confirmation for logged-out session '{session_id}'.")
                except Exception as e:
                    logger.warning(f"Error clearing pending confirmation on logout ({session_id}): {e}")
            logger.info(f"🔒 User session '{session_id}' logged out and disconnected.")
            return True
        return False


# Global singleton session manager
session_manager = UserSessionManager()
