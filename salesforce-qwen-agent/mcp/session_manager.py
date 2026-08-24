"""
User Session Manager — handles per-session Salesforce authentication,
user profile isolation, and session-specific Agent & MCP client routing.
"""

import logging
import os
from typing import Any, Dict, Optional

from mcp.client import SalesforceMCPClient
from mcp.registry import ToolRegistry
from mcp.executor import ToolExecutor
from agent.agent import SalesforceAgent
from llm.base import BaseLLM

logger = logging.getLogger(__name__)


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

    def get_session(self, session_id: str = "default") -> Optional[Dict[str, Any]]:
        """Get raw session dictionary by session_id."""
        return self._sessions.get(session_id)

    async def get_or_create_agent(self, session_id: str = "default") -> SalesforceAgent:
        """
        Get or create the SalesforceAgent instance for a given session.
        If user has logged in via OAuth, returns their isolated Agent connected to THEIR org.
        Otherwise, returns the default Agent.
        """
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
    ) -> SalesforceAgent:
        """
        Register a newly authenticated Salesforce user session from OAuth flow.
        Creates an isolated MCP client, ToolExecutor, and SalesforceAgent for this user.
        """
        logger.info(f"🔑 Registering OAuth session '{session_id}' for user: {user_info.get('display_name')} ({instance_url})")

        # 1. Create session-specific MCP client with user's access token
        user_mcp_client = SalesforceMCPClient(
            mcp_url=os.getenv("SALESFORCE_MCP_URL", ""),
            instance_url=instance_url,
            client_id=os.getenv("SALESFORCE_CLIENT_ID", ""),
            client_secret=os.getenv("SALESFORCE_CLIENT_SECRET", ""),
            username=user_info.get("username", ""),
            password="",
            security_token="",
            domain="login",
            access_token=access_token,
            refresh_token=refresh_token,
        )

        try:
            await user_mcp_client.connect()
            logger.info(f"✅ User MCP Client connected for session '{session_id}'.")
        except Exception as e:
            logger.warning(f"⚠️ User MCP connection notice for session '{session_id}': {e}")

        # 2. Use existing registry or create tool executor
        registry = self._default_tool_registry or ToolRegistry()
        executor = ToolExecutor(user_mcp_client, registry)

        # 3. Create isolated agent for this user
        user_agent = SalesforceAgent(
            llm=self._llm,
            executor=executor,
            max_iterations=int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "10")),
            max_history=int(os.getenv("MAX_CONVERSATION_HISTORY", "20")),
        )

        # 4. Save session state
        self._sessions[session_id] = {
            "session_id": session_id,
            "authenticated": True,
            "credentials": {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "instance_url": instance_url,
            },
            "user_info": user_info,
            "mcp_client": user_mcp_client,
            "executor": executor,
            "agent": user_agent,
        }

        return user_agent

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
            logger.info(f"🔒 User session '{session_id}' logged out and disconnected.")
            return True
        return False


# Global singleton session manager
session_manager = UserSessionManager()
