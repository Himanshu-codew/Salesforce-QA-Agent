"""
Automated Test Suite for Multi-User Salesforce OAuth & Per-Session Agent Routing.
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.session_manager import UserSessionManager
from agent.agent import SalesforceAgent
from mcp.executor import ToolExecutor
from mcp.client import SalesforceMCPClient


class TestMultiUserSessionManager(unittest.TestCase):

    def setUp(self):
        self.mgr = UserSessionManager()
        self.mock_client = MagicMock(spec=SalesforceMCPClient)
        self.mock_client.username = "default_admin@sf.com"
        self.mock_executor = MagicMock(spec=ToolExecutor)
        self.mock_agent = MagicMock(spec=SalesforceAgent)
        self.mock_llm = MagicMock()

        self.mgr.initialize_defaults(
            default_mcp_client=self.mock_client,
            default_tool_registry=MagicMock(),
            default_executor=self.mock_executor,
            default_agent=self.mock_agent,
            llm=self.mock_llm,
        )

    def test_unauthenticated_session_returns_default_agent(self):
        """Unauthenticated user session should route to default server agent."""
        async def run_test():
            agent = await self.mgr.get_or_create_agent("session_123")
            self.assertEqual(agent, self.mock_agent)

            info = self.mgr.get_user_info("session_123")
            self.assertFalse(info["authenticated"])
            self.assertEqual(info["session_id"], "session_123")

        asyncio.run(run_test())

    def test_register_oauth_session_isolates_user(self):
        """OAuth session registration should create isolated Agent connected to user's org."""
        async def run_test():
            user_info = {
                "display_name": "Himanshu Developer",
                "email": "himanshu@dev.com",
                "username": "himanshu@dev.com",
                "org_id": "00D000000001234",
                "org_name": "Himanshu Dev Org",
                "authenticated": True,
            }

            with patch("mcp.session_manager.SalesforceMCPClient") as mock_client_cls:
                mock_inst = AsyncMock()
                mock_client_cls.return_value = mock_inst

                user_agent = await self.mgr.register_oauth_session(
                    session_id="user_session_abc",
                    access_token="00D_mock_token",
                    refresh_token="mock_refresh",
                    instance_url="https://himanshu-dev.my.salesforce.com",
                    user_info=user_info,
                )

                self.assertIsNotNone(user_agent)
                self.assertNotEqual(user_agent, self.mock_agent)

                # Check user info lookup
                info = self.mgr.get_user_info("user_session_abc")
                self.assertTrue(info["authenticated"])
                self.assertEqual(info["user"]["display_name"], "Himanshu Developer")
                self.assertEqual(info["instance_url"], "https://himanshu-dev.my.salesforce.com")

                # Test logout
                logged_out = await self.mgr.logout_session("user_session_abc")
                self.assertTrue(logged_out)

                info_after = self.mgr.get_user_info("user_session_abc")
                self.assertFalse(info_after["authenticated"])

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
