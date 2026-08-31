"""
Automated Test Suite for the 7 Benchmark Chatbot Queries.
Verifies RAG tool retrieval, safety confirmation handling, and multi-step agent flow for complex requests.
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock

# Ensure agent modules are in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import SalesforceAgent
from agent.rag import ToolRAGRetriever
from sfmcp.executor import ToolExecutor
from llm.base import BaseLLM


class MockLLM(BaseLLM):
    """Mock LLM for testing deterministic agent turn responses."""

    def __init__(self):
        self.responses = []
        self.call_count = 0

    async def chat(self, messages, temperature=0.0, max_tokens=1024):
        return "Summary of completed action."

    async def chat_with_tools(self, messages, tools, temperature=0.0, max_tokens=1024):
        if self.call_count < len(self.responses):
            res = self.responses[self.call_count]
            self.call_count += 1
            return res
        return {"content": "Task complete.", "tool_calls": [], "finish_reason": "stop"}

    async def close(self):
        pass


class Test7BenchmarkQueries(unittest.TestCase):

    def setUp(self):
        self.rag = ToolRAGRetriever()
        self.executor = MagicMock(spec=ToolExecutor)
        self.executor.execute = AsyncMock(return_value='{"status": "success"}')
        self.llm = MockLLM()
        self.agent = SalesforceAgent(llm=self.llm, executor=self.executor)

    def test_query_1_rag_tools(self):
        """1. Show me all Accounts AND tell me how many Leads I have"""
        tools = self.rag.get_relevant_tools("Show me all Accounts AND tell me how many Leads I have")
        tool_names = [t["function"]["name"] for t in tools]
        self.assertIn("soqlQuery", tool_names)

    def test_query_2_rag_tools(self):
        """2. Find ABC Technologies, show its Opportunities, AND count its Contacts"""
        tools = self.rag.get_relevant_tools("Find ABC Technologies, show its Opportunities, AND count its Contacts")
        tool_names = [t["function"]["name"] for t in tools]
        self.assertIn("soqlQuery", tool_names)

    def test_query_3_rag_tools(self):
        """3. Show me Contacts at John Doe, update phone, delete oldest Lead"""
        tools = self.rag.get_relevant_tools("Show me Contacts at John Doe, update the phone on the first one to 555-1111, and then delete the oldest Lead")
        tool_names = [t["function"]["name"] for t in tools]
        self.assertIn("updateSobjectRecord", tool_names)
        self.assertIn("deleteSobjectRecord", tool_names)
        self.assertIn("listRecentSobjectRecords", tool_names)

    def test_query_4_rag_tools(self):
        """4. Find newest Lead and delete it, then create Account"""
        tools = self.rag.get_relevant_tools("Find the newest Lead and delete it, then create a new Account for whatever company it was from")
        tool_names = [t["function"]["name"] for t in tools]
        self.assertIn("deleteSobjectRecord", tool_names)
        self.assertIn("createSobjectRecord", tool_names)

    def test_query_5_rag_tools(self):
        """5. Find every Account and list Opportunities and Contacts"""
        tools = self.rag.get_relevant_tools("Find every Account and for each one list its Opportunities and Contacts")
        tool_names = [t["function"]["name"] for t in tools]
        self.assertIn("soqlQuery", tool_names)

    def test_query_6_rag_tools(self):
        """6. Show Tasks and Events together next 7 days"""
        tools = self.rag.get_relevant_tools("Show me Tasks and Events together, sorted by date, for the next 7 days")
        tool_names = [t["function"]["name"] for t in tools]
        self.assertIn("soqlQuery", tool_names)

    def test_query_7_rag_tools(self):
        """7. Opportunities where Industry is Technology OR Amount > $50,000"""
        tools = self.rag.get_relevant_tools("Show me Opportunities where the Account's Industry is Technology OR the Amount is over $50,000")
        tool_names = [t["function"]["name"] for t in tools]
        self.assertIn("soqlQuery", tool_names)

    def test_safe_first_execution_and_confirmation(self):
        """Verify safe tools run first before asking confirmation for destructive tools."""
        async def run_test():
            self.llm.responses = [
                {
                    "content": "",
                    "tool_calls": [
                        {"id": "tc1", "name": "updateSobjectRecord", "arguments": {"sobject-name": "Contact", "id": "003123", "body": {"Phone": "555-1111"}}},
                        {"id": "tc2", "name": "deleteSobjectRecord", "arguments": {"sobject-name": "Lead", "id": "00Q999"}},
                    ],
                    "finish_reason": "tool_calls"
                }
            ]

            events = []
            async for event in self.agent.process_message("Update contact and delete lead", session_id="test_session"):
                events.append(event)

            # Check that updateSobjectRecord executed
            tool_calls_executed = [e["data"]["name"] for e in events if e["type"] == "tool_call"]
            self.assertIn("updateSobjectRecord", tool_calls_executed)
            self.assertNotIn("deleteSobjectRecord", tool_calls_executed)

            # Check that confirmation was requested for deleteSobjectRecord
            confirmations = [e for e in events if e["type"] == "confirmation"]
            self.assertEqual(len(confirmations), 1)
            self.assertTrue(self.agent.planner.has_pending_confirmation("test_session"))

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
