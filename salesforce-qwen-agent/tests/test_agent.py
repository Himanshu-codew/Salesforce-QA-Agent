"""
Unit tests for the Salesforce Qwen Agent.
Tests cover memory management, task planner, tool registry, and tool definitions.
"""

import json
import pytest

# ──────────────────────────────────────────────────────────────
# Memory Tests
# ──────────────────────────────────────────────────────────────

class TestConversationMemory:
    """Tests for ConversationMemory."""

    def setup_method(self):
        from agent.memory import ConversationMemory
        self.memory = ConversationMemory(max_messages=10)

    def test_add_user_message(self):
        self.memory.add_user_message("Hello")
        assert len(self.memory) == 1
        assert self.memory.messages[0]["role"] == "user"
        assert self.memory.messages[0]["content"] == "Hello"

    def test_add_assistant_message(self):
        self.memory.add_assistant_message("Hi there!")
        assert len(self.memory) == 1
        assert self.memory.messages[0]["role"] == "assistant"

    def test_add_tool_calls_and_results(self):
        tool_calls = [{
            "id": "tc_123",
            "name": "soqlQuery",
            "arguments": {"q": "SELECT Id FROM Account LIMIT 5"},
        }]
        self.memory.add_assistant_tool_calls(tool_calls)
        self.memory.add_tool_result("tc_123", "soqlQuery", '{"records": []}')

        assert len(self.memory) == 2
        assert self.memory.messages[0]["role"] == "assistant"
        assert self.memory.messages[0]["tool_calls"] is not None
        assert self.memory.messages[1]["role"] == "tool"

    def test_sliding_window_trim(self):
        from agent.memory import ConversationMemory
        mem = ConversationMemory(max_messages=4)

        for i in range(10):
            mem.add_user_message(f"Message {i}")

        assert len(mem) <= 4

    def test_get_messages_for_llm(self):
        self.memory.add_user_message("Test")
        messages = self.memory.get_messages_for_llm("You are a helper.")

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helper."
        assert messages[1]["role"] == "user"

    def test_clear(self):
        self.memory.add_user_message("Test")
        self.memory.add_assistant_message("Response")
        self.memory.clear()
        assert len(self.memory) == 0


# ──────────────────────────────────────────────────────────────
# Planner Tests
# ──────────────────────────────────────────────────────────────

class TestTaskPlanner:
    """Tests for TaskPlanner safety guardrails."""

    def setup_method(self):
        from agent.planner import TaskPlanner
        self.planner = TaskPlanner()

    def test_read_only_tool_is_safe(self):
        result = self.planner.check_tool_safety(
            "soqlQuery",
            {"q": "SELECT Id FROM Account"},
            "session1",
        )
        assert result["safe"] is True
        assert result["requires_confirmation"] is False

    def test_destructive_tool_requires_confirmation(self):
        result = self.planner.check_tool_safety(
            "deleteSobjectRecord",
            {"sobject-name": "Account", "id": "001xxx"},
            "session1",
        )
        assert result["safe"] is False
        assert result["requires_confirmation"] is True
        assert "Delete" in result["confirmation_message"]

    def test_confirm_pending_action(self):
        # Set up a pending delete
        self.planner.check_tool_safety(
            "deleteSobjectRecord",
            {"sobject-name": "Account", "id": "001xxx"},
            "session1",
        )

        # Confirm
        action = self.planner.process_confirmation("yes", "session1")
        assert action is not None
        assert action["tool_name"] == "deleteSobjectRecord"

    def test_decline_pending_action(self):
        self.planner.check_tool_safety(
            "deleteSobjectRecord",
            {"sobject-name": "Account", "id": "001xxx"},
            "session1",
        )

        action = self.planner.process_confirmation("no", "session1")
        assert action is None
        assert not self.planner.has_pending_confirmation("session1")

    def test_classify_intent_read_only(self):
        tool_calls = [
            {"name": "soqlQuery", "arguments": {}},
            {"name": "getUserInfo", "arguments": {}},
        ]
        assert self.planner.classify_intent(tool_calls) == "read_only"

    def test_classify_intent_mutating(self):
        tool_calls = [
            {"name": "createSobjectRecord", "arguments": {}},
        ]
        assert self.planner.classify_intent(tool_calls) == "mutating"

    def test_classify_intent_destructive(self):
        tool_calls = [
            {"name": "deleteSobjectRecord", "arguments": {}},
        ]
        assert self.planner.classify_intent(tool_calls) == "destructive"


# ──────────────────────────────────────────────────────────────
# Tool Definition Tests
# ──────────────────────────────────────────────────────────────

class TestToolDefinitions:
    """Tests for Salesforce tool definitions."""

    def test_all_11_tools_defined(self):
        from tools.salesforce import SALESFORCE_TOOLS
        assert len(SALESFORCE_TOOLS) == 12

    def test_tool_format_valid(self):
        from tools.salesforce import SALESFORCE_TOOLS
        for tool in SALESFORCE_TOOLS:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]

    def test_tool_categories(self):
        from tools.salesforce import (
            READ_ONLY_TOOLS, MUTATING_TOOLS, DESTRUCTIVE_TOOLS,
            is_read_only, is_mutating, is_destructive,
        )

        assert len(READ_ONLY_TOOLS) == 6
        assert len(MUTATING_TOOLS) == 4
        assert len(DESTRUCTIVE_TOOLS) == 2

        assert is_read_only("soqlQuery")
        assert is_mutating("createSobjectRecord")
        assert is_destructive("deleteSobjectRecord")
        assert not is_read_only("deleteSobjectRecord")

    def test_get_tool_definitions_returns_list(self):
        from tools.salesforce import get_tool_definitions
        tools = get_tool_definitions()
        assert isinstance(tools, list)
        assert len(tools) == 12


# ──────────────────────────────────────────────────────────────
# Tool Registry Tests
# ──────────────────────────────────────────────────────────────

class TestToolRegistry:
    """Tests for ToolRegistry."""

    @pytest.mark.asyncio
    async def test_initialize_with_local_tools(self):
        from mcp.registry import ToolRegistry
        registry = ToolRegistry()
        await registry.initialize(mcp_client=None)

        assert registry.is_initialized
        assert len(registry) == 12

    @pytest.mark.asyncio
    async def test_lookup_tool(self):
        from mcp.registry import ToolRegistry
        registry = ToolRegistry()
        await registry.initialize(mcp_client=None)

        tool = registry.get_tool("soqlQuery")
        assert tool is not None
        assert tool["function"]["name"] == "soqlQuery"

    @pytest.mark.asyncio
    async def test_has_tool(self):
        from mcp.registry import ToolRegistry
        registry = ToolRegistry()
        await registry.initialize(mcp_client=None)

        assert registry.has_tool("soqlQuery")
        assert not registry.has_tool("nonExistentTool")

    @pytest.mark.asyncio
    async def test_list_tool_names(self):
        from mcp.registry import ToolRegistry
        registry = ToolRegistry()
        await registry.initialize(mcp_client=None)

        names = registry.list_tool_names()
        assert "soqlQuery" in names
        assert "find" in names
        assert "getUserInfo" in names
        assert "deleteSobjectRecord" in names

    @pytest.mark.asyncio
    async def test_mcp_to_openai_conversion(self):
        from mcp.registry import ToolRegistry

        mcp_tool = {
            "name": "testTool",
            "description": "A test tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                },
                "required": ["param1"],
            },
        }

        result = ToolRegistry._mcp_to_openai(mcp_tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "testTool"
        assert result["function"]["parameters"]["type"] == "object"


# ──────────────────────────────────────────────────────────────
# RAG Retriever Tests
# ──────────────────────────────────────────────────────────────

class TestToolRAGRetriever:
    """Tests for ToolRAGRetriever tool filtering optimization."""

    def setup_method(self):
        from agent.rag import ToolRAGRetriever
        self.retriever = ToolRAGRetriever(default_top_k=4)

    def test_rag_filters_tools_for_query(self):
        tools = self.retriever.get_relevant_tools("Show me accounts with SOQL query", top_k=3)
        assert len(tools) <= 3
        tool_names = [t["function"]["name"] for t in tools]
        assert "soqlQuery" in tool_names

    def test_rag_greeting_returns_small_subset(self):
        tools = self.retriever.get_relevant_tools("hi", top_k=3)
        assert len(tools) <= 3
