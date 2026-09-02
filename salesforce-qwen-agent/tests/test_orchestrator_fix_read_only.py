"""
Focused offline tests for Fix A in the Orchestrator (agent/multi_agent.py).

Fix A: when READ_ONLY_MODE=true and the user request is read-only, mutation/
destructive tool schemas are removed from the RAG-selected tool list BEFORE they
reach the planner/worker tool-selection step (llm.chat_with_tools). Explicit
write/compound requests keep the full tool set. READ_ONLY_MODE planner/executor
gates are untouched.

All tests use mocks - no live LLM / Salesforce / embedding model.
"""

import os
import sys
import asyncio
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.multi_agent import Orchestrator

SAFE = {"safe": True, "requires_confirmation": False, "confirmation_message": "", "pending_action": None}

_READ = ["soqlQuery", "listRecentSobjectRecords", "getRelatedRecords", "getUserInfo"]
_WRITE = ["createSobjectRecord", "updateRelatedRecord", "deleteSobjectRecord"]


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": f"Tool {name}", "parameters": {"type": "object", "properties": {}}}}


class _CaptureLLM:
    """Captures the tools passed to chat_with_tools so we can assert what reached Qwen."""
    def __init__(self, tool_call):
        self.tools_seen = []
        self._tool_call = tool_call

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        return "[]"

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        if tools is not None:
            self.tools_seen.append([t["function"]["name"] for t in tools])
        return {"content": "", "tool_calls": list(self._tool_call), "finish_reason": "tool_calls"}


class _Exec:
    def __init__(self):
        self.executed = []

    async def execute(self, name, arguments):
        self.executed.append((name, arguments))
        return "{}"


class _Planner:
    def has_pending_confirmation(self, session_id):
        return False

    def check_tool_safety(self, tool_name, arguments, session_id="default"):
        return dict(SAFE)


def _build(message, rag_names, tool_call_name="soqlQuery", plan=None):
    llm = _CaptureLLM([{"id": "t1", "name": tool_call_name, "arguments": {"q": "SELECT Id FROM Account"}}])
    exec_ = _Exec()
    orch = Orchestrator(llm=llm, executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _Planner()

    rag = MagicMock()
    rag.get_relevant_tools = MagicMock(return_value=[_tool(n) for n in rag_names])
    orch.rag_retriever = rag

    orch._generate_plan = AsyncMock(return_value=plan if plan is not None else [{
        "task_id": 1, "description": message, "agent": "DataAgent", "depends_on": []
    }])

    return orch, llm, exec_


def _run(orch, message):
    async def _go():
        events = []
        async for ev in orch.process_message(message, "default"):
            events.append(ev)
        return events
    return asyncio.run(_go())


def _capture_tools(orch, llm):
    # tools reach chat_with_tools only when a tool call is requested and executed.
    return llm.tools_seen[-1] if llm.tools_seen else []


def test_read_only_request_filters_mutation_tools():
    orch, llm, exec_ = _build(
        "How many Account records do we have?", _READ + _WRITE,
    )
    _run(orch, "How many Account records do we have?")
    passed = _capture_tools(orch, llm)
    assert passed is not None
    assert "soqlQuery" in passed
    assert "createSobjectRecord" not in passed
    assert "updateRelatedRecord" not in passed
    assert "deleteSobjectRecord" not in passed


def test_read_only_request_keeps_read_tools():
    orch, llm, exec_ = _build("Show me all Accounts", _READ + _WRITE)
    _run(orch, "Show me all Accounts")
    passed = _capture_tools(orch, llm)
    for rt in _READ:
        assert rt in passed, f"{rt} should remain available"


def test_write_intent_not_filtered():
    orch, llm, exec_ = _build("Create a new Account", _READ + _WRITE)
    _run(orch, "Create a new Account")
    passed = _capture_tools(orch, llm)
    assert "createSobjectRecord" in passed, "write intent must keep mutation tools"
    assert "updateRelatedRecord" in passed


def test_compound_write_keeps_mutation_tools():
    orch, llm, exec_ = _build("Create an Account and list all Contacts", _READ + _WRITE)
    _run(orch, "Create an Account and list all Contacts")
    passed = _capture_tools(orch, llm)
    assert "createSobjectRecord" in passed
    assert "soqlQuery" in passed


def test_read_only_mode_planner_gates_reference_unchanged():
    # Import the gate flags to confirm the fix does not alter them.
    import agent.planner as planner_module
    import sfmcp.executor as executor_module
    before_p = planner_module.READ_ONLY_MODE
    before_e = executor_module.READ_ONLY_MODE
    orch, llm, exec_ = _build("How many Accounts?", _READ + _WRITE)
    _run(orch, "How many Accounts?")
    assert planner_module.READ_ONLY_MODE == before_p
    assert executor_module.READ_ONLY_MODE == before_e


if __name__ == "__main__":
    import unittest
    unittest.main(module=__name__)