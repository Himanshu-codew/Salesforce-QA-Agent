"""
Focused offline tests for Fix B in the Orchestrator (agent/multi_agent.py).

Fix B: when a read-only Salesforce COUNT query is executed and Salesforce
explicitly returns totalSize==0, the Orchestrator normalizes that zero-result
into the project's count line ("**Total Count:** 0") BEFORE synthesis, so the
synthesizer can state an explicit zero instead of treating an empty records
array as missing/unknown data.

Invariants preserved:
- Non-COUNT empty results must NOT become "Total Count: 0".
- Non-zero COUNT results keep their real count.
- COUNT results with records keep existing behavior.
- Synthesizer (planner -> worker -> executor -> synthesizer) flow is unchanged.

All tests use mocks - no live LLM / Salesforce / embedding model.
"""

import os
import sys
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.multi_agent import (
    Orchestrator,
    _normalize_zero_count_result,
)

SAFE = {"safe": True, "requires_confirmation": False, "confirmation_message": "", "pending_action": None}


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": f"Tool {name}", "parameters": {"type": "object", "properties": {}}}}


class _RunLLM:
    """Captures the raw tool results that reach the synthesizer's user message."""
    def __init__(self, tool_calls, chat_result="[final answer]"):
        self._tool_calls = tool_calls
        self._chat_result = chat_result
        self.synthesis_user_msg = None

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        for m in (messages or []):
            if m.get("role") == "user":
                self.synthesis_user_msg = m.get("content", "")
        return self._chat_result

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        return {"content": "", "tool_calls": list(self._tool_calls), "finish_reason": "tool_calls"}


class _Exec:
    def __init__(self, result):
        self.result = result
        self.executed = []

    async def execute(self, name, arguments):
        self.executed.append((name, arguments))
        return self.result


class _Planner:
    def has_pending_confirmation(self, session_id):
        return False

    def check_tool_safety(self, tool_name, arguments, session_id="default"):
        return dict(SAFE)


def _build(result_json, tool_name="soqlQuery", soql="SELECT COUNT(Id) FROM Account", plan=None):
    llm = _RunLLM([{"id": "t1", "name": tool_name, "arguments": {"q": soql}}])
    exec_ = _Exec(result_json)
    orch = Orchestrator(llm=llm, executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _Planner()

    rag = MagicMock()
    rag.get_relevant_tools = MagicMock(return_value=[_tool("soqlQuery")])
    # Fix A filter must leave the single read tool available.
    orch.rag_retriever = rag

    orch._generate_plan = AsyncMock(return_value=plan if plan is not None else [{
        "task_id": 1, "description": "count accounts", "agent": "DataAgent", "depends_on": [],
    }])

    return orch, llm, exec_


def _run(orch, message):
    async def _go():
        events = []
        async for ev in orch.process_message(message, "default"):
            events.append(ev)
        return events
    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# Unit tests for the normalization helper
# ---------------------------------------------------------------------------


def test_helper_count_zero_total_size_is_normalized():
    raw = json.dumps({"totalSize": 0, "records": [], "total_count": 0})
    out = _normalize_zero_count_result("soqlQuery", raw, {"q": "SELECT COUNT(Id) FROM Account"})
    assert out == "**Total Count:** 0"


def test_helper_non_count_empty_is_unchanged():
    raw = json.dumps({"totalSize": 0, "records": [], "done": True})
    out = _normalize_zero_count_result("soqlQuery", raw, {"q": "SELECT Id FROM Account LIMIT 10"})
    assert out == raw


def test_helper_non_zero_count_is_unchanged():
    raw = json.dumps({"totalSize": 1, "records": [{"attributes": {"type": "Account"}, "expr0": 22}]})
    out = _normalize_zero_count_result("soqlQuery", raw, {"q": "SELECT COUNT(Id) FROM Account"})
    assert out == raw


def test_helper_count_with_records_is_unchanged():
    raw = json.dumps({"totalSize": 2, "records": [{"Id": "a"}, {"Id": "b"}]})
    out = _normalize_zero_count_result("soqlQuery", raw, {"q": "SELECT COUNT(Id) FROM Account"})
    assert out == raw


def test_helper_non_soql_tool_is_unchanged():
    raw = json.dumps({"records": [], "totalSize": 0})
    out = _normalize_zero_count_result("listRecentSobjectRecords", raw, {"q": "SELECT COUNT(Id) FROM Account"})
    assert out == raw


def test_helper_unparseable_is_unchanged():
    raw = "not json"
    out = _normalize_zero_count_result("soqlQuery", raw, {"q": "SELECT COUNT(Id) FROM Account"})
    assert out == raw


def test_helper_missing_soql_arg_is_unchanged():
    raw = json.dumps({"totalSize": 0, "records": []})
    out = _normalize_zero_count_result("soqlQuery", raw, {"q": ""})
    assert out == raw


# ---------------------------------------------------------------------------
# Integration tests: what actually reaches the synthesizer
# ---------------------------------------------------------------------------


def test_count_zero_reaches_synthesis_as_count_line():
    raw = json.dumps({"totalSize": 0, "records": [], "total_count": 0})
    orch, llm, _ = _build(raw)
    _run(orch, "How many Account records do we have?")
    assert llm.synthesis_user_msg is not None
    assert "**Total Count:** 0" in llm.synthesis_user_msg


def test_non_count_empty_not_turned_into_zero():
    raw = json.dumps({"totalSize": 0, "records": [], "done": True})
    orch, llm, _ = _build(raw, soql="SELECT Id FROM Account LIMIT 10")
    _run(orch, "Show me recent Accounts")
    assert llm.synthesis_user_msg is not None
    assert "**Total Count:** 0" not in llm.synthesis_user_msg


def test_non_zero_count_keeps_real_value():
    raw = json.dumps({"totalSize": 1, "records": [{"attributes": {"type": "Account"}, "expr0": 22}]})
    orch, llm, _ = _build(raw)
    _run(orch, "How many Account records do we have?")
    assert llm.synthesis_user_msg is not None
    assert "**Total Count:** 0" not in llm.synthesis_user_msg
    assert "22" in llm.synthesis_user_msg


def test_count_with_records_keeps_existing_behavior():
    raw = json.dumps({"totalSize": 2, "records": [{"Id": "a"}, {"Id": "b"}]})
    orch, llm, _ = _build(raw)
    _run(orch, "List Account records")
    assert llm.synthesis_user_msg is not None
    assert "**Total Count:** 0" not in llm.synthesis_user_msg
    assert "totalSize" in llm.synthesis_user_msg


def test_synthesizer_still_invoked_for_zero_count():
    raw = json.dumps({"totalSize": 0, "records": [], "total_count": 0})
    orch, llm, _ = _build(raw)
    events = _run(orch, "How many Account records do we have?")
    # The synthesizer generates a final response event.
    assert any(e.get("type") == "response" for e in events)
    assert any(e.get("type") == "thinking" and "Synthesizer" in str(e.get("data")) for e in events)


if __name__ == "__main__":
    import unittest
    unittest.main(module=__name__)