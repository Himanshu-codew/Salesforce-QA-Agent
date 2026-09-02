"""
Focused offline tests for the duplicate-count guard in agent/multi_agent.py.

Guard: when the worker LLM autonomously adds an aggregate `SELECT COUNT(...)`
tool call on top of a plain list request (the soqlQuery schema advertises COUNT
support), the Orchestrator SKIPS that unrequested COUNT call so only the
requested list query executes. Genuine count requests and explicit list+count
requests still run their COUNT unchanged.

Data-fidelity invariants preserved:
- Pure list request -> list query executes, no COUNT, no invented total.
- Genuine COUNT request -> COUNT executes, "**Total Count:** N" preserved.
- Explicit compound list + count -> both operations allowed.
- Reference tables stay verbatim and Id/Name remain separate cells.

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
    _has_count_intent,
    _split_reference_results,
)

SAFE = {"safe": True, "requires_confirmation": False, "confirmation_message": "", "pending_action": None}


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": f"Tool {name}", "parameters": {"type": "object", "properties": {}}}}


class _RunLLM:
    """Returns a fixed list of tool calls for the worker stage."""

    def __init__(self, tool_calls, chat_result="[final answer]"):
        self._tool_calls = list(tool_calls)
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
    def __init__(self, results_by_q):
        self.results_by_q = results_by_q
        self.executed = []

    async def execute(self, name, arguments):
        self.executed.append((name, arguments))
        q = arguments.get("q", "")
        return self.results_by_q.get(q, "{}")


class _Planner:
    def has_pending_confirmation(self, session_id):
        return False

    def check_tool_safety(self, tool_name, arguments, session_id="default"):
        return dict(SAFE)


def _build(tool_calls, results_by_q, message, plan_tasks):
    llm = _RunLLM(tool_calls)
    exec_ = _Exec(results_by_q)
    orch = Orchestrator(llm=llm, executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _Planner()

    rag = MagicMock()
    rag.get_relevant_tools = MagicMock(return_value=[_tool("soqlQuery")])
    orch.rag_retriever = rag
    orch._generate_plan = AsyncMock(return_value=plan_tasks)

    return orch, llm, exec_


def _run(orch, message):
    async def _go():
        events = []
        async for ev in orch.process_message(message, "default"):
            events.append(ev)
        return events

    return asyncio.run(_go())


_LIST_TASK = [{
    "task_id": 1, "description": "list accounts", "agent": "DataAgent", "depends_on": [],
}]

_LIST_RESULTS = {
    "SELECT Id, Name FROM Account LIMIT 3": json.dumps({
        "totalSize": 3,
        "records": [
            {"attributes": {"type": "Account"}, "Id": "001A", "Name": "Acme"},
            {"attributes": {"type": "Account"}, "Id": "001B", "Name": "Globex"},
            {"attributes": {"type": "Account"}, "Id": "001C", "Name": "Initech"},
        ],
        "done": True,
    }),
    "SELECT COUNT(Id) FROM Account": json.dumps({
        "totalSize": 1, "records": [{"attributes": {"type": "Account"}, "expr0": 22}],
    }),
}


def test_list_only_does_not_trigger_count():
    """Pure list request: the worker's unrequested COUNT call is dropped."""
    tool_calls = [
        {"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id, Name FROM Account LIMIT 3"}},
        {"id": "t2", "name": "soqlQuery", "arguments": {"q": "SELECT COUNT(Id) FROM Account"}},
    ]
    orch, _llm, exec_ = _build(tool_calls, _LIST_RESULTS, "", _LIST_TASK)
    events = _run(orch, "List the first 3 Accounts and show their Id and Name.")

    executed_qs = [args.get("q") for name, args in exec_.executed]
    assert "SELECT COUNT(Id) FROM Account" not in executed_qs
    assert executed_qs.count("SELECT Id, Name FROM Account LIMIT 3") == 1


def test_genuine_count_query_still_works():
    """Genuine count request: COUNT tool call still executes."""
    tool_calls = [
        {"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT COUNT(Id) FROM Account"}},
    ]
    orch, _llm, exec_ = _build(
        tool_calls, _LIST_RESULTS, "",
        [{"task_id": 1, "description": "count accounts", "agent": "DataAgent", "depends_on": []}],
    )
    events = _run(orch, "How many Account records are there?")

    executed_qs = [args.get("q") for name, args in exec_.executed]
    assert "SELECT COUNT(Id) FROM Account" in executed_qs


def test_explicit_compound_list_and_count_allows_both():
    """Explicit 'list AND also count' still runs both operations."""
    tool_calls = [
        {"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id, Name FROM Account LIMIT 3"}},
        {"id": "t2", "name": "soqlQuery", "arguments": {"q": "SELECT COUNT(Id) FROM Account"}},
    ]
    orch, _llm, exec_ = _build(tool_calls, _LIST_RESULTS, "", _LIST_TASK)
    events = _run(orch, "List the first 3 Accounts and also tell me how many Accounts there are.")

    executed_qs = [args.get("q") for name, args in exec_.executed]
    assert "SELECT Id, Name FROM Account LIMIT 3" in executed_qs
    assert "SELECT COUNT(Id) FROM Account" in executed_qs


def test_reference_table_remains_verbatim_and_columns_separate():
    """The pre-built reference table keeps Id and Name as separate columns."""
    tool_results = [
        {"tool": "soqlQuery", "result": _LIST_RESULTS["SELECT Id, Name FROM Account LIMIT 3"]},
    ]
    ref_tables, _ = _split_reference_results(tool_results)
    assert ref_tables, "expected at least one reference table"
    table = ref_tables[0]
    assert "| Id | Name |" in table
    assert "IdName" not in table
    assert "001A" in table and "Acme" in table


def test_no_duplicate_count_block_in_synthesis():
    """With the unrequested COUNT dropped, no count result reaches synthesis."""
    tool_calls = [
        {"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id, Name FROM Account LIMIT 3"}},
        {"id": "t2", "name": "soqlQuery", "arguments": {"q": "SELECT COUNT(Id) FROM Account"}},
    ]
    orch, llm, exec_ = _build(tool_calls, _LIST_RESULTS, "", _LIST_TASK)
    events = _run(orch, "List the first 3 Accounts and show their Id and Name.")

    tool_results = [ev["data"]["result"] for ev in events if ev.get("type") == "tool_result"]
    assert tool_results, "expected at least one tool result"
    assert len(tool_results) == 1
    assert "expr0" not in tool_results[0]


# ---------------------------------------------------------------------------
# Unit tests for the count-intent classifier
# ---------------------------------------------------------------------------


def test_count_intent_helper_positive():
    assert _has_count_intent("how many accounts are there")
    assert _has_count_intent("count my leads")
    assert _has_count_intent("what is the total number of opportunities")
    assert _has_count_intent("list accounts and tell me how many there are")


def test_count_intent_helper_negative():
    assert not _has_count_intent("list the first 3 accounts and show their id and name")
    assert not _has_count_intent("show me all accounts")
    assert not _has_count_intent("find my leads")
    assert not _has_count_intent("")
    assert not _has_count_intent(None)