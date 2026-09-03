"""
Focused offline tests for the Planner fast-path latency optimization in
agent/multi_agent.py.

Fast path: a clearly identifiable, READ-ONLY Salesforce/data request is routed
to a single implicit DataAgent task WITHOUT a standalone Planner LLM call.
Write/delete requests, ambiguous requests, and general questions keep the
Planner. All downstream safety/correctness (Fix A read-only filtering, Fix B
COUNT, D1 errors, E5 RAG fallback/grounding, F5 confirmation, P0 timeouts,
duplicate-COUNT guard, data fidelity) is unchanged.

Each test MOCKS the LLM and explicitly asserts whether the Planner call occurs
or is skipped (we assert on the AsyncMock `_generate_plan` call count, not just
response text).

All tests use mocks - no live LLM / Salesforce / embedding model.
"""

import os
import sys
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.multi_agent import Orchestrator

SAFE = {"safe": True, "requires_confirmation": False, "confirmation_message": "", "pending_action": None}
CONFIRM = {"safe": True, "requires_confirmation": True, "confirmation_message": "Confirm delete?", "pending_action": None}


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": f"Tool {name}", "parameters": {"type": "object", "properties": {}}}}


class _RunLLM:
    def __init__(self, tool_calls, chat_result="[final answer]"):
        self._tool_calls = list(tool_calls)
        self._chat_result = chat_result
        self.chat_calls = []
        self.chat_with_tools_calls = []

    async def chat(self, messages=None, temperature=0.0, max_tokens=8192):
        self.chat_calls.append(messages)
        return self._chat_result

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=8192):
        self.chat_with_tools_calls.append((messages, tools))
        return {"content": "", "tool_calls": list(self._tool_calls), "finish_reason": "tool_calls"}


class _Exec:
    def __init__(self, result):
        self.result = result
        self.executed = []

    async def execute(self, name, arguments):
        self.executed.append((name, arguments))
        return self.result


class _Planner:
    def __init__(self, safety=SAFE):
        self._safety = safety

    def has_pending_confirmation(self, session_id):
        return False

    def check_tool_safety(self, tool_name, arguments, session_id="default"):
        return dict(self._safety)


def _build(llm, exec_, rag_tools, plan_return, safety=SAFE, require_confirmation=False):
    orch = Orchestrator(llm=llm, executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _Planner(safety)

    rag = MagicMock()
    rag.get_relevant_tools = MagicMock(return_value=rag_tools)
    orch.rag_retriever = rag

    orch._generate_plan = AsyncMock(return_value=plan_return)
    return orch


def _run(orch, message, session_id="default"):
    async def _go():
        events = []
        async for ev in orch.process_message(message, session_id):
            events.append(ev)
        return events
    return asyncio.run(_go())


def _planner_call_count(orch):
    mock = orch._generate_plan
    return mock.call_count


# ---------------------------------------------------------------------------
# a) simple Salesforce read -> Planner skipped
# ---------------------------------------------------------------------------


def test_simple_read_skips_planner():
    llm = _RunLLM([{"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id, Name FROM Account LIMIT 3"}}])
    exec_ = _Exec(json.dumps({"totalSize": 3, "records": [{"attributes": {"type": "Account"}, "Id": "a", "Name": "Acme"}], "done": True}))
    orch = _build(llm, exec_, [_tool("soqlQuery")], plan_return=[])
    events = _run(orch, "Show me the Account records")
    assert _planner_call_count(orch) == 0
    assert any(e.get("type") == "tool_call" for e in events)
    assert exec_.executed


def test_simple_read_with_object_name_skips_planner():
    llm = _RunLLM([{"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id, Name FROM Account LIMIT 5"}}])
    exec_ = _Exec(json.dumps({"totalSize": 3, "records": [{"Id": "a", "Name": "Acme"}], "done": True}))
    orch = _build(llm, exec_, [_tool("soqlQuery")], plan_return=[])
    events = _run(orch, "Show me the Accounts")
    assert _planner_call_count(orch) == 0
    assert any(e.get("type") == "tool_call" for e in events)


# b) Salesforce list -> Planner skipped


def test_list_skips_planner():
    llm = _RunLLM([{"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id, Name FROM Account LIMIT 3"}}])
    exec_ = _Exec(json.dumps({"totalSize": 3, "records": [{"attributes": {"type": "Account"}, "Id": "a", "Name": "Acme"}], "done": True}))
    orch = _build(llm, exec_, [_tool("soqlQuery")], plan_return=[])
    events = _run(orch, "List the first 3 Accounts and show their Id and Name.")
    assert _planner_call_count(orch) == 0
    assert any(e.get("type") == "tool_call" for e in events)


# c) genuine COUNT -> Planner skipped


def test_count_skips_planner():
    llm = _RunLLM([{"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT COUNT(Id) FROM Account"}}])
    exec_ = _Exec(json.dumps({"totalSize": 1, "records": [{"attributes": {"type": "Account"}, "expr0": 22}]}))
    orch = _build(llm, exec_, [_tool("soqlQuery")], plan_return=[])
    events = _run(orch, "How many Account records are there?")
    assert _planner_call_count(orch) == 0
    executed_qs = [a.get("q") for n, a in exec_.executed]
    assert "SELECT COUNT(Id) FROM Account" in executed_qs


# d) explicit compound list + count -> Planner skipped, both operations preserved


def test_compound_list_and_count_skips_planner_and_runs_both():
    llm = _RunLLM([
        {"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id, Name FROM Account LIMIT 3"}},
        {"id": "t2", "name": "soqlQuery", "arguments": {"q": "SELECT COUNT(Id) FROM Account"}},
    ])
    exec_ = _Exec(None)

    async def fake_execute(name, arguments):
        q = arguments.get("q", "")
        exec_.executed.append((name, arguments))
        if "COUNT" in q:
            return json.dumps({"totalSize": 1, "records": [{"expr0": 22}]})
        return json.dumps({"totalSize": 3, "records": [{"Id": "a", "Name": "Acme"}], "done": True})

    exec_.execute = fake_execute
    orch = _build(llm, exec_, [_tool("soqlQuery")], plan_return=[])
    events = _run(orch, "List the first 3 Accounts and also tell me how many Accounts there are.")
    assert _planner_call_count(orch) == 0
    executed_qs = [a.get("q") for n, a in exec_.executed]
    assert "SELECT Id, Name FROM Account LIMIT 3" in executed_qs
    assert "SELECT COUNT(Id) FROM Account" in executed_qs


# e) Salesforce write/delete -> Planner still used (confirmation preserved)


def test_write_delete_keeps_planner_and_confirmation():
    llm = _RunLLM([{"id": "t1", "name": "deleteSobjectRecord", "arguments": {"id": "001x"}}])
    exec_ = _Exec("{}")
    orch = _build(
        llm, exec_, [_tool("deleteSobjectRecord")],
        plan_return=[{"task_id": 1, "description": "delete", "agent": "ActionAgent", "depends_on": []}],
        safety=CONFIRM,
    )
    events = _run(orch, "Delete Account 001x")
    assert _planner_call_count(orch) == 1
    # Confirmation gate fires (worker safety requires confirmation -> no execute).
    assert any("Confirm delete?" in str(e.get("data", "")) for e in events)
    assert not exec_.executed


# f) ambiguous Salesforce request -> Planner still used


def test_ambiguous_request_keeps_planner():
    llm = _RunLLM([])
    exec_ = _Exec("{}")
    orch = _build(llm, exec_, [_tool("soqlQuery")], plan_return=[])
    # "hello" has no Salesforce/data keyword -> not fast-pathed.
    events = _run(orch, "hello there, can you help me")
    assert _planner_call_count(orch) == 1


# g) general question -> existing general path preserved


def test_general_question_keeps_planner_and_general_path():
    llm = _RunLLM([])
    exec_ = _Exec("{}")
    orch = _build(llm, exec_, [], plan_return=[])
    events = _run(orch, "What is the capital of France?")
    assert _planner_call_count(orch) == 1
    assert any(e.get("type") == "response" for e in events)
    # No tool starallele call was executed for a general question.
    assert not exec_.executed


# h) read-only mutation filtering still works


def test_read_only_mutation_filtering_still_works():
    # RAG returns BOTH a read tool and a mutation tool; read-only intent must
    # strip the mutation tool from the schemas offered to the worker.
    llm = _RunLLM([{"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Account LIMIT 5"}}])
    exec_ = _Exec(json.dumps({"totalSize": 1, "records": [{"Id": "a"}], "done": True}))
    orch = _build(llm, exec_, [_tool("soqlQuery"), _tool("createSobjectRecord")], plan_return=[])
    events = _run(orch, "Show me some Accounts")
    assert _planner_call_count(orch) == 0
    # Filtering happened BEFORE the worker: worker only saw the read tool schema.
    assert len(llm.chat_with_tools_calls) == 1
    offered = [t["function"]["name"] for t in (llm.chat_with_tools_calls[0][1] or [])]
    assert "soqlQuery" in offered
    assert "createSobjectRecord" not in offered


# i) duplicate COUNT protection still works


def test_duplicate_count_protection_still_works():
    # List-only request: worker returns list + a redundant COUNT -> guard drops
    # the redundant COUNT (no count intent, sibling list query present).
    llm = _RunLLM([
        {"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id, Name FROM Account LIMIT 3"}},
        {"id": "t2", "name": "soqlQuery", "arguments": {"q": "SELECT COUNT(Id) FROM Account"}},
    ])
    exec_ = _Exec(None)

    async def fake_execute(name, arguments):
        q = arguments.get("q", "")
        exec_.executed.append((name, arguments))
        if "COUNT" in q:
            return json.dumps({"totalSize": 1, "records": [{"expr0": 22}]})
        return json.dumps({"totalSize": 3, "records": [{"Id": "a", "Name": "Acme"}], "done": True})

    exec_.execute = fake_execute
    orch = _build(llm, exec_, [_tool("soqlQuery")], plan_return=[])
    events = _run(orch, "List the first 3 Accounts and show their Id and Name.")
    assert _planner_call_count(orch) == 0
    executed_qs = [a.get("q") for n, a in exec_.executed]
    assert "SELECT COUNT(Id) FROM Account" not in executed_qs
    assert executed_qs.count("SELECT Id, Name FROM Account LIMIT 3") == 1


# j) Salesforce failure still returns controlled error (D1)


def test_salesforce_failure_returns_controlled_error():
    llm = _RunLLM([{"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Account"}}])
    exec_ = _Exec(json.dumps({"error": "ProviderError: INVALID_TYPE", "tool": "soqlQuery"}))
    orch = _build(llm, exec_, [_tool("soqlQuery")], plan_return=[])
    events = _run(orch, "Show me Accounts")
    assert _planner_call_count(orch) == 0
    errs = [e for e in events if e.get("type") == "error"]
    assert errs, "expected a controlled error event"
    assert errs[0].get("code") == "SALESFORCE_FAILED"


# k) timeout safety still works (P0)


def test_timeout_safety_still_works():
    class TimeoutLLM(_RunLLM):
        async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=8192):
            raise asyncio.TimeoutError()

    llm = TimeoutLLM([])
    exec_ = _Exec("{}")
    orch = _build(llm, exec_, [_tool("soqlQuery")], plan_return=[])
    events = _run(orch, "Show me Accounts")
    errs = [e for e in events if e.get("type") == "error"]
    assert errs and errs[0].get("code") == "TIMEOUT"


# l) RAG empty/exception fallback still grounds Salesforce requests (E5)


def test_rag_empty_fallback_grounds_salesforce_request():
    # Fast-path request: RAG returns [] but Salesforce intent -> fall back to the
    # complete tool registry so the request is still grounded.
    from agent.multi_agent import get_tool_definitions  # noqa: F401 (registry access)

    llm = _RunLLM([{"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Account LIMIT 2"}}])
    exec_ = _Exec(json.dumps({"totalSize": 1, "records": [{"Id": "a", "Name": "Acme"}], "done": True}))
    orch = _build(llm, exec_, [], plan_return=[])  # RAG returns []
    events = _run(orch, "Show me some Account records")
    assert _planner_call_count(orch) == 0
    # The worker went ahead because fallback found tools (not an empty-tool general path).
    assert llm.chat_with_tools_calls, "expected worker call thanks to fallback tools"


def test_general_question_rag_empty_keeps_general_path():
    # E5: general non-Salesforce query with empty RAG must NOT fast-path and must
    # keep the general answer path (no Salesforce tools, planner still used).
    llm = _RunLLM([])
    exec_ = _Exec("{}")
    orch = _build(llm, exec_, [], plan_return=[])
    events = _run(orch, "Tell me a joke")
    assert _planner_call_count(orch) == 1
    assert not llm.chat_with_tools_calls


# m) data fidelity / reference table behavior intact


def test_data_fidelity_reference_table_intact():
    from agent.multi_agent import _split_reference_results

    result = json.dumps({
        "totalSize": 2,
        "records": [
            {"attributes": {"type": "Account"}, "Id": "001A", "Name": "Acme"},
            {"attributes": {"type": "Account"}, "Id": "001B", "Name": "Globex"},
        ],
        "done": True,
    })
    ref_tables, _ = _split_reference_results([{"tool": "soqlQuery", "result": result}])
    assert ref_tables
    table = ref_tables[0]
    assert "| Id | Name |" in table
    assert "IdName" not in table
    assert "001A" in table and "Acme" in table


if __name__ == "__main__":
    import unittest
    unittest.main(module=__name__)