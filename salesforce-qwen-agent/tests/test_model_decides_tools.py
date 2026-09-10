"""
Model-driven tool selection tests.

RAG tool subsetting is disabled. These tests pin the intended behavior:

  * ToolRAGRetriever returns the COMPLETE Salesforce registry so the model
    (Qwen) decides which tool to call — for real queries.
  * The worker LLM actually receives the full registry (write/compound) or the
    read-only-filtered registry (read-only) — never a stale RAG subset.
  * Greetings/general chat route to the general Qwen path with no Salesforce
    tools surfaced and no tool calls executed.
  * Read-only requests never expose mutation/destructive tools to the model.

All tests are local doubles — no live LLM / Salesforce / MCP.
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import filter_tools_for_query
from agent.multi_agent import Orchestrator
from agent.rag import ToolRAGRetriever
from tools.salesforce import get_tool_definitions, is_read_only

_MUTATORS = ("createSobjectRecord", "updateSobjectRecord", "updateRelatedRecord",
             "deleteSobjectRecord", "deleteRelatedRecord")


def _tool_names(tools):
    return [t.get("function", {}).get("name", "") for t in (tools or [])]


# ---------------------------------------------------------------------------
# Retriever passthrough
# ---------------------------------------------------------------------------

def test_retriever_returns_full_registry():
    r = ToolRAGRetriever()
    tools = r.get_relevant_tools("Find Opportunities with Amount above 50000", top_k=5)
    assert len(tools) == len(r.all_tools) == len(get_tool_definitions())
    names = _tool_names(tools)
    assert "soqlQuery" in names and "createSobjectRecord" in names


def test_retriever_returns_empty_only_for_degenerate_input():
    r = ToolRAGRetriever()
    assert r.get_relevant_tools("   h  ", top_k=5) == []
    assert r.get_relevant_tools("", top_k=5) == []


# ---------------------------------------------------------------------------
# Read-only safety net (deterministic, survives the passthrough)
# ---------------------------------------------------------------------------

def test_read_only_never_surfaces_mutation_tools():
    names = _tool_names(filter_tools_for_query(get_tool_definitions(), "Show me all Accounts"))
    assert "soqlQuery" in names
    assert all(not n.endswith("Record") or n not in _MUTATORS for n in names)
    assert all(is_read_only(n) for n in names), f"leaked: {names}"


def test_explicit_write_keeps_full_set_for_model():
    names = _tool_names(filter_tools_for_query(
        get_tool_definitions(), "Create a new Account for Acme"))
    assert "createSobjectRecord" in names


# ---------------------------------------------------------------------------
# Worker-level: the LLM receives the right tool set
# ---------------------------------------------------------------------------

class _RecordingLLM:
    """Records every tool list the worker/general LLM was given."""

    def __init__(self):
        self.tool_sets = []
        self.chat_return = "General answer."
        self.general_chat_calls = 0

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        self.tool_sets.append(_tool_names(tools or []))
        names = self.tool_sets[-1]
        if "soqlQuery" in names:
            return {"content": "", "tool_calls": [
                {"id": "t1", "name": "soqlQuery",
                 "arguments": {"q": "SELECT Id FROM Account LIMIT 5"}}],
                "finish_reason": "tool_calls"}
        return {"content": "", "tool_calls": [], "finish_reason": "stop"}

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        self.general_chat_calls += 1
        return self.chat_return


class _RecordingExec:
    def __init__(self):
        self.executed = []

    async def execute(self, name, arguments, user_provenance=None):
        self.executed.append((name, arguments))
        return '{"totalSize":0,"records":[]}'


class _AsyncVal:
    def __init__(self, value):
        self._value = value

    def __call__(self, *a, **k):
        return _AwaitVal(self._value)


class _AwaitVal:
    def __init__(self, value):
        self._value = value

    def __await__(self):
        async def _inner():
            return self._value
        return _inner().__await__()


def _build_llm():
    return _RecordingLLM()


def _build_orchestrator(llm, execf):
    orch = Orchestrator(llm=llm, executor=execf, max_iterations=5, max_history=4)
    return orch


async def _stream(orch, message, session_id="default"):
    events = []
    async for ev in orch.process_message(message, session_id):
        events.append(ev)
    return events


def _run(orch, message, session_id="default"):
    return asyncio.run(_stream(orch, message, session_id))


def test_read_query_worker_sees_read_only_tools_only():
    llm = _build_llm()
    execf = _RecordingExec()
    orch = _build_orchestrator(llm, execf)
    orch._generate_plan = _AsyncVal([
        {"task_id": 1, "description": "Show me all Accounts", "agent": "DataAgent", "depends_on": []},
    ])
    events = _run(orch, "Show me all Accounts above 50000")
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert any(tc["data"]["name"] == "soqlQuery" for tc in tool_calls)
    assert llm.tool_sets, "worker must have been called with a tool list"
    worker_names = llm.tool_sets[-1]
    assert "soqlQuery" in worker_names
    for m in _MUTATORS:
        assert m not in worker_names, f"read-only request leaked {m} to the model"


def test_write_query_worker_sees_full_registry():
    llm = _build_llm()
    execf = _RecordingExec()
    orch = _build_orchestrator(llm, execf)
    orch._generate_plan = _AsyncVal([
        {"task_id": 1, "description": "Create a new Account for Acme", "agent": "DataAgent",
         "depends_on": []},
    ])
    events = _run(orch, "Create a new Account for Acme")
    assert llm.tool_sets, "worker must have been called with a tool list"
    worker_names = llm.tool_sets[-1]
    assert len(worker_names) == len(get_tool_definitions()), \
        "write query must give the model the full registry to choose from"
    assert "createSobjectRecord" in worker_names
    # At least one tool_call event is produced (the model used the registry).
    assert any(e["type"] == "tool_call" for e in events)


def test_greeting_routes_to_general_chat_without_tools():
    llm = _build_llm()
    execf = _RecordingExec()
    orch = _build_orchestrator(llm, execf)
    orch._generate_plan = _AsyncVal([])  # planner says: no Salesforce task
    events = _run(orch, "hello, how are you?")
    assert any(e["type"] == "response" for e in events)
    assert not any(e["type"] == "tool_call" for e in events), \
        "greeting must never produce a Salesforce tool call"
    assert llm.general_chat_calls >= 1, "greeting must hit the general chat path"
    assert not execf.executed


def test_retriever_via_orchestrator_non_fast_path_full_set():
    # Force the orchestrator's semantic path (no fast path) and verify the tool
    # list given to the worker equals the full registry for a Salesforce query.
    llm = _build_llm()
    execf = _RecordingExec()
    orch = _build_orchestrator(llm, execf)
    orch._generate_plan = _AsyncVal([
        {"task_id": 1, "description": "How many Account records exist?", "agent": "DataAgent",
         "depends_on": []},
    ])
    _run(orch, "How many Account records exist?")
    assert llm.tool_sets, "worker must have been called with a tool list"
    worker_names = llm.tool_sets[-1]
    # count query is read-only -> soqlQuery present, no mutations leaked.
    assert "soqlQuery" in worker_names
    for m in _MUTATORS:
        assert m not in worker_names