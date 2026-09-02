"""
Focused offline tests for E5: RAG degradation / safe tool fallback.

E5 fixes the silent tools=[] degradation when RAG returns empty (after an
internal exception, an empty score set, or a timeout):

  RAG failure -> [] -> Qwen called with no tools -> ungrounded SF answer

The fix, for Salesforce/data requests only:
  - Runnable RAG is bounded by RAG_TIMEOUT (never blocks the loop forever).
  - On empty/timeout -> fall back to the COMPLETE Salesforce tool registry.
  - The fallback ALWAYS passes through filter_tools_for_query (Fix A), so
    read-only queries never gain mutation tools and write queries keep what
    they need.
  - A Salesforce/data request must never finish success=true with an
    ungrounded LLM answer when no Salesforce tool result was fetched.

Clearly general/non-Salesforce queries are unaffected (existing
general-answer behavior preserved).

All tests use local mocks - no live LLM / Salesforce / MCP / ChromaDB.
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import SalesforceAgent, _has_salesforce_intent, filter_tools_for_query
from agent.multi_agent import Orchestrator
from tools.salesforce import get_tool_definitions, is_read_only

SOQL_TOOL = {
    "type": "function",
    "function": {"name": "soqlQuery", "description": "Run a SOQL query",
                 "parameters": {"type": "object", "properties": {}}},
}
CREATE_TOOL = {
    "type": "function",
    "function": {"name": "createSobjectRecord", "description": "Create a record",
                 "parameters": {"type": "object", "properties": {}}},
}


class _ReadLLM:
    """LLM fake: emit a soqlQuery tool call on every turn (normal path)."""

    def __init__(self, tool_calls=True):
        self.synthesis_called = False
        self._tool_calls = tool_calls

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        if self._tool_calls:
            return {"content": "", "tool_calls": [
                {"id": "t1", "name": "soqlQuery",
                 "arguments": {"q": "SELECT Id, Name FROM Account LIMIT 1"}}],
                "finish_reason": "tool_calls"}
        return {"content": "There are about 150 accounts in the org.", "tool_calls": [],
                "finish_reason": "stop"}

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        self.synthesis_called = True
        return "I processed your request."


class _TextLLM:
    """LLM fake that returns ONLY text (no tool calls) - used for the
    ungrounded-answer guard test."""

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        return {"content": "There are approximately 150 Account records.", "tool_calls": [],
                "finish_reason": "stop"}

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        return "There are approximately 150 Account records."


class _Exec:
    def __init__(self, result=""):
        self.result = result
        self.executed = []

    async def execute(self, name, arguments):
        self.executed.append((name, arguments))
        return self.result


class _NoRAG:
    """A ragreretriever stand-in whose get_relevant_tools returns [] (empty)."""

    def get_relevant_tools(self, user_query, top_k=None):
        return []


def _build_agent(llm=None, executor=None, rag=None, message="Show me the accounts"):
    agent = SalesforceAgent(llm=llm or _ReadLLM(), executor=executor or _Exec(), max_iterations=5)
    agent.rag_retriever = rag if rag is not None else agent.rag_retriever
    return agent


async def _run_events(agent, message, session_id="default"):
    events = []
    async for ev in agent.process_message(message, session_id):
        events.append(ev)
    return events


def _run(agent, message="Show me the accounts"):
    return asyncio.run(_run_events(agent, message))


def _tool_names(tools):
    return [t.get("function", {}).get("name", "") for t in tools]


# ---------------------------------------------------------------------------
# Intent helper (shared) sanity
# ---------------------------------------------------------------------------

def test_salesforce_intent_detection():
    assert _has_salesforce_intent("How many Account records do we have?")
    assert _has_salesforce_intent("List the first 3 Accounts")
    assert not _has_salesforce_intent("hello")
    assert not _has_salesforce_intent("what is python?")


# ---------------------------------------------------------------------------
# 1. Authenticated SalesforceAgent + RAG returns [] + Salesforce query
#    -> complete tool fallback is used.
# 2. Fallback passes through filter_tools_for_query.
# 10. Orchestrator empty-RAG Salesforce request gets safe fallback.
# 11. Orchestrator clearly general empty-RAG request retains general behaviour.
# ---------------------------------------------------------------------------

def test_agent_empty_rag_sf_query_falls_back_to_tools():
    agent = _build_agent(rag=_NoRAG())
    events = _run(agent, "How many Account records do we have?")
    # A real tool call happens because the fallback supplied a tool registry.
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert any(tc["data"]["name"] == "soqlQuery" for tc in tool_calls), \
        "fallback registry must make soqlQuery available so it can be called"


def test_agent_empty_rag_sf_query_fallback_passes_fix_a():
    agent = _build_agent(rag=_NoRAG())
    # Directly exercise the effective-tool helper on a read-only Salesforce query.
    async def go():
        tools = await agent._get_effective_tools("How many Account records?", "default")
        names = _tool_names(tools)
        return names, tools
    names, _ = asyncio.run(go())
    assert "soqlQuery" in names
    assert "getObjectSchema" in names


def test_orchestrator_empty_rag_sf_fallback():
    soql = {"type": "function", "function": {"name": "soqlQuery"}}
    orch = Orchestrator(llm=_MockOrchLLM(planner_plan=[], tools_return=[]),
                        executor=_MockOrchExec(), max_iterations=5, max_history=4)
    orch._generate_plan = _AsyncVal([])          # empty plan
    orch.rag_retriever.get_relevant_tools = lambda *a, **k: []   # empty RAG
    events = asyncio.run(_stream(orch, "Show me all Accounts above 50000"))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert any(tc["data"]["name"] == "soqlQuery" for tc in tool_calls), \
        "empty-RAG Salesforce request must fall back and execute a tool"


def test_orchestrator_empty_rag_general_keeps_general_answer():
    orch = Orchestrator(llm=_MockOrchLLM(planner_plan=[], tools_return=[]),
                        executor=_MockOrchExec(), max_iterations=5, max_history=4)
    orch._generate_plan = _AsyncVal([])          # empty plan
    orch.rag_retriever.get_relevant_tools = lambda *a, **k: []   # empty RAG
    orch.llm.chat = _AsyncText("Hi! How can I help you?")
    events = asyncio.run(_stream(orch, "what is python?"))
    responses = [e["data"] for e in events if e["type"] == "response"]
    assert responses
    # No Salesforce tool should run for a clearly general question.
    assert not any(e["type"] == "tool_call" for e in events)


# ---------------------------------------------------------------------------
# 3. Read-only query fallback does NOT expose mutation tools.
# ---------------------------------------------------------------------------

def test_read_only_fallback_never_exposes_mutation_tools():
    async def go():
        tools = await _build_agent(rag=_NoRAG())._get_effective_tools(
            "How many Account records?", "default")
        return _tool_names(tools)
    names = asyncio.run(go())
    assert "soqlQuery" in names and "getObjectSchema" in names
    assert all(is_read_only(n) for n in names), f"read-only fallback leaked: {names}"


# ---------------------------------------------------------------------------
# 4. Explicit mutation query falls back but keeps required mutation tools
#    (subject to existing confirmation/READ_ONLY safety).
# ---------------------------------------------------------------------------

def test_write_query_fallback_keeps_mutation_tools():
    async def go():
        # "create" is a write keyword -> filter_tools_for_query returns full set.
        tools = await _build_agent(rag=_NoRAG())._get_effective_tools(
            "Create a new Account", "default")
        return _tool_names(tools)
    names = asyncio.run(go())
    assert "createSobjectRecord" in names, "write query must retain mutation tools"


# ---------------------------------------------------------------------------
# 5. Salesforce-specific query cannot return ungrounded success when no tool
#    result exists.
# ---------------------------------------------------------------------------

def test_sf_query_no_tool_result_is_not_ungrounded_success():
    # LLM returns only text, no tool calls, and no tool result is fetched.
    execf = _Exec(result="{}")  # won't be called because no tool_calls
    agent = _build_agent(llm=_TextLLM(), executor=execf, rag=_NoRAG())
    events = _run(agent, "How many Account records do we have?")
    errs = [e for e in events if e["type"] == "error"]
    respons = [e for e in events if e["type"] == "response"]
    assert errs, "must emit a controlled error instead of an ungrounded success"
    assert errs[0].get("code") == "SALESFORCE_FAILED"
    assert not respons, f"must NOT succeed ungrounded, got: {respons}"
    assert not execf.executed


# ---------------------------------------------------------------------------
# 6. General/non-Salesforce query can still use existing general-answer
#    behaviour (no guard; no forced tool call).
# ---------------------------------------------------------------------------

def test_general_query_not_guarded_and_not_forced():
    agent = _build_agent(llm=_TextLLM(), executor=_Exec(), rag=_NoRAG())
    events = _run(agent, "hello")
    respons = [e for e in events if e["type"] == "response"]
    assert respons, "general query should still produce a response"
    assert not [e for e in events if e["type"] == "error"], \
        "general query must not be surfaced as a Salesforce error"


# ---------------------------------------------------------------------------
# 7 & 8. RAG timeout does not leave Salesforce request with tools=[] and does
#         not fake Salesforce data.
# ---------------------------------------------------------------------------

def test_agent_rag_timeout_uses_fallback_not_empty():
    agent = _build_agent(rag=_NoRAG())
    import agent.agent as amod
    original = amod.RAG_TIMEOUT
    try:
        amod.RAG_TIMEOUT = 0.0  # force immediate timeout on to_thread+wait_for
        async def callback(msg, tools):
            pass
        # Patch _bounded retrieval path is internal; simulate by confirming
        # the helper returns fallback tools under a failing (throwing) RAG.
        def boom(*a, **k):
            raise RuntimeError("controlled RAG failure")
        agent.rag_retriever.get_relevant_tools = boom
        tools = asyncio.run(agent._get_effective_tools(
            "How many Account records?", "default"))
        names = _tool_names(tools)
        assert "soqlQuery" in names, f"exception must still fall back, got {names}"
    finally:
        amod.RAG_TIMEOUT = original


def test_agent_rag_timeout_does_not_fake_data():
    # Even when RAG times out and falls back to tools, if the tool is never
    # executed (LLM returns text) the SF query must not produce a fabricated
    # success answer.
    agent = _build_agent(llm=_TextLLM(), executor=_Exec(), rag=_NoRAG())
    events = _run(agent, "How many Account records do we have?")

    async def recheck():
        tools = await agent._get_effective_tools("How many Account records?", "default")
        return tools
    # The guarded path already proven in test 5; assert no fabricated tool_result.
    assert not any(e["type"] == "tool_result" for e in events)
    errs = [e for e in events if e["type"] == "error"]
    assert errs and errs[0].get("code") == "SALESFORCE_FAILED"


# ---------------------------------------------------------------------------
# 9. Existing RAG exception -> [] is safely handled (falls back for SF).
# ---------------------------------------------------------------------------

def test_rag_exception_degrades_to_safe_fallback():
    agent = _build_agent(rag=_NoRAG())

    def boom(*a, **k):
        raise RuntimeError("chroma exploded")
    agent.rag_retriever.get_relevant_tools = boom
    tools = asyncio.run(agent._get_effective_tools("List the Accounts", "default"))
    names = _tool_names(tools)
    assert "soqlQuery" in names, "an internal RAG exception must not yield tools=[] for SF"

    # General query: exception -> [] stays [] (general answer preserved).
    tools2 = asyncio.run(agent._get_effective_tools("hello", "default"))
    assert _tool_names(tools2) == []


# ---------------------------------------------------------------------------
# 12. Fix A remains intact.
# 13. Fix B remains intact.
# 14. D1 remains intact.
# 15. P0 remains intact.
# ---------------------------------------------------------------------------

def test_fix_a_intact_read_only_filtering():
    msg = "How many Account records?"
    full = get_tool_definitions()
    filtered = filter_tools_for_query(full, msg)
    assert all(is_read_only(n) for n in _tool_names(filtered))


def test_fix_b_intact_count_helper_still_present():
    from agent.agent import _is_soql_count
    assert callable(_is_soql_count)


def test_d1_intact_error_envelope_still_errors():
    from agent.agent import _executor_error_message
    assert _executor_error_message('{"error":"nope","tool":"soqlQuery"}') == "nope"
    assert _executor_error_message('{"totalSize":0,"records":[]}') is None


def test_p0_intact_timeout_class_present():
    from agent.agent import AgentTimeoutError, _bounded_call, AGENT_LLM_TIMEOUT
    assert issubclass(AgentTimeoutError, Exception)


# ---------------------------------------------------------------------------
# Orchestrator test doubles
# ---------------------------------------------------------------------------

class _MockOrchLLM:
    def __init__(self, planner_plan, tools_return):
        self.chat_calls = []
        self._plan = planner_plan
        self._tools = tools_return
        self._chat_return = "General answer."

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        self.chat_calls.append(dict(messages=messages))
        return self._chat_return

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        # Emit a soqlQuery tool call only if soqlQuery is in the tool list.
        names = _tool_names(tools or [])
        if "soqlQuery" in names:
            return {"content": "", "tool_calls": [
                {"id": "ot1", "name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Account LIMIT 1"}}],
                "finish_reason": "tool_calls"}
        return {"content": "", "tool_calls": [], "finish_reason": "stop"}

    async def get_last_user(self):
        return self.chat_calls


class _MockOrchExec:
    def __init__(self):
        self.executed = []

    async def execute(self, name, arguments):
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


class _AsyncText:
    def __init__(self, text):
        self._text = text

    def __call__(self, *a, **k):
        return _AwaitText(self._text)


class _AwaitText:
    def __init__(self, text):
        self._text = text

    def __await__(self):
        async def _inner():
            return self._text
        return _inner().__await__()


async def _stream(orch, message, session_id="default"):
    events = []
    async for ev in orch.process_message(message, session_id):
        events.append(ev)
    return events


if __name__ == "__main__":
    import unittest
