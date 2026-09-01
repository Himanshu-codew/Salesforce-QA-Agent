"""
Regression tests for the hardened multi-agent pipeline (17 categories).

These tests exercise the GENERIC pipeline only — no query-specific routing rules
are added anywhere. Arbitrary new queries must work through the same code paths
(planner -> semantic RAG -> Qwen -> executor -> synthesizer).

All tests use mocks (no live LLM / Salesforce / MCP / embedding model / network).

Validated behaviors:
  - bounded overall process time (no indefinite hangs)
  - bounded per-stage timeouts (LLM / executor)
  - bounded multi-tool iteration loop
  - structured terminal errors {code, message} (no raw JSON leaks)
  - clean natural-language output on success
  - a NEW arbitrary query needs NO new routing rule
"""
import os
import sys
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.multi_agent as ma
from agent.multi_agent import Orchestrator

SAFE = {"safe": True, "requires_confirmation": False, "confirmation_message": "", "pending_action": None}


# ---------------------------------------------------------------------------
# Flexible mocks
# ---------------------------------------------------------------------------

class _LLM:
    def __init__(self):
        self.chat_result = "How can I help you with your Salesforce data today?"
        self.chat_exc = None
        self.chat_never = False
        self.with_tools_result = {"content": "", "tool_calls": [], "finish_reason": "stop"}
        self.with_tools_exc = None
        self.with_tools_never = False
        self.chat_calls = []

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        self.chat_calls.append(messages)
        if self.chat_exc:
            raise self.chat_exc
        if self.chat_never:
            await asyncio.sleep(30)
        return self.chat_result

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        if self.with_tools_exc:
            raise self.with_tools_exc
        if self.with_tools_never:
            await asyncio.sleep(30)
        return self.with_tools_result


class _Executor:
    def __init__(self):
        self.result = "[]"
        self.exc = None
        self.never = False
        self.executed = []

    async def execute(self, name, arguments):
        self.executed.append((name, arguments))
        if self.exc:
            raise self.exc
        if self.never:
            await asyncio.sleep(30)
        return self.result


class _Planner:
    def __init__(self, pending=False):
        self._pending = pending

    def has_pending_confirmation(self, session_id):
        return self._pending

    def check_tool_safety(self, tool_name, arguments, session_id="default"):
        return dict(SAFE)

    def process_confirmation(self, user_response, session_id):
        return None

    def clear_pending(self, session_id):
        return None


def make(
    plan=None,
    rag=(),
    chat=None,
    with_tools=None,
    executor_result="[]",
    max_iters=12,
):
    llm = _LLM()
    if chat is not None:
        llm.chat_result = chat
    if with_tools is not None:
        llm.with_tools_result = with_tools
    ex = _Executor()
    ex.result = executor_result

    orch = Orchestrator(llm=llm, executor=ex, max_iterations=max_iters, max_history=4)
    orch._generate_plan = AsyncMock(return_value=plan if plan is not None else [])
    orch._max_tool_iterations = max_iters

    ragm = MagicMock()
    ragm.get_relevant_tools.return_value = list(rag)
    orch.rag_retriever = ragm

    planner = _Planner()
    orch.safety_planner = planner
    return orch, llm, ex, planner


def run(orch, message, session_id="default"):
    return asyncio.run(_collect(orch, message, session_id))


async def _collect(orch, message, session_id):
    events = []
    async for e in orch.process_message(message, session_id):
        events.append(e)
    return events


def _ev(events, etype):
    return [e for e in events if e.get("type") == etype]


def _responses(events):
    return [e["data"] for e in events if e.get("type") == "response"]


def _errors(events):
    return [e for e in events if e.get("type") == "error"]


def _tool_calls(events):
    return [e for e in events if e.get("type") == "tool_call"]


SOQL = {"type": "function", "function": {"name": "soqlQuery"}}
FIND = {"type": "function", "function": {"name": "find"}}


def _soql_tc(name="soqlQuery"):
    return {"id": "t1", "name": name, "arguments": {"q": "SELECT Id FROM Account LIMIT 5"}}


# ---------------------------------------------------------------------------
# 1. Greeting
# ---------------------------------------------------------------------------

def test_greeting_routes_to_general_qwen():
    orch, llm, ex, _ = make(plan=[], rag=(), chat="Hi there! How can I help?")
    events = run(orch, "hi")
    responses = _responses(events)
    assert responses and responses[0] == "Hi there! How can I help?"
    assert not _tool_calls(events)
    assert not _errors(events)


# ---------------------------------------------------------------------------
# 2. General knowledge
# ---------------------------------------------------------------------------

def test_general_knowledge_no_tools():
    plan = None
    orch, llm, ex, _ = make(plan=[], rag=(), chat="Python is a programming language.")
    events = run(orch, "what is Python?")
    responses = _responses(events)
    assert responses and responses[0] == "Python is a programming language."
    assert not _tool_calls(events)


# ---------------------------------------------------------------------------
# 3. Simple Salesforce query
# ---------------------------------------------------------------------------

def test_simple_salesforce_query():
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "list accounts", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
        with_tools={"content": "", "tool_calls": [_soql_tc()], "finish_reason": "tool_calls"},
        chat="Here are your accounts:",
    )
    events = run(orch, "show my recent Accounts")
    assert any(tc["data"]["name"] == "soqlQuery" for tc in _tool_calls(events))
    assert _responses(events), "expected a synthesized response"
    assert not _errors(events)


# ---------------------------------------------------------------------------
# 4. Multi-step Salesforce query (dependent tasks)
# ---------------------------------------------------------------------------

def test_multi_step_salesforce_query():
    plan = [
        {"task_id": 1, "description": "find account id", "agent": "DataAgent", "depends_on": []},
        {"task_id": 2, "description": "query opportunities", "agent": "DataAgent", "depends_on": [1]},
    ]
    orch, llm, ex, _ = make(
        plan=plan,
        rag=(SOQL,),
        with_tools={"content": "", "tool_calls": [_soql_tc()], "finish_reason": "tool_calls"},
        chat="Here are the results.",
        max_iters=12,
    )
    events = run(orch, "show opportunities for each of my accounts")
    calls = _tool_calls(events)
    assert len(calls) >= 2, "expected sequential tool calls across dependent tasks"
    assert _responses(events)
    assert not _errors(events)


# ---------------------------------------------------------------------------
# 5. Ambiguous query
# ---------------------------------------------------------------------------

def test_ambiguous_query_never_crashes():
    orch, llm, ex, _ = make(plan=[], rag=(), chat="Can you clarify what you'd like?")
    events = run(orch, "Can you help me?")
    # Either a clean response OR a structured error — never an unhandled crash,
    # and never the old generic "An error occurred while formatting the response."
    assert _responses(events) or _errors(events)
    assert "An error occurred while formatting the response." not in [e["data"] for e in events]


# ---------------------------------------------------------------------------
# 6. Unknown Salesforce object / term
# ---------------------------------------------------------------------------

def test_unknown_object_returns_controlled_error():
    failure = json.dumps({"error": "Invalid type: Widget__x", "tool": "soqlQuery",
                          "suggestion": "Check the supported object types."})
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "query widgets", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
        with_tools={"content": "", "tool_calls": [_soql_tc()], "finish_reason": "tool_calls"},
        executor_result=failure,
    )
    events = run(orch, "select my Widget records")
    errors = _errors(events)
    assert errors, "expected a structured error for unknown object"
    assert errors[0]["code"] == "SALESFORCE_FAILED"
    assert errors[0]["message"] and "Invalid type" in errors[0]["message"]
    assert "An error occurred while formatting the response." not in errors[0]["message"]


# ---------------------------------------------------------------------------
# 7. Malformed Qwen response (tool call without a valid name)
# ---------------------------------------------------------------------------

def test_malformed_qwen_toolcall_is_structured_error():
    bad = {"content": "", "tool_calls": [{"id": "x", "name": "", "arguments": {}}],
           "finish_reason": "tool_calls"}
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "do something", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
        with_tools=bad,
    )
    events = run(orch, "do something with accounts")
    errors = _errors(events)
    assert errors and errors[0]["code"] == "INVALID_TOOL_CALL"


# ---------------------------------------------------------------------------
# 8. Malformed tool arguments (raw string, unparseable)
# ---------------------------------------------------------------------------

def test_malformed_tool_arguments_is_structured_error():
    bad = {"content": "", "tool_calls": [{"id": "x", "name": "soqlQuery", "arguments": "{not json!!"}],
           "finish_reason": "tool_calls"}
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "query", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
        with_tools=bad,
    )
    events = run(orch, "query accounts")
    errors = _errors(events)
    assert errors and errors[0]["code"] == "INVALID_TOOL_CALL"


def test_valid_string_arguments_are_normalized():
    tc = {"id": "x", "name": "soqlQuery", "arguments": json.dumps({"q": "SELECT Id FROM Account"})}
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "query", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
        with_tools={"content": "", "tool_calls": [tc], "finish_reason": "tool_calls"},
        chat="Here are results.",
    )
    events = run(orch, "query accounts")
    assert any(tc["data"]["name"] == "soqlQuery" for tc in _tool_calls(events))
    assert not _errors(events)


# ---------------------------------------------------------------------------
# 9. MCP timeout (bounded executor)
# ---------------------------------------------------------------------------

def test_executor_timeout_is_structured_error():
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "query", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
        with_tools={"content": "", "tool_calls": [_soql_tc()], "finish_reason": "tool_calls"},
    )
    ex.never = True
    with patch.object(ma, "EXECUTOR_TIMEOUT", 0.2):
        events = run(orch, "query accounts")
    errors = _errors(events)
    assert errors and errors[0]["code"] == "TIMEOUT"


# ---------------------------------------------------------------------------
# 10. Salesforce auth failure
# ---------------------------------------------------------------------------

def test_salesforce_auth_failure_is_structured_error():
    failure = json.dumps({"error": "401 Unauthorized: INVALID_SESSION_ID", "tool": "soqlQuery",
                          "suggestion": "Your session may have expired. Try reconnecting."})
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "query", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
        with_tools={"content": "", "tool_calls": [_soql_tc()], "finish_reason": "tool_calls"},
        executor_result=failure,
    )
    events = run(orch, "query accounts")
    errors = _errors(events)
    assert errors and errors[0]["code"] == "SALESFORCE_FAILED"


# ---------------------------------------------------------------------------
# 11. Salesforce API failure (non-auth)
# ---------------------------------------------------------------------------

def test_salesforce_api_failure_is_structured_error():
    failure = json.dumps({"error": "Malformed query: unexpected token", "tool": "soqlQuery",
                          "suggestion": "Check the SOQL syntax."})
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "query", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
        with_tools={"content": "", "tool_calls": [_soql_tc()], "finish_reason": "tool_calls"},
        executor_result=failure,
    )
    events = run(orch, "query accounts")
    errors = _errors(events)
    assert errors and errors[0]["code"] == "SALESFORCE_FAILED"
    assert not _responses(events)


# ---------------------------------------------------------------------------
# 12. Qwen timeout (bounded synthesis / general)
# ---------------------------------------------------------------------------

def test_qwen_timeout_is_structured_error():
    orch, llm, ex, _ = make(plan=[], rag=(), chat=None)
    llm.chat_never = True
    with patch.object(ma, "LLM_STAGE_TIMEOUT", 0.2):
        events = run(orch, "hi")
    errors = _errors(events)
    assert errors and errors[0]["code"] == "TIMEOUT"


def test_qwen_tool_selection_timeout_is_structured_error():
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "query", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
    )
    llm.with_tools_never = True
    with patch.object(ma, "LLM_STAGE_TIMEOUT", 0.2):
        events = run(orch, "query accounts")
    errors = _errors(events)
    assert errors and errors[0]["code"] == "TIMEOUT"


# ---------------------------------------------------------------------------
# 13. Empty planner result (two branches)
# ---------------------------------------------------------------------------

def test_empty_plan_no_tools_routes_to_general():
    orch, llm, ex, _ = make(plan=[], rag=(), chat="Sure, how can I help?")
    events = run(orch, "thanks")
    assert _responses(events) and not _tool_calls(events)


def test_empty_plan_with_tools_uses_implicit_task():
    orch, llm, ex, _ = make(
        plan=[], rag=(SOQL,),
        with_tools={"content": "", "tool_calls": [_soql_tc()], "finish_reason": "tool_calls"},
        chat="Here are your accounts.",
    )
    events = run(orch, "Accounts above 50000")
    assert any(tc["data"]["name"] == "soqlQuery" for tc in _tool_calls(events))
    assert _responses(events)


# ---------------------------------------------------------------------------
# 14 + 15. Multiple tool calls / tool loop protection
# ---------------------------------------------------------------------------

def test_multiple_tool_calls_in_one_task():
    two = {"content": "", "tool_calls": [_soql_tc("soqlQuery"), _soql_tc("find")],
           "finish_reason": "tool_calls"}
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "query", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL, FIND),
        with_tools=two,
        chat="Results.",
        max_iters=12,
    )
    events = run(orch, "show accounts and find anything else")
    assert len(_tool_calls(events)) == 2
    assert _responses(events)


def test_tool_loop_protection_stops_runaway():
    two = {"content": "", "tool_calls": [_soql_tc("soqlQuery"), _soql_tc("find")],
           "finish_reason": "tool_calls"}
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "query", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL, FIND),
        with_tools=two,
        chat="Results.",
        max_iters=2,
    )
    events = run(orch, "show accounts and find anything else")
    errors = _errors(events)
    assert errors and errors[0]["code"] == "TOO_MANY_STEPS"
    # The loop was stopped — no unbounded execution.
    assert len(ex.executed) < 2


# ---------------------------------------------------------------------------
# 16. Frontend response contract
# ---------------------------------------------------------------------------

def test_response_contract_only_clean_answer():
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "query", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
        with_tools={"content": "", "tool_calls": [_soql_tc()], "finish_reason": "tool_calls"},
        chat="Here is your clean answer.",
    )
    events = run(orch, "query accounts")
    for e in events:
        if e["type"] == "response":
            # Frontend renders only event.data — must be clean natural language,
            # never raw Qwen/tool/MCP JSON or planner payloads.
            assert e["data"] == "Here is your clean answer."
        if e["type"] == "error":
            assert isinstance(e.get("code"), str) and e.get("code")
            assert isinstance(e.get("message"), str) and e.get("message")


def test_error_contract_has_code_and_message():
    failure = json.dumps({"error": "boom", "tool": "soqlQuery", "suggestion": "retry"})
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "query", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
        with_tools={"content": "", "tool_calls": [_soql_tc()], "finish_reason": "tool_calls"},
        executor_result=failure,
    )
    events = run(orch, "query accounts")
    errs = _errors(events)
    assert errs
    assert errs[0].get("code") == "SALESFORCE_FAILED"
    assert errs[0].get("message")


# ---------------------------------------------------------------------------
# 17. Clean natural-language output (no raw JSON anywhere in user-facing events)
# ---------------------------------------------------------------------------

def test_no_raw_json_leaks_to_user_facing_events():
    raw_leak_markers = ('"name"', '"arguments"', '"tool_calls"', "soqlQuery(", "SELECT ")
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "query", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
        with_tools={"content": "", "tool_calls": [_soql_tc()], "finish_reason": "tool_calls"},
        chat="All good. I found the records you asked about.",
    )
    events = run(orch, "query accounts")
    for e in events:
        if e["type"] == "response":
            assert e["data"] == "All good. I found the records you asked about."
            assert not any(m in e["data"].lower() for m in raw_leak_markers)
    # Error events carry a clean message, not a raw JSON envelope.
    failure = json.dumps({"error": "auth failed", "tool": "soqlQuery", "suggestion": "reconnect"})
    orch2, llm2, ex2, _ = make(
        plan=[{"task_id": 1, "description": "query", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
        with_tools={"content": "", "tool_calls": [_soql_tc()], "finish_reason": "tool_calls"},
        executor_result=failure,
    )
    events2 = run(orch2, "query accounts")
    errs = _errors(events2)
    assert errs
    assert '"tool"' not in errs[0]["message"] and 'arguments' not in errs[0]["message"]


# ---------------------------------------------------------------------------
# New arbitrary query needs NO new routing rule
# ---------------------------------------------------------------------------

def test_arbitrary_salesforce_query_needs_no_rule():
    # A brand-new phrasing not present anywhere in the codebase's prompts/tests.
    novel_query = "dig out every deal that has not closed yet in my org"
    orch, llm, ex, _ = make(
        plan=[{"task_id": 1, "description": "query open opps", "agent": "DataAgent", "depends_on": []}],
        rag=(SOQL,),
        with_tools={"content": "", "tool_calls": [_soql_tc()], "finish_reason": "tool_calls"},
        chat="Here are the open deals.",
    )
    events = run(orch, novel_query)
    assert any(tc["data"]["name"] == "soqlQuery" for tc in _tool_calls(events))
    assert _responses(events)


def test_arbitrary_general_query_needs_no_rule():
    novel_query = "explain the trade-off between microservices and monoliths"
    orch, llm, ex, _ = make(plan=[], rag=(), chat="Great question about software architecture.")
    events = run(orch, novel_query)
    assert _responses(events) and not _tool_calls(events)


def test_overall_process_timeout_bounds_runaway_generator():
    # A stage that never resolves AND is resistant to the per-stage timeout id
    # handled by OVERALL_PROCESS_TIMEOUT as a final safety net.
    orch, llm, ex, _ = make(plan=[], rag=())
    llm.chat_never = True
    with patch.object(ma, "LLM_STAGE_TIMEOUT", 30.0), patch.object(ma, "OVERALL_PROCESS_TIMEOUT", 0.3):
        events = run(orch, "hi")
    errors = _errors(events)
    assert errors and errors[0]["code"] == "TIMEOUT"


if __name__ == "__main__":
    import unittest
    unittest.main(module=__name__)
