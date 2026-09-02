"""
Focused offline tests for D1: executor error-envelope handling in the
authenticated SalesforceAgent path (agent/agent.py).

D1 roots:
- ToolExecutor.execute() returns failure envelopes as JSON *strings*, e.g.
    {"error": "..."}
    {"error": "...", "tool": "..."}
    {"error": "...", "tool": "...", "suggestion": "..."}
- agent/agent.py must DETECT these, emit a controlled {"type":"error",
  "code":"SALESFORCE_FAILED", ...} event, and NEVER let the envelope reach
  tool_results_fetched / memory / table-formatting / synthesis as normal data.
- The Orchestrator already has this via multi_agent._executor_error_message();
  this is the agent.agent equivalent (deliberately not imported to avoid a
  circular dependency).

Invariants preserved:
- Normal Salesforce results (records, COUNT, zero-count) are NOT detected.
- Malformed/non-JSON results are NOT treated as errors and do not throw.
- P0 AgentTimeoutError still yields TIMEOUT, never SALESFORCE_FAILED.
- Fix A (read-only filter) / Fix B (COUNT/zero-count) remain untouched.

All tests use mocks - no live LLM / Salesforce / MCP.
"""

import os
import sys
import asyncio
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import agent as agent_mod
from agent.agent import (
    SalesforceAgent,
    _executor_error_message,
    _salesforce_failed_event,
)

SOQL_TOOL = {
    "type": "function",
    "function": {"name": "soqlQuery", "description": "Run a SOQL query",
                 "parameters": {"type": "object", "properties": {}}},
}


class _ReadLLM:
    """LLM fake: emit one soqlQuery tool call, then nothing (only called once
    because a D1 error short-circuits before a synthesis step)."""

    def __init__(self):
        self.synthesis_called = False
        self.tool_call = {
            "id": "t1", "name": "soqlQuery",
            "arguments": {"q": "SELECT Id, Name FROM Account LIMIT 1"},
        }

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        return {"content": "", "tool_calls": [self.tool_call], "finish_reason": "tool_calls"}

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        self.synthesis_called = True
        return "I processed your request."


class _Exec:
    def __init__(self, result):
        self.result = result
        self.executed = []

    async def execute(self, name, arguments):
        self.executed.append((name, arguments))
        return self.result


def _build_agent(executor_result, llm=None):
    """Build a SalesforceAgent with mocks for the normal (non-confirmed) path."""
    agent = SalesforceAgent(llm=llm or _ReadLLM(), executor=_Exec(executor_result), max_iterations=5)
    agent.rag_retriever.get_relevant_tools = lambda *a, **k: [SOQL_TOOL]
    return agent


def _run(agent, message="Show me the accounts"):
    async def _go():
        events = []
        async for ev in agent.process_message(message, "default"):
            events.append(ev)
        return events
    return asyncio.run(_go())


def _only_errors(events):
    return [e for e in events if e.get("type") == "error"]


def _memory_has_error(agent, session="default"):
    mem = agent._get_memory(session)
    for m in mem.get_messages_for_llm(""):
        if isinstance(m, dict) and m.get("role") == "tool":
            try:
                parsed = json.loads(m.get("content", ""))
                if isinstance(parsed, dict) and parsed.get("error"):
                    return True
            except Exception:
                pass
    return False


# ---------------------------------------------------------------------------
# 1. Detector: executor error envelope shapes (unit tests)
# ---------------------------------------------------------------------------


def test_detector_plain_error():
    assert _executor_error_message(json.dumps({"error": "Salesforce query failed"})) == "Salesforce query failed"


def test_detector_error_with_tool():
    assert _executor_error_message(json.dumps({"error": "Salesforce query failed", "tool": "soqlQuery"})) == "Salesforce query failed"


def test_detector_error_with_suggestion():
    msg = _executor_error_message(json.dumps({"error": "Invalid SOQL", "tool": "soqlQuery", "suggestion": "Check object name"}))
    assert "Invalid SOQL" in msg
    assert "Check object name" in msg


def test_detector_empty_error_is_not_error():
    assert _executor_error_message(json.dumps({"error": ""})) is None


def test_detector_non_string_error_is_not_error():
    assert _executor_error_message(json.dumps({"error": 123})) is None


# ---------------------------------------------------------------------------
# 2. Detector: normal Salesforce results are NOT errors
# ---------------------------------------------------------------------------


def test_detector_normal_records_result_not_error():
    raw = json.dumps({"totalSize": 1, "records": [{"Id": "a", "Name": "Acme",
                                                    "attributes": {"type": "Account"}}], "done": True})
    assert _executor_error_message(raw) is None


def test_detector_count_result_not_error():
    raw = json.dumps({"totalSize": 1, "records": [{"attributes": {"type": "Account"}, "expr0": 22}]})
    assert _executor_error_message(raw) is None


def test_detector_zero_count_result_not_error():
    raw = json.dumps({"totalSize": 0, "records": []})
    assert _executor_error_message(raw) is None


# ---------------------------------------------------------------------------
# 3. Detector: malformed / non-JSON input
# ---------------------------------------------------------------------------


def test_detector_non_json_returns_none():
    assert _executor_error_message("some ordinary plain string") is None


def test_detector_empty_string_returns_none():
    assert _executor_error_message("") is None


def test_detector_non_string_returns_none():
    assert _executor_error_message(42) is None


# ---------------------------------------------------------------------------
# 4. Normal tool execution: executor error envelope -> SALESFORCE_FAILED
# ---------------------------------------------------------------------------


def test_normal_tool_error_yields_controlled_error(monkeypatch):
    agent = _build_agent(json.dumps({"error": "Salesforce query failed", "tool": "soqlQuery"}))
    monkeypatch.setattr(agent_mod, "AGENT_EXECUTOR_TIMEOUT", 5.0)
    events = _run(agent)

    errors = _only_errors(events)
    assert errors, "expected a controlled SALESFORCE_FAILED error event"
    assert errors[0]["code"] == "SALESFORCE_FAILED"
    assert "Salesforce query failed" in errors[0]["message"]

    # Nothing synthesized / emitted as a fake success.
    assert not [e for e in events if e.get("type") == "response"]
    assert not [e for e in events if e.get("type") == "tool_result"]

    # The failed envelope was not stored in memory as a successful tool result.
    assert not _memory_has_error(agent)


def test_normal_tool_error_is_not_stored_as_fetched_data():
    raw = json.dumps({"error": "Salesforce query failed", "tool": "soqlQuery"})

    class _ProbeLLM(_ReadLLM):
        pass

    agent = _build_agent(raw, llm=_ProbeLLM())
    events = _run(agent)
    errors = _only_errors(events)
    assert errors and errors[0]["code"] == "SALESFORCE_FAILED"
    # No synthesis call: the LLM was never asked to write an answer for the failure.
    assert not agent.llm.synthesis_called


# ---------------------------------------------------------------------------
# 5. Tool-not-found envelope ({"error": ...} without a "tool" field)
# ---------------------------------------------------------------------------


def test_tool_not_found_envelope_detected(monkeypatch):
    agent = _build_agent(json.dumps({"error": "Tool 'foo' not found"}))
    monkeypatch.setattr(agent_mod, "AGENT_EXECUTOR_TIMEOUT", 5.0)
    events = _run(agent)
    errors = _only_errors(events)
    assert errors, "tool-not-found envelope (no 'tool' field) must be detected"
    assert errors[0]["code"] == "SALESFORCE_FAILED"
    assert "Tool 'foo' not found" in errors[0]["message"]
    assert not agent.llm.synthesis_called


# ---------------------------------------------------------------------------
# 6. Normal healthy Salesforce read still works (not detected)
# ---------------------------------------------------------------------------


def test_normal_read_not_detected_and_returns_response(monkeypatch):
    raw = json.dumps({"totalSize": 1, "records": [{"Id": "a", "Name": "Acme",
                                                    "attributes": {"type": "Account"}}], "done": True})
    agent = _build_agent(raw)
    monkeypatch.setattr(agent_mod, "AGENT_EXECUTOR_TIMEOUT", 5.0)
    events = _run(agent)
    assert not _only_errors(events), "normal result must NOT be treated as an error"
    # The direct-response fast path should produce a response.
    responses = [e for e in events if e.get("type") == "response"]
    assert responses


# ---------------------------------------------------------------------------
# 7. Fix B preserved: COUNT and zero-count results are NOT detected
# ---------------------------------------------------------------------------


def test_count_result_not_detected(monkeypatch):
    raw = json.dumps({"totalSize": 1, "records": [{"attributes": {"type": "Account"}, "expr0": 22}]})
    agent = _build_agent(raw)
    monkeypatch.setattr(agent_mod, "AGENT_EXECUTOR_TIMEOUT", 5.0)
    events = _run(agent)
    assert not _only_errors(events)
    responses = [e for e in events if e.get("type") == "response"]
    assert responses


def test_zero_count_result_not_detected(monkeypatch):
    raw = json.dumps({"totalSize": 0, "records": []})
    agent = _build_agent(raw)
    agent._get_memory("default")  # ensure memory exists
    monkeypatch.setattr(agent_mod, "AGENT_EXECUTOR_TIMEOUT", 5.0)
    events = _run(agent)
    assert not _only_errors(events)


# ---------------------------------------------------------------------------
# 8. Confirmed mutation: executor error envelope -> SALESFORCE_FAILED
# ---------------------------------------------------------------------------


class _StubPlannerConfirmed:
    def has_pending_confirmation(self, session_id="default"):
        return True

    def process_confirmation(self, user_message, session_id="default"):
        return {"tool_name": "updateRelatedRecord",
                "arguments": {"record-id": "001xx", "fields": {"Name": "x"}},
                "type": "update"}


class _MutatingLLM:
    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        return {"content": "", "tool_calls": [], "finish_reason": "stop"}

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        return "done"


def test_confirmed_mutation_error_yields_controlled_error(monkeypatch):
    err = json.dumps({"error": "Update failed", "tool": "updateRelatedRecord"})
    agent = SalesforceAgent(llm=_MutatingLLM(), executor=_Exec(err), max_iterations=5)
    agent.planner = _StubPlannerConfirmed()
    monkeypatch.setattr(agent_mod, "AGENT_EXECUTOR_TIMEOUT", 5.0)

    events = _run(agent, message="yes")
    errors = _only_errors(events)
    assert errors and errors[0]["code"] == "SALESFORCE_FAILED"
    assert "Update failed" in errors[0]["message"]
    # No fake success / no memory pollution as a successful tool result.
    assert not [e for e in events if e.get("type") == "response"]
    assert not _memory_has_error(agent)


# ---------------------------------------------------------------------------
# 9. P0 regression: AgentTimeoutError still yields TIMEOUT, not SALESFORCE_FAILED
# ---------------------------------------------------------------------------


def test_p0_timeout_is_not_salesforce_failed(monkeypatch):
    class _SlowExec:
        async def execute(self, name, arguments):
            await asyncio.sleep(60.0)
            return "{}"

    agent = SalesforceAgent(llm=object(), executor=_SlowExec(), max_iterations=5)
    agent.planner = _StubPlannerConfirmed()  # confirmed path fastest to trigger
    monkeypatch.setattr(agent_mod, "AGENT_EXECUTOR_TIMEOUT", 0.05)

    events = _run(agent, message="yes")
    errors = _only_errors(events)
    assert errors, "expected a timeout error event"
    assert errors[0]["code"] == "TIMEOUT"
    assert errors[0]["code"] != "SALESFORCE_FAILED"


# ---------------------------------------------------------------------------
# event builder shape
# ---------------------------------------------------------------------------


def test_salesforce_failed_event_shape():
    ev = _salesforce_failed_event("boom")
    assert ev["type"] == "error"
    assert ev["code"] == "SALESFORCE_FAILED"
    assert ev["message"] == "boom"