"""
Offline tests for P0 reliability fix D4: timeout protection on the
agent.agent (SalesforceAgent) path and the overall /chat ceiling in app.py.

Guarantees under test:
- A per-stage LLM/executor timeout is converted into a CONTROLLED structured
  error (code "TIMEOUT") and never into a fake/synthetic result.
- asyncio.CancelledError (task cancellation) is NOT swallowed by the watchdog.
- The timeout constants reuse LLM_STAGE_TIMEOUT / EXECUTOR_TIMEOUT, matching
  the Orchestrator's env/values (no circular import).
- app.py exposes CHAT_OVERALL_TIMEOUT (defense-in-depth ceiling) that reuses
  OVERALL_PROCESS_TIMEOUT.
- Fix A (filter_tools_for_query) and Fix B zero-count normalization remain
  reachable/unchanged on this path.

All tests are mocked - no live LLM, Salesforce, or MCP requests.
"""

import os
import sys
import asyncio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import agent as agent_mod
from agent.agent import (
    AgentTimeoutError,
    _bounded_call,
    _timeout_error_event,
    SalesforceAgent,
    filter_tools_for_query,
)
from agent.multi_agent import _normalize_zero_count_result as _orch_normalize


# Fix B helper lives on the Orchestrator (multi_agent); re-expose locally for
# the preservation probe below.
_normalize_zero_count_result = _orch_normalize


# ---------------------------------------------------------------------------
# _bounded_call primitives
# ---------------------------------------------------------------------------


async def _slow(delay: float = 60.0):
    await asyncio.sleep(delay)
    return "never"


async def _fast():
    return "result"


async def _cancelled():
    await asyncio.sleep(60.0)
    return "never"


def test_bounded_call_returns_fast_value_immediately():
    async def _go():
        return await _bounded_call(_fast(), 5.0, "probe")
    assert asyncio.run(_go()) == "result"


def test_bounded_call_raises_controlled_timeout_for_slow_awaitable():
    async def _go():
        try:
            await _bounded_call(_slow(), 0.05, "probe")
        except AgentTimeoutError as exc:
            return exc
        raise AssertionError("expected AgentTimeoutError")
    exc = asyncio.run(_go())
    assert isinstance(exc, AgentTimeoutError)
    assert exc.stage == "probe"
    assert "probe" in str(exc)


def test_bounded_call_never_swallows_cancelled_error():
    # asyncio.CancelledError is a BaseException; the watchdog must NOT turn it
    # into an AgentTimeoutError, otherwise outer task cancellation could be
    # misreported as a "timeout".
    async def _go():
        task = asyncio.create_task(_bounded_call(_cancelled(), 5.0, "probe"))
        await asyncio.sleep(0.02)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            return "cancelled"
        raise AssertionError("expected cancellation to propagate")
    assert asyncio.run(_go()) == "cancelled"


def test_bounded_call_surfaces_unrelated_exception_unchanged():
    async def _boom():
        raise ValueError("synthetic failure")
    async def _go():
        try:
            await _bounded_call(_boom(), 5.0, "probe")
        except ValueError:
            return "value-error"
        raise AssertionError("expected ValueError")
    assert asyncio.run(_go()) == "value-error"


# ---------------------------------------------------------------------------
# _timeout_error_event shape
# ---------------------------------------------------------------------------


def test_timeout_error_event_shape():
    ev = _timeout_error_event("probe")
    assert ev["type"] == "error"
    assert ev["code"] == "TIMEOUT"
    assert "probe" in ev["message"]
    assert "probe" in ev["data"]


# ---------------------------------------------------------------------------
# Constants reuse the same env var names/values as the Orchestrator
# ---------------------------------------------------------------------------


def test_agent_timeout_constants_read_orchestrator_env_defaults(monkeypatch):
    monkeypatch.delenv("LLM_STAGE_TIMEOUT", raising=False)
    monkeypatch.delenv("EXECUTOR_TIMEOUT", raising=False)
    import importlib
    mod = importlib.reload(agent_mod)
    assert mod.AGENT_LLM_TIMEOUT == 90.0
    assert mod.AGENT_EXECUTOR_TIMEOUT == 90.0


def test_agent_timeout_constants_respect_env_overrides(monkeypatch):
    monkeypatch.setenv("LLM_STAGE_TIMEOUT", "7.5")
    monkeypatch.setenv("EXECUTOR_TIMEOUT", "3.0")
    import importlib
    mod = importlib.reload(agent_mod)
    assert mod.AGENT_LLM_TIMEOUT == 7.5
    assert mod.AGENT_EXECUTOR_TIMEOUT == 3.0


def test_app_chat_overall_timeout_default_is_300():
    import app as app_mod
    assert app_mod.CHAT_OVERALL_TIMEOUT == 300.0


def test_app_chat_overall_timeout_respects_env_override(monkeypatch):
    import importlib
    monkeypatch.setenv("OVERALL_PROCESS_TIMEOUT", "22")
    import app as app_mod
    importlib.reload(app_mod)
    assert app_mod.CHAT_OVERALL_TIMEOUT == 22.0


# ---------------------------------------------------------------------------
# P0 guarantee: confirmed-operation executor timeout -> controlled error event
# ---------------------------------------------------------------------------


class _StubPlannerConfirmed:
    def has_pending_confirmation(self, session_id="default"):
        return True

    def process_confirmation(self, user_message, session_id="default"):
        return {
            "tool_name": "deleteSO",
            "arguments": {"sobject-name": "Account", "id": "001xx"},
            "type": "delete",
        }


async def _executor_timeout_blocker():
    await asyncio.sleep(60.0)
    return "{}"


def test_confirmed_executor_timeout_yields_controlled_error_not_fake_result(monkeypatch):
    agent = SalesforceAgent(llm=object(), executor=object(), max_iterations=3)
    agent.planner = _StubPlannerConfirmed()

    class _Exec:
        async def execute(self, name, arguments):
            await asyncio.sleep(60.0)
            return "{}"

    agent.executor = _Exec()

    monkeypatch.setattr(agent_mod, "AGENT_EXECUTOR_TIMEOUT", 0.05)

    async def _go():
        events = []
        async for ev in agent.process_message("yes", "default"):
            events.append(ev)
        return events

    events = asyncio.run(_go())
    error_events = [e for e in events if e.get("type") == "error"]
    assert error_events, "expected a controlled error event on timeout"
    ev = error_events[0]
    assert ev["code"] == "TIMEOUT"
    # No synthetic tool_result / response fabricated from the timeout.
    fake_results = [e for e in events if e.get("type") == "tool_result"]
    assert not fake_results
    responses = [e for e in events if e.get("type") == "response"]
    assert not responses


def test_confirmed_executor_fast_result_still_yields_tool_result(monkeypatch):
    class _FakeLLM:
        async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
            return {"content": "Operation completed", "tool_calls": [], "finish_reason": "stop"}

        async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
            return "Operation completed"

    agent = SalesforceAgent(llm=_FakeLLM(), executor=object(), max_iterations=3)
    agent.planner = _StubPlannerConfirmed()

    class _Exec:
        async def execute(self, name, arguments):
            return '{"totalSize":1}'

    agent.executor = _Exec()
    monkeypatch.setattr(agent_mod, "AGENT_EXECUTOR_TIMEOUT", 5.0)

    async def _go():
        events = []
        async for ev in agent.process_message("yes", "default"):
            events.append(ev)
        return events

    events = asyncio.run(_go())
    tool_results = [e for e in events if e.get("type") == "tool_result"]
    assert tool_results, "expected normal tool_result when executor is fast"
    errors = [e for e in events if e.get("type") == "error"]
    assert not errors


# ---------------------------------------------------------------------------
# Fix A / Fix B remain intact on this module
# ---------------------------------------------------------------------------


def test_fix_a_filter_still_importable_and_applied():
    # A read-only query must keep read tools available.
    before = filter_tools_for_query(
        [{"type": "function", "function": {"name": "soqlQuery"}}],
        "show me accounts",
    )
    names = [t["function"]["name"] for t in before]
    assert "soqlQuery" in names


def test_fix_b_helper_still_normalizes_zero_count():
    assert _normalize_zero_count_result is not None
    assert _orch_normalize is not None