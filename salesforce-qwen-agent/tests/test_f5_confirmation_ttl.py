"""
Focused offline tests for F5: pending-confirmation TTL + exact action binding.

F5 roots:
- Pending destructive-action confirmations had no TTL: a stale "yes"/"ok"
  could execute an old delete hours later.
- Same-turn multiple destructive tool calls could overwrite the first pending
  action (X creates pending X, Y overwrites with Y, user is asked to confirm X,
  then Y could execute).
- Pending confirmations were never cleared on expiry or on logout.

Implemented:
- agent/planner.py: PENDING_CONFIRMATION_TTL (env, default 300s) with
  time.monotonic(); pending records carry created_at; process_confirmation
  returns a distinct {"status":"expired"} result instead of a confirmed action;
  check_tool_safety never overwrites an existing pending action (first action
  stays bound).
- agent/agent.py + agent/multi_agent.py: expired branch handled BEFORE the
  truthy-confirmed branch -> controlled response, executor never invoked.
- sfmcp/session_manager.py: logout clears the session agent's pending
  confirmation.

Invariants preserved:
- Unexpired confirm still returns the EXACT original action.
- Decline still clears the pending action and returns None.
- Cross-session isolation unchanged.
- Fix A/B/D1/P0/E5 behavior untouched.

All tests use mocks/fakes - no live LLM / Salesforce / MCP / mutations.
"""

import asyncio
import os
import sys
import time
import importlib
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import planner as planner_mod  # noqa: E402
from agent.planner import TaskPlanner, PENDING_CONFIRMATION_TTL  # noqa: E402
from agent.agent import SalesforceAgent  # noqa: E402
from agent.multi_agent import Orchestrator  # noqa: E402

DELETE_ARGS = {"sobject-name": "Account", "id": "001X"}
DELETE_ARGS_Y = {"sobject-name": "Account", "id": "001Y"}
DELETE_ARGS_Z = {"sobject-name": "Account", "id": "001Z"}


class _Exec:
    """Never performs real Salesforce operations; records what it is asked to do."""

    def __init__(self, result='{"totalSize": 1, "records": []}'):
        self.result = result
        self.executed = []

    async def execute(self, name, arguments):
        self.executed.append((name, arguments))
        return self.result


def _run(agent, message="yes"):
    async def _go():
        events = []
        async for ev in agent.process_message(message, "default"):
            events.append(ev)
        return events

    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# A. created_at is stored
# ---------------------------------------------------------------------------

def test_pending_confirmation_stores_created_at():
    p = TaskPlanner()
    p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "sA")
    pending = p.get_pending_confirmation("sA")
    assert pending is not None
    assert "created_at" in pending
    assert isinstance(pending["created_at"], float)


# ---------------------------------------------------------------------------
# B. Unexpired confirmation returns the EXACT original action
# ---------------------------------------------------------------------------

def test_unexpired_confirmation_returns_exact_original_action():
    p = TaskPlanner()
    p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "sA")
    action = p.process_confirmation("yes", "sA")
    assert action is not None
    assert not action.get("status") == "expired"
    assert action["tool_name"] == "deleteSobjectRecord"
    assert action["arguments"] == DELETE_ARGS
    assert not p.has_pending_confirmation("sA")


# ---------------------------------------------------------------------------
# C. Expired confirmation: cleared, distinct expired result, never confirmed
# ---------------------------------------------------------------------------

def test_expired_confirmation_cleared_and_distinct():
    p = TaskPlanner()
    p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "sA")
    p._pending_confirmations["sA"]["created_at"] = time.monotonic() - 9999.0

    result = p.process_confirmation("yes", "sA")
    assert result is not None
    assert result.get("status") == "expired"
    assert "expired" in result.get("message", "").lower()
    assert not p.has_pending_confirmation("sA")

    # A real (non-expired) lookup after expiry must NOT return the old action.
    assert p.process_confirmation("yes", "sA") is None


# ---------------------------------------------------------------------------
# D. Expired "yes" NEVER reaches the executor (agent level)
# ---------------------------------------------------------------------------

def test_expired_yes_never_reaches_executor_agent_level():
    exec_ = _Exec()
    agent = SalesforceAgent(llm=object(), executor=exec_, max_iterations=5)
    agent.planner._pending_confirmations["default"] = {
        "tool_name": "deleteSobjectRecord",
        "arguments": dict(DELETE_ARGS),
        "type": "delete",
        "created_at": time.monotonic() - 9999.0,
        "confirmation_message": "Delete Account?",
    }

    events = _run(agent, message="yes")

    responses = [e for e in events if e.get("type") == "response"]
    assert responses, "expected a controlled response for expired confirmation"
    assert "expired" in responses[0]["data"].lower()
    assert not exec_.executed, "executor must never be called for an expired confirmation"
    assert not [e for e in events if e.get("type") == "tool_call"]


# ---------------------------------------------------------------------------
# E. Expired "ok"/other affirmative response also cannot execute
# ---------------------------------------------------------------------------

def test_expired_ok_also_cannot_execute_planner_level():
    p = TaskPlanner()
    p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "sA")
    p._pending_confirmations["sA"]["created_at"] = time.monotonic() - 9999.0
    result = p.process_confirmation("ok", "sA")
    assert result is not None and result.get("status") == "expired"
    assert not p.has_pending_confirmation("sA")


def test_expired_ok_never_executes_agent_level():
    exec_ = _Exec()
    agent = SalesforceAgent(llm=object(), executor=exec_, max_iterations=5)
    agent.planner._pending_confirmations["default"] = {
        "tool_name": "deleteSobjectRecord",
        "arguments": dict(DELETE_ARGS),
        "type": "delete",
        "created_at": time.monotonic() - 9999.0,
        "confirmation_message": "Delete Account?",
    }

    events = _run(agent, message="ok")

    responses = [e for e in events if e.get("type") == "response"]
    assert responses and "expired" in responses[0]["data"].lower()
    assert not exec_.executed


# ---------------------------------------------------------------------------
# F. Decline still clears pending and preserves existing behavior
# ---------------------------------------------------------------------------

def test_decline_clears_pending():
    p = TaskPlanner()
    p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "sA")
    result = p.process_confirmation("no", "sA")
    assert result is None
    assert not p.has_pending_confirmation("sA")


def test_decline_empty_pending_returns_none():
    p = TaskPlanner()
    assert p.process_confirmation("no", "sA") is None


def test_decline_agent_level_still_cancels():
    exec_ = _Exec()
    agent = SalesforceAgent(llm=object(), executor=exec_, max_iterations=5)
    agent.planner.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "default")

    events = _run(agent, message="no")

    responses = [e for e in events if e.get("type") == "response"]
    assert responses and "cancelled" in responses[0]["data"].lower()
    assert not exec_.executed
    assert not agent.planner.has_pending_confirmation("default")


# ---------------------------------------------------------------------------
# G. Cross-session isolation
# ---------------------------------------------------------------------------

def test_cross_session_isolation():
    p = TaskPlanner()
    p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "sA")

    assert not p.has_pending_confirmation("sB")
    # Session B confirming nothing must not touch A's pending action.
    assert p.process_confirmation("yes", "sB") is None
    # A's action is still pending and bound to A.
    assert p.has_pending_confirmation("sA")
    assert p.get_pending_confirmation("sA")["arguments"] == DELETE_ARGS

    # Session A can still confirm its own action normally.
    action = p.process_confirmation("yes", "sA")
    assert action is not None and action["arguments"] == DELETE_ARGS


# ---------------------------------------------------------------------------
# H. Same-turn multiple destructive actions: first stays bound
# ---------------------------------------------------------------------------

def test_same_turn_multiple_destructive_first_stays_bound():
    p = TaskPlanner()
    safety_x = p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "sA")
    safety_y = p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS_Y, "sA")

    # Second call must not replace the first pending action.
    pending = p.get_pending_confirmation("sA")
    assert pending["tool_name"] == "deleteSobjectRecord"
    assert pending["arguments"] == DELETE_ARGS
    assert safety_y["pending_action"]["arguments"] == DELETE_ARGS

    # Confirmation remains bound to X; Y is never substituted.
    action = p.process_confirmation("yes", "sA")
    assert action is not None
    assert action["arguments"] == DELETE_ARGS


# ---------------------------------------------------------------------------
# I. At-most-once pending behavior (repeated checks do not replace)
# ---------------------------------------------------------------------------

def test_at_most_once_pending_behavior():
    p = TaskPlanner()
    p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "sA")
    p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS_Y, "sA")
    p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS_Z, "sA")

    pending = p.get_pending_confirmation("sA")
    assert pending["arguments"] == DELETE_ARGS

    action = p.process_confirmation("yes", "sA")
    assert action is not None and action["arguments"] == DELETE_ARGS


# ---------------------------------------------------------------------------
# J. TTL configuration: default 300s + env override
# ---------------------------------------------------------------------------

def test_ttl_default_is_300_seconds(monkeypatch):
    monkeypatch.delenv("PENDING_CONFIRMATION_TTL", raising=False)
    reloaded = importlib.reload(planner_mod)
    assert reloaded.PENDING_CONFIRMATION_TTL == 300.0


def test_ttl_environment_override_works(monkeypatch):
    monkeypatch.setenv("PENDING_CONFIRMATION_TTL", "10")
    reloaded = importlib.reload(planner_mod)
    assert reloaded.PENDING_CONFIRMATION_TTL == 10.0
    monkeypatch.delenv("PENDING_CONFIRMATION_TTL", raising=False)
    importlib.reload(planner_mod)


def test_ttl_global_constant_default():
    assert PENDING_CONFIRMATION_TTL == 300.0


# ---------------------------------------------------------------------------
# K. TTL expiry uses monotonic time (patching time.monotonic, no sleeping)
# ---------------------------------------------------------------------------

def test_expiry_uses_monotonic_clock():
    clock = {"t": 1000.0}

    def fake_monotonic():
        return clock["t"]

    with patch.object(planner_mod.time, "monotonic", side_effect=fake_monotonic):
        p = TaskPlanner()
        p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "sA")
        # Advance the monotonic clock past the TTL.
        clock["t"] += 9999.0
        result = p.process_confirmation("yes", "sA")
    assert result is not None and result.get("status") == "expired"


def test_confirmation_below_ttl_is_valid_on_monotonic_clock():
    clock = {"t": 5000.0}

    def fake_monotonic():
        return clock["t"]

    with patch.object(planner_mod.time, "monotonic", side_effect=fake_monotonic):
        p = TaskPlanner()
        p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "sA")
        clock["t"] += 1.0  # still far below 300s TTL
        action = p.process_confirmation("yes", "sA")
    assert action is not None
    assert action["arguments"] == DELETE_ARGS


def test_ttl_zero_expires_immediately(monkeypatch):
    monkeypatch.setattr(planner_mod, "PENDING_CONFIRMATION_TTL", 0.0)
    p = TaskPlanner()
    p.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "sA")
    result = p.process_confirmation("yes", "sA")
    assert result is not None and result.get("status") == "expired"
    assert not p.has_pending_confirmation("sA")


# ---------------------------------------------------------------------------
# L. Missing/legacy pending without created_at does not crash
# ---------------------------------------------------------------------------

def test_legacy_pending_without_created_at_does_not_crash():
    p = TaskPlanner()
    p._pending_confirmations["sA"] = {
        "tool_name": "deleteSobjectRecord",
        "arguments": dict(DELETE_ARGS),
        "type": "delete",
    }
    # Backward-compatible: missing created_at is treated as not-yet-expired,
    # so the flow neither crashes nor silently refuses a valid confirmation.
    action = p.process_confirmation("yes", "sA")
    assert action is not None
    assert action["arguments"] == DELETE_ARGS
    assert not p.has_pending_confirmation("sA")


# ---------------------------------------------------------------------------
# M. Logout clears the pending confirmation (safer re-login)
# ---------------------------------------------------------------------------

def test_logout_clears_pending_confirmation():
    from unittest.mock import AsyncMock, MagicMock
    from sfmcp.session_manager import UserSessionManager

    mgr = UserSessionManager()
    mgr.initialize_defaults(
        default_mcp_client=MagicMock(),
        default_tool_registry=MagicMock(),
        default_executor=MagicMock(),
        default_agent=MagicMock(),
        llm=MagicMock(),
    )

    async def run():
        with patch("sfmcp.session_manager.SalesforceMCPClient") as mock_client_cls:
            mock_inst = AsyncMock()
            mock_client_cls.return_value = mock_inst
            await mgr.register_oauth_session(
                session_id="f5_sid",
                access_token="tok",
                refresh_token="ref",
                instance_url="https://x.my.salesforce.com",
                user_info={"display_name": "F5", "username": "u@x.com", "authenticated": True},
            )
            session = mgr._sessions["f5_sid"]
            assert "agent" in session
            user_agent = session["agent"]
            # Place an orphaned pending destructive confirmation on the planner.
            user_agent.planner.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "f5_sid")
            assert user_agent.planner.has_pending_confirmation("f5_sid")

            logged_out = await mgr.logout_session("f5_sid")
            assert logged_out is True
            assert not user_agent.planner.has_pending_confirmation("f5_sid")
            # A later confirmation cannot execute the orphaned action.
            assert user_agent.planner.process_confirmation("yes", "f5_sid") is None

    asyncio.run(run())


# ---------------------------------------------------------------------------
# N. Agent-level expired confirmation: controlled response, executor not called
# ---------------------------------------------------------------------------

def test_agent_level_expired_yields_controlled_response_without_execution():
    exec_ = _Exec()
    agent = SalesforceAgent(llm=object(), executor=exec_, max_iterations=5)
    agent.planner._pending_confirmations["default"] = {
        "tool_name": "deleteSobjectRecord",
        "arguments": dict(DELETE_ARGS),
        "type": "delete",
        "created_at": time.monotonic() - 9999.0,
        "confirmation_message": "Delete Account?",
    }

    events = _run(agent, message="yes")

    assert not exec_.executed
    actions = [e for e in events if e.get("type") == "tool_call"]
    assert not actions
    errors = [e for e in events if e.get("type") == "error"]
    assert not errors, "expired confirmation is a controlled response, not an error"
    responses = [e for e in events if e.get("type") == "response"]
    assert responses, "expected a controlled response for expired confirmation"
    assert "expired" in responses[0]["data"].lower()
    assert "No action was executed" in responses[0]["data"]


def test_agent_level_unexpired_confirmation_still_executes():
    exec_ = _Exec()
    agent = SalesforceAgent(llm=object(), executor=exec_, max_iterations=5)
    agent.planner.check_tool_safety("deleteSobjectRecord", DELETE_ARGS, "default")

    # Unexpired confirmation flow must keep working (existing behavior).
    action = agent.planner.process_confirmation("yes", "default")
    assert action is not None
    assert action["arguments"] == DELETE_ARGS
    assert not agent.planner.has_pending_confirmation("default")


# ---------------------------------------------------------------------------
# O. Orchestrator-level expired confirmation: controlled response, no execution
# ---------------------------------------------------------------------------

class _ExpiredSafetyPlanner:
    def has_pending_confirmation(self, session_id="default"):
        return True

    def process_confirmation(self, user_response, session_id="default"):
        return {
            "status": "expired",
            "tool_name": "deleteSobjectRecord",
            "message": "This confirmation has expired. No action was executed.",
        }


class _QuietLLM:
    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        return {"content": "", "tool_calls": [], "finish_reason": "stop"}

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        return "done"


def test_orchestrator_level_expired_yields_controlled_response_without_execution():
    exec_ = _Exec()
    orch = Orchestrator(llm=_QuietLLM(), executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _ExpiredSafetyPlanner()

    events = _run(orch, message="yes")

    assert not exec_.executed
    assert not [e for e in events if e.get("type") == "tool_call"]
    responses = [e for e in events if e.get("type") == "response"]
    assert responses, "expected a controlled response for expired confirmation"
    assert "expired" in responses[0]["data"].lower()
    assert "No action was executed" in responses[0]["data"]