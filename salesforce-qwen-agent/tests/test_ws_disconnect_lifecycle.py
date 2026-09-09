"""
Regression tests: WebSocket disconnect lifecycle (root-cause fix).

The recurring production symptom is the browser banner
"Disconnected — Reconnecting...". Auditing the lifecycle found that the
SERVER itself was force-closing still-healthy sockets:

1. Any unexpected exception inside ``websocket_chat`` (a malformed JSON frame
   from ``json.loads``, or a transient ``session_manager.get_or_create_agent``
   failure) escaped into the generic exception handler, and the ``finally``
   block then called ``await websocket.close()`` on a still-CONNECTED socket —
   pushing a close frame to the browser and producing a spurious disconnect
   even though neither client nor network had a problem.

2. During a long in-flight turn, the wait loop only polled producer/heartbeat
   and, on a failed send (Starlette flips state to DISCONNECTED), waited for
   the whole minute(s)-long turn to finish instead of aborting immediately,
   and the busy flag was not released until the very end.

These tests pin the fix:

- unexpected exceptions no longer tear down a connected socket;
- malformed frames and agent-init failures are surfaced and the connection
  keeps running;
- a failed send aborts the in-flight turn, cancels its tasks, releases the
  busy flag and sends nothing after the disconnect;
- the handler only closes a socket the client already abandoned.
"""

import asyncio
import json
import os
import sys

from starlette.websockets import WebSocketState

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import WebSocketDisconnect  # noqa: E402

import app as app_module  # noqa: E402


class _FakeWebSocket:
    """Starlette-like double for websocket_chat.

    ``flip_on_disconnect=True`` mirrors Starlette: when `receive_text`
    terminates in a WebSocketDisconnect the application_state flips to
    DISCONNECTED. With ``flip_on_disconnect=False`` the socket stays
    CONNECTED so the test can assert the handler does NOT close it.
    """

    def __init__(
        self,
        messages,
        application_state=WebSocketState.CONNECTED,
        fail_send_after=None,
        receive_error=None,
        flip_on_disconnect=True,
    ):
        self._messages = list(messages)
        self.application_state = application_state
        self.fail_send_after = fail_send_after
        self.receive_error = receive_error
        self.flip_on_disconnect = flip_on_disconnect
        self.sent: list[dict] = []
        self.send_attempts = 0
        self.closed = False

    async def accept(self):
        pass

    async def receive_text(self):
        if self.receive_error == "runtime-not-connected" and not self._messages:
            self.application_state = WebSocketState.DISCONNECTED
            raise RuntimeError('WebSocket is not connected. Need to call "accept" first.')
        if self.receive_error == "runtime-other" and not self._messages:
            raise RuntimeError("boom")
        if self.receive_error == "disconnect" and not self._messages:
            if self.flip_on_disconnect:
                self.application_state = WebSocketState.DISCONNECTED
            raise WebSocketDisconnect(1000)
        if self.application_state != WebSocketState.CONNECTED:
            raise RuntimeError('WebSocket is not connected. Need to call "accept" first.')
        if self._messages:
            return self._messages.pop(0)
        if self.flip_on_disconnect:
            self.application_state = WebSocketState.DISCONNECTED
        raise WebSocketDisconnect(1000)

    async def send_json(self, payload):
        self.send_attempts += 1
        if self.fail_send_after is not None and self.send_attempts > self.fail_send_after:
            self.application_state = WebSocketState.DISCONNECTED
            raise WebSocketDisconnect(1006)
        self.sent.append(payload)

    async def close(self):
        self.closed = True


class _StreamingAgent:
    """Fake agent that streams events; counts invocations."""

    def __init__(self, events):
        self.events = events
        self.invocations = 0

    def clear_session(self, session_id):  # noqa: ARG002
        pass

    async def process_message(self, user_message, session_id):  # noqa: ARG002
        self.invocations += 1
        for ev in self.events:
            yield ev


class _StubSessionManager:
    def __init__(self, agent):
        self._agent = agent

    async def get_or_create_agent(self, session_id):  # noqa: ARG002
        return self._agent


class _FailingThenStubSessionManager:
    """Agent init fails on the first call, succeeds afterwards (cold instance)."""

    def __init__(self, agent, error: Exception = RuntimeError("agent init boom")):
        self._agent = agent
        self._error = error
        self.calls = 0

    async def get_or_create_agent(self, session_id):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            raise self._error
        return self._agent


def _clean_globals(monkeypatch):
    monkeypatch.setattr(app_module, "_session_busy", {})
    monkeypatch.setattr(app_module, "_recent_requests", {})
    monkeypatch.setattr(app_module, "session_files", {})
    monkeypatch.setattr(app_module, "WS_HEARTBEAT_SECONDS", 0.05)


def _msg(content="hi"):
    return json.dumps({"type": "message", "content": content, "session_id": "s"})


def _run_handler(ws, session_id="s"):
    """Run websocket_chat to completion; return leftover live tasks."""

    async def _go():
        await app_module.websocket_chat(ws, session_id)
        now = asyncio.current_task()
        return [t for t in asyncio.all_tasks() if t is not now and not t.done()]

    return asyncio.run(_go())


# ─────────────────────────────────────────────────────────────
# Root cause: unexpected inputs/exceptions must NOT close a
# still-connected socket (that caused spurious disconnects).
# ─────────────────────────────────────────────────────────────

def test_malformed_frame_does_not_kill_connection(monkeypatch):
    _clean_globals(monkeypatch)
    agent = _StreamingAgent([{"type": "response", "data": "Fine."}])
    monkeypatch.setattr(app_module, "session_manager", _StubSessionManager(agent))
    ws = _FakeWebSocket(["this is not json", _msg("hi")], flip_on_disconnect=False)

    leftover = _run_handler(ws)

    assert leftover == [], "no background task may leak"
    assert app_module._session_busy == {}, "busy flag must be cleared"
    errors = [p for p in ws.sent if p.get("type") == "error"]
    assert len(errors) == 1, "malformed frame surfaced as an error, not a teardown"
    responses = [p for p in ws.sent if p.get("type") == "response"]
    assert len(responses) == 1, "the next valid message was still processed"
    assert not ws.closed, "a still-connected socket must never be closed by the server"


def test_agent_init_failure_does_not_kill_connection(monkeypatch):
    _clean_globals(monkeypatch)
    agent = _StreamingAgent([{"type": "response", "data": "Good to go."}])
    sm = _FailingThenStubSessionManager(agent)
    monkeypatch.setattr(app_module, "session_manager", sm)
    ws = _FakeWebSocket([_msg("one"), _msg("two")], flip_on_disconnect=False)

    leftover = _run_handler(ws)

    assert leftover == []
    assert sm.calls == 2, "second message retried agent init on the SAME socket"
    errors = [p for p in ws.sent if p.get("type") == "error"]
    assert len(errors) == 1, "init failure surfaced as an error, not a teardown"
    responses = [p for p in ws.sent if p.get("type") == "response"]
    assert len(responses) == 1, "the retried message was processed"
    assert not ws.closed, "a still-connected socket must never be closed by the server"


def test_unrelated_error_never_closes_connected_socket(monkeypatch):
    # The OLD behavior: any exception escaping the loop hit the generic
    # handler and the finally block force-closed a CONNECTED socket -> browser
    # showed "Disconnected — Reconnecting..." with server at fault.
    _clean_globals(monkeypatch)
    agent = _StreamingAgent([{"type": "response", "data": "ok."}])
    monkeypatch.setattr(app_module, "session_manager", _StubSessionManager(agent))
    ws = _FakeWebSocket([_msg("hi")], receive_error="runtime-other", flip_on_disconnect=False)

    leftover = _run_handler(ws)

    assert leftover == []
    assert app_module._session_busy == {}
    assert not ws.closed, "handler-level error must not close a healthy socket"


# ─────────────────────────────────────────────────────────────
# Lifecycle: a failed send aborts the turn, cleans up, and sends
# nothing afterwards.
# ─────────────────────────────────────────────────────────────

def test_failed_send_aborts_in_flight_turn_and_cleans_up(monkeypatch):
    _clean_globals(monkeypatch)
    agent = _StreamingAgent([{"type": "response", "data": "Late."}])
    monkeypatch.setattr(app_module, "session_manager", _StubSessionManager(agent))
    ws = _FakeWebSocket([_msg("hello")], fail_send_after=0)

    leftover = _run_handler(ws)

    assert leftover == [], "producer/heartbeat tasks were cancelled and reaped"
    assert app_module._session_busy == {}, "busy flag released immediately"
    assert ws.closed, "abandoned socket is closed as part of cleanup"
    # Every attempt before the disconnect was refused: nothing after disconnect.
    for p in ws.sent:
        assert p.get("type") != "idle", "no cosmetic idle frame after disconnect"
    assert ws.send_attempts == 1, "only the first (failed) send was attempted"


# ─────────────────────────────────────────────────────────────
# Clean shutdown bookkeeping already pinned by test_ws_receive_safety:
# leftover==[], busy=={}, response delivered once, socket closed when the
# client is gone.
# ─────────────────────────────────────────────────────────────

def test_clean_disconnect_still_closes_abandoned_socket(monkeypatch):
    _clean_globals(monkeypatch)
    agent = _StreamingAgent([{"type": "response", "data": "Two Accounts match."}])
    monkeypatch.setattr(app_module, "session_manager", _StubSessionManager(agent))
    ws = _FakeWebSocket([_msg("Show Accounts.")])

    leftover = _run_handler(ws)

    assert leftover == []
    assert app_module._session_busy == {}
    responses = [p for p in ws.sent if p.get("type") == "response"]
    assert len(responses) == 1
    assert responses[0]["data"] == "Two Accounts match."
    assert ws.closed, "an ABANDONED socket is closed for cleanup"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))