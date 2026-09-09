"""
Regression tests: WebSocket receive/loop safety (Issue 2).

Previously the handler's outer loop called ``websocket.receive_text()``
directly. Once ANY prior send hit an OSError (client gone), Starlette flips
``application_state`` to DISCONNECTED and raises WebSocketDisconnect — which
the send wrapper swallows as a normal "[WS] Client disconnected during event
delivery". The NEXT loop iteration then hit
``RuntimeError: WebSocket is not connected. Need to call "accept" first.``
inside ``receive_text()``, escaped the WebSocketDisconnect handler into the
generic exception handler, and logged a noisy traceback for a normal
disconnect.

These tests pin the fix: the receive loop now goes through ``_ws_receive_text``,
treats a dead/closed socket or the state-guard RuntimeError as a normal
disconnect (None -> clean break), re-raises any OTHER RuntimeError, delivers
the final response exactly once, clears the busy flag and leaves no background
task behind.
"""

import asyncio
import json
import os
import sys

from starlette.websockets import WebSocketState

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import WebSocketDisconnect  # noqa: E402

import app as app_module  # noqa: E402
from app import _ws_receive_text  # noqa: E402


class _FakeWebSocket:
    """Minimal Starlette-like WebSocket double for websocket_chat.

    ``fail_send_after``: the N-th and later sends fail — mirroring Starlette,
    which flips ``application_state`` to DISCONNECTED and raises
    WebSocketDisconnect(1006) when a send hits an OSError.
    ``receive_error``: exception the receive raises ONCE the message queue is
    empty (the production state-guard RuntimeError / a disconnect / a bug).
    """

    def __init__(
        self,
        messages,
        application_state=WebSocketState.CONNECTED,
        fail_send_after=None,
        receive_error=None,
    ):
        self._messages = list(messages)
        self.application_state = application_state
        self.fail_send_after = fail_send_after
        self.receive_error = receive_error
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
            self.application_state = WebSocketState.DISCONNECTED
            raise WebSocketDisconnect(1000)
        if self.application_state != WebSocketState.CONNECTED:
            raise RuntimeError('WebSocket is not connected. Need to call "accept" first.')
        if self._messages:
            return self._messages.pop(0)
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


def _clean_globals(monkeypatch):
    monkeypatch.setattr(app_module, "_session_busy", {})
    monkeypatch.setattr(app_module, "_recent_requests", {})
    monkeypatch.setattr(app_module, "session_files", {})
    monkeypatch.setattr(app_module, "WS_HEARTBEAT_SECONDS", 0.05)


def _msg(content="hi"):
    return json.dumps({"type": "message", "content": content, "session_id": "s"})


def _run_handler(ws, session_id="s"):
    """Run websocket_chat to completion; return the remaining live tasks."""
    async def _go():
        await app_module.websocket_chat(ws, session_id)
        now = asyncio.current_task()
        return [t for t in asyncio.all_tasks() if t is not now and not t.done()]

    return asyncio.run(_go())


# ─────────────────────────────────────────────────────────────
# _ws_receive_text unit behavior
# ─────────────────────────────────────────────────────────────

def test_receive_returns_message_when_connected():
    ws = _FakeWebSocket(["hello"], application_state=WebSocketState.CONNECTED)
    raw = asyncio.run(_ws_receive_text(ws))
    assert raw == "hello"


def test_receive_none_when_already_disconnected():
    # The production race: state already flipped to DISCONNECTED by a failed
    # send. receive_text must NOT be called; None signals end-of-connection.
    ws = _FakeWebSocket([], application_state=WebSocketState.DISCONNECTED)
    assert asyncio.run(_ws_receive_text(ws)) is None
    assert ws._messages == [], "receive_text was not attempted on a dead socket"


def test_receive_none_on_websocket_disconnect():
    ws = _FakeWebSocket([], application_state=WebSocketState.CONNECTED,
                        receive_error="disconnect")
    assert asyncio.run(_ws_receive_text(ws)) is None


def test_receive_none_on_not_connected_runtime_error():
    # The EXACT production regression: receive_text raises
    # RuntimeError('WebSocket is not connected...') — must be a normal
    # disconnect, not a traceback.
    ws = _FakeWebSocket([], application_state=WebSocketState.CONNECTED,
                        receive_error="runtime-not-connected")
    assert asyncio.run(_ws_receive_text(ws)) is None


def test_receive_reraises_unrelated_runtime_error():
    # Only the state-guard RuntimeError is treated as a disconnect; a genuine
    # bug must not be swallowed blind.
    ws = _FakeWebSocket([], application_state=WebSocketState.CONNECTED,
                        receive_error="runtime-other")
    try:
        asyncio.run(_ws_receive_text(ws))
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("unrelated RuntimeError must propagate")


# ─────────────────────────────────────────────────────────────
# websocket_chat end-to-end
# ─────────────────────────────────────────────────────────────

def test_chat_clean_disconnect_after_final_response(monkeypatch):
    _clean_globals(monkeypatch)
    agent = _StreamingAgent([
        {"type": "tool_call", "data": {"name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Account"}}},
        {"type": "response", "data": "Two Accounts match."},
    ])
    monkeypatch.setattr(app_module, "session_manager", _StubSessionManager(agent))
    ws = _FakeWebSocket([_msg("Show Accounts.")])

    leftover = _run_handler(ws)

    assert leftover == [], "no background task may leak after the handler returns"
    assert agent.invocations == 1
    assert app_module._session_busy == {}, "busy flag must be cleared"
    responses = [p for p in ws.sent if p.get("type") == "response"]
    assert len(responses) == 1, "final response delivered exactly once"
    assert responses[0]["data"] == "Two Accounts match."
    assert ws.closed


def test_chat_disconnect_during_processing_no_runtimeerror(monkeypatch):
    # Send fails mid-stream (client gone -> Starlette flips state to
    # DISCONNECTED). The producer stops; the next receive must end cleanly
    # instead of raising the state-guard RuntimeError.
    _clean_globals(monkeypatch)
    agent = _StreamingAgent([
        {"type": "tool_call", "data": {"name": "createSobjectRecord", "arguments": {}}},
        {"type": "response", "data": "Please provide the required fields."},
    ])
    monkeypatch.setattr(app_module, "session_manager", _StubSessionManager(agent))
    ws = _FakeWebSocket([_msg("create a lead")], fail_send_after=0)

    leftover = _run_handler(ws)

    assert leftover == [], "no background task may leak"
    assert app_module._session_busy == {}, "busy flag must be cleared"
    assert ws.send_attempts >= 1, "delivery was attempted before the disconnect"


def test_chat_receive_runtime_error_after_delivery_clean(monkeypatch):
    # Response was fully delivered, then the client disappears and the next
    # receive_text raises the state-guard RuntimeError — previously a traceback.
    _clean_globals(monkeypatch)
    agent = _StreamingAgent([
        {"type": "response", "data": "Two Accounts match."},
    ])
    monkeypatch.setattr(app_module, "session_manager", _StubSessionManager(agent))
    ws = _FakeWebSocket([_msg("Show Accounts.")], receive_error="runtime-not-connected")

    leftover = _run_handler(ws)

    assert leftover == [], "no background task may leak"
    assert app_module._session_busy == {}, "busy flag must be cleared"
    responses = [p for p in ws.sent if p.get("type") == "response"]
    assert len(responses) == 1, "final response delivered exactly once, no duplicate"


def test_chat_multiple_messages_same_socket_no_duplicate(monkeypatch):
    # A connected client sends two messages: both are processed on the SAME
    # socket, each response is delivered once, and only exhaustion of the queue
    # ends the connection.
    _clean_globals(monkeypatch)
    agent = _StreamingAgent([{"type": "response", "data": "ok."}])
    monkeypatch.setattr(app_module, "session_manager", _StubSessionManager(agent))
    ws = _FakeWebSocket([_msg("one"), _msg("two")])

    leftover = _run_handler(ws)

    assert leftover == []
    assert agent.invocations == 2, "both messages processed on the same socket"
    responses = [p for p in ws.sent if p.get("type") == "response"]
    assert len(responses) == 2, "each message produced exactly one final response"
    assert app_module._session_busy == {}


def test_chat_unrelated_runtime_error_does_not_crash_handler(monkeypatch):
    # A genuine bug surfaced through receive is logged as a connection error
    # and the handler still returns; it is NOT converted into a user error.
    _clean_globals(monkeypatch)
    agent = _StreamingAgent([{"type": "response", "data": "ok."}])
    monkeypatch.setattr(app_module, "session_manager", _StubSessionManager(agent))
    ws = _FakeWebSocket([_msg("hi")], receive_error="runtime-other")

    leftover = _run_handler(ws)

    assert leftover == []
    assert app_module._session_busy == {}


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))