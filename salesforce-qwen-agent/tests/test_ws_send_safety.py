"""
Regression tests: WebSocket send safety (Issue 2).

Previously the websocket_chat handler had two UNGUARDED `await websocket.send_json`
calls (the agent-not-initialized error and the "previous request still processing"
busy-ack). If the client disconnected exactly then, Starlette raised
`RuntimeError: WebSocket is not connected. Need to call "accept" first.` which
escaped into the generic exception handler and logged a noisy traceback
("[WS] Connection error (...)").

These tests pin the fix: those sends now go through _ws_send_json, so a
disconnect/send race is swallowed instead of raised, while a connected socket
still receives the frame.
"""

import asyncio
import json
import os
import sys

from starlette.websockets import WebSocketState

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import WebSocketDisconnect

import app as app_module
from app import _ws_send_json


class _FakeWebSocket:
    """Minimal Starlette-like WebSocket double for websocket_chat."""

    def __init__(self, messages, application_state=WebSocketState.CONNECTED, fail_send=False):
        self._messages = list(messages)
        self.application_state = application_state
        self.fail_send = fail_send
        self.sent: list[dict] = []
        self.send_attempts = 0
        self.closed = False

    async def accept(self):
        pass

    async def receive_text(self):
        if self._messages:
            return self._messages.pop(0)
        raise WebSocketDisconnect(1000)

    async def send_json(self, payload):
        self.send_attempts += 1
        if self.fail_send:
            raise RuntimeError('WebSocket is not connected. Need to call "accept" first.')
        self.sent.append(payload)

    async def close(self):
        self.closed = True


class _StubSessionManager:
    def __init__(self, agent):
        self._agent = agent

    async def get_or_create_agent(self, session_id):  # noqa: ARG002
        return self._agent


def _clean_globals(monkeypatch):
    monkeypatch.setattr(app_module, "_session_busy", {})
    monkeypatch.setattr(app_module, "_recent_requests", {})
    monkeypatch.setattr(app_module, "session_files", {})


def _msg(content="hi"):  # noqa: ARG001
    return json.dumps({"type": "message", "content": content, "session_id": "s"})


# ─────────────────────────────────────────────────────────────
# _ws_send_json unit behavior
# ─────────────────────────────────────────────────────────────

def test_send_json_catches_runtime_error_despite_connected_state():
    # application_state still says CONNECTED (Starlette state does not flip
    # instantly), yet the underlying send raises RuntimeError — the exact race
    # that previously produced the noisy traceback.
    ws = _FakeWebSocket([], application_state=WebSocketState.CONNECTED, fail_send=True)
    ok = asyncio.run(_ws_send_json(ws, {"type": "ping"}))
    assert ok is False
    # No exception was raised (the old raw send would have propagated it).


def test_send_json_status_checked_before_send():
    ws = _FakeWebSocket([], application_state=WebSocketState.DISCONNECTED)
    ok = asyncio.run(_ws_send_json(ws, {"type": "ping"}))
    assert ok is False
    assert ws.sent == []


def test_send_json_succeeds_when_connected():
    ws = _FakeWebSocket([], application_state=WebSocketState.CONNECTED)
    ok = asyncio.run(_ws_send_json(ws, {"type": "ping"}))
    assert ok is True
    assert ws.sent == [{"type": "ping"}]


# ─────────────────────────────────────────────────────────────
# websocket_chat end-to-end (agent-not-initialized & busy branches)
# ─────────────────────────────────────────────────────────────

def test_agent_none_error_send_is_disconnect_safe(monkeypatch):
    _clean_globals(monkeypatch)
    monkeypatch.setattr(
        app_module, "session_manager", _StubSessionManager(None)
    )
    ws = _FakeWebSocket([_msg()], fail_send=True)
    # Previously this raised RuntimeError out of the handler; now it exits cleanly.
    asyncio.run(app_module.websocket_chat(ws, "s"))
    assert ws.send_attempts == 1, "the agent-not-initialized error frame was attempted"
    assert ws.sent == [], "nothing was delivered to a disconnected client"


def test_agent_none_error_still_delivered_when_connected(monkeypatch):
    _clean_globals(monkeypatch)
    monkeypatch.setattr(
        app_module, "session_manager", _StubSessionManager(None)
    )
    ws = _FakeWebSocket([_msg()], fail_send=False)
    asyncio.run(app_module.websocket_chat(ws, "s"))
    assert ws.sent and ws.sent[0]["type"] == "error"


def test_busy_ack_send_is_disconnect_safe(monkeypatch):
    _clean_globals(monkeypatch)
    monkeypatch.setattr(app_module, "_session_busy", {"s": True})
    monkeypatch.setattr(
        app_module, "session_manager", _StubSessionManager(object())
    )
    ws = _FakeWebSocket([_msg()], fail_send=True)
    asyncio.run(app_module.websocket_chat(ws, "s"))
    assert ws.send_attempts == 1, "the busy-ack frame was attempted"
    assert ws.sent == [], "nothing was delivered to a disconnected client"


def test_busy_ack_still_delivered_when_connected(monkeypatch):
    _clean_globals(monkeypatch)
    monkeypatch.setattr(app_module, "_session_busy", {"s": True})
    monkeypatch.setattr(
        app_module, "session_manager", _StubSessionManager(object())
    )
    ws = _FakeWebSocket([_msg()], fail_send=False)
    asyncio.run(app_module.websocket_chat(ws, "s"))
    assert ws.sent and ws.sent[0]["type"] == "progress"
    assert "previous request" in ws.sent[0]["data"]