"""
WebSocket idle-keepalive tests.

The turn-scoped heartbeat only runs while a Qwen request is active; when a
connection is idle it otherwise carries zero server->client traffic, so an edge
or proxy idle timeout can silently close the socket. ``_idle_keepalive`` pushes
lightweight ``ping`` frames while idle (skipping active turns) so idle sockets
stay alive. These tests exercise the helper in isolation and end-to-end through
the real ``websocket_chat`` handler, without a live Salesforce org or LLM.
"""

import asyncio
import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402


class _FakeWS:
    """Minimal stand-in for the real WebSocket receive/send contract."""

    def __init__(self, state):
        self.application_state = state
        self.sent = []

    async def send_json(self, payload):
        self.sent.append(payload)


class _SlowAgent:
    """Fake agent whose process_message is silent for `silence_s` then responds."""

    def __init__(self, silence_s: float, events):
        self._silence = silence_s
        self._events = events

    def clear_session(self, session_id):  # noqa: ARG002
        pass

    async def process_message(self, user_message, session_id):  # noqa: ARG002
        await asyncio.sleep(self._silence)
        for ev in self._events:
            yield ev


def _fake_session_manager_returning(agent):
    class _SM:
        async def get_or_create_agent(self, session_id):  # noqa: ARG002
            return agent

    return _SM()


def _make_app(monkeypatch, silence_s: float = 0.5, heartbeat_s: float = 0.2,
              keepalive_s: float = 0.1):
    """Build a fresh FastAPI app that reuses the real websocket_chat handler."""
    monkeypatch.setattr(app_module, "WS_HEARTBEAT_SECONDS", heartbeat_s)
    monkeypatch.setattr(app_module, "WS_KEEPALIVE_SECONDS", keepalive_s)
    monkeypatch.setattr(app_module, "_session_busy", {})

    agent = _SlowAgent(
        silence_s,
        [
            {"type": "tool_call", "data": {"name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Contact LIMIT 10"}}},
            {"type": "tool_result", "data": {"name": "soqlQuery", "result": "[]"}},
            {"type": "response", "data": "Done."},
        ],
    )
    monkeypatch.setattr(app_module, "session_manager", _fake_session_manager_returning(agent))

    test_app = FastAPI()
    test_app.websocket("/ws/{session_id}")(app_module.websocket_chat)
    return test_app


def _run_briefly(coro, seconds: float):
    """Run a coroutine for `seconds`, then cancel it (isolated loop per test)."""
    async def _run():
        task = asyncio.create_task(coro)
        await asyncio.sleep(seconds)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())


def test_idle_keepalive_emits_pings(monkeypatch):
    monkeypatch.setattr(app_module, "WS_KEEPALIVE_SECONDS", 0.01)
    monkeypatch.setattr(app_module, "_session_busy", {})
    ws = _FakeWS(app_module.WebSocketState.CONNECTED)

    _run_briefly(app_module._idle_keepalive(ws, "sess"), seconds=0.06)

    assert len(ws.sent) >= 2, ws.sent
    assert all(frame["type"] == "ping" for frame in ws.sent)
    assert all("ts" in frame for frame in ws.sent)


def test_idle_keepalive_skips_pings_while_busy(monkeypatch):
    monkeypatch.setattr(app_module, "WS_KEEPALIVE_SECONDS", 0.01)
    monkeypatch.setattr(app_module, "_session_busy", {"sess": True})
    ws = _FakeWS(app_module.WebSocketState.CONNECTED)

    _run_briefly(app_module._idle_keepalive(ws, "sess"), seconds=0.06)

    assert ws.sent == [], "no pings may be sent while a turn is in progress"


def test_idle_keepalive_stops_when_socket_disconnected(monkeypatch):
    monkeypatch.setattr(app_module, "WS_KEEPALIVE_SECONDS", 0.01)
    monkeypatch.setattr(app_module, "_session_busy", {})
    ws = _FakeWS(app_module.WebSocketState.DISCONNECTED)

    async def _run():
        await app_module._idle_keepalive(ws, "sess")

    asyncio.run(_run())
    assert ws.sent == []


def test_idle_keepalive_stops_after_failed_send(monkeypatch):
    monkeypatch.setattr(app_module, "WS_KEEPALIVE_SECONDS", 0.01)
    monkeypatch.setattr(app_module, "_session_busy", {})

    class _DroppedWS(_FakeWS):
        async def send_json(self, payload):  # noqa: ARG002
            self.sent.append(payload)
            raise RuntimeError("send failed")

    ws = _DroppedWS(app_module.WebSocketState.CONNECTED)

    async def _run():
        await app_module._idle_keepalive(ws, "sess")

    asyncio.run(_run())  # must terminate, not spin forever
    assert len(ws.sent) == 1


def test_idle_socket_receives_keepalive_pings_end_to_end(monkeypatch):
    test_app = _make_app(monkeypatch, keepalive_s=0.2)
    with TestClient(test_app) as client:
        with client.websocket_connect("/ws/idle_1") as ws:
            # No message is sent; the socket is idle. It must still receive
            # server ping frames instead of sitting silent.
            got_ping = False
            for _ in range(30):
                raw = ws.receive_text()
                if json.loads(raw).get("type") == "ping":
                    got_ping = True
                    break
            assert got_ping


def test_no_keepalive_ping_during_active_turn(monkeypatch):
    test_app = _make_app(monkeypatch, silence_s=2.0, heartbeat_s=0.1, keepalive_s=0.05)
    with TestClient(test_app) as client:
        with client.websocket_connect("/ws/busy_1") as ws:
            ws.send_text(json.dumps({
                "type": "message",
                "content": "run",
                "session_id": "busy_1",
            }))

            types = []
            for _ in range(120):
                raw = ws.receive_text()
                types.append(json.loads(raw).get("type"))
                if types[-1] in ("response", "error"):
                    break

    assert "response" in types
    assert "ping" not in types, f"keepalive ping leaked into active turn: {types}"