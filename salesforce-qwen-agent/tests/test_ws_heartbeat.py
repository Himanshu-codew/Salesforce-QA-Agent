"""
WebSocket keep-alive / heartbeat tests.

These verify the application-level heartbeat mechanism that keeps a WebSocket
alive during a long (blocking) Qwen request, without requiring a live Salesforce
org or a real LLM. A fake agent's ``process_message`` deliberately emits no
events for a while (mimicking a slow Qwen call), and the test asserts the same
socket still delivers a ``progress`` heartbeat and then the final ``response``.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402


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


def _make_app(monkeypatch, silence_s: float = 2.0, heartbeat_s: float = 0.2):
    """Build a fresh FastAPI app that reuses the real websocket_chat handler."""
    monkeypatch.setattr(app_module, "WS_HEARTBEAT_SECONDS", heartbeat_s)
    monkeypatch.setattr(app_module, "_session_busy", {})

    agent = _SlowAgent(
        silence_s,
        [
            {"type": "tool_call", "data": {"name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Contact LIMIT 10"}}},
            {"type": "tool_result", "data": {"name": "soqlQuery", "result": "[]"}},
            {"type": "response", "data": "Two Contacts match."},
        ],
    )
    monkeypatch.setattr(app_module, "session_manager", _fake_session_manager_returning(agent))

    test_app = FastAPI()
    test_app.websocket("/ws/{session_id}")(app_module.websocket_chat)
    return test_app, agent


def test_heartbeat_keeps_socket_alive_during_slow_request(monkeypatch):
    test_app, _ = _make_app(monkeypatch, silence_s=2.0, heartbeat_s=0.2)
    with TestClient(test_app) as client:
        with client.websocket_connect("/ws/slow_1") as ws:
            ws.send_text(json.dumps({"type": "message", "content": "Show my Contacts.", "session_id": "slow_1"}))

            types = []
            saw_heartbeat = False
            saw_response = False
            saw_idle = False
            # Bound the read so a leak never hangs the suite.
            for _ in range(60):
                raw = ws.receive_text()
                data = json.loads(raw)
                types.append(data.get("type"))
                if data.get("type") == "progress":
                    saw_heartbeat = True
                if data.get("type") == "response":
                    saw_response = True
                if data.get("type") == "idle":
                    saw_idle = True
                    break
            assert saw_heartbeat, f"expected a progress heartbeat frame, got {types}"
            assert saw_response, f"expected a final response on the SAME socket, got {types}"
            assert saw_idle, f"expected idle completion marker, got {types}"


def test_short_request_sends_heartbeat_and_response_on_same_socket(monkeypatch):
    test_app, _ = _make_app(monkeypatch, silence_s=0.8, heartbeat_s=0.2)
    with TestClient(test_app) as client:
        with client.websocket_connect("/ws/slow_2") as ws:
            ws.send_text(json.dumps({"type": "message", "content": "Show my Accounts.", "session_id": "slow_2"}))
            saw_heartbeat = False
            saw_response = False
            saw_idle = False
            for _ in range(60):
                raw = ws.receive_text()
                data = json.loads(raw)
                if data.get("type") == "progress":
                    saw_heartbeat = True
                if data.get("type") == "response":
                    saw_response = True
                if data.get("type") == "idle":
                    saw_idle = True
                    break
            assert saw_heartbeat
            assert saw_response
            assert saw_idle


def test_duplicate_message_does_not_create_second_concurrent_run(monkeypatch):
    """A second socket (e.g. a reconnect) on a busy session is nacked, not run twice."""
    test_app, agent = _make_app(monkeypatch, silence_s=1.0, heartbeat_s=0.25)
    with TestClient(test_app) as client:
        with client.websocket_connect("/ws/dup_1") as ws1:
            ws1.send_text(json.dumps({"type": "message", "content": "Show my Contacts.", "session_id": "dup_1"}))
            # A reconnect/replay socket appears while the first is still busy.
            with client.websocket_connect("/ws/dup_1") as ws2:
                ws2.send_text(json.dumps({"type": "message", "content": "Show my Contacts.", "session_id": "dup_1"}))
                # The second socket only ever sees the busy-ack (nack) frame.
                raw = ws2.receive_text()
                data = json.loads(raw)
                assert data.get("type") == "progress"
                assert "previous request" in data.get("data", "")
            # The FIRST socket continues and delivers the real response on the same connection.
            saw_response = False
            saw_idle = False
            for _ in range(60):
                raw = ws1.receive_text()
                data = json.loads(raw)
                if data.get("type") == "response":
                    saw_response = True
                if data.get("type") == "idle":
                    saw_idle = True
                    break
            assert saw_response, "first socket must still receive its final response"
            assert saw_idle