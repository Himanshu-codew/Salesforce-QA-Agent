"""
Regression tests: per-request dedupe (Issue 3 — one submission == one execution).

The frontend tags each submission with a unique request_id. A re-sent duplicate
(reconnect replay, double-submit, auto-resend) carrying the SAME request_id must
be dropped server-side instead of executed again — the defensive layer that
stops a single "create a lead" from creating two Leads. These tests verify the
WebSocket handler drops an identical request_id within the TTL while a fresh
request_id still executes normally.

All tests use a fake counting agent (no live LLM / Salesforce).
"""

import json
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402


class _CountingAgent:
    """Fake agent that counts how many times process_message is entered."""

    def __init__(self, events):
        self._events = events
        self.calls = 0

    def clear_session(self, session_id):  # noqa: ARG002
        pass

    async def process_message(self, user_message, session_id):  # noqa: ARG002
        self.calls += 1
        for ev in self._events:
            yield ev


def _fake_session_manager_returning(agent):
    class _SM:
        async def get_or_create_agent(self, session_id):  # noqa: ARG002
            return agent

    return _SM()


def _make_app(monkeypatch, agent):
    monkeypatch.setattr(app_module, "session_manager", _fake_session_manager_returning(agent))
    monkeypatch.setattr(app_module, "_session_busy", {})
    monkeypatch.setattr(app_module, "_recent_requests", {})
    test_app = FastAPI()
    test_app.websocket("/ws/{session_id}")(app_module.websocket_chat)
    return test_app


def _read_until(ws, stop_type="idle", max_iters=80):
    """Read frames until `stop_type` or bounded iteration; returns seen types."""
    types = []
    for _ in range(max_iters):
        raw = ws.receive_text()
        data = json.loads(raw)
        types.append(data.get("type"))
        if data.get("type") == stop_type:
            break
    return types


def _agent_events(response_text):
    return [
        {"type": "tool_call", "data": {"name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Lead LIMIT 1"}}},
        {"type": "tool_result", "data": {"name": "soqlQuery", "result": "[]"}},
        {"type": "response", "data": response_text},
    ]


def test_same_request_id_resubmission_is_dropped(monkeypatch):
    agent = _CountingAgent(_agent_events("Lead created"))
    test_app = _make_app(monkeypatch, agent)

    with TestClient(test_app) as client:
        with client.websocket_connect("/ws/dedupe_1") as ws:
            ws.send_text(json.dumps({
                "type": "message",
                "content": "create a lead",
                "session_id": "dedupe_1",
                "request_id": "REQ-AAA",
            }))
            first = _read_until(ws)
            assert "response" in first

            # Same request_id re-sent (e.g. an automatic resend after reconnect).
            ws.send_text(json.dumps({
                "type": "message",
                "content": "create a lead",
                "session_id": "dedupe_1",
                "request_id": "REQ-AAA",
            }))
            raw = ws.receive_text()
            data = json.loads(raw)
            assert data.get("type") == "progress"
            assert "already processed" in data.get("data", "")
            assert agent.calls == 1, "the duplicate must NOT be executed again"


def test_fresh_request_id_still_executes(monkeypatch):
    agent = _CountingAgent(_agent_events("Done"))
    test_app = _make_app(monkeypatch, agent)

    with TestClient(test_app) as client:
        with client.websocket_connect("/ws/dedupe_2") as ws:
            ws.send_text(json.dumps({
                "type": "message",
                "content": "create a lead",
                "session_id": "dedupe_2",
                "request_id": "REQ-BBB",
            }))
            first = _read_until(ws)
            assert "response" in first

            # A DIFFERENT request id with the same text is a genuinely new
            # submission and must execute (matches "create another lead").
            ws.send_text(json.dumps({
                "type": "message",
                "content": "create a lead",
                "session_id": "dedupe_2",
                "request_id": "REQ-CCC",
            }))
            second = _read_until(ws)
            assert "response" in second
            assert agent.calls == 2