"""
Regression tests: replay of a completed turn on reconnect-resend.

When a turn COMPLETES but the client disconnects before the terminal response
is delivered ("final response not deliverable"), the frontend reconnects and
auto-resends the same request_id. Instead of silently dropping the duplicate
(which previously left the user without the answer), the server replays the
cached terminal response/error. The "one submission == one execution" safety
guarantee is unchanged — the agent is never run a second time.

All tests use a fake agent (no live LLM / Salesforce).
"""

import json
import sys
from pathlib import Path

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
    monkeypatch.setattr(app_module, "_session_outputs", {})
    test_app = FastAPI()
    test_app.websocket("/ws/{session_id}")(app_module.websocket_chat)
    return test_app


def _read_until(ws, stop_type="idle", max_iters=80):
    """Read frames until `stop_type`; returns list of (type, data) tuples."""
    frames = []
    for _ in range(max_iters):
        raw = ws.receive_text()
        data = json.loads(raw)
        frames.append((data.get("type"), data.get("data")))
        if data.get("type") == stop_type:
            break
    return frames


def _agent_events(response_text):
    return [
        {"type": "tool_call", "data": {"name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Lead LIMIT 1"}}},
        {"type": "tool_result", "data": {"name": "soqlQuery", "result": "[]"}},
        {"type": "response", "data": response_text},
    ]


def test_completed_turn_is_replayed_on_resend(monkeypatch):
    agent = _CountingAgent(_agent_events("There are 42 Accounts."))
    test_app = _make_app(monkeypatch, agent)

    session = "replay_1"
    with TestClient(test_app) as client:
        with client.websocket_connect(f"/ws/{session}") as ws:
            ws.send_text(json.dumps({
                "type": "message",
                "content": "how many accounts?",
                "session_id": session,
                "request_id": "REQ-REPLAY",
            }))
            first = _read_until(ws)
            assert ("response", "There are 42 Accounts.") in first, first
            assert agent.calls == 1

            # Same request_id resend after a disconnect: the terminal response
            # is replayed instead of dropped, and the agent is NOT run again.
            ws.send_text(json.dumps({
                "type": "message",
                "content": "how many accounts?",
                "session_id": session,
                "request_id": "REQ-REPLAY",
            }))
            replayed = _read_until(ws)
            assert ("response", "There are 42 Accounts.") in replayed, replayed
            assert agent.calls == 1, "replay must NEVER re-execute the turn"


def test_replay_followed_by_idle_frame(monkeypatch):
    agent = _CountingAgent(_agent_events("Done."))
    test_app = _make_app(monkeypatch, agent)

    session = "replay_2"
    with TestClient(test_app) as client:
        with client.websocket_connect(f"/ws/{session}") as ws:
            ws.send_text(json.dumps({
                "type": "message",
                "content": "what is up",
                "session_id": session,
                "request_id": "REQ-1",
            }))
            _read_until(ws)
            ws.send_text(json.dumps({
                "type": "message",
                "content": "what is up",
                "session_id": session,
                "request_id": "REQ-1",
            }))
            frames = _read_until(ws)
            types = [t for t, _ in frames]
            assert "response" in types
            assert "idle" in types, "replay must terminate with an idle frame"


def test_no_cache_yields_dedupe_frame(monkeypatch):
    agent = _CountingAgent(_agent_events("Done."))
    test_app = _make_app(monkeypatch, agent)

    session = "replay_3"
    with TestClient(test_app) as client:
        with client.websocket_connect(f"/ws/{session}") as ws:
            ws.send_text(json.dumps({
                "type": "message",
                "content": "hi",
                "session_id": session,
                "request_id": "REQ-2",
            }))
            _read_until(ws)
            # Evict the completed turn's output so no replay is available.
            app_module._session_outputs.pop(session, None)
            ws.send_text(json.dumps({
                "type": "message",
                "content": "hi",
                "session_id": session,
                "request_id": "REQ-2",
            }))
            raw = ws.receive_text()
            data = json.loads(raw)
            assert data.get("type") == "dedupe", data
            assert agent.calls == 1


def test_cached_output_helpers_roundtrip(monkeypatch):
    monkeypatch.setattr(app_module, "_session_outputs", {})
    capture = []
    app_module._capture_event(capture, {"type": "tool_call", "data": {"name": "soqlQuery"}})
    app_module._capture_event(capture, {"type": "response", "data": "Final answer."})
    assert app_module._terminal_event(capture) == {"type": "response", "data": "Final answer."}

    app_module._remember_output("sess_cache", "REQ-X", capture)
    cached = app_module._cached_output("sess_cache", "REQ-X")
    assert cached == {"type": "response", "data": "Final answer."}
    # A different request_id never matches, and clearing behaves:
    assert app_module._cached_output("sess_cache", "REQ-Y") is None
    app_module._session_outputs.pop("sess_cache", None)
    assert app_module._cached_output("sess_cache", "REQ-X") is None

    # A turn that ended WITHOUT a terminal event must not be cached (replay of
    # a mid-flight tool-only turn would be meaningless).
    partial = [{"type": "tool_call", "data": {"name": "soqlQuery"}}]
    app_module._remember_output("sess_partial", "REQ-Z", partial)
    assert app_module._cached_output("sess_partial", "REQ-Z") is None