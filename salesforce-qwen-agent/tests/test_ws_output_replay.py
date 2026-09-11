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
import time
from pathlib import Path

import asyncio

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


class _GoneWebSocket:
    """A socket that is no longer connected: any send attempt fails fast."""

    def __init__(self):
        self.application_state = "GONE"
        self.send_attempts = 0

    async def send_json(self, payload):  # noqa: ARG002
        self.send_attempts += 1
        raise RuntimeError("socket is gone")


class _StreamingAgent:
    """Fake agent that streams events (optionally failing mid-stream)."""

    def __init__(self, events, fail_after=None, raiser=None):
        self._events = list(events)
        self.fail_after = fail_after
        self.raiser = raiser

    def clear_session(self, session_id):  # noqa: ARG002
        pass

    async def process_message(self, user_message, session_id):  # noqa: ARG002
        for i, ev in enumerate(self._events):
            if self.raiser is not None and (self.fail_after == i or self.fail_after is None):
                raise self.raiser("boom")
            yield ev


def test_ws_produce_drains_and_captures_after_disconnect():
    """Regression: a failed send must NOT abandon the agent generator.

    Previously _ws_produce `break`-ed on the first failed send, so when the
    client disconnected mid-turn (e.g. during the long LLM call) the final
    response was never generated — there was nothing left to cache or replay and
    the reconnect-resend got the "already processed" message. The producer now
    drains the generator silently, capturing every event including the terminal
    response.
    """
    events = [
        {"type": "tool_call", "data": {"name": "getUserInfo"}},
        {"type": "tool_result", "data": {"name": "getUserInfo", "result": "{}"}},
        {"type": "response", "data": "You are Himanshu Swami."},
    ]
    ws = _GoneWebSocket()
    capture = []

    async def _run():
        await app_module._ws_produce(ws, _StreamingAgent(events), "sess-drain", "Who am I?", capture)

    asyncio.run(_run())

    assert ws.send_attempts == 0, "no sends may be attempted on a dead socket"
    assert app_module._terminal_event(capture) == {
        "type": "response",
        "data": "You are Himanshu Swami.",
    }
    assert len(capture) == 3, "every streamed event must be captured"


def test_ws_produce_captures_error_event_when_agent_raises():
    ws = _GoneWebSocket()
    capture = []
    agent = _StreamingAgent(
        [{"type": "tool_call", "data": {"name": "soqlQuery"}}],
        raiser=RuntimeError,
    )

    async def _run():
        await app_module._ws_produce(ws, agent, "sess-err", "hi", capture)

    asyncio.run(_run())

    terminal = app_module._terminal_event(capture)
    assert terminal is not None and terminal["type"] == "error"
    assert terminal.get("code") == "INTERNAL_ERROR"


def test_replay_or_wait_waits_for_inflight_turn(monkeypatch):
    """A fast reconnect-resend must receive the real answer once the abandoned
    turn finishes, instead of an immediate 'already processed' fallback."""
    monkeypatch.setattr(app_module, "_session_outputs", {})
    monkeypatch.setattr(app_module, "_session_busy", {"sess_inflight": True})
    monkeypatch.setattr(app_module, "_REPLAY_WAIT_SECONDS", 5)

    async def _populate():
        await asyncio.sleep(0.2)
        capture = [{"type": "tool_call", "data": {"name": "getUserInfo"}},
                   {"type": "response", "data": "You are Himanshu Swami."}]
        app_module._remember_output("sess_inflight", "REQ-W", capture)
        app_module._session_busy["sess_inflight"] = False

    async def _scenario():
        filler = asyncio.create_task(_populate())
        try:
            return await app_module._replay_or_wait("sess_inflight", "REQ-W")
        finally:
            await filler

    result = asyncio.run(_scenario())
    assert result == {"type": "response", "data": "You are Himanshu Swami."}


def test_replay_or_wait_immediate_miss_returns_none(monkeypatch):
    monkeypatch.setattr(app_module, "_session_outputs", {})
    monkeypatch.setattr(app_module, "_session_busy", {})

    async def _run():
        return await app_module._replay_or_wait("sess_nothing", "REQ-NOPE")

    assert asyncio.run(_run()) is None