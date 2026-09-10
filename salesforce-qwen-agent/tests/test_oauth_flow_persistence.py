"""
Regression tests: pending OAuth flows survive process restarts.

The callback 400s with "state not found" whenever the process restarts between
the login 307 and Salesforce's redirect (uvicorn --reload, Render container
recycle, crash). Pending flows are now mirrored to an encrypted file so the
callback state is restored on startup. These tests verify persist/load round
trips, expiry filtering, and that in-memory states always win over disk.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402


def _make_flow(**overrides):
    flow = {
        "session_id": "sess-1",
        "auth_host": "login.salesforce.com",
        "client_id": "CLIENT_ID",
        "client_secret": "TOP-SECRET-SECRET",
        "redirect_uri": "http://localhost:8000/api/auth/callback",
        "code_verifier": "VERIFIER-123",
        "scope": "api",
        "created_at": time.time(),
    }
    flow.update(overrides)
    return flow


def test_persist_load_round_trip_encrypted(tmp_path, monkeypatch):
    store_path = tmp_path / "oauth_pending.enc"
    monkeypatch.setattr(app_module, "_OAUTH_FLOW_STORE_PATH", str(store_path))

    flows = {
        "state-1": _make_flow(),
        "state-2": _make_flow(session_id="sess-2", client_secret="ANOTHER-SECRET"),
    }
    monkeypatch.setattr(app_module, "_oauth_pending_flows", dict(flows))

    app_module._persist_oauth_flows()
    assert store_path.exists(), "persisted mirror must exist on disk"

    raw = store_path.read_text(encoding="utf-8")
    blob = json.loads(raw)
    assert "state-1" not in raw, "flows must be encrypted, never plaintext on disk"
    assert "TOP-SECRET-SECRET" not in raw, "secrets must not be stored in cleartext"
    assert blob.get("v") == 1

    # Simulate a restart: fresh (empty) in-memory store, restored from disk.
    monkeypatch.setattr(app_module, "_oauth_pending_flows", {})
    app_module._load_oauth_flows()

    restored = app_module._oauth_pending_flows
    assert restored["state-1"]["code_verifier"] == "VERIFIER-123"
    assert restored["state-1"]["client_secret"] == "TOP-SECRET-SECRET"
    assert restored["state-2"]["client_secret"] == "ANOTHER-SECRET"


def test_load_skips_expired_states(tmp_path, monkeypatch):
    store_path = tmp_path / "oauth_pending.enc"
    monkeypatch.setattr(app_module, "_OAUTH_FLOW_STORE_PATH", str(store_path))

    expired_ts = time.time() - 3600  # older than the 600s TTL
    flows = {
        "stale-state": _make_flow(created_at=expired_ts),
        "fresh-state": _make_flow(created_at=time.time()),
    }
    monkeypatch.setattr(app_module, "_oauth_pending_flows", dict(flows))
    app_module._persist_oauth_flows()

    monkeypatch.setattr(app_module, "_oauth_pending_flows", {})
    app_module._load_oauth_flows()
    restored = app_module._oauth_pending_flows
    assert "fresh-state" in restored
    assert "stale-state" not in restored, "expired states must never be restored"


def test_load_never_overwrites_in_memory_states(tmp_path, monkeypatch):
    store_path = tmp_path / "oauth_pending.enc"
    monkeypatch.setattr(app_module, "_OAUTH_FLOW_STORE_PATH", str(store_path))

    monkeypatch.setattr(app_module, "_oauth_pending_flows", {
        "state-X": _make_flow(session_id="disk-vers"),
    })
    app_module._persist_oauth_flows()

    # In-memory already holds a NEWER flow for the same state (new process won).
    monkeypatch.setattr(app_module, "_oauth_pending_flows", {
        "state-X": _make_flow(session_id="memory-vers"),
    })
    app_module._load_oauth_flows()
    assert app_module._oauth_pending_flows["state-X"]["session_id"] == "memory-vers"


def test_cleanup_persists_removals(tmp_path, monkeypatch):
    store_path = tmp_path / "oauth_pending.enc"
    monkeypatch.setattr(app_module, "_OAUTH_FLOW_STORE_PATH", str(store_path))

    monkeypatch.setattr(app_module, "_oauth_pending_flows", {
        "expired-a": _make_flow(created_at=time.time() - 3600),
    })
    app_module._prune_expired_states()
    assert app_module._oauth_pending_flows == {}
    assert store_path.exists(), "prune must mirror the removal to disk"


def test_oauth_login_persists_flow(tmp_path, monkeypatch):
    """The real login route writes the flow and the encrypted mirror."""
    store_path = tmp_path / "oauth_pending.enc"
    monkeypatch.setattr(app_module, "_OAUTH_FLOW_STORE_PATH", str(store_path))
    monkeypatch.setattr(app_module, "_oauth_pending_flows", {})

    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    from app import app as real_app

    test_app = FastAPI()
    test_app.get("/api/auth/login")(app_module.oauth_login)
    client = TestClient(test_app, follow_redirects=False)
    res = client.get(
        "/api/auth/login",
        params={"session_id": "sess-login", "domain": "login"},
    )
    assert res.status_code == 307
    assert store_path.exists(), "login must persist its pending flow to disk"
    raw = store_path.read_text(encoding="utf-8")
    assert "code_verifier" not in raw, f"flow must be encrypted: {raw[:80]}"
    assert len(app_module._oauth_pending_flows) == 1