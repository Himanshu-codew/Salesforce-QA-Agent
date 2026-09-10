"""
OAuth URL / security hardening tests (M1-M6).

Exercises the REAL handlers registered on the production application against a
TestClient. The app lifecycle is never entered (no MCP/LLM startup), so these
tests only verify URL construction, redirect targets, credential handling,
header emission and fail-closed validation — no network calls are made.

Covered:
  M1  Open redirect: only Salesforce-owned OAuth hosts may receive a redirect;
      any other host fails closed with a 400 and NO Location header.
  M2  Credentials never travel in a URL: BYO setup is a short-lived nonce flow;
      client_id/client_secret in the query string are rejected outright.
  M3  No source-code Consumer Key/Secret fallback; unconfigured server app
      fails safely with guided instructions.
  M4  postMessage handshake is same-origin only (static JS, no '*', origin check).
  M5  session_id is charset-validated at login, setup and callback; never
      rendered unsafely and no longer logged.
  M6  Security headers (CSP without script 'unsafe-inline', nosniff,
      X-Frame-Options, Referrer-Policy) on every response.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402

_APP_PY = Path(app_module.__file__).resolve()
_STATIC_DIR = _APP_PY.parent / "static"
_AUTH_HOST = "https://login.salesforce.com/services/oauth2/authorize?"
_TEST_KEY = "3MVG9TestConsumerKey0000000000"
_TEST_SECRET = "TestConsumerSecret0000008888"


@pytest.fixture()
def client():
    return TestClient(app_module.app)


@pytest.fixture()
def client_with_env(client, monkeypatch):
    monkeypatch.setenv("SALESFORCE_CLIENT_ID", _TEST_KEY)
    monkeypatch.setenv("SALESFORCE_CLIENT_SECRET", _TEST_SECRET)
    return client


@pytest.fixture()
def client_without_env(client, monkeypatch):
    monkeypatch.delenv("SALESFORCE_CLIENT_ID", raising=False)
    monkeypatch.delenv("SALESFORCE_CLIENT_SECRET", raising=False)
    return client


def _has_location(response) -> bool:
    return bool(response.headers.get("location"))


# ── M1: host resolution must fail closed ──


def test_resolve_auth_host_friendly_aliases():
    for alias in ("", "login", "production", "prod", "developer", "dev", "LOGIN", "Production"):
        assert app_module._resolve_auth_host(alias) == "login.salesforce.com"
    assert app_module._resolve_auth_host(None) == "login.salesforce.com"
    assert app_module._resolve_auth_host("test") == "test.salesforce.com"
    assert app_module._resolve_auth_host("TEST") == "test.salesforce.com"


def test_resolve_auth_host_my_domains():
    assert app_module._resolve_auth_host("acme.my.salesforce.com") == "acme.my.salesforce.com"
    assert app_module._resolve_auth_host("acme.develop.my.salesforce.com") == "acme.develop.my.salesforce.com"
    assert app_module._resolve_auth_host("acme.sandbox.my.salesforce.com") == "acme.sandbox.my.salesforce.com"
    assert app_module._resolve_auth_host("https://acme.my.salesforce.com/") == "acme.my.salesforce.com"
    assert app_module._resolve_auth_host(" https://test.salesforce.com ") == "test.salesforce.com"


@pytest.mark.parametrize(
    "evil",
    [
        "attacker.invalid",
        "evil.salesforce.com",
        "salesforce.com.attacker.invalid",
        "user@evil.com",
        "evil.com:8443",
        "evil.com/path",
        "evil.com?domain=x",
        "evil.com#frag",
        "//evil.com",
        "https:////evil.com",
        "ftp://evil.com",
        "javascript:alert(1)",
        "evil%2Ecom",
        "login.salesforce.com.evil.com",
        "login.salesforce.com/path",
        "test.salesforce.com:443",
        "acme.my.salesforce.com.evil.com",
        "login.salesforce.com.",
        "..my.salesforce.com",
    ],
)
def test_resolve_auth_host_rejects_external_origins(evil):
    assert app_module._resolve_auth_host(evil) is None, f"should reject: {evil}"


@pytest.mark.parametrize(
    "evil_domain",
    [
        "attacker.invalid",
        "evil.salesforce.com",
        "salesforce.com.attacker.invalid",
        "https://attacker.invalid/path",
        "login.salesforce.com/..//attacker.invalid",
        "//evil.com",
        "evil.com/path",
    ],
)
def test_login_external_domain_never_redirects(client_with_env, evil_domain):
    r = client_with_env.get(
        "/api/auth/login", params={"session_id": "abc123", "domain": evil_domain}
    )
    assert r.status_code == 400
    assert not _has_location(r)
    assert b"Unsupported Login Domain" in r.content


def test_login_allowed_host_redirect_target_is_salesforce(client_with_env):
    r = client_with_env.get(
        "/api/auth/login", params={"session_id": "abc123", "domain": "login"}, follow_redirects=False
    )
    assert r.status_code == 307
    loc = r.headers["location"]
    assert loc.startswith(_AUTH_HOST)
    assert "redirect_uri" in loc


def test_login_my_domain_redirect_target_is_that_my_domain(client_with_env):
    r = client_with_env.get(
        "/api/auth/login",
        params={"session_id": "abc123", "domain": "https://acme.my.salesforce.com/"},
        follow_redirects=False,
    )
    assert r.status_code == 307
    assert r.headers["location"].startswith("https://acme.my.salesforce.com/services/oauth2/authorize?")


# ── M2: credentials never travel in a URL ──


def test_login_rejects_client_secret_in_url(client, monkeypatch):
    r = client.get(
        "/api/auth/login",
        params={"session_id": "abc123", "domain": "login", "client_secret": _TEST_SECRET},
    )
    assert r.status_code == 400
    assert not _has_location(r)
    assert b"Credentials Must Not Travel in the URL" in r.content
    assert _TEST_SECRET.encode() not in r.content
    assert _TEST_SECRET not in str(r.headers)


def test_login_rejects_client_id_in_url(client):
    r = client.get(
        "/api/auth/login",
        params={"session_id": "abc123", "domain": "login", "client_id": _TEST_KEY},
    )
    assert r.status_code == 400
    assert not _has_location(r)


def test_oauth_setup_roundtrip_never_leaks_secret(client):
    setup = client.post(
        "/api/auth/oauth_setup",
        json={
            "session_id": "abc123",
            "domain": "acme.my.salesforce.com",
            "client_id": _TEST_KEY,
            "client_secret": _TEST_SECRET,
        },
    )
    assert setup.status_code == 200
    nonce = setup.json()["nonce"]
    assert _TEST_SECRET not in setup.text
    assert _TEST_KEY not in setup.text

    r = client.get(
        "/api/auth/login", params={"session_id": "abc123", "oauth_setup": nonce}, follow_redirects=False
    )
    assert r.status_code == 307
    loc = r.headers["location"]
    assert loc.startswith("https://acme.my.salesforce.com/services/oauth2/authorize?")
    assert _TEST_KEY in loc  # consumer key is public and expected
    assert _TEST_SECRET not in loc
    assert _TEST_SECRET not in str(r.headers)
    assert _TEST_SECRET.encode() not in r.content


def test_oauth_setup_nonce_is_single_use(client):
    setup = client.post(
        "/api/auth/oauth_setup",
        json={
            "session_id": "abc123",
            "domain": "login",
            "client_id": _TEST_KEY,
            "client_secret": _TEST_SECRET,
        },
    )
    nonce = setup.json()["nonce"]
    first = client.get(
        "/api/auth/login", params={"session_id": "abc123", "oauth_setup": nonce}, follow_redirects=False
    )
    assert first.status_code == 307
    second = client.get("/api/auth/login", params={"session_id": "abc123", "oauth_setup": nonce})
    assert second.status_code == 400
    assert not _has_location(second)


def test_login_rejects_bogus_nonce(client):
    r = client.get(
        "/api/auth/login", params={"session_id": "abc123", "oauth_setup": "definitely-not-real"}
    )
    assert r.status_code == 400
    assert not _has_location(r)


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "abc123", "domain": "attacker.invalid", "client_id": _TEST_KEY, "client_secret": _TEST_SECRET},
        {"session_id": "abc123", "domain": "login", "client_id": _TEST_KEY, "client_secret": ""},
        {"session_id": "abc123", "domain": "login", "client_id": "", "client_secret": _TEST_SECRET},
        {"session_id": "</script>", "domain": "login", "client_id": _TEST_KEY, "client_secret": _TEST_SECRET},
    ],
)
def test_oauth_setup_rejects_invalid_payloads(client, payload):
    r = client.post("/api/auth/oauth_setup", json=payload)
    assert r.status_code == 400
    assert "nonce" not in r.text


# ── M3: no hardcoded source-code credential fallback ──


def test_no_hardcoded_consumer_key_or_secret_in_source():
    assert not hasattr(app_module, "_DEFAULT_APP_KEY")
    assert not hasattr(app_module, "_DEFAULT_APP_SECRET")
    src = _APP_PY.read_text(encoding="utf-8")
    assert "OEZBQzMyMUJGRjc" not in src  # any residue of the removed default secret
    assert "_DEFAULT_APP_SECRET" not in src


def test_login_unconfigured_server_app_fails_closed(client_without_env):
    r = client_without_env.get("/api/auth/login", params={"session_id": "abc123", "domain": "login"})
    assert r.status_code == 400
    assert not _has_location(r)
    assert b"OAuth Is Not Configured" in r.content


def test_login_custom_my_domain_without_byo_fails_closed(client_without_env):
    r = client_without_env.get(
        "/api/auth/login", params={"session_id": "abc123", "domain": "acme.my.salesforce.com"}
    )
    assert r.status_code == 400
    assert not _has_location(r)


# ── M4: postMessage handshake stays same-origin ──


def test_popup_script_posts_to_own_origin_only():
    popup_js = (_STATIC_DIR / "oauth_popup.js").read_text(encoding="utf-8")
    assert "window.opener.postMessage" in popup_js
    assert "window.location.origin" in popup_js
    assert "'*'" not in popup_js
    assert '"*"' not in popup_js


def test_main_window_checks_message_origin():
    script_js = (_STATIC_DIR / "script.js").read_text(encoding="utf-8")
    assert "event.origin !== window.location.origin" in script_js
    assert "oauth_success" in script_js


def test_callback_uses_external_popup_script_not_inline():
    src = _APP_PY.read_text(encoding="utf-8")
    assert '/static/oauth_popup.js' in src
    assert "window.opener.postMessage" not in src
    assert "postMessage" not in src


# ── M5: session_id validation and safe rendering ──


@pytest.mark.parametrize(
    "session_id",
    [
        "</script><script>alert(1)</script>",
        '"><script>alert(1)</script>',
        "with space",
        "a/b",
        "a?b",
        "a#b",
        "a%3Cb",
        "x" * 65,
        "a@b",
        "a;b",
    ],
)
def test_login_invalid_session_id_fails_closed(client_with_env, session_id):
    r = client_with_env.get("/api/auth/login", params={"session_id": session_id, "domain": "login"})
    assert r.status_code == 400
    assert not _has_location(r)
    assert session_id.encode() not in r.content
    assert b"Invalid Session" in r.content


def test_login_valid_session_id_boundaries(client_with_env):
    for sid in ("a", "_", "-", "default", "a" * 64, "A1_b-2Z"):
        r = client_with_env.get(
            "/api/auth/login", params={"session_id": sid, "domain": "login"}, follow_redirects=False
        )
        assert r.status_code == 307, sid


def test_callback_invalid_session_id_fails_closed(client):
    evil = "</script><script>alert(1)</script>"
    state = "sv1"
    app_module._oauth_pending_flows[state] = {
        "session_id": evil,
        "auth_host": "login.salesforce.com",
        "client_id": _TEST_KEY,
        "client_secret": _TEST_SECRET,
        "redirect_uri": "http://testserver/api/auth/callback",
        "code_verifier": "0000000000000000000000000000000000000000000000",
        "scope": "api",
        "created_at": 9999999999999,
    }
    r = client.get("/api/auth/callback", params={"state": state, "code": "fake"})
    assert r.status_code == 400
    assert evil.encode() not in r.content
    assert b"data-session-id" not in r.content
    assert app_module._oauth_pending_flows.get(state) is None


def test_login_no_longer_logs_session_id():
    src = _APP_PY.read_text(encoding="utf-8")
    assert "session '{session_id}'" not in src


# ── M6: security headers ──


def _csp(client):
    r = client.get("/health")
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert r.headers.get("x-frame-options") == "DENY"
    assert r.headers.get("referrer-policy") == "no-referrer"
    return r.headers.get("content-security-policy", "")


def test_security_headers_present(client):
    csp = _csp(client)
    assert "default-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "object-src 'none'" in csp
    assert "base-uri 'none'" in csp


def test_csp_blocks_inline_scripts_and_unsafe_constructs(client):
    csp = _csp(client)
    script_src = [s for s in csp.split(";") if "script-src" in s][0]
    assert "'self'" in script_src
    assert "unsafe-inline" not in script_src
    assert "unsafe-eval" not in script_src


def test_csp_keeps_google_fonts_and_ui_styles(client):
    csp = _csp(client)
    style_src = [s for s in csp.split(";") if "style-src" in s][0]
    font_src = [s for s in csp.split(";") if "font-src" in s][0]
    assert "https://fonts.googleapis.com" in style_src
    assert "https://fonts.gstatic.com" in font_src


def test_error_popup_and_login_bear_security_headers(client):
    r = client.get("/api/auth/login", params={"session_id": "abc123", "domain": "attacker.invalid"})
    assert r.status_code == 400
    assert r.headers.get("referrer-policy") == "no-referrer"
    assert r.headers.get("x-content-type-options") == "nosniff"
    assert "Content-Security-Policy" in r.headers