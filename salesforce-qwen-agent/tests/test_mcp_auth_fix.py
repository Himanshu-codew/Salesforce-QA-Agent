"""
Regression tests: MCP auth preference fix (Option 1) + one-time 401 noise reduction.

Problem: with SALESFORCE_OAUTH_SCOPE configured carrying MCP scopes
(mcp_api / sfap_api / legacy sfap:mcp:*), the hosted Salesforce MCP server still
returned 401 on every boot. Root cause: `authenticate()` preferred the SOAP
partner login whenever username+password+security_token were present, and a SOAP
session token carries no MCP scope -> MCP endpoint rejects it with 401 -> silent
REST fallback.

Fix:
  - `authenticate()` prefers the OAuth password grant when an MCP-capable scope
    is configured (the grant forwards `scope`, so the token is MCP-capable),
    keeping SOAP-first for REST-only (no-scope) setups.
  - The password-grant path now captures `refresh_token` so later refreshes stay
    MCP-capable.
  - `_needs_mcp_token()` treats an unexpired SOAP-derived token as needing
    replacement when an MCP scope is configured, so it is never blindly reused
    against the MCP endpoint.
  - A persistent 401 during MCP session init is reported with ONE actionable
    warning; subsequent boots/retries stay quiet (DEBUG).

Pure unit tests (mocked HTTP / MCP internals, no live credentials).
"""

import asyncio
import logging
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sfmcp.client import SalesforceMCPClient

MCP_SCOPE = "api sfap_api refresh_token offline_access mcp_api"
REST_ONLY_SCOPE = "api refresh_token offline_access"


def _make_client():
    """A bare client with the attrs `authenticate`/`_ensure_fresh_token` touch."""
    client = SalesforceMCPClient.__new__(SalesforceMCPClient)
    client._access_token = None
    client._refresh_token = None
    client._expires_at = 0.0
    client._access_token_is_soap = False
    client.oauth_scope = None
    client.auth_host = None
    client.domain = "login"
    client.instance_url = "https://x.salesforce.com"
    client.client_id = "cid"
    client.client_secret = "csec"
    client.username = "u@x.com"
    client.password = "pwd"
    client.security_token = "tok"
    client.mcp_url = "https://api.salesforce.com/platform/mcp/v1/platform/sobject-all"
    client.token_vault = None
    client.session_id = None
    client.mcp_required = False
    client.mcp_transport = "REST"
    client._mcp_401_warned = False
    client._oauth_unavailable = False
    client._oauth_reason = ""
    client._mcp_unavailable_reason = ""
    client._mcp_enabled = True
    client._http_client = AsyncMock()
    client._persist_tokens = Mock()
    client._soap_authenticate = AsyncMock(return_value="soap_tok")
    return client


def _oauth_response(access_token="oauth_tok", refresh_token="rt", expires_in=3600):
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.status_code = 200
    resp.json = Mock(return_value={
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": expires_in,
        "instance_url": "https://x.salesforce.com",
    })
    return resp


def _oauth_error_response(status=400, err="invalid_grant", desc="expired access/refresh token"):
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.status_code = status
    resp.text = f'{{"error":"{err}","error_description":"{desc}"}}'
    resp.json = Mock(return_value={"error": err, "error_description": desc})
    err_inst = Exception(f"Client error '{status}' for url 'https://login.salesforce.com/oauth2/token'")
    err_inst.response = SimpleNamespace(status_code=status, text=resp.text)
    err_inst.response.raise_for_status = resp.raise_for_status
    resp.raise_for_status.side_effect = err_inst
    return resp


def _make_http401():
    err = Exception("Client error '401 Unauthorized' for url 'https://api.salesforce.com/'")
    err.response = SimpleNamespace(status_code=401)
    return err


# ─────────────────────────────────────────────
# authenticate(): OAuth-first when MCP scope set
# ─────────────────────────────────────────────

def test_authenticate_uses_oauth_password_grant_when_mcp_scope():
    client = _make_client()
    client.oauth_scope = MCP_SCOPE
    client._http_client.post = AsyncMock(return_value=_oauth_response())

    result = asyncio.run(client.authenticate())

    assert result == "oauth_tok"
    assert client._access_token == "oauth_tok"
    assert client._access_token_is_soap is False
    assert client._refresh_token == "rt"
    client._soap_authenticate.assert_not_awaited()
    client._persist_tokens.assert_called_once()
    # The grant must forward the configured MCP scope.
    call_kwargs = client._http_client.post.call_args
    payload = call_kwargs.kwargs.get("data") or call_kwargs.args[1]
    assert payload["grant_type"] == "password"
    assert payload["scope"] == MCP_SCOPE


def test_authenticate_captures_refresh_token_from_grant():
    client = _make_client()
    client.oauth_scope = MCP_SCOPE
    client._http_client.post = AsyncMock(return_value=_oauth_response(refresh_token="new_rt"))

    asyncio.run(client.authenticate())

    assert client._refresh_token == "new_rt"


def test_authenticate_falls_back_to_soap_when_oauth_grant_fails():
    client = _make_client()
    client.oauth_scope = MCP_SCOPE
    client._http_client.post = AsyncMock(side_effect=RuntimeError("oauth down"))

    result = asyncio.run(client.authenticate())

    assert result == "soap_tok"
    client._soap_authenticate.assert_awaited_once()


# ─────────────────────────────────────────────
# authenticate(): graceful degradation when OAuth grant rejected
# ─────────────────────────────────────────────

def test_authenticate_records_4xx_oauth_failure_and_falls_back_to_soap():
    client = _make_client()
    client.oauth_scope = MCP_SCOPE
    client._http_client.post = AsyncMock(return_value=_oauth_error_response(
        status=400, err="invalid_grant"
    ))

    result = asyncio.run(client.authenticate())

    assert result == "soap_tok"
    client._soap_authenticate.assert_awaited_once()
    assert client._oauth_unavailable is True
    assert "HTTP 400" in client._oauth_reason
    assert "invalid_grant" in client._oauth_reason


def test_authenticate_skips_oauth_entirely_once_marked_unavailable():
    client = _make_client()
    client.oauth_scope = MCP_SCOPE
    client._oauth_unavailable = True
    client._oauth_reason = "OAuth grant rejected (HTTP 400): error=invalid_grant"

    result = asyncio.run(client.authenticate(force_oauth=True))

    assert result == "soap_tok"
    client._soap_authenticate.assert_awaited_once()
    client._http_client.post.assert_not_awaited()


def test_authenticate_does_not_cache_transient_oauth_failure():
    client = _make_client()
    client.oauth_scope = MCP_SCOPE
    client._http_client.post = AsyncMock(side_effect=RuntimeError("network down"))

    result = asyncio.run(client.authenticate())

    assert result == "soap_tok"
    assert client._oauth_unavailable is False


def test_authenticate_raises_without_soap_creds_when_oauth_rejected():
    client = _make_client()
    client.oauth_scope = MCP_SCOPE
    client.username = None
    client.password = None
    client.security_token = None
    client._http_client.post = AsyncMock(return_value=_oauth_error_response(
        status=400, err="invalid_client"
    ))

    with pytest.raises(RuntimeError, match="OAuth authentication failed"):
        asyncio.run(client.authenticate())


# ─────────────────────────────────────────────
# _try_oauth_refresh / _needs_mcp_token: no per-turn churn
# ─────────────────────────────────────────────

def test_oauth_failure_cached_stops_repeat_grant_attempts():
    client = _make_client()
    client.oauth_scope = MCP_SCOPE
    client._http_client.post = AsyncMock(return_value=_oauth_error_response(
        status=400, err="invalid_grant"
    ))

    first = asyncio.run(client.authenticate())
    second = asyncio.run(client.authenticate(force_oauth=True))

    assert first == "soap_tok"
    assert second == "soap_tok"
    assert client._oauth_unavailable is True
    # Exactly ONE OAuth grant attempt across both calls — the second never hits
    # the token endpoint.
    assert client._http_client.post.await_count == 1
    assert client._soap_authenticate.await_count == 2


def test_refresh_skipped_when_oauth_unavailable():
    client = _make_client()
    client.oauth_scope = MCP_SCOPE
    client._oauth_unavailable = True
    client._refresh_token = "rt"

    assert asyncio.run(client._try_oauth_refresh()) is False
    client._http_client.post.assert_not_awaited()


def test_needs_mcp_token_false_for_soap_token_when_oauth_unavailable():
    client = _make_client()
    client._access_token = "soap_sid"
    client._access_token_is_soap = True
    client._expires_at = time.time() + 3600
    client.oauth_scope = MCP_SCOPE
    client._oauth_unavailable = True

    assert client._needs_mcp_token() is False


def test_ensure_fresh_token_reuses_soap_token_for_rest_when_oauth_unavailable():
    client = _make_client()
    client._access_token = "soap_sid"
    client._access_token_is_soap = True
    client._expires_at = time.time() + 3600
    client.oauth_scope = MCP_SCOPE
    client._oauth_unavailable = True
    client._load_mcp_scoped_token_from_vault = Mock(return_value=False)
    client._try_oauth_refresh = AsyncMock(return_value=False)
    client.authenticate = AsyncMock(return_value="soap_tok")

    asyncio.run(client._ensure_fresh_token())

    client._try_oauth_refresh.assert_not_awaited()
    client.authenticate.assert_not_awaited()


def test_ensure_fresh_token_still_replaces_soap_token_when_oauth_works():
    client = _make_client()
    client._access_token = "soap_sid"
    client._access_token_is_soap = True
    client._expires_at = time.time() + 3600
    client.oauth_scope = MCP_SCOPE
    client._load_mcp_scoped_token_from_vault = Mock(return_value=False)
    client._try_oauth_refresh = AsyncMock(return_value=False)
    client.authenticate = AsyncMock(return_value="oauth_tok")

    asyncio.run(client._ensure_fresh_token())

    client.authenticate.assert_awaited_once()


# ─────────────────────────────────────────────
# authenticate(): SOAP-first preserved for REST-only
# ─────────────────────────────────────────────

def test_authenticate_keeps_soap_first_when_no_mcp_scope():
    client = _make_client()
    client.oauth_scope = None

    result = asyncio.run(client.authenticate())

    assert result == "soap_tok"
    client._soap_authenticate.assert_awaited_once()
    client._http_client.post.assert_not_awaited()


def test_authenticate_falls_to_oauth_when_soap_fails_without_scope():
    client = _make_client()
    client.oauth_scope = REST_ONLY_SCOPE
    client._soap_authenticate = AsyncMock(side_effect=RuntimeError("soap down"))
    client._http_client.post = AsyncMock(return_value=_oauth_response())

    result = asyncio.run(client.authenticate())

    assert result == "oauth_tok"
    client._soap_authenticate.assert_awaited_once()


# ─────────────────────────────────────────────
# _scope_has_mcp_capability
# ─────────────────────────────────────────────

def test_scope_has_mcp_capability():
    assert SalesforceMCPClient._scope_has_mcp_capability("sfap:mcp:all sfap:mcp:remote api")
    assert SalesforceMCPClient._scope_has_mcp_capability(MCP_SCOPE)
    assert SalesforceMCPClient._scope_has_mcp_capability("api sfap_api mcp_api")
    assert not SalesforceMCPClient._scope_has_mcp_capability(REST_ONLY_SCOPE)
    assert not SalesforceMCPClient._scope_has_mcp_capability("")
    assert not SalesforceMCPClient._scope_has_mcp_capability("api")


# ─────────────────────────────────────────────
# _needs_mcp_token: SOAP token must be replaced for MCP
# ─────────────────────────────────────────────

def test_needs_mcp_token_true_for_soap_token_when_mcp_scope():
    client = _make_client()
    client._access_token = "soap_sid"
    client._access_token_is_soap = True
    client._expires_at = time.time() + 3600
    client.oauth_scope = MCP_SCOPE
    assert client._needs_mcp_token() is True


def test_needs_mcp_token_false_for_soap_token_without_scope():
    client = _make_client()
    client._access_token = "soap_sid"
    client._access_token_is_soap = True
    client._expires_at = time.time() + 3600
    client.oauth_scope = None
    assert client._needs_mcp_token() is False


def test_needs_mcp_token_false_for_oauth_token_when_mcp_scope():
    client = _make_client()
    client._access_token = "oauth_tok"
    client._access_token_is_soap = False
    client._expires_at = time.time() + 3600
    client.oauth_scope = MCP_SCOPE
    assert client._needs_mcp_token() is False


def test_needs_mcp_token_true_when_missing_or_expired():
    client = _make_client()
    client.oauth_scope = MCP_SCOPE
    assert client._needs_mcp_token() is True  # no access token
    client._access_token = "tok"
    client._access_token_is_soap = False
    client._expires_at = time.time() - 10
    assert client._needs_mcp_token() is True  # expired


# ─────────────────────────────────────────────
# _ensure_fresh_token: routes SOAP token to refresh/re-auth
# ─────────────────────────────────────────────

def test_ensure_fresh_token_refreshes_soap_token_when_mcp_scope():
    client = _make_client()
    client._access_token = "soap_sid"
    client._access_token_is_soap = True
    client._expires_at = time.time() + 3600
    client.oauth_scope = MCP_SCOPE
    client._refresh_token = "rt"
    client._load_mcp_scoped_token_from_vault = Mock(return_value=False)
    client._try_oauth_refresh = AsyncMock(return_value=True)
    client.authenticate = AsyncMock(return_value="oauth_tok")

    asyncio.run(client._ensure_fresh_token())

    client._try_oauth_refresh.assert_awaited_once()
    client.authenticate.assert_not_awaited()


def test_ensure_fresh_token_reauths_soap_token_via_authenticate_without_refresh():
    client = _make_client()
    client._access_token = "soap_sid"
    client._access_token_is_soap = True
    client._expires_at = time.time() + 3600
    client.oauth_scope = MCP_SCOPE
    client._refresh_token = None
    client._load_mcp_scoped_token_from_vault = Mock(return_value=False)
    client._try_oauth_refresh = AsyncMock(return_value=False)
    client.authenticate = AsyncMock(return_value="oauth_tok")

    asyncio.run(client._ensure_fresh_token())

    client.authenticate.assert_awaited_once()


def test_ensure_fresh_token_noop_when_token_valid():
    client = _make_client()
    client._access_token = "oauth_tok"
    client._access_token_is_soap = False
    client._expires_at = time.time() + 3600
    client.oauth_scope = MCP_SCOPE
    client._load_mcp_scoped_token_from_vault = Mock(return_value=False)
    client._try_oauth_refresh = AsyncMock(return_value=True)
    client.authenticate = AsyncMock(return_value="oauth_tok")

    asyncio.run(client._ensure_fresh_token())

    client._try_oauth_refresh.assert_not_awaited()
    client.authenticate.assert_not_awaited()
    client._load_mcp_scoped_token_from_vault.assert_not_called()


# ─────────────────────────────────────────────
# One-time 401 noise reduction in _ensure_connected
# ─────────────────────────────────────────────

class _FakeMcpContext:
    def __init__(self, initialize_success=False):
        self._initialize_success = initialize_success
        self.entered = False

    async def __aenter__(self):
        self.entered = True
        return (SimpleNamespace(), SimpleNamespace())

    async def __aexit__(self, *args):
        return False


def _fake_streamable_http_client():
    def _factory(*args, **kwargs):
        return _FakeMcpContext()
    return _factory


class _FakeClientSession401:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def initialize(self):
        raise _make_http401()


class _FakeClientSessionOk:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def initialize(self):
        return None


def _make_connected_client():
    client = _make_client()
    client._session = None
    client._mcp_ctx = None
    client._mcp_read = None
    client._mcp_write = None
    client._connected = False
    client._access_token = "tok"
    client.mcp_required = False
    client._ensure_fresh_token = AsyncMock()
    client._close_mcp_session = AsyncMock()
    return client


def test_mcp_401_warned_at_most_once(caplog):
    client = _make_connected_client()
    with patch("sfmcp.client.streamable_http_client", _fake_streamable_http_client()), \
         patch("sfmcp.client.ClientSession", _FakeClientSession401):
        with caplog.at_level(logging.WARNING, logger="sfmcp.client"):
            asyncio.run(client._ensure_connected())
            asyncio.run(client._ensure_connected())

    full_warnings = [
        r for r in caplog.records
        if "Session init rejected (401 Unauthorized)" in r.getMessage()
    ]
    assert len(full_warnings) == 1
    assert "Verify" in full_warnings[0].getMessage()
    assert client._mcp_401_warned is True


def test_mcp_success_clears_401_warning_flag(caplog):
    client = _make_connected_client()
    # First: a 401 rejection.
    with patch("sfmcp.client.streamable_http_client", _fake_streamable_http_client()), \
         patch("sfmcp.client.ClientSession", _FakeClientSession401):
        asyncio.run(client._ensure_connected())
    assert client._mcp_401_warned is True

    # Now a successful init on the SAME client resets the flag.
    client._session = None
    client._mcp_ctx = None
    with patch("sfmcp.client.streamable_http_client", _fake_streamable_http_client()), \
         patch("sfmcp.client.ClientSession", _FakeClientSessionOk):
        with caplog.at_level(logging.WARNING, logger="sfmcp.client"):
            asyncio.run(client._ensure_connected())

    assert client._mcp_401_warned is False
    assert client.mcp_transport == "MCP"


# ─────────────────────────────────────────────
# _ensure_connected: MCP gated unless a real OAuth token exists
# ─────────────────────────────────────────────

def test_ensure_connected_gates_mcp_when_oauth_unavailable():
    client = _make_connected_client()
    client._access_token = "soap_sid"
    client._access_token_is_soap = True
    client._oauth_unavailable = True
    client._oauth_reason = "OAuth grant rejected (HTTP 400): error=invalid_grant"

    with patch("sfmcp.client.streamable_http_client", side_effect=AssertionError("MCP must not connect")):
        asyncio.run(client._ensure_connected())

    assert client._session is None
    assert client.mcp_transport == "REST"
    assert "rejected" in client._mcp_unavailable_reason


def test_ensure_connected_gates_mcp_when_disabled():
    client = _make_connected_client()
    client._mcp_enabled = False
    client._access_token = "tok"
    client._access_token_is_soap = False

    with patch("sfmcp.client.streamable_http_client", side_effect=AssertionError("MCP must not connect")):
        asyncio.run(client._ensure_connected())

    assert client._session is None
    assert "SALESFORCE_MCP_ENABLED=false" in client._mcp_unavailable_reason
    assert client.mcp_transport == "REST"


def test_ensure_connected_gates_mcp_with_no_url():
    client = _make_connected_client()
    client.mcp_url = ""

    with patch("sfmcp.client.streamable_http_client", side_effect=AssertionError("MCP must not connect")):
        asyncio.run(client._ensure_connected())

    assert client._session is None
    assert "SALESFORCE_MCP_URL" in client._mcp_unavailable_reason


def test_ensure_connected_gates_mcp_when_only_soap_token_with_guidance():
    client = _make_connected_client()
    client._access_token = "soap_sid"
    client._access_token_is_soap = True
    client._oauth_unavailable = False

    with patch("sfmcp.client.streamable_http_client", side_effect=AssertionError("MCP must not connect")):
        asyncio.run(client._ensure_connected())

    assert client._session is None
    assert "interactive /api/auth/login" in client._mcp_unavailable_reason


def test_ensure_connected_reconnects_mcp_when_vault_oauth_token_available():
    client = _make_client()
    client._session = None
    client._mcp_ctx = None
    client._mcp_read = None
    client._mcp_write = None
    client._connected = False
    client.oauth_scope = MCP_SCOPE
    client._access_token = "soap_sid"
    client._access_token_is_soap = True
    client._expires_at = time.time() + 3600
    client.mcp_required = False
    client._close_mcp_session = AsyncMock()

    def _load_vault_oauth_token():
        client._access_token = "oauth_vault_tok"
        client._access_token_is_soap = False
        client._expires_at = time.time() + 3600
        return True

    with patch.object(client, "_load_mcp_scoped_token_from_vault", side_effect=_load_vault_oauth_token), \
         patch("sfmcp.client.streamable_http_client", _fake_streamable_http_client()), \
         patch("sfmcp.client.ClientSession", _FakeClientSessionOk):
        asyncio.run(client._ensure_connected())

    assert client._session is not None
    assert client.mcp_transport == "MCP"
    assert client._mcp_unavailable_reason == ""


def test_call_tool_rest_only_and_single_ensure_when_oauth_unavailable():
    import sfmcp.client as mod
    client = _make_client()
    client.oauth_scope = MCP_SCOPE
    client._oauth_unavailable = True
    client._access_token = "soap_sid"
    client._access_token_is_soap = True
    client._expires_at = time.time() + 3600
    client._session = None
    client._mcp_ctx = None
    client._connected = False
    client._load_mcp_scoped_token_from_vault = Mock(return_value=False)
    client._fallback_rest_api = AsyncMock(return_value={"totalSize": 1, "records": []})

    mcp_connects = []
    real_ensure = mod.SalesforceMCPClient._ensure_connected

    async def counting_ensure(self):
        mcp_connects.append(1)
        return await real_ensure(self)

    with patch.object(mod.SalesforceMCPClient, "_ensure_connected", counting_ensure), \
         patch("sfmcp.client.streamable_http_client", side_effect=AssertionError("MCP must not connect")):
        result = asyncio.run(client.call_tool("soqlQuery", {"query": "SELECT Id FROM Account"}))

    assert result == {"totalSize": 1, "records": []}
    assert len(mcp_connects) == 1
    assert client.mcp_transport == "REST"
    assert "rejected" in client._mcp_unavailable_reason