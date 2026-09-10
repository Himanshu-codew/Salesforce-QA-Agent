"""
Regression tests: PKCE-only auth simplification.

The auth system now uses ONLY OAuth 2.0 Authorization Code + PKCE. The
client-credentials grant, OAuth password grant, and SOAP partner login have
all been removed. Authentication flows are:
  - Interactive browser: /api/auth/login (PKCE) -> token stored in vault
  - Server-side clients: load a PKCE token from vault OR use refresh_token

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
    """A bare client with the attrs touch_authenticate paths."""
    client = SalesforceMCPClient.__new__(SalesforceMCPClient)
    client._access_token = None
    client._refresh_token = None
    client._expires_at = 0.0
    client.oauth_scope = None
    client.auth_host = None
    client.domain = "login"
    client.instance_url = "https://x.salesforce.com"
    client.client_id = "cid"
    client.client_secret = "csec"
    client.username = ""
    client.password = ""
    client.security_token = ""
    client.mcp_url = "https://api.salesforce.com/platform/mcp/v1/platform/sobject-all"
    client.token_vault = None
    client.session_id = None
    client.mcp_required = False
    client.mcp_transport = "REST"
    client._mcp_401_warned = False
    client._http_client = AsyncMock()
    client._persist_tokens = Mock()
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


def _make_http401():
    err = Exception("Client error '401 Unauthorized' for url 'https://api.salesforce.com/'")
    err.response = SimpleNamespace(status_code=401)
    return err


# ─────────────────────────────────────────────
# authenticate(): PKCE-only — no client-credentials/password/SOAP
# ─────────────────────────────────────────────

def test_authenticate_uses_refresh_token_grant_when_available():
    client = _make_client()
    client._refresh_token = "rt"

    async def _fake_refresh():
        client._access_token = "refreshed_tok"
        return True

    client._try_oauth_refresh = AsyncMock(side_effect=_fake_refresh)
    client._load_mcp_scoped_token_from_vault = Mock(return_value=False)

    result = asyncio.run(client.authenticate())

    assert result == "refreshed_tok"
    client._try_oauth_refresh.assert_awaited_once()
    client._load_mcp_scoped_token_from_vault.assert_not_called()


def test_authenticate_loads_from_vault_when_no_refresh_token():
    client = _make_client()
    client._access_token = "vault_tok"
    client._load_mcp_scoped_token_from_vault = Mock(return_value=True)
    client._try_oauth_refresh = AsyncMock(return_value=False)

    result = asyncio.run(client.authenticate())

    assert result == "vault_tok"
    client._load_mcp_scoped_token_from_vault.assert_called_once()


def test_authenticate_warns_when_no_token_available(caplog):
    client = _make_client()
    client._load_mcp_scoped_token_from_vault = Mock(return_value=False)
    client._try_oauth_refresh = AsyncMock(return_value=False)

    with caplog.at_level(logging.WARNING, logger="sfmcp.client"):
        result = asyncio.run(client.authenticate())

    assert result == ""
    assert "No PKCE token available" in caplog.text


def test_authenticate_never_calls_legacy_methods():
    """Guard: client-credentials / password / SOAP must not exist."""
    client = _make_client()
    assert not hasattr(client, "_client_credentials_authenticate")
    assert not hasattr(client, "_soap_authenticate")
    client._load_mcp_scoped_token_from_vault = Mock(return_value=False)
    client._try_oauth_refresh = AsyncMock(return_value=False)
    result = asyncio.run(client.authenticate())
    assert result == ""
    client._http_client.post.assert_not_awaited()


# ─────────────────────────────────────────────
# _needs_mcp_token: simple missing/expired check
# ─────────────────────────────────────────────

def test_needs_mcp_token_false_when_token_valid():
    client = _make_client()
    client._access_token = "oauth_tok"
    client._expires_at = time.time() + 3600
    assert client._needs_mcp_token() is False


def test_needs_mcp_token_true_when_missing():
    client = _make_client()
    assert client._needs_mcp_token() is True


def test_needs_mcp_token_true_when_expired():
    client = _make_client()
    client._access_token = "tok"
    client._expires_at = time.time() - 10
    assert client._needs_mcp_token() is True


# ─────────────────────────────────────────────
# _ensure_fresh_token: refresh or re-auth when needed
# ─────────────────────────────────────────────

def test_ensure_fresh_token_noop_when_token_valid():
    client = _make_client()
    client._access_token = "oauth_tok"
    client._expires_at = time.time() + 3600
    client._load_mcp_scoped_token_from_vault = Mock(return_value=False)
    client._try_oauth_refresh = AsyncMock(return_value=True)
    client.authenticate = AsyncMock(return_value="oauth_tok")

    asyncio.run(client._ensure_fresh_token())

    client._try_oauth_refresh.assert_not_awaited()
    client.authenticate.assert_not_awaited()
    client._load_mcp_scoped_token_from_vault.assert_not_called()


def test_ensure_fresh_token_refreshes_expired_token():
    client = _make_client()
    client._access_token = "expired_tok"
    client._expires_at = time.time() - 10
    client._refresh_token = "rt"
    client._load_mcp_scoped_token_from_vault = Mock(return_value=False)
    client._try_oauth_refresh = AsyncMock(return_value=True)
    client.authenticate = AsyncMock(return_value="oauth_tok")

    asyncio.run(client._ensure_fresh_token())

    client._try_oauth_refresh.assert_awaited_once()
    client.authenticate.assert_not_awaited()


def test_ensure_fresh_token_reauths_when_refresh_fails():
    client = _make_client()
    client._expires_at = 0.0
    client._refresh_token = None
    client._load_mcp_scoped_token_from_vault = Mock(return_value=False)
    client._try_oauth_refresh = AsyncMock(return_value=False)
    client.authenticate = AsyncMock(return_value="oauth_tok")

    asyncio.run(client._ensure_fresh_token())

    client.authenticate.assert_awaited_once()


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