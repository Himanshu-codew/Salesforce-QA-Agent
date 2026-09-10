"""
Regression tests: MCP client write-safety (Issue 3 / Issue 5).

A mutating tool (create/update/delete/upload) whose MCP call fails WITHOUT a
definitive signal (connection drop, timeout, transport error) has an UNKNOWN
outcome — Salesforce may or may not have executed it. Re-invoking it then (via
the MCP reconnect-retry or the REST fallback) duplicates the record. These tests
pin the fix: uncertain writes raise a controlled error and are NEVER
auto-re-invoked, while reads keep the reconnect-then-REST fallback and 401
auth-retry behavior that the production REST fallback depends on.

Pure unit tests (mocked MCP session / internals).
"""

import asyncio
import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sfmcp.client import SalesforceMCPClient


class _OkResult:
    isError = False
    structuredContent = {"ok": True}
    content = []


class _ErrResult:
    isError = True
    structuredContent = None
    content = [SimpleNamespace(text='{"error": "boom"}')]


class _Http401Error(Exception):
    response = SimpleNamespace(status_code=401)


def _make_client(session_call, mcp_required=False):
    client = SalesforceMCPClient.__new__(SalesforceMCPClient)
    client._access_token = "tok"
    client._name_map = {}
    client.mcp_required = mcp_required
    client.mcp_transport = "MCP"
    client._session = SimpleNamespace(call_tool=session_call)
    client._ensure_fresh_token = AsyncMock()
    client._ensure_connected = AsyncMock()
    client._close_mcp_session = AsyncMock()
    client._try_oauth_refresh = AsyncMock(return_value=True)
    client._fallback_rest_api = AsyncMock(return_value={"fallback": "ok"})
    return client


# ─────────────────────────────────────────────
# Uncertain write failures: must NOT re-invoke
# ─────────────────────────────────────────────

def test_write_uncertain_failure_raises_without_retry_or_fallback():
    session_call = AsyncMock(side_effect=httpx.ConnectError("connection dropped"))
    client = _make_client(session_call)

    with pytest.raises(RuntimeError, match="unknown outcome"):
        asyncio.run(client.call_tool(
            "createSobjectRecord", {"sobject-name": "Lead", "LastName": "Test"}
        ))

    assert session_call.call_count == 1, "a mutation must never be invoked twice"
    client._fallback_rest_api.assert_not_called()
    client._close_mcp_session.assert_not_called()


def test_update_uncertain_failure_raises_without_retry_or_fallback():
    session_call = AsyncMock(side_effect=RuntimeError("stream disconnected mid-request"))
    client = _make_client(session_call)

    with pytest.raises(RuntimeError, match="unknown outcome"):
        asyncio.run(client.call_tool(
            "updateSobjectRecord", {"sobject-name": "Lead", "id": "00Qx", "body": {"Company": "NewCo"}}
        ))

    assert session_call.call_count == 1
    client._fallback_rest_api.assert_not_called()


# ─────────────────────────────────────────────
# Definitive write failures: REST fallback safe
# ─────────────────────────────────────────────

def test_write_definitive_tool_error_can_fall_back_to_rest():
    # isError means the MASTER tool execution failed — nothing was created, so
    # the REST fallback is safe and the Issue-5 fallback keeps working.
    session_call = AsyncMock(return_value=_ErrResult())
    client = _make_client(session_call)

    out = asyncio.run(client.call_tool(
        "createSobjectRecord", {"sobject-name": "Lead", "LastName": "Test"}
    ))

    assert out == {"fallback": "ok"}
    assert session_call.call_count == 1
    client._fallback_rest_api.assert_awaited_once()


# ─────────────────────────────────────────────
# Reads keep reconnect + REST fallback
# ─────────────────────────────────────────────

def test_read_transient_failure_reconnects_once_then_succeeds():
    session_call = AsyncMock(side_effect=[httpx.ConnectError("drop"), _OkResult()])
    client = _make_client(session_call)

    out = asyncio.run(client.call_tool("soqlQuery", {"q": "SELECT Id FROM Account LIMIT 5"}))

    assert out == {"ok": True}
    assert session_call.call_count == 2
    client._fallback_rest_api.assert_not_called()


def test_read_transient_failure_falls_back_to_rest_after_one_reconnect():
    session_call = AsyncMock(side_effect=[httpx.ConnectError("drop"), httpx.ConnectError("drop")])
    client = _make_client(session_call)

    out = asyncio.run(client.call_tool("soqlQuery", {"q": "SELECT Id FROM Account LIMIT 5"}))

    assert out == {"fallback": "ok"}
    assert session_call.call_count == 2
    client._fallback_rest_api.assert_awaited_once()


def test_read_401_refreshes_and_retries_mcp_once():
    session_call = AsyncMock(side_effect=[_Http401Error(), _OkResult()])
    client = _make_client(session_call)

    out = asyncio.run(client.call_tool("soqlQuery", {"q": "SELECT Id FROM Account LIMIT 5"}))

    assert out == {"ok": True}
    assert session_call.call_count == 2
    client._try_oauth_refresh.assert_awaited_once()
    client._fallback_rest_api.assert_not_called()


# ─────────────────────────────────────────────
# Writes: 401 auth rejection is safe to refresh+retry
# ─────────────────────────────────────────────

def test_write_auth_rejection_refreshes_and_retries_safely():
    # A 401 means the request was refused before execution — refreshing and
    # retrying a mutation is therefore safe (no duplicate possible).
    session_call = AsyncMock(side_effect=[_Http401Error(), _OkResult()])
    client = _make_client(session_call)

    out = asyncio.run(client.call_tool(
        "createSobjectRecord", {"sobject-name": "Lead", "LastName": "Test"}
    ))

    assert out == {"ok": True}
    assert session_call.call_count == 2
    client._try_oauth_refresh.assert_awaited_once()
    client._fallback_rest_api.assert_not_called()