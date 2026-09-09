"""
Regression tests: executor-level mutation idempotency (Issue 3 — one submission
== one Lead creation).

An identical mutating/destructive tool call (same tool name AND same arguments)
repeated within the dedupe window must not execute a second time — it returns
the previous result instead. This catches a planner/LLM re-issue or double
invocation before it creates a duplicate Salesforce record. Read-only tools are
never deduplicated (re-listing is harmless).

Pure unit tests: no live Salesforce / MCP / LLM.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sfmcp.executor import ToolExecutor, _normalize_value
from sfmcp.registry import ToolRegistry
import sfmcp.executor as executor_module


class _FakeMcpClient:
    """Records every call_tool invocation and returns a fixed create result."""

    def __init__(self, result=None):
        self.calls: list[tuple[str, dict]] = []
        self.result = result if result is not None else json.dumps({"id": "00Qfake"})

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return self.result


def _executor(mcp_client) -> ToolExecutor:
    registry = ToolRegistry()
    registry._load_local_tools()
    return ToolExecutor(mcp_client, registry)


def _prov(arguments):
    """Body-mirrored provenance so these IDEMPOTENCY tests exercise the dedupe
    rules (not provenance semantics — those live in test_mutation_validation)."""
    skip = {"sobject-name", "sobject_name", "sobject", "object", "sobjectName", "objectName"}
    return {
        api: frozenset({_normalize_value(v)})
        for api, v in arguments.items()
        if api not in skip and v is not None and str(v).strip()
    }


@pytest.fixture(autouse=True)
def _clean_mutation_store():
    executor_module._recent_mutations.clear()
    yield
    executor_module._recent_mutations.clear()


def test_identical_create_executes_once():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    args = {"sobject-name": "Lead", "LastName": "Farhan", "Company": "Acme"}

    r1 = asyncio.run(exec_.execute("createSobjectRecord", dict(args), user_provenance=_prov(args)))
    r2 = asyncio.run(exec_.execute("createSobjectRecord", dict(args), user_provenance=_prov(args)))

    assert r1 == r2
    assert len(mcp.calls) == 1, "identical mutation must execute exactly once"


def test_different_args_are_distinct_submissions():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)

    asyncio.run(exec_.execute(
        "createSobjectRecord",
        {"sobject-name": "Lead", "LastName": "One", "Company": "Acme"},
        user_provenance={"LastName": {"one"}, "Company": {"acme"}},
    ))
    asyncio.run(exec_.execute(
        "createSobjectRecord",
        {"sobject-name": "Lead", "LastName": "Two", "Company": "Acme"},
        user_provenance={"LastName": {"two"}, "Company": {"acme"}},
    ))

    assert len(mcp.calls) == 2, "different create args are two real submissions"


def test_arg_key_order_does_not_defeat_dedupe():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)

    first_args = {"sobject-name": "Lead", "LastName": "Ali", "Company": "X"}
    asyncio.run(exec_.execute(
        "createSobjectRecord",
        {"sobject-name": "Lead", "LastName": "Ali", "Company": "X"},
        user_provenance=_prov(first_args),
    ))
    asyncio.run(exec_.execute(
        "createSobjectRecord",
        {"LastName": "Ali", "Company": "X", "sobject-name": "Lead"},
        user_provenance=_prov(first_args),
    ))

    assert len(mcp.calls) == 1


def test_destructive_tool_is_deduplicated():
    mcp = _FakeMcpClient(result=json.dumps({"success": True, "deleted": "00Q000000000001"}))
    exec_ = _executor(mcp)
    args = {"sobject-name": "Lead", "id": "00Q000000000001"}

    asyncio.run(exec_.execute("deleteSobjectRecord", dict(args)))
    asyncio.run(exec_.execute("deleteSobjectRecord", dict(args)))

    assert len(mcp.calls) == 1


def test_read_only_tools_are_never_deduplicated():
    mcp = _FakeMcpClient(result=json.dumps({"totalSize": 0, "records": []}))
    exec_ = _executor(mcp)

    asyncio.run(exec_.execute("soqlQuery", {"q": "SELECT Id FROM Account LIMIT 5"}))
    asyncio.run(exec_.execute("soqlQuery", {"q": "SELECT Id FROM Account LIMIT 5"}))

    assert len(mcp.calls) == 2, "repeated reads must still hit Salesforce"


def test_key_scope_is_per_details():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)

    lead_args = {"sobject-name": "Lead", "LastName": "Zara", "Company": "Acme"}
    contact_args = {"sobject-name": "Contact", "LastName": "Zara"}
    asyncio.run(exec_.execute(
        "createSobjectRecord", lead_args, user_provenance=_prov(lead_args)
    ))
    asyncio.run(exec_.execute(
        "createSobjectRecord", contact_args, user_provenance=_prov(contact_args)
    ))

    assert len(mcp.calls) == 2, "different sobject names are different operations"