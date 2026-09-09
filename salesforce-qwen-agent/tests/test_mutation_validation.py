"""
Production Mutation Safety Validation tests (approved specification, section K1).

Proves FAIL-CLOSED behavior:
- vague "create a Lead/Contact/Account/Opportunity/custom object" never reaches MCP
  or REST.
- empty/null/whitespace required fields are rejected.
- valid mutations still execute exactly once (and stay idempotent).
- custom objects validate against live Describe metadata; Describe failure fails
  closed.
- validation failures are never cached as successful idempotent mutations.
- after validation failure there is NO second tool-calling LLM invocation and the
  synthesizer (no tools) receives the validation result and asks for input.
- fabricated defaults are never inserted.
- delete confirmation and existing MCP write-safety remain intact.

Pure unit tests: no live Salesforce / MCP / LLM.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.mutation_validation import (
    OBJECT_REQUIRED_FIELDS,
    is_valid_salesforce_id,
    validate_mutation_fields,
)
from sfmcp.executor import (
    ToolExecutor,
    _extract_user_provided_fields,
    _normalize_value,
)
from sfmcp.registry import ToolRegistry
import sfmcp.executor as executor_module


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

_VALID_LEAD_ID = "00Q000000000001"
_VALID_CONTACT_ID = "003000000000001"
_VALID_ACCOUNT_ID = "001000000000001"
_VALID_RECORD_IDS = [_VALID_LEAD_ID, _VALID_CONTACT_ID, _VALID_ACCOUNT_ID]


# ─────────────────────────────────────────────────────────────
# Per-object Describe metadata returned by the fake MCP client.
# Mirrors real Salesforce API requirements so the resolver-first
# validation (describe for BOTH standard and custom objects) can be
# exercised deterministically in tests.
# ─────────────────────────────────────────────────────────────
_DESCRIBE_FIELDS: dict[str, list[tuple[str, str]]] = {
    "Lead":        [("LastName", "Last Name"), ("Company", "Company Name")],
    "Contact":     [("LastName", "Last Name")],
    "Account":     [("Name", "Account Name")],
    "Opportunity": [("Name", "Opportunity Name"), ("StageName", "Stage"),
                    ("CloseDate", "Close Date")],
    "Case":        [],
    "Task":        [("Subject", "Subject")],
    "CustomTenant__c": [("Tenant__c", "Tenant"), ("Name", "Name")],
}


class _FakeMcpClient:
    """Records every call_tool invocation and returns a fixed create result.

    describe_required_fields is OBJECT-AWARE (resolver-first): per-object
    required fields mirror Salesforce metadata. Describe returns None for an
    object with no entry (forcing the static-registry fallback / fail-closed)."""

    def __init__(self, result=None):
        self.calls: list[tuple[str, dict]] = []
        self.result = result if result is not None else json.dumps({"id": "00Qfake"})
        self.describe_calls: list[str] = []
        self.describe_required: list[tuple[str, str]] | None = None
        self.describe_overrides: dict[str, list[tuple[str, str]] | None] = {}

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return self.result

    async def describe_required_fields(self, sobject_name):
        self.describe_calls.append(sobject_name)
        if sobject_name in self.describe_overrides:
            return self.describe_overrides[sobject_name]
        return _DESCRIBE_FIELDS.get(sobject_name)


class _FakeMcpClientNoDescribe:
    """MCP client whose describe resolution FAILS (schema cannot be established)."""

    def __init__(self, result=None):
        self.calls: list[tuple[str, dict]] = []
        self.result = result if result is not None else json.dumps({"id": "00Qfake"})
        self.describe_calls: list[str] = []

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return self.result

    async def describe_required_fields(self, sobject_name):
        self.describe_calls.append(sobject_name)
        return None


def _executor(mcp_client) -> ToolExecutor:
    registry = ToolRegistry()
    registry._load_local_tools()
    return ToolExecutor(mcp_client, registry)


async def _exec_with_prov(exec_, tool_name, arguments):
    """Run a tool through the executor passing body-mirrored provenance.

    These unit tests validate the PRESENCE and STRUCTURAL rules, so the body's
    own values are modeled as user-supplied (provenance = the body). Provenance
    semantics themselves are covered separately by the A–P matrix, the
    fail-closed-on-None case, and the orchestrator-level tests."""
    body = arguments.get("body") if isinstance(arguments.get("body"), dict) else {}
    prov = {
        str(api): frozenset({_normalize_value(v)})
        for api, v in body.items()
        if v is not None and str(v).strip()
    }
    return await exec_.execute(tool_name, arguments, user_provenance=prov)


@pytest.fixture(autouse=True)
def _clean_mutation_store():
    executor_module._recent_mutations.clear()
    yield
    executor_module._recent_mutations.clear()


def _parse(result: str) -> dict:
    return json.loads(result)


def _assert_validation_error(result: str, tool: str, missing_api: list[str] | None = None):
    parsed = _parse(result)
    assert parsed.get("validation_error") is True
    assert parsed.get("retry_allowed") is False
    assert parsed.get("requires_user_input") is True
    assert "tool" in parsed
    assert "error" in parsed
    assert "suggestion" in parsed
    assert "sobject_name" in parsed
    assert "missing_fields" in parsed
    assert "missing_fields_human" in parsed
    if missing_api is not None:
        assert set(parsed.get("missing_fields", [])).issubset(set(missing_api))


# ─────────────────────────────────────────────────────────────
# createSobjectRecord — vague requests must never reach Salesforce
# ─────────────────────────────────────────────────────────────

def test_create_lead_empty_body_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Lead", "body": {}}))
    _assert_validation_error(res, "createSobjectRecord", ["LastName", "Company"])
    assert mcp.calls == [], "vague create must never call MCP"


def test_create_lead_vague_request_never_reaches_salesforce():
    # "create a lead" with no fields at all.
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Lead"}))
    _assert_validation_error(res, "createSobjectRecord", ["LastName", "Company"])
    assert mcp.calls == []


def test_create_lead_only_first_name_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "createSobjectRecord", {"sobject-name": "Lead", "body": {"FirstName": "John"}}
    ))
    _assert_validation_error(res, "createSobjectRecord", ["LastName", "Company"])
    assert mcp.calls == []


def test_create_lead_null_company_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "createSobjectRecord", {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": None}}
    ))
    _assert_validation_error(res, "createSobjectRecord", ["Company"])
    assert mcp.calls == []


def test_create_lead_whitespace_last_name_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "createSobjectRecord", {"sobject-name": "Lead", "body": {"LastName": "   ", "Company": "Acme"}}
    ))
    _assert_validation_error(res, "createSobjectRecord", ["LastName"])
    assert mcp.calls == []


def test_create_lead_complete_accepted_executes_once():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    args = {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": "Acme", "FirstName": "John"}}
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", dict(args)))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


# ─────────────────────────────────────────────────────────────
# Contact / Account / Opportunity / Case / Task validation
# ─────────────────────────────────────────────────────────────

def test_create_contact_empty_body_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Contact", "body": {}}))
    _assert_validation_error(res, "createSobjectRecord", ["LastName"])
    assert mcp.calls == []


def test_create_contact_vague_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Contact", "body": {"FirstName": "Jane"}}))
    _assert_validation_error(res, "createSobjectRecord", ["LastName"])
    assert mcp.calls == []


def test_create_account_missing_name_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Account", "body": {}}))
    _assert_validation_error(res, "createSobjectRecord", ["Name"])
    assert mcp.calls == []


def test_create_account_vague_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Account"}))
    _assert_validation_error(res, "createSobjectRecord", ["Name"])
    assert mcp.calls == []


def test_create_account_complete_accepted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Account", "body": {"Name": "Acme"}}))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


def test_create_opportunity_vague_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Opportunity", "body": {}}))
    _assert_validation_error(res, "createSobjectRecord", ["Name", "StageName", "CloseDate"])
    assert mcp.calls == []


def test_create_opportunity_incomplete_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "createSobjectRecord", {"sobject-name": "Opportunity", "body": {"Name": "Deal"}}
    ))
    _assert_validation_error(res, "createSobjectRecord", ["StageName", "CloseDate"])
    assert mcp.calls == []


def test_create_opportunity_complete_accepted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "createSobjectRecord",
        {"sobject-name": "Opportunity", "body": {"Name": "Deal", "StageName": "Prospecting", "CloseDate": "2026-12-31"}},
    ))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


def test_create_case_non_empty_body_allowed_when_no_required_fields():
    # Case has no field Salesforce requires on create (Subject optional, Status
    # defaulted) — a non-empty case body must NOT be over-rejected (#5).
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "createSobjectRecord", {"sobject-name": "Case", "body": {"Description": "Login issue"}}
    ))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


def test_create_case_empty_body_rejected():
    # Even with no required fields, a ZERO-field create remains vague and fails closed.
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Case", "body": {}}))
    _assert_validation_error(res, "createSobjectRecord")
    assert mcp.calls == [], "empty-body create must never reach Salesforce"


def test_create_task_incomplete_rejected():
    # Task requires Subject; Status is DEFAULTED on create so it is not required (#5).
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "createSobjectRecord", {"sobject-name": "Task", "body": {"Description": "Follow up"}}
    ))
    _assert_validation_error(res, "createSobjectRecord", ["Subject"])
    assert mcp.calls == []


def test_create_task_with_subject_accepted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "createSobjectRecord", {"sobject-name": "Task", "body": {"Subject": "Follow up"}}
    ))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


# ─────────────────────────────────────────────────────────────
# Structural argument errors
# ─────────────────────────────────────────────────────────────

def test_create_no_sobject_name_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"body": {"LastName": "Smith"}}))
    _assert_validation_error(res, "createSobjectRecord")
    assert mcp.calls == []


def test_create_body_not_dict_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Lead", "body": "not-a-dict"}))
    _assert_validation_error(res, "createSobjectRecord")
    assert mcp.calls == []


# ─────────────────────────────────────────────────────────────
# Custom objects use live Describe; Describe failure fails closed
# ─────────────────────────────────────────────────────────────

def test_create_custom_object_vague_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "CustomTenant__c", "body": {}}))
    _assert_validation_error(res, "createSobjectRecord", ["Tenant__c", "Name"])
    assert mcp.calls == []
    assert mcp.describe_calls == ["CustomTenant__c"], "custom object must be described"


def test_create_custom_object_complete_accepted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "createSobjectRecord",
        {"sobject-name": "CustomTenant__c", "body": {"Tenant__c": "Acme", "Name": "Main"}},
    ))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1
    assert mcp.describe_calls == ["CustomTenant__c"]


def test_create_custom_object_describe_fails_closed():
    mcp = _FakeMcpClientNoDescribe()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "CustomTenant__c", "body": {"Name": "Acme"}}))
    _assert_validation_error(res, "createSobjectRecord")
    assert mcp.calls == [], "schema-unresolved mutation must fail closed (no MCP call)"
    assert mcp.describe_calls == ["CustomTenant__c"]


# ─────────────────────────────────────────────────────────────
# Resolver-first validation for STANDARD objects (#5)
# ─────────────────────────────────────────────────────────────

def test_resolver_first_uses_describe_for_standard_objects():
    # A per-org custom REQUIRED field on a standard object (Lead) must be honored
    # via Describe even though Lead is in the static registry — the live metadata
    # is authoritative, never blindly overridden by the hard-coded registry.
    mcp = _FakeMcpClient()
    mcp.describe_overrides["Lead"] = [
        ("LastName", "Last Name"),
        ("Company", "Company Name"),
        ("CustomLeadFlag__c", "Custom Lead Flag"),
    ]
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "createSobjectRecord", {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": "Acme"}}
    ))
    _assert_validation_error(res, "createSobjectRecord", ["CustomLeadFlag__c"])
    assert mcp.calls == [], "missing custom required field must block the create"
    assert mcp.describe_calls == ["Lead"]

    ok = asyncio.run(_exec_with_prov(exec_,
        "createSobjectRecord", {"sobject-name": "Lead",
                                "body": {"LastName": "Smith", "Company": "Acme",
                                         "CustomLeadFlag__c": "Required"}}
    ))
    assert "validation_error" not in _parse(ok)
    assert len(mcp.calls) == 1


def test_static_fallback_when_describe_unavailable_for_standard_object():
    # When Describe is unavailable (returns None), the static registry is the
    # zero-I/O fallback: a vague Lead is still rejected and a valid one executes.
    mcp = _FakeMcpClient()
    mcp.describe_overrides["Lead"] = None
    exec_ = _executor(mcp)

    bad = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Lead", "body": {}}))
    _assert_validation_error(bad, "createSobjectRecord", ["LastName", "Company"])
    assert mcp.calls == [], "static fallback must still reject a vague lead"

    good = asyncio.run(_exec_with_prov(exec_,
        "createSobjectRecord",
        {"sobject-name": "Lead", "body": {"LastName": "Sharma", "Company": "Tech Solutions"}},
    ))
    assert "validation_error" not in _parse(good)
    assert len(mcp.calls) == 1


def test_unknown_object_without_describe_falls_back_then_fails_closed():
    # Object is in neither Describe nor the static registry -> fail closed.
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Mystery__c", "body": {"Name": "X"}}))
    _assert_validation_error(res, "createSobjectRecord")
    assert mcp.calls == [], "unknown object with unresolved schema must fail closed"


# ─────────────────────────────────────────────────────────────
# Public executor.validate_mutation (pre-flight API used by the
# agent loops for #8 stop-on-validation-failure)
# ─────────────────────────────────────────────────────────────

def test_executor_validate_mutation_public_preflight():
    # validate_mutation() is the public fail-closed pre-flight used by agent.py to
    # stop remaining mutations after the first validation failure (#8).
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)

    bad = asyncio.run(exec_.validate_mutation(
        "createSobjectRecord", {"sobject-name": "Lead", "body": {}}
    ))
    assert bad is not None
    assert _parse(bad)["validation_error"] is True

    good = asyncio.run(exec_.validate_mutation(
        "createSobjectRecord",
        {"sobject-name": "Lead", "body": {"LastName": "Sharma", "Company": "Tech Solutions"}},
        user_provenance=_extract_user_provided_fields(
            "Last Name: Sharma, Company: Tech Solutions"
        ),
    ))
    assert good is None

    # Reads are never validated (pre-flight no-op).
    assert asyncio.run(exec_.validate_mutation("soqlQuery", {"q": "SELECT Id FROM Account LIMIT 1"})) is None
    assert mcp.calls == [], "pre-flight validation must never call Salesforce"


# ─────────────────────────────────────────────────────────────
# update / related / upload / delete validation
# ─────────────────────────────────────────────────────────────

def test_update_valid_accepted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "updateSobjectRecord",
        {"sobject-name": "Lead", "id": _VALID_LEAD_ID, "body": {"Status": "Working"}},
    ))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


def test_update_empty_id_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "updateSobjectRecord", {"sobject-name": "Lead", "id": "", "body": {"Status": "Working"}}
    ))
    _assert_validation_error(res, "updateSobjectRecord")
    assert mcp.calls == []


def test_update_invalid_id_format_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "updateSobjectRecord", {"sobject-name": "Lead", "id": "not-an-id", "body": {"Status": "Working"}}
    ))
    _assert_validation_error(res, "updateSobjectRecord")
    assert mcp.calls == []


def test_update_empty_body_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "updateSobjectRecord", {"sobject-name": "Lead", "id": _VALID_LEAD_ID, "body": {}}
    ))
    _assert_validation_error(res, "updateSobjectRecord")
    assert mcp.calls == []


def test_update_related_valid_accepted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "updateRelatedRecord",
        {"id": _VALID_CONTACT_ID, "relationship-path": "Account", "body": {"Phone": "555-0100"}},
    ))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


def test_update_related_missing_rel_path_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "updateRelatedRecord",
        {"id": _VALID_CONTACT_ID, "body": {"Phone": "555-0100"}},
    ))
    _assert_validation_error(res, "updateRelatedRecord")
    assert mcp.calls == []


def test_delete_valid_id_accepted():
    mcp = _FakeMcpClient(result=json.dumps({"success": True}))
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"deleteSobjectRecord", {"sobject-name": "Lead", "id": _VALID_LEAD_ID}))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


def test_delete_invalid_id_rejected():
    mcp = _FakeMcpClient(result=json.dumps({"success": True}))
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"deleteSobjectRecord", {"sobject-name": "Lead", "id": "bad"}))
    _assert_validation_error(res, "deleteSobjectRecord")
    assert mcp.calls == []


def test_upload_valid_accepted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "uploadRecordAttachment",
        {"record_id": _VALID_ACCOUNT_ID, "file_name": "test.pdf", "file_content_base64": "dGVzdA=="},
    ))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


def test_upload_missing_record_id_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "uploadRecordAttachment", {"file_name": "test.pdf", "file_content_base64": "dGVzdA=="}
    ))
    _assert_validation_error(res, "uploadRecordAttachment")
    assert mcp.calls == []


def test_upload_missing_file_content_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,
        "uploadRecordAttachment", {"record_id": _VALID_ACCOUNT_ID, "file_name": "test.pdf"}
    ))
    _assert_validation_error(res, "uploadRecordAttachment")
    assert mcp.calls == []


# ─────────────────────────────────────────────────────────────
# Idempotency interaction
# ─────────────────────────────────────────────────────────────

def test_validation_error_not_cached_as_success():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    bad_args = {"sobject-name": "Lead", "body": {}}

    r1 = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", dict(bad_args)))
    r2 = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", dict(bad_args)))

    assert "validation_error" in _parse(r1)
    assert "validation_error" in _parse(r2)
    assert mcp.calls == [], "rejected mutations must never reach Salesforce"
    # And a correct submission is NOT falsely deduplicated against the rejected one.
    good_args = {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": "Acme"}}
    r3 = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", dict(good_args)))
    assert "validation_error" not in _parse(r3)
    assert len(mcp.calls) == 1, "corrected submission must execute"


def test_valid_mutation_remains_idempotent_after_validation_pass():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    args = {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": "Acme"}}

    r1 = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", dict(args)))
    r2 = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", dict(args)))

    assert r1 == r2
    assert len(mcp.calls) == 1, "valid identical mutation must execute exactly once"


# ─────────────────────────────────────────────────────────────
# Read-only tools are unaffected
# ─────────────────────────────────────────────────────────────

def test_read_only_tools_unaffected():
    mcp = _FakeMcpClient(result=json.dumps({"totalSize": 0, "records": []}))
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"soqlQuery", {"q": "SELECT Id FROM Account LIMIT 5"}))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


# ─────────────────────────────────────────────────────────────
# Error envelope contract
# ─────────────────────────────────────────────────────────────

def test_error_envelope_full_fields_present():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Lead", "body": {}}))
    parsed = _parse(res)
    for key in ("validation_error", "retry_allowed", "requires_user_input", "missing_fields",
                "missing_fields_human", "sobject_name", "tool", "error", "suggestion"):
        assert key in parsed, f"missing field {key}"


# ─────────────────────────────────────────────────────────────
# No fabricated/default values ever inserted
# ─────────────────────────────────────────────────────────────

def test_fabricated_defaults_never_submitted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    # A vague lead is rejected outright — nothing reaches MCP.
    asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Lead"}))
    assert mcp.calls == []
    # A valid lead forwards EXACTLY the user-provided body — no invented defaults.
    asyncio.run(_exec_with_prov(exec_,"createSobjectRecord", {"sobject-name": "Lead",
                                                       "body": {"LastName": "Smith", "Company": "Acme"}}))
    assert len(mcp.calls) == 1
    submitted_body = mcp.calls[0][1].get("body", {})
    assert "Unknown" not in str(submitted_body)
    assert "Individual" not in str(submitted_body)
    assert "New Account" not in str(submitted_body)
    assert "New Opportunity" not in str(submitted_body)
    assert submitted_body.get("LastName") == "Smith"
    assert submitted_body.get("Company") == "Acme"


# ─────────────────────────────────────────────────────────────
# Salesforce ID format unit checks
# ─────────────────────────────────────────────────────────────

def test_salesforce_id_format_validity():
    assert is_valid_salesforce_id("00Q000000000001")
    assert is_valid_salesforce_id("001000000000001")
    assert is_valid_salesforce_id("003000000000001")
    assert is_valid_salesforce_id("00Q000000000001AAA")
    assert not is_valid_salesforce_id("")
    assert not is_valid_salesforce_id("00Qdel")
    assert not is_valid_salesforce_id("not-an-id")
    assert not is_valid_salesforce_id(1234)


# ─────────────────────────────────────────────────────────────
# Static registry sanity
# ─────────────────────────────────────────────────────────────

def test_static_registry_has_expected_standard_objects():
    assert "lead" in OBJECT_REQUIRED_FIELDS
    assert "contact" in OBJECT_REQUIRED_FIELDS
    assert "account" in OBJECT_REQUIRED_FIELDS
    assert "opportunity" in OBJECT_REQUIRED_FIELDS
    assert "case" in OBJECT_REQUIRED_FIELDS
    assert "task" in OBJECT_REQUIRED_FIELDS


def test_validate_mutation_fields_direct_unit():
    # Direct module-level validation (no executor) — proves the gate logic itself.
    err = validate_mutation_fields("createSobjectRecord", {"sobject-name": "Lead", "body": {}}, None)
    assert err is not None
    parsed = json.loads(err)
    assert parsed["validation_error"] is True
    assert set(parsed["missing_fields"]) == {"LastName", "Company"}

    ok = validate_mutation_fields("createSobjectRecord", {"sobject-name": "Lead", "body": {"LastName": "A", "Company": "B"}}, None)
    assert ok is None

    ok2 = validate_mutation_fields("soqlQuery", {"q": "SELECT Id FROM Account"}, None)
    assert ok2 is None, "read-only tools are not validated"


# ─────────────────────────────────────────────────────────────
# Orchestrator-level: no automatic mutation retry + synthesizer handoff
# ─────────────────────────────────────────────────────────────

def test_validation_failure_does_not_reinvoke_tool_calling_llm():
    """
    The ActionAgent (chat_with_tools) must be invoked exactly once. After a
    validation failure the result flows to the no-tool synthesizer — the
    tool-calling model is never re-invoked with the failure, so no second
    mutation attempt can occur.
    """
    from agent.multi_agent import Orchestrator

    class _OneShotToolLLM:
        """Emit a single vague create tool-call, then hand off; count invocations."""

        def __init__(self):
            self.tool_llm_calls = 0
            self.synthesis_user_msg = ""
            self._tool_calls = [
                {"id": "t1", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Lead", "body": {}}},
            ]

        async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
            self.tool_llm_calls += 1
            return {"content": "", "tool_calls": list(self._tool_calls), "finish_reason": "tool_calls"}

        async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
            for m in messages:
                if m.get("role") == "user":
                    self.synthesis_user_msg = m.get("content", "")
            return "Please provide the missing required fields: Last Name and Company."

    llm = _OneShotToolLLM()
    mcp = _FakeMcpClient()
    exec_ = ToolExecutor(mcp, exec_registry())
    orch = Orchestrator(llm=llm, executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _SafePlanner()
    orch._generate_plan = AsyncMockMock(return_value=[{
        "task_id": 1, "description": "create lead", "agent": "ActionAgent", "depends_on": [],
    }])
    orch._get_relevant_tools_or_fallback = AsyncMockMock(return_value=[])

    events = _run_orchestrator(orch, "Create a lead")

    # The tool-calling LLM emitted its calls once and is never re-invoked.
    assert llm.tool_llm_calls == 1, "tool-calling LLM must be invoked exactly once"
    # The vague create never reached Salesforce.
    assert mcp.calls == []
    # The validation error was handed to the (no-tool) synthesizer.
    assert "validation_error" in llm.synthesis_user_msg or "Please provide" in llm.synthesis_user_msg or _has_validation_error(events)
    # No second mutation tool call event after the failure.
    tool_call_events = [e for e in events if e.get("type") == "tool_call"]
    assert len(tool_call_events) == 1, "no second tool call after validation failure"


def test_validation_failure_returns_deterministic_ask_not_fatal_error():
    """validation_error=true results are normal tool results, NOT AgentError, and
    the vague-create ask is rendered deterministically — the synthesizer LLM is
    not needed for the wording."""
    from agent.multi_agent import Orchestrator

    class _LLM:
        def __init__(self):
            self.tool_calls = [
                {"id": "t1", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Lead", "body": {}}},
            ]
            self.synthesis_user_msg = ""
            self.chat_calls = 0

        async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
            return {"content": "", "tool_calls": list(self.tool_calls), "finish_reason": "tool_calls"}

        async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
            self.chat_calls += 1
            for m in messages:
                if m.get("role") == "user":
                    self.synthesis_user_msg = m.get("content", "")
            return "Please provide Last Name and Company."

    llm = _LLM()
    mcp = _FakeMcpClient()
    exec_ = ToolExecutor(mcp, exec_registry())
    orch = Orchestrator(llm=llm, executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _SafePlanner()
    orch._generate_plan = AsyncMockMock(return_value=[{
        "task_id": 1, "description": "create lead", "agent": "ActionAgent", "depends_on": [],
    }])
    orch._get_relevant_tools_or_fallback = AsyncMockMock(return_value=[])

    events = _run_orchestrator(orch, "Create a lead")

    # No SALESFORCE_FAILED fatal error event for a validation failure.
    errors = [e for e in events if e.get("type") == "error"]
    assert not errors, "validation failure must not be a terminal error"
    # The user-facing response is the deterministic ask listing the required fields.
    responses = [e for e in events if e.get("type") == "response"]
    assert responses, "a user-facing response must be produced"
    text = responses[-1]["data"]
    assert "Sure. To create the Lead, please provide:" in text
    assert "- Last Name" in text and "- Company Name" in text
    # The ask is deterministic: the synthesizer LLM is never invoked.
    assert llm.chat_calls == 0, "no LLM-written ask for a blocked create"


# ─────────────────────────────────────────────────────────────
# #8 MULTI-TOOL-CALL SAFETY: one validation failure stops the remaining
# mutations in the same response — even when the later ones are valid.
# ─────────────────────────────────────────────────────────────

def test_multi_mutation_validation_failure_stops_remaining_calls():
    """The ActionAgent returns THREE create calls in one response: the FIRST
    (vague Lead) fails validation, so the Account+Opportunity creates that follow
    MUST NOT execute — fail-closed for the whole batch."""
    from agent.multi_agent import Orchestrator

    class _LLM:
        def __init__(self):
            self.tool_calls = [
                {"id": "t1", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Lead", "body": {}}},
                {"id": "t2", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Account", "body": {"Name": "Acme"}}},
                {"id": "t3", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Opportunity",
                               "body": {"Name": "Deal", "StageName": "Prospecting",
                                        "CloseDate": "2026-12-31"}}},
            ]
            self.tool_llm_calls = 0
            self.chat_calls = 0

        async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
            self.tool_llm_calls += 1
            return {"content": "", "tool_calls": list(self.tool_calls), "finish_reason": "tool_calls"}

        async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
            self.chat_calls += 1
            return "Please provide the missing required fields (Last Name and Company Name)."

    llm = _LLM()
    mcp = _FakeMcpClient()
    exec_ = ToolExecutor(mcp, exec_registry())
    orch = Orchestrator(llm=llm, executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _SafePlanner()
    orch._generate_plan = AsyncMockMock(return_value=[{
        "task_id": 1, "description": "create records", "agent": "ActionAgent", "depends_on": [],
    }])
    orch._get_relevant_tools_or_fallback = AsyncMockMock(return_value=[])

    events = _run_orchestrator(orch, "Create a lead, an account, and an opportunity")

    # ZERO mutations from the response reached Salesforce — not even the (valid)
    # Account/Opportunity creates that followed the failing Lead.
    assert mcp.calls == [], "a validation failure must stop ALL mutations in the response"
    created = [c for c in mcp.calls if c[0] == "createSobjectRecord"]
    assert created == [], "later valid mutations in a failed batch must not run"
    # EVERY non-executed mutation yields a validation/blocked envelope to synthesis.
    tool_results = [e for e in events if e.get("type") == "tool_result"]
    assert len(tool_results) == 3, "all three mutations must return failure envelopes"
    for e in tool_results:
        env = json.loads(e["data"]["result"])
        assert env["validation_error"] is True
        assert env.get("tool") == "createSobjectRecord"
        assert env["retry_allowed"] is False
    assert _has_validation_error(events)
    # The tool-calling LLM is still invoked exactly once (no automatic retry).
    assert llm.tool_llm_calls == 1


def test_multi_mutation_first_valid_then_invalid_blocks_everything():
    """REGRESSION (#8 all-or-nothing): FIRST mutation VALID (Account) + SECOND
    invalid/vague (Lead). The valid Account create must NOT execute either —
    if ANY mutation in the response fails validation, ZERO mutations run."""
    from agent.multi_agent import Orchestrator

    class _LLM:
        def __init__(self):
            self.tool_calls = [
                {"id": "t1", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Account", "body": {"Name": "Acme"}}},
                {"id": "t2", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Lead", "body": {}}},
            ]
            self.tool_llm_calls = 0

        async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
            self.tool_llm_calls += 1
            return {"content": "", "tool_calls": list(self.tool_calls), "finish_reason": "tool_calls"}

        async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
            return "To create the lead I need Last Name and Company Name. Please provide them."

    llm = _LLM()
    mcp = _FakeMcpClient()
    exec_ = ToolExecutor(mcp, exec_registry())
    orch = Orchestrator(llm=llm, executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _SafePlanner()
    orch._generate_plan = AsyncMockMock(return_value=[{
        "task_id": 1, "description": "create records", "agent": "ActionAgent", "depends_on": [],
    }])
    orch._get_relevant_tools_or_fallback = AsyncMockMock(return_value=[])

    events = _run_orchestrator(orch, "Create an account and a lead")

    # The valid Account create executes ZERO times because the batch failed.
    assert mcp.calls == [], "any validation failure must block even an earlier valid mutation"
    # Both mutations returned failure envelopes to synthesis.
    tool_results = [e for e in events if e.get("type") == "tool_result"]
    assert len(tool_results) == 2
    for e in tool_results:
        env = json.loads(e["data"]["result"])
        assert env["validation_error"] is True
        assert env["retry_allowed"] is False
    # The LLM asks the user for the missing Lead fields; it is not auto-retried.
    responses = [e for e in events if e.get("type") == "response"]
    assert responses and "Last Name" in responses[-1]["data"]
    assert llm.tool_llm_calls == 1, "no automatic tool-calling retry after failure"


def test_multi_mutation_all_valid_executes_all_in_order():
    """Success case: when every mutation passes validation, ALL run (in order)."""
    from agent.multi_agent import Orchestrator

    class _LLM:
        def __init__(self):
            self.tool_calls = [
                {"id": "t1", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Lead",
                               "body": {"LastName": "Sharma", "Company": "Tech Solutions"}}},
                {"id": "t2", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Account", "body": {"Name": "Acme"}}},
            ]

        async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
            return {"content": "", "tool_calls": list(self.tool_calls), "finish_reason": "tool_calls"}

        async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
            return "Created the lead and the account."

    llm = _LLM()
    mcp = _FakeMcpClient()
    exec_ = ToolExecutor(mcp, exec_registry())
    orch = Orchestrator(llm=llm, executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _SafePlanner()
    orch._generate_plan = AsyncMockMock(return_value=[{
        "task_id": 1, "description": "create records", "agent": "ActionAgent", "depends_on": [],
    }])
    orch._get_relevant_tools_or_fallback = AsyncMockMock(return_value=[])

    events = _run_orchestrator(orch,
        "Create a lead. Last Name: Sharma, Company: Tech Solutions, and create an account with Name: Acme")

    assert len(mcp.calls) == 2
    assert mcp.calls[0][1].get("body", {}).get("LastName") == "Sharma"
    assert mcp.calls[1][1].get("body", {}).get("Name") == "Acme"


# ─────────────────────────────────────────────────────────────
# TEST B: a VALID create request works end-to-end (Orchestrator).
# ─────────────────────────────────────────────────────────────

def test_orchestrator_valid_create_executes_once():
    """'Create a lead. Last Name: Sharma, Company: Tech Solutions.' must reach
    Salesforce exactly once and forward the exact user-provided body."""
    from agent.multi_agent import Orchestrator

    class _LLM:
        def __init__(self):
            self.tool_calls = [
                {"id": "t1", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Lead",
                               "body": {"LastName": "Sharma", "Company": "Tech Solutions"}}},
            ]
            self.chat_calls = 0

        async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
            return {"content": "", "tool_calls": list(self.tool_calls), "finish_reason": "tool_calls"}

        async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
            self.chat_calls += 1
            return "Created the lead for Sharma at Tech Solutions."

    llm = _LLM()
    mcp = _FakeMcpClient()
    exec_ = ToolExecutor(mcp, exec_registry())
    orch = Orchestrator(llm=llm, executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _SafePlanner()
    orch._generate_plan = AsyncMockMock(return_value=[{
        "task_id": 1, "description": "create lead", "agent": "ActionAgent", "depends_on": [],
    }])
    orch._get_relevant_tools_or_fallback = AsyncMockMock(return_value=[])

    events = _run_orchestrator(orch, "Create a lead. Last Name: Sharma, Company: Tech Solutions.")

    assert len(mcp.calls) == 1
    assert mcp.calls[0][0] == "createSobjectRecord"
    assert mcp.calls[0][1]["body"] == {"LastName": "Sharma", "Company": "Tech Solutions"}
    responses = [e for e in events if e.get("type") == "response"]
    assert responses, "synthesizer must produce a user-facing response"


# ─────────────────────────────────────────────────────────────
# agent.agent path: the D1 envelope detector treats validation errors
# as normal tool results, and blocked mutations carry envelopes (#8).
# ─────────────────────────────────────────────────────────────

def test_agent_path_validation_error_not_fatal_and_blocked_envelope():
    from agent.agent import _blocked_mutation_result, _executor_error_message

    env = json.dumps({
        "error": "Cannot createSobjectRecord: required fields are missing.",
        "validation_error": True,
        "retry_allowed": False,
        "requires_user_input": True,
        "tool": "createSobjectRecord",
    })
    assert _executor_error_message(env) is None, \
        "validation errors must flow to synthesis, not abort the turn"

    block = _blocked_mutation_result({
        "name": "createSobjectRecord",
        "arguments": {"sobject-name": "Account", "body": {"Name": "Acme"}},
    })
    parsed = _parse(block)
    assert parsed["validation_error"] is True
    assert parsed["tool"] == "createSobjectRecord"
    assert parsed["sobject_name"] == "Account"
    assert parsed["retry_allowed"] is False
    assert "NOT executed" in parsed["error"]
    assert "ZERO mutations" in parsed["error"]


def test_agent_path_first_valid_then_invalid_blocks_everything():
    """REGRESSION (#8 all-or-nothing) on the authenticated SalesforceAgent
    (agent.agent) path: FIRST mutation VALID (Account) + SECOND invalid/vague
    (Lead). Pre-flight validates ALL mutations BEFORE burning any — so the valid
    Account create must NOT execute, ZERO mutations reach Salesforce, and the
    agent asks the user for the missing Lead fields instead of retrying."""
    from types import SimpleNamespace

    from agent.agent import SalesforceAgent

    _CREATE_TOOL = {
        "type": "function",
        "function": {"name": "createSobjectRecord", "description": "Create a record",
                     "parameters": {"type": "object", "properties": {}}},
    }

    class _BatchLLM:
        def __init__(self):
            self.tool_llm_calls = 0
            self._tool_calls = [
                {"id": "t1", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Account", "body": {"Name": "Acme"}}},
                {"id": "t2", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Lead", "body": {}}},
            ]

        async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
            self.tool_llm_calls += 1
            if self.tool_llm_calls == 1:
                return {"content": "", "tool_calls": list(self._tool_calls), "finish_reason": "tool_calls"}
            # The model decides to ask for input rather than re-attempting a mutation.
            return {"content": "To create the lead I need Last Name and Company Name. Please provide them.",
                    "tool_calls": [], "finish_reason": "stop"}

        async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
            return "To create the lead I need Last Name and Company Name. Please provide them."

    llm = _BatchLLM()
    mcp = _FakeMcpClient()
    exec_ = ToolExecutor(mcp, exec_registry())
    agent = SalesforceAgent(llm=llm, executor=exec_, max_iterations=5)
    agent.rag_retriever.get_relevant_tools = lambda *a, **k: [_CREATE_TOOL]

    async def _go():
        events = []
        async for ev in agent.process_message("Create an account and a lead", "sess"):
            events.append(ev)
        return events

    events = asyncio.run(_go())

    # ZERO mutations from this response reached Salesforce — the valid Account
    # create did NOT execute just because it came first.
    assert mcp.calls == [], "any validation failure must block even an earlier valid mutation"
    # The account never executed: only failure envelopes were produced.
    tool_results = [e for e in events if e.get("type") == "tool_result"]
    assert len(tool_results) == 2
    for e in tool_results:
        parsed = _parse(e["data"]["result"])
        assert parsed["validation_error"] is True
        assert parsed["retry_allowed"] is False
    # The agent asks the user for the missing fields (no auto retry, no canned
    # SALESFORCE_FAILED / ungrounded guard).
    responses = [e for e in events if e.get("type") == "response"]
    assert responses and "Last Name" in responses[-1]["data"] and "Company" in responses[-1]["data"]
    errors = [e for e in events if e.get("type") == "error"]
    assert not errors
    # Exactly one tool-calling decision, then a text answer — the valid mutation
    # was never re-attempted.
    assert llm.tool_llm_calls == 2


# ─────────────────────────────────────────────────────────────
# Existing safety mechanisms remain intact
# ─────────────────────────────────────────────────────────────

def test_destructive_delete_still_requires_confirmation_before_executor():
    """Planner confirmation for deletes runs BEFORE the executor — intact."""
    from agent.planner import TaskPlanner

    planner = TaskPlanner()
    safety = planner.check_tool_safety("deleteSobjectRecord", {"sobject-name": "Lead", "id": _VALID_LEAD_ID}, "sX")
    assert safety.get("requires_confirmation") is True
    assert safety.get("safe") is False


def test_mcp_write_safety_still_intact_after_validation():
    """A valid mutation that hits an uncertain MCP failure still raises RuntimeError
    (no retry / no REST fallback for writes) — the write-safety gate is independent
    of validation and untouched."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    import httpx

    from sfmcp.client import SalesforceMCPClient

    session_call = AsyncMock(side_effect=httpx.ConnectError("connection dropped"))
    client = SalesforceMCPClient.__new__(SalesforceMCPClient)
    client._access_token = "tok"
    client._name_map = {}
    client.mcp_required = False
    client.mcp_transport = "MCP"
    client._session = SimpleNamespace(call_tool=session_call)
    client._ensure_fresh_token = AsyncMock()
    client._ensure_connected = AsyncMock()
    client._close_mcp_session = AsyncMock()
    client._try_oauth_refresh = AsyncMock(return_value=True)
    client._fallback_rest_api = AsyncMock(return_value={"fallback": "ok"})

    with pytest.raises(RuntimeError, match="unknown outcome"):
        asyncio.run(client.call_tool(
            "createSobjectRecord", {"sobject-name": "Lead", "LastName": "Test", "Company": "Acme"}
        ))
    assert session_call.call_count == 1, "a mutation must never be invoked twice"
    client._fallback_rest_api.assert_not_called()


# ─────────────────────────────────────────────────────────────
# Orchestrator test helpers
# ─────────────────────────────────────────────────────────────

class _SafePlanner:
    def has_pending_confirmation(self, session_id="default"):
        return False

    def check_tool_safety(self, tool_name, arguments, session_id="default"):
        return {"safe": True, "requires_confirmation": False, "confirmation_message": "",
                "pending_action": None, "blocked_message": ""}


class AsyncMockMock:
    def __init__(self, return_value=None):
        self.return_value = return_value

    async def __call__(self, *args, **kwargs):
        return self.return_value


def exec_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry._load_local_tools()
    return registry


def _run_orchestrator(orch, message):
    async def _go():
        events = []
        async for ev in orch.process_message(message, "default"):
            events.append(ev)
        return events

    return asyncio.run(_go())


def _has_validation_error(events) -> bool:
    for e in events:
        if e.get("type") == "tool_result":
            try:
                parsed = json.loads(e["data"]["result"])
                if parsed.get("validation_error") is True:
                    return True
            except Exception:
                continue
    return False


# ─────────────────────────────────────────────────────────────
# MUTATION PROVENANCE (deterministic, no heuristics):
# required/optional field values must be traceable to explicit
# user input. LLM-fabricated values are REJECTED.
# ─────────────────────────────────────────────────────────────

def _find_provenance_envelope(events) -> dict | None:
    for e in events:
        if e.get("type") == "tool_result":
            try:
                parsed = json.loads(e["data"]["result"])
            except Exception:
                continue
            if parsed.get("validation_error") is True:
                return parsed
    return None


def _last_response_text(events) -> str:
    responses = [e for e in events if e.get("type") == "response"]
    return responses[-1]["data"] if responses else ""


class _FabricatingLeadLLM:
    """Scripted agent-path LLM that FABRICATES the production incident body:
    'create a lead' -> Doe / Acme Corp / john.doe@acme.com, exactly as Qwen
    did against the deployed app. Only supply_company_at controls Company."""

    def __init__(self, tool_calls=None, chat_text=None):
        self.tool_calls = tool_calls if tool_calls is not None else [
            {"id": "t1", "name": "createSobjectRecord",
             "arguments": {"sobject-name": "Lead",
                           "body": {"LastName": "Doe", "Company": "Acme Corp",
                                    "Email": "john.doe@acme.com"}}},
        ]
        self.chat_text = chat_text if chat_text is not None else (
            "To create the lead I need Last Name and Company Name. Please provide them."
        )
        self.tool_llm_calls = 0
        self.chat_calls = 0

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        self.tool_llm_calls += 1
        return {"content": "", "tool_calls": list(self.tool_calls), "finish_reason": "tool_calls"}

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        self.chat_calls += 1
        return self.chat_text


def _new_orchestrator(llm, mcp):
    from agent.multi_agent import Orchestrator
    exec_ = ToolExecutor(mcp, exec_registry())
    orch = Orchestrator(llm=llm, executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _SafePlanner()
    orch._generate_plan = AsyncMockMock(return_value=[{
        "task_id": 1, "description": "create lead", "agent": "ActionAgent", "depends_on": [],
    }])
    orch._get_relevant_tools_or_fallback = AsyncMockMock(return_value=[])
    return orch


def test_provenance_vague_create_lead_fabricated_values_rejected():
    """PROVENANCE #1: 'create a lead' must NEVER reach Salesforce. The LLM's
    fabricated Doe/Acme Corp/john.doe@acme.com is rejected deterministically;
    the assistant asks for the CREATEABLE-AND-REQUIRED fields (not the invented
    ones) and there is NO automatic retry and NO LLM-written ask."""
    llm = _FabricatingLeadLLM()
    mcp = _FakeMcpClient()
    orch = _new_orchestrator(llm, mcp)

    events = _run_orchestrator(orch, "create a lead")

    assert mcp.calls == [], "fabricated create must never reach Salesforce"
    env = _find_provenance_envelope(events)
    assert env is not None, "a validation envelope must be returned"
    # The ask must be the TRUE required set (Describe/static), NOT the fabricated
    # field list — Email was invented by the LLM but is optional for Lead.
    assert env.get("missing_required") is True
    assert set(env["missing_fields"]) == {"LastName", "Company"}
    assert set(env["missing_fields_human"]) == {"Last Name", "Company Name"}
    assert env["retry_allowed"] is False
    text = _last_response_text(events)
    assert "Sure. To create the Lead, please provide:" in text
    assert "- Last Name" in text and "- Company Name" in text
    # No fabricated values leak into the deterministic ask.
    for invented in ("Doe", "Acme", "john.doe", "Acme Corp"):
        assert invented not in text, f"fabricated value {invented!r} must never appear in the ask"
    assert llm.tool_llm_calls == 1, "no automatic tool-calling retry after rejection"
    assert llm.chat_calls == 0, "the ask is deterministic — the synthesizer LLM is not invoked"


def test_provenance_create_fabricated_ask_lists_custom_required_fields_deterministically():
    """PROBLEM 1 (Lead UX): a fabricated create of a CUSTOM object must ask for
    the CUSTOM required fields resolved from Describe (never guessed from a
    static list) — deterministically, with ZERO mutations and ZERO invented
    values. Describe for CustomTenant__c requires Tenant__c + Name."""
    llm = _FabricatingLeadLLM(tool_calls=[
        {"id": "t1", "name": "createSobjectRecord",
         "arguments": {"sobject-name": "CustomTenant__c",
                       "body": {"Tenant__c": "Acme Tenants", "Name": "HQ"}}},
    ])
    mcp = _FakeMcpClient()
    orch = _new_orchestrator(llm, mcp)

    events = _run_orchestrator(orch, "create a custom tenant")

    assert mcp.calls == [], "fabricated custom-object create must never reach Salesforce"
    env = _find_provenance_envelope(events)
    assert env is not None
    assert env.get("missing_required") is True
    # Custom required fields come from the live Describe metadata, not the
    # fabricated body.
    assert set(env["missing_fields"]) == {"Tenant__c", "Name"}
    text = _last_response_text(events)
    assert "Sure. To create the CustomTenant__c, please provide:" in text
    assert "- Tenant" in text and "- Name" in text
    for invented in ("Acme Tenants", "HQ"):
        assert invented not in text, f"fabricated value {invented!r} must never appear in the ask"
    assert llm.chat_calls == 0, "custom-field ask is deterministic — synthesizer LLM not invoked"


def test_provenance_executor_unit_fabricated_lead_rejected():
    """The hard provenance gate lives in ToolExecutor.validate_mutation (used by
    both pre-flight and execute) and runs BEFORE idempotency/transport: with an
    EXPLICIT EMPTY provenance map ({} — a caller carrying no authored values), a
    fabricated Lead body is rejected and MCP is never called."""
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)

    async def _go():
        return await exec_.execute(
            "createSobjectRecord",
            {"sobject-name": "Lead",
             "body": {"LastName": "Doe", "Company": "Acme Corp"}},
            user_provenance={},
        )

    res = asyncio.run(_go())
    env = json.loads(res)
    assert env["validation_error"] is True
    assert set(env["missing_fields"]) == {"LastName", "Company"}
    assert mcp.calls == []


def test_provenance_executor_unit_body_matching_none_fails_closed():
    """A caller that supplies NO provenance object at all (None) FAILS CLOSED
    for a body-bearing mutation — there must be no bypass and no backwards-
    compatible hole: no provenance ⇒ un-authored values ⇒ do not write."""
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)

    async def _go():
        return await exec_.execute(
            "createSobjectRecord",
            {"sobject-name": "Lead",
             "body": {"LastName": "Sharma", "Company": "Tech Solutions"}},
            user_provenance=None,
        )

    res = asyncio.run(_go())
    env = json.loads(res)
    assert env["validation_error"] is True
    assert env.get("requires_user_input") is True
    assert set(env["missing_fields"]) == {"LastName", "Company"}
    assert mcp.calls == []


def test_provenance_executor_unit_explicit_values_allowed():
    """Same gate, same body, but the user explicitly supplied the values in a
    labeled message: the create proceeds and reaches the transport exactly once."""
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    prov = _extract_user_provided_fields("Last Name: Sharma, Company: Tech Solutions")

    async def _go():
        return await exec_.execute(
            "createSobjectRecord",
            {"sobject-name": "Lead",
             "body": {"LastName": "Sharma", "Company": "Tech Solutions"}},
            user_provenance=prov,
        )

    res = asyncio.run(_go())
    parsed = json.loads(res)
    assert parsed.get("validation_error") is not True
    assert len(mcp.calls) == 1


def test_provenance_create_lead_for_john_does_not_infer_company():
    """PROVENANCE #2: 'Create a lead. Last Name: John' establishes only the last
    name. The LLM's invented Company must be rejected — we never infer/provide
    Company on the user's behalf. Only Company is asked for.
    NOTE: an UNLABELED 'for John' / 'named John' phrase establishes NO
    provenance (deterministically mapping it would be guessing); the user must
    restate the value with its label."""
    llm = _FabricatingLeadLLM(tool_calls=[
        {"id": "t1", "name": "createSobjectRecord",
         "arguments": {"sobject-name": "Lead",
                       "body": {"LastName": "John", "Company": "Acme Partners"}}},
    ], chat_text="I need the Company Name. Please provide it.")
    mcp = _FakeMcpClient()
    orch = _new_orchestrator(llm, mcp)

    events = _run_orchestrator(orch, "Create a lead. Last Name: John")

    assert mcp.calls == [], "never execute a create with an inferred Company"
    env = _find_provenance_envelope(events)
    assert env is not None
    assert set(env["missing_fields"]) == {"Company"}
    assert "LastName" not in env["missing_fields"], "John's name was user-provided"
    assert "Company" in _last_response_text(events)


def test_provenance_explicit_field_bindings_execute_exact_body():
    """PROVENANCE #3: explicit user bindings ('Last Name: Sharma. Company:
    Tech Solutions.') make the create legitimate — it executes exactly once with
    the exact user-supplied values and nothing else."""
    llm = _FabricatingLeadLLM(tool_calls=[
        {"id": "t1", "name": "createSobjectRecord",
         "arguments": {"sobject-name": "Lead",
                       "body": {"LastName": "Sharma", "Company": "Tech Solutions"}}},
    ], chat_text="Created the lead for Sharma at Tech Solutions.")
    mcp = _FakeMcpClient()
    orch = _new_orchestrator(llm, mcp)

    events = _run_orchestrator(orch, "Create a lead. Last Name: Sharma. Company: Tech Solutions.")

    assert len(mcp.calls) == 1
    assert mcp.calls[0][0] == "createSobjectRecord"
    assert mcp.calls[0][1]["body"] == {"LastName": "Sharma", "Company": "Tech Solutions"}
    assert _find_provenance_envelope(events) is None


def test_provenance_multi_turn_bare_values_never_auto_mapped():
    """PROVENANCE #4 (multi-turn): 'Sharma, Tech Solutions' carries NO field
    labels, so the parser cannot deterministically map it. The second create is
    STILL rejected — we never guess/auto-map bare values to fields. Both turns
    together produce ZERO Salesforce calls."""
    llm_turn1 = _FabricatingLeadLLM(chat_text="I need the Last Name and Company Name. Please provide them.")
    mcp1 = _FakeMcpClient()
    _run_orchestrator(_new_orchestrator(llm_turn1, mcp1), "create a lead")
    assert mcp1.calls == []

    llm_turn2 = _FabricatingLeadLLM(tool_calls=[
        {"id": "t1", "name": "createSobjectRecord",
         "arguments": {"sobject-name": "Lead",
                       "body": {"LastName": "Sharma", "Company": "Tech Solutions"}}},
    ], chat_text="I need the fields labeled explicitly, e.g. 'Last Name: Sharma, Company: Tech Solutions'.")
    mcp2 = _FakeMcpClient()
    events2 = _run_orchestrator(_new_orchestrator(llm_turn2, mcp2), "Sharma, Tech Solutions")

    assert mcp2.calls == [], "bare unlabeled values must never auto-map to a create"
    env = _find_provenance_envelope(events2)
    assert env is not None
    assert set(env["missing_fields"]) == {"LastName", "Company"}
    assert "Last Name" in _last_response_text(events2)


def test_provenance_optional_fields_omitted_when_absent():
    """PROVENANCE #5a: the LLM supplies ONLY the user-provided required fields —
    no invented FirstName/Email. The create succeeds with exactly that body."""
    llm = _FabricatingLeadLLM(tool_calls=[
        {"id": "t1", "name": "createSobjectRecord",
         "arguments": {"sobject-name": "Lead",
                       "body": {"LastName": "Sharma", "Company": "Tech Solutions"}}},
    ], chat_text="Created the lead.")
    mcp = _FakeMcpClient()
    orch = _new_orchestrator(llm, mcp)

    events = _run_orchestrator(orch, "Create a lead. Last Name: Sharma. Company: Tech Solutions.")

    assert len(mcp.calls) == 1
    assert mcp.calls[0][1]["body"] == {"LastName": "Sharma", "Company": "Tech Solutions"}


def test_provenance_fabricated_optional_fields_rejected():
    """PROVENANCE #5b: when the LLM INVENTS optional fields (FirstName, Email)
    the user never supplied, the create is rejected — optional fields must be
    omitted, never fabricated."""
    llm = _FabricatingLeadLLM(tool_calls=[
        {"id": "t1", "name": "createSobjectRecord",
         "arguments": {"sobject-name": "Lead",
                       "body": {"LastName": "Sharma", "Company": "Tech Solutions",
                                "FirstName": "John", "Email": "john.doe@acme.com"}}},
    ], chat_text="I did not receive First Name and Email from you. Please provide them if you want them set.")
    mcp = _FakeMcpClient()
    orch = _new_orchestrator(llm, mcp)

    events = _run_orchestrator(orch, "Create a lead. Last Name: Sharma. Company: Tech Solutions.")

    assert mcp.calls == [], "fabricated optional fields must not reach Salesforce"
    env = _find_provenance_envelope(events)
    assert env is not None
    assert set(env["missing_fields"]) == {"FirstName", "Email"}
    # All REQUIRED fields were user-provided, so this is NOT a required-fields
    # ask (missing_required absent) — the deterministic ask must NOT fire and
    # list optional First Name / Email as if they were required.
    assert env.get("missing_required") is not True
    assert llm.chat_calls == 1, "synthesizer LLM handles the optional-field note"
    assert "First Name and Email" in _last_response_text(events)


def test_provenance_mixed_valid_and_fabricated_blocks_everything():
    """PROVENANCE #6 (all-or-nothing): a VALID lead create (Sharma / Tech
    Solutions) in the same response as a FABRICATED account ('Global Corp') means
    ZERO mutations execute — the valid lead is blocked too, as a sibling of the
    rejected mutation."""
    from agent.multi_agent import Orchestrator

    class _MixedLLM:
        def __init__(self):
            self.tool_calls = [
                {"id": "t1", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Lead",
                               "body": {"LastName": "Sharma", "Company": "Tech Solutions"}}},
                {"id": "t2", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Account", "body": {"Name": "Global Corp"}}},
            ]
            self.tool_llm_calls = 0

        async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
            self.tool_llm_calls += 1
            return {"content": "", "tool_calls": list(self.tool_calls), "finish_reason": "tool_calls"}

        async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
            return ("I need the Account Name. Please provide it explicitly.")

    llm = _MixedLLM()
    mcp = _FakeMcpClient()
    exec_ = ToolExecutor(mcp, exec_registry())
    orch = Orchestrator(llm=llm, executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _SafePlanner()
    orch._generate_plan = AsyncMockMock(return_value=[{
        "task_id": 1, "description": "create records", "agent": "ActionAgent", "depends_on": [],
    }])
    orch._get_relevant_tools_or_fallback = AsyncMockMock(return_value=[])

    events = _run_orchestrator(
        orch,
        "Create a lead. Last Name: Sharma. Company: Tech Solutions. Also create an account named Acme.",
    )

    assert mcp.calls == [], "a fabricated mutation must block the whole response (all-or-nothing)"
    envelopes = [e for e in events if e.get("type") == "tool_result"]
    assert len(envelopes) == 2, "both mutations yield rejection envelopes"
    account_env = None
    for e in envelopes:
        env = json.loads(e["data"]["result"])
        assert env["validation_error"] is True
        assert env["retry_allowed"] is False
        if env["sobject_name"].lower() == "account":
            account_env = env
    assert account_env is not None
    assert "Name" in account_env["missing_fields"]
    assert llm.tool_llm_calls == 1


# ─────────────────────────────────────────────────────────────
# PROVENANCE MATRIX (A–P): field-scoped, request-scoped, FULL-VALUE
# equality. Locked design decisions: no substring/token matching, no
# cross-field reuse, no truncation, no unlabeled auto-mapping, no
# global/context store (per-message provenance only), and explicit
# fail-closed on None.
# ─────────────────────────────────────────────────────────────

def _provenance_matrix_case(user_message, body, expect_pass):
    """Run one matrix case through validate_mutation against the Lead/Account
    describe schema and return (passed, missing_fields)."""
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    prov = _extract_user_provided_fields(user_message)
    res = asyncio.run(exec_.validate_mutation(
        "createSobjectRecord", body, user_provenance=prov
    ))
    if res is None:
        assert expect_pass, f"expected FAIL for {user_message!r} -> passed"
        return True, []
    parsed = json.loads(res)
    assert not expect_pass, f"expected PASS for {user_message!r} -> {parsed.get('error')}"
    assert parsed["validation_error"] is True
    assert parsed["retry_allowed"] is False
    assert parsed["requires_user_input"] is True
    return False, parsed.get("missing_fields", [])


def test_provenance_matrix_a_p():
    cases = [
        # A. Labeled values match -> pass
        ("Last Name: Sharma, Company: Tech Solutions",
         {"sobject-name": "Lead", "body": {"LastName": "Sharma", "Company": "Tech Solutions"}}, True),
        # B. Cross-field swap -> fail (Last 'Tech Solutions' is the Company value)
        ("Last Name: Sharma, Company: Tech Solutions",
         {"sobject-name": "Lead", "body": {"LastName": "Tech Solutions", "Company": "Sharma"}}, False),
        # C. Truncation -> fail (user said Acme Technologies, LLM wrote Acme)
        ("Company: Acme Technologies",
         {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": "Acme"}}, False),
        # D. Unlabeled bare values -> fail (deterministic mapping impossible)
        ("Sharma, Tech Solutions",
         {"sobject-name": "Lead", "body": {"LastName": "Sharma", "Company": "Tech Solutions"}}, False),
        # E. API-name + label mix -> pass
        ("LastName: Sharma, company is Tech Solutions",
         {"sobject-name": "Lead", "body": {"LastName": "Sharma", "Company": "Tech Solutions"}}, True),
        # F. Stale/multi-turn values do NOT authorize a NEW message -> fail
        #   (this message carries no bindings, even though a prior turn had them)
        ("create a lead",
         {"sobject-name": "Lead", "body": {"LastName": "Sharma", "Company": "Tech Solutions"}}, False),
        # G. Quoted value containing a conjunction -> pass
        ('Company: "Research and Development", Last Name: Smith',
         {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": "Research and Development"}}, True),
        # H. Unbound conjunction in plain text -> binds only the head word (quotes
        #    required for full phrase) — design tradeoff
        ("Company is Research and Development",
         {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": "Research and Development"}}, False),
        # I. Empty body -> provenance layer passes but the mutation is still
        #    rejected by the presence gate (vague create) -> overall FAIL.
        ("create a lead", {"sobject-name": "Lead", "body": {}}, False),
        # J. Sentence-period boundaries + internal email dot -> pass
        ("Last Name: Smith. Company: Acme. Email: bob.d@acme.com.",
         {"sobject-name": "Lead",
          "body": {"LastName": "Smith", "Company": "Acme", "Email": "bob.d@acme.com"}}, True),
        # K. Full-width label/punctuation separators -> pass
        ("Last Name：Sharma。 Company：Tech Solutions",
         {"sobject-name": "Lead", "body": {"LastName": "Sharma", "Company": "Tech Solutions"}}, True),
        # L. Flat single-key args (no 'body' container) -> pass
        ("Status: Working, Last Name: Smith, Company: Acme",
         {"sobject-name": "Lead", "LastName": "Smith", "Status": "Working", "Company": "Acme"}, True),
        # M. Fabricated email (user never supplied it) -> fail
        ("Last Name: Sharma, Company: Tech Solutions",
         {"sobject-name": "Lead",
          "body": {"LastName": "Sharma", "Company": "Tech Solutions",
                   "Email": "john.doe@acme.com"}}, False),
        # N. Case/whitespace-insensitive full-value match -> pass
        ("Company: TECH SOLUTIONS, Last Name: Smith",
         {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": "TECH SOLUTIONS"}}, True),
        # O. Trailing-period normalization on both sides -> pass
        ("Company: Acme., Last Name: Smith.",
         {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": "Acme"}}, True),
        # P. 'for <name>' / 'named <name>' establish NO provenance -> fail
        ("create a lead for Sharma",
         {"sobject-name": "Lead", "body": {"LastName": "Sharma", "Company": "Acme Corp"}}, False),
    ]
    for message, body, expect in cases:
        passed, missing = _provenance_matrix_case(message, body, expect)
        assert passed == expect, f"matrix case {message!r}: passed={passed} expected={expect}"


def test_soql_auto_fix_never_executes_a_mutating_tool():
    """The SOQL auto-fix path runs OUTSIDE the mutation pre-flight/provenance
    block. A mutating/destructive tool returned by the fix LLM must NEVER be
    executed — only read-only tools may be auto-corrected."""
    from agent.agent import SalesforceAgent

    _SOQL_TOOL = {
        "type": "function",
        "function": {"name": "soqlQuery", "description": "Run a SOQL query",
                     "parameters": {"type": "object", "properties": {}}},
    }
    _CREATE_TOOL = {
        "type": "function",
        "function": {"name": "createSobjectRecord", "description": "Create a record",
                     "parameters": {"type": "object", "properties": {}}},
    }

    class _FixLLM:
        def __init__(self):
            self.tool_llm_calls = 0

        async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
            self.tool_llm_calls += 1
            if self.tool_llm_calls == 1:
                # First decision: a MALFORMED soqlQuery (triggers the auto-fix path).
                return {"content": "", "tool_calls": [
                    {"id": "t1", "name": "soqlQuery",
                     "arguments": {"q": "SELECT Id FROM Lead WHERE"}},
                ], "finish_reason": "tool_calls"}
            # The auto-fix LLM tries to smuggle in a CREATE. Must be refused.
            return {"content": "", "tool_calls": [
                {"id": "t2", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Lead",
                               "body": {"LastName": "Doe", "Company": "Acme"}}},
            ], "finish_reason": "tool_calls"}

        async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
            return "I could not correct the query."

    mcp = _FakeMcpClient(result="Malformed Query: expected EOF at 'WHERE'")
    exec_ = _executor(mcp)
    agent = SalesforceAgent(llm=_FixLLM(), executor=exec_, max_iterations=5)
    agent.rag_retriever.get_relevant_tools = lambda *a, **k: [_SOQL_TOOL, _CREATE_TOOL]

    # RAG fast-path bypasses the planner for read-only intents, so ensure the
    # first tool-call response reaches the executor through the normal path.
    events = []
    async def _go():
        async for ev in agent.process_message("Show me the malformed leads", "soql-guard-session"):
            events.append(ev)
    asyncio.run(_go())

    assert mcp.calls == [("soqlQuery", {"q": "SELECT Id FROM Lead WHERE"})], \
        "only the read-only query may be attempted; the mutation must never execute"
    assert all(c[0] != "createSobjectRecord" for c in mcp.calls)
    errors = [e for e in events if e.get("type") == "error"]
    assert errors, "the refusal must surface as a controlled error event"
    assert "refused" in errors[0]["message"].lower()