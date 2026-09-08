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
from sfmcp.executor import ToolExecutor
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
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Lead", "body": {}}))
    _assert_validation_error(res, "createSobjectRecord", ["LastName", "Company"])
    assert mcp.calls == [], "vague create must never call MCP"


def test_create_lead_vague_request_never_reaches_salesforce():
    # "create a lead" with no fields at all.
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Lead"}))
    _assert_validation_error(res, "createSobjectRecord", ["LastName", "Company"])
    assert mcp.calls == []


def test_create_lead_only_first_name_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
        "createSobjectRecord", {"sobject-name": "Lead", "body": {"FirstName": "John"}}
    ))
    _assert_validation_error(res, "createSobjectRecord", ["LastName", "Company"])
    assert mcp.calls == []


def test_create_lead_null_company_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
        "createSobjectRecord", {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": None}}
    ))
    _assert_validation_error(res, "createSobjectRecord", ["Company"])
    assert mcp.calls == []


def test_create_lead_whitespace_last_name_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
        "createSobjectRecord", {"sobject-name": "Lead", "body": {"LastName": "   ", "Company": "Acme"}}
    ))
    _assert_validation_error(res, "createSobjectRecord", ["LastName"])
    assert mcp.calls == []


def test_create_lead_complete_accepted_executes_once():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    args = {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": "Acme", "FirstName": "John"}}
    res = asyncio.run(exec_.execute("createSobjectRecord", dict(args)))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


# ─────────────────────────────────────────────────────────────
# Contact / Account / Opportunity / Case / Task validation
# ─────────────────────────────────────────────────────────────

def test_create_contact_empty_body_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Contact", "body": {}}))
    _assert_validation_error(res, "createSobjectRecord", ["LastName"])
    assert mcp.calls == []


def test_create_contact_vague_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Contact", "body": {"FirstName": "Jane"}}))
    _assert_validation_error(res, "createSobjectRecord", ["LastName"])
    assert mcp.calls == []


def test_create_account_missing_name_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Account", "body": {}}))
    _assert_validation_error(res, "createSobjectRecord", ["Name"])
    assert mcp.calls == []


def test_create_account_vague_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Account"}))
    _assert_validation_error(res, "createSobjectRecord", ["Name"])
    assert mcp.calls == []


def test_create_account_complete_accepted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Account", "body": {"Name": "Acme"}}))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


def test_create_opportunity_vague_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Opportunity", "body": {}}))
    _assert_validation_error(res, "createSobjectRecord", ["Name", "StageName", "CloseDate"])
    assert mcp.calls == []


def test_create_opportunity_incomplete_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
        "createSobjectRecord", {"sobject-name": "Opportunity", "body": {"Name": "Deal"}}
    ))
    _assert_validation_error(res, "createSobjectRecord", ["StageName", "CloseDate"])
    assert mcp.calls == []


def test_create_opportunity_complete_accepted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
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
    res = asyncio.run(exec_.execute(
        "createSobjectRecord", {"sobject-name": "Case", "body": {"Description": "Login issue"}}
    ))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


def test_create_case_empty_body_rejected():
    # Even with no required fields, a ZERO-field create remains vague and fails closed.
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Case", "body": {}}))
    _assert_validation_error(res, "createSobjectRecord")
    assert mcp.calls == [], "empty-body create must never reach Salesforce"


def test_create_task_incomplete_rejected():
    # Task requires Subject; Status is DEFAULTED on create so it is not required (#5).
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
        "createSobjectRecord", {"sobject-name": "Task", "body": {"Description": "Follow up"}}
    ))
    _assert_validation_error(res, "createSobjectRecord", ["Subject"])
    assert mcp.calls == []


def test_create_task_with_subject_accepted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
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
    res = asyncio.run(exec_.execute("createSobjectRecord", {"body": {"LastName": "Smith"}}))
    _assert_validation_error(res, "createSobjectRecord")
    assert mcp.calls == []


def test_create_body_not_dict_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Lead", "body": "not-a-dict"}))
    _assert_validation_error(res, "createSobjectRecord")
    assert mcp.calls == []


# ─────────────────────────────────────────────────────────────
# Custom objects use live Describe; Describe failure fails closed
# ─────────────────────────────────────────────────────────────

def test_create_custom_object_vague_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "CustomTenant__c", "body": {}}))
    _assert_validation_error(res, "createSobjectRecord", ["Tenant__c", "Name"])
    assert mcp.calls == []
    assert mcp.describe_calls == ["CustomTenant__c"], "custom object must be described"


def test_create_custom_object_complete_accepted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
        "createSobjectRecord",
        {"sobject-name": "CustomTenant__c", "body": {"Tenant__c": "Acme", "Name": "Main"}},
    ))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1
    assert mcp.describe_calls == ["CustomTenant__c"]


def test_create_custom_object_describe_fails_closed():
    mcp = _FakeMcpClientNoDescribe()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "CustomTenant__c", "body": {"Name": "Acme"}}))
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
    res = asyncio.run(exec_.execute(
        "createSobjectRecord", {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": "Acme"}}
    ))
    _assert_validation_error(res, "createSobjectRecord", ["CustomLeadFlag__c"])
    assert mcp.calls == [], "missing custom required field must block the create"
    assert mcp.describe_calls == ["Lead"]

    ok = asyncio.run(exec_.execute(
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

    bad = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Lead", "body": {}}))
    _assert_validation_error(bad, "createSobjectRecord", ["LastName", "Company"])
    assert mcp.calls == [], "static fallback must still reject a vague lead"

    good = asyncio.run(exec_.execute(
        "createSobjectRecord",
        {"sobject-name": "Lead", "body": {"LastName": "Sharma", "Company": "Tech Solutions"}},
    ))
    assert "validation_error" not in _parse(good)
    assert len(mcp.calls) == 1


def test_unknown_object_without_describe_falls_back_then_fails_closed():
    # Object is in neither Describe nor the static registry -> fail closed.
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Mystery__c", "body": {"Name": "X"}}))
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
    res = asyncio.run(exec_.execute(
        "updateSobjectRecord",
        {"sobject-name": "Lead", "id": _VALID_LEAD_ID, "body": {"Status": "Working"}},
    ))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


def test_update_empty_id_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
        "updateSobjectRecord", {"sobject-name": "Lead", "id": "", "body": {"Status": "Working"}}
    ))
    _assert_validation_error(res, "updateSobjectRecord")
    assert mcp.calls == []


def test_update_invalid_id_format_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
        "updateSobjectRecord", {"sobject-name": "Lead", "id": "not-an-id", "body": {"Status": "Working"}}
    ))
    _assert_validation_error(res, "updateSobjectRecord")
    assert mcp.calls == []


def test_update_empty_body_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
        "updateSobjectRecord", {"sobject-name": "Lead", "id": _VALID_LEAD_ID, "body": {}}
    ))
    _assert_validation_error(res, "updateSobjectRecord")
    assert mcp.calls == []


def test_update_related_valid_accepted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
        "updateRelatedRecord",
        {"id": _VALID_CONTACT_ID, "relationship-path": "Account", "body": {"Phone": "555-0100"}},
    ))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


def test_update_related_missing_rel_path_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
        "updateRelatedRecord",
        {"id": _VALID_CONTACT_ID, "body": {"Phone": "555-0100"}},
    ))
    _assert_validation_error(res, "updateRelatedRecord")
    assert mcp.calls == []


def test_delete_valid_id_accepted():
    mcp = _FakeMcpClient(result=json.dumps({"success": True}))
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("deleteSobjectRecord", {"sobject-name": "Lead", "id": _VALID_LEAD_ID}))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


def test_delete_invalid_id_rejected():
    mcp = _FakeMcpClient(result=json.dumps({"success": True}))
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("deleteSobjectRecord", {"sobject-name": "Lead", "id": "bad"}))
    _assert_validation_error(res, "deleteSobjectRecord")
    assert mcp.calls == []


def test_upload_valid_accepted():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
        "uploadRecordAttachment",
        {"record_id": _VALID_ACCOUNT_ID, "file_name": "test.pdf", "file_content_base64": "dGVzdA=="},
    ))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


def test_upload_missing_record_id_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
        "uploadRecordAttachment", {"file_name": "test.pdf", "file_content_base64": "dGVzdA=="}
    ))
    _assert_validation_error(res, "uploadRecordAttachment")
    assert mcp.calls == []


def test_upload_missing_file_content_rejected():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute(
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

    r1 = asyncio.run(exec_.execute("createSobjectRecord", dict(bad_args)))
    r2 = asyncio.run(exec_.execute("createSobjectRecord", dict(bad_args)))

    assert "validation_error" in _parse(r1)
    assert "validation_error" in _parse(r2)
    assert mcp.calls == [], "rejected mutations must never reach Salesforce"
    # And a correct submission is NOT falsely deduplicated against the rejected one.
    good_args = {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": "Acme"}}
    r3 = asyncio.run(exec_.execute("createSobjectRecord", dict(good_args)))
    assert "validation_error" not in _parse(r3)
    assert len(mcp.calls) == 1, "corrected submission must execute"


def test_valid_mutation_remains_idempotent_after_validation_pass():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    args = {"sobject-name": "Lead", "body": {"LastName": "Smith", "Company": "Acme"}}

    r1 = asyncio.run(exec_.execute("createSobjectRecord", dict(args)))
    r2 = asyncio.run(exec_.execute("createSobjectRecord", dict(args)))

    assert r1 == r2
    assert len(mcp.calls) == 1, "valid identical mutation must execute exactly once"


# ─────────────────────────────────────────────────────────────
# Read-only tools are unaffected
# ─────────────────────────────────────────────────────────────

def test_read_only_tools_unaffected():
    mcp = _FakeMcpClient(result=json.dumps({"totalSize": 0, "records": []}))
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("soqlQuery", {"q": "SELECT Id FROM Account LIMIT 5"}))
    assert "validation_error" not in _parse(res)
    assert len(mcp.calls) == 1


# ─────────────────────────────────────────────────────────────
# Error envelope contract
# ─────────────────────────────────────────────────────────────

def test_error_envelope_full_fields_present():
    mcp = _FakeMcpClient()
    exec_ = _executor(mcp)
    res = asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Lead", "body": {}}))
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
    asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Lead"}))
    assert mcp.calls == []
    # A valid lead forwards EXACTLY the user-provided body — no invented defaults.
    asyncio.run(exec_.execute("createSobjectRecord", {"sobject-name": "Lead",
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


def test_validation_failure_flows_to_synthesizer_not_fatal_error():
    """validation_error=true results are normal tool results, NOT AgentError."""
    from agent.multi_agent import Orchestrator

    class _LLM:
        def __init__(self):
            self.tool_calls = [
                {"id": "t1", "name": "createSobjectRecord",
                 "arguments": {"sobject-name": "Lead", "body": {}}},
            ]
            self.synthesis_user_msg = ""

        async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
            return {"content": "", "tool_calls": list(self.tool_calls), "finish_reason": "tool_calls"}

        async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
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
    # The synthesizer was invoked with the validation tool result and produced a
    # user-facing "provide the fields" response.
    assert "validation_error" in llm.synthesis_user_msg or "required" in llm.synthesis_user_msg.lower() or "provide" in llm.synthesis_user_msg.lower()
    responses = [e for e in events if e.get("type") == "response"]
    assert responses, "synthesizer must produce a user-facing response"
    assert "Please provide" in responses[-1]["data"]


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

    # The vague Lead create was blocked BEFORE Salesforce...
    assert mcp.calls == [], "a validation failure must stop ALL mutations in the response"
    # ...so the (valid) Account and Opportunity creates were NOT executed either.
    created = [c for c in mcp.calls if c[0] == "createSobjectRecord"]
    assert created == [], "later valid mutations in a failed batch must not run"
    # Only ONE tool call actually produced a result (the failing Lead); the rest
    # were stopped.
    tool_results = [e for e in events if e.get("type") == "tool_result"]
    assert len(tool_results) == 1, "after the failed mutation, no further tool may run"
    assert _has_validation_error(events)
    # The tool-calling LLM is still invoked exactly once (no automatic retry).
    assert llm.tool_llm_calls == 1


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

    events = _run_orchestrator(orch, "Create a lead and an account")

    assert len(mcp.calls) == 2
    assert mcp.calls[0][1].get("body", {}).get("LastName") == "Sharma"
    assert mcp.calls[1][1].get("body", {}).get("Name") == "Acme"


# ─────────────────────────────────────────────────────────────
# TEST B: a VALID create request works end-to-end (Orchestrator).
# ─────────────────────────────────────────────────────────────

def test_orchestrator_valid_create_executes_once():
    """'Create a lead. Last Name Sharma, Company Tech Solutions.' must reach
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

    events = _run_orchestrator(orch, "Create a lead. Last Name Sharma, Company Tech Solutions.")

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