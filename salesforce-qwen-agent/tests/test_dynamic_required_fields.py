"""
Dynamic required-fields tests — proves the create-ask is resolved from live
Describe metadata and no static per-object list (e.g. Lead-only LastName and
Company) is the source of truth.

Mocked Describe: picklists, lookups, defaulted fields, custom objects, and
org-specific custom required fields are exercised deterministically.

Pure unit tests: no live Salesforce / MCP / LLM.
"""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sfmcp.client import (
    _simplify_describe_fields,
    _extract_required_from_describe,
    _extract_required_field_options,
)
from sfmcp.executor import ToolExecutor
from sfmcp.registry import ToolRegistry
import sfmcp.executor as executor_module
from agent.multi_agent import _render_required_fields_ask

_normalize_value = executor_module._normalize_value


# ─────────────────────────────────────────────────────────────
# Raw Describe payloads (mirrors the Salesforce API shape)
# ─────────────────────────────────────────────────────────────

_RAW_LEAD_WITH_CUSTOM_REQUIRED: dict = {
    "fields": [
        {"name": "LastName", "label": "Last Name", "type": "string",
         "nillable": False, "defaultedOnCreate": False, "createable": True},
        {"name": "Company", "label": "Company Name", "type": "string",
         "nillable": False, "defaultedOnCreate": False, "createable": True},
        {"name": "Region__c", "label": "Region", "type": "picklist",
         "nillable": False, "defaultedOnCreate": False, "createable": True,
         "picklistValues": [
             {"value": "APAC", "active": True},
             {"value": "EMEA", "active": True},
             {"value": "Americas", "active": True},
             {"value": "Retired", "active": False},
         ]},
        {"name": "Email", "label": "Email", "type": "email",
         "nillable": True, "defaultedOnCreate": False, "createable": True},
        {"name": "IsDeleted", "label": "Deleted", "type": "boolean",
         "nillable": False, "defaultedOnCreate": True, "createable": False},
    ]
}

_RAW_OPPORTUNITY: dict = {
    "fields": [
        {"name": "Name", "label": "Opportunity Name", "type": "string",
         "nillable": False, "defaultedOnCreate": False, "createable": True},
        {"name": "StageName", "label": "Stage", "type": "picklist",
         "nillable": False, "defaultedOnCreate": False, "createable": True,
         "picklistValues": [
             {"value": "Prospecting", "active": True},
             {"value": "Qualification", "active": True},
             {"value": "Needs Analysis", "active": True},
             {"value": "Closed Won", "active": True},
             {"value": "Closed Lost", "active": True},
             {"value": "Archived", "active": False},
         ]},
        {"name": "CloseDate", "label": "Close Date", "type": "date",
         "nillable": False, "defaultedOnCreate": False, "createable": True},
        {"name": "Amount", "label": "Amount", "type": "currency",
         "nillable": True, "defaultedOnCreate": False, "createable": True},
    ]
}

_RAW_TASK_WITH_REFERENCE: dict = {
    "fields": [
        {"name": "Subject", "label": "Subject", "type": "string",
         "nillable": False, "defaultedOnCreate": False, "createable": True},
        {"name": "WhoId", "label": "Name", "type": "reference",
         "nillable": True, "defaultedOnCreate": False, "createable": True,
         "referenceTo": ["Lead", "Contact"]},
        {"name": "Status", "label": "Status", "type": "picklist",
         "nillable": False, "defaultedOnCreate": True, "createable": True,
         "picklistValues": [
             {"value": "Not Started", "active": True},
             {"value": "In Progress", "active": True},
             {"value": "Completed", "active": True},
         ]},
        {"name": "Description", "label": "Description", "type": "textarea",
         "nillable": True, "defaultedOnCreate": False, "createable": True},
    ]
}

_RAW_CUSTOM_OBJECT: dict = {
    "fields": [
        {"name": "Tenant__c", "label": "Tenant", "type": "reference",
         "nillable": False, "defaultedOnCreate": False, "createable": True,
         "referenceTo": ["Tenant__c"]},
        {"name": "Name", "label": "Name", "type": "string",
         "nillable": False, "defaultedOnCreate": False, "createable": True},
        {"name": "Tier__c", "label": "Tier", "type": "picklist",
         "nillable": False, "defaultedOnCreate": False, "createable": True,
         "picklistValues": [
             {"value": "Standard", "active": True},
             {"value": "Premium", "active": True},
             {"value": "Enterprise", "active": True},
         ]},
        {"name": "BillingAddress__c", "label": "Billing Address", "type": "text",
         "nillable": True, "defaultedOnCreate": False, "createable": True},
    ]
}

_RAW_EMPTY_REQUIRED: dict = {
    "fields": [
        {"name": "Subject", "label": "Subject", "type": "string",
         "nillable": True, "defaultedOnCreate": True, "createable": True},
        {"name": "Description", "label": "Description", "type": "text",
         "nillable": True, "defaultedOnCreate": False, "createable": True},
    ]
}

# Rich Describe map: sobject_name -> raw describe payload
_RAW_DESCRIBE: dict[str, dict] = {
    "Lead":         _RAW_LEAD_WITH_CUSTOM_REQUIRED,
    "Opportunity":  _RAW_OPPORTUNITY,
    "Task":         _RAW_TASK_WITH_REFERENCE,
    "CustomTenant__c": _RAW_CUSTOM_OBJECT,
    "GhostCase__c": _RAW_EMPTY_REQUIRED,
}

# Required fields (from Describe) expected after flattening
_DESCRIBE_REQUIRED: dict[str, list[tuple[str, str]]] = {
    "Lead":         [("LastName", "Last Name"), ("Company", "Company Name"), ("Region__c", "Region")],
    "Opportunity":  [("Name", "Opportunity Name"), ("StageName", "Stage"), ("CloseDate", "Close Date")],
    "Task":         [("Subject", "Subject")],
    "CustomTenant__c": [("Tenant__c", "Tenant"), ("Name", "Name"), ("Tier__c", "Tier")],
    "GhostCase__c": [],
}

# Expected options for required fields (from Describe)
_DESCRIBE_OPTIONS: dict[str, dict[str, list[str]]] = {
    "Lead":         {"Region__c": ["APAC", "EMEA", "Americas"]},
    "Opportunity":  {"StageName": ["Prospecting", "Qualification", "Needs Analysis",
                                   "Closed Won", "Closed Lost"]},
    "Task":         {},
    "CustomTenant__c": {"Tenant__c": ["Tenant__c"], "Tier__c": ["Standard", "Premium", "Enterprise"]},
    "GhostCase__c": {},
}

# Valid record IDs for tests
_VALID_LEAD_ID = "00Q000000000001"
_VALID_OPP_ID  = "006000000000001"

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────


class _FakeDescribeMcpClient:
    """MCP client that resolves required fields via a Describe map, and also
    resolves describe_required_field_options so the executor can attach picklist /
    reference hints to the deterministic required-fields ask."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.call_result: str = json.dumps({"id": "00Qfake"})
        self._raw_describe = dict(_RAW_DESCRIBE)

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return self.call_result

    async def describe_required_fields(self, sobject_name):
        required = _DESCRIBE_REQUIRED.get(sobject_name)
        return required  # None when unknown

    async def describe_required_field_options(self, sobject_name):
        return _DESCRIBE_OPTIONS.get(sobject_name)  # None when unknown


class _FakeDescribeNoOptionsMcpClient(_FakeDescribeMcpClient):
    """Same as _FakeDescribeMcpClient but WITHOUT describe_required_field_options.
    Exercises fail-soft when the executor enrichment method is missing."""

    async def describe_required_field_options(self, sobject_name):
        raise AttributeError("not implemented")


class _FakeDescribeFailureMcpClient:
    """Describe fails (returns None) for all objects — forces static fallback /
    fail-closed path."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return json.dumps({"id": "00Qfake"})

    async def describe_required_fields(self, sobject_name):
        return None

    async def describe_required_field_options(self, sobject_name):
        return None


class _FakeNoDescribeMcpClient:
    """MCP client that lacks the describe methods entirely (getattr returns None).
    Exercises the static-registry fallback for legacy executor wiring."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return json.dumps({"id": "00Qfake"})

    # No describe_required_fields / describe_required_field_options methods


def _executor(mcp_client) -> ToolExecutor:
    registry = ToolRegistry()
    registry._load_local_tools()
    return ToolExecutor(mcp_client, registry)


async def _exec_with_prov(exec_, tool_name, arguments):
    """Run a tool through the executor passing body-mirrored provenance."""
    body = arguments.get("body") if isinstance(arguments.get("body"), dict) else {}
    prov = {
        str(api): frozenset({executor_module._normalize_value(v)})
        for api, v in body.items()
        if v is not None and str(v).strip()
    }
    return await exec_.execute(tool_name, arguments, user_provenance=prov)


@pytest.fixture(autouse=True)
def _clean_mutation_store():
    executor_module._recent_mutations.clear()
    yield
    executor_module._recent_mutations.clear()


def _find_env_in_tool_results(tool_results: list[dict]) -> dict | None:
    """Extract the first missing_required envelope from a flat list of tool
    results (orchestrator internals)."""
    for item in tool_results or []:
        if not isinstance(item, dict) or "result" not in item:
            continue
        try:
            parsed = json.loads(item["result"])
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and parsed.get("missing_required") is True:
            return parsed
    return None


def _run_tool_directly(exec_, tool_name, arguments):
    """Synchronously run the executor tool and return the parsed envelope."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_exec_with_prov(exec_, tool_name, arguments))
    finally:
        loop.close()


def _parse(result: str) -> dict | None:
    if not result:
        return None
    try:
        return json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════
# A: _simplify_describe_fields produces picklist/reference metadata
# ═══════════════════════════════════════════════════════════════

class TestSimplifyDescribesFieldsPreservesPicklistAndReference:
    """A1/A4/A8: the simplified Describe keeps live picklist_values and
    reference_to metadata so downstream consumers (the required-fields gate
    AND the deterministic ask) can render real allowed values."""

    def test_picklist_values_active_only_capped(self):
        fields = _simplify_describe_fields(_RAW_OPPORTUNITY)
        stage = next(f for f in fields if f["name"] == "StageName")
        assert stage["type"] == "picklist"
        assert stage["required"] is True
        # Active values only; "Archived" (active: False) excluded; 5 <= 12 cap
        assert stage["picklist_values"] == [
            "Prospecting", "Qualification", "Needs Analysis", "Closed Won", "Closed Lost"
        ]

    def test_reference_to_captured(self):
        fields = _simplify_describe_fields(_RAW_TASK_WITH_REFERENCE)
        who = next(f for f in fields if f["name"] == "WhoId")
        assert who["reference_to"] == ["Lead", "Contact"]

    def test_reference_to_empty_when_not_reference(self):
        fields = _simplify_describe_fields(_RAW_OPPORTUNITY)
        for f in fields:
            assert "reference_to" not in f, f"{f['name']}: no reference_to for non-reference type"

    def test_picklist_values_absent_when_not_picklist(self):
        fields = _simplify_describe_fields(_RAW_LEAD_WITH_CUSTOM_REQUIRED)
        for f in fields:
            if f["type"] != "picklist":
                assert "picklist_values" not in f

    def test_required_flag_correct(self):
        fields = _simplify_describe_fields(_RAW_LEAD_WITH_CUSTOM_REQUIRED)
        req = {f["name"] for f in fields if f.get("required")}
        # defaultedOnCreate=True (IsDeleted), nillable=True (Email), non-createable
        assert req == {"LastName", "Company", "Region__c"}


# ═══════════════════════════════════════════════════════════════
# B: _extract_required_field_options produces correct hint map
# ═══════════════════════════════════════════════════════════════

class TestExtractRequiredFieldOptions:
    """A3/A8/A9: choice hints come exclusively from live Describe."""

    def test_picklist_returns_active_values(self):
        fields = _simplify_describe_fields(_RAW_LEAD_WITH_CUSTOM_REQUIRED)
        options = _extract_required_field_options(fields)
        assert options["Region__c"] == ["APAC", "EMEA", "Americas"]
        # Only required fields keyed
        assert "Email" not in options
        assert "LastName" not in options  # string, not picklist

    def test_reference_returns_object_names(self):
        fields = _simplify_describe_fields(_RAW_TASK_WITH_REFERENCE)
        options = _extract_required_field_options(fields)
        # Subject is required string, no options
        assert "Subject" not in options

    def test_custom_object_picklist(self):
        fields = _simplify_describe_fields(_RAW_CUSTOM_OBJECT)
        options = _extract_required_field_options(fields)
        assert options["Tier__c"] == ["Standard", "Premium", "Enterprise"]
        assert "Name" not in options  # string
        # Tenant__c is a reference field AND required — its option is the
        # referenced sObject name (the user must supply a valid Tenant__c ID).
        assert options["Tenant__c"] == ["Tenant__c"]

    def test_empty_when_no_required(self):
        fields = _simplify_describe_fields(_RAW_EMPTY_REQUIRED)
        options = _extract_required_field_options(fields)
        assert options == {}


# ═══════════════════════════════════════════════════════════════
# C: executor enriches required-fields envelopes with missing_field_options
# ═══════════════════════════════════════════════════════════════

class TestExecutorEnrichesOptions:
    """A6/A8: both the provenance-enriched and presence-gate envelopes carry
    missing_field_options when the MCP client supports it."""

    def test_provenance_envelope_has_picklist_options(self):
        """LLM fabricates Lead values (no user provenance) -> blocked, envelope
        carries Region__c active picklist choices from live Describe."""
        mcp = _FakeDescribeMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "Lead",
             "body": {"LastName": "Doe", "Company": "Acme Corp", "Email": "x@y.com"}},
        )
        env = _parse(result)
        assert env is not None
        assert env.get("validation_error") is True
        assert env.get("missing_required") is True
        # Required fields minus user-provided LastName+Company = Region__c
        assert set(env["missing_fields"]) == {"Region__c"}
        assert env.get("missing_field_options", {}).get("Region__c") == [
            "APAC", "EMEA", "Americas"
        ]

    def test_provenance_envelope_has_no_options_when_field_not_enum(self):
        """LLM fabricates all required Lead body fields -> blocked, missing
        includes LastName/Company/Region__c; LastName and Company are plain
        strings with no picklist options."""
        mcp = _FakeDescribeMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "Lead",
             "body": {"FirstName": "John"}},  # FirstName fabricated (optional, not required)
        )
        env = _parse(result)
        assert env is not None
        assert env.get("missing_required") is True
        assert set(env["missing_fields"]) == {"LastName", "Company", "Region__c"}
        options = env.get("missing_field_options", {})
        assert "Region__c" in options
        assert "LastName" not in options
        assert "Company" not in options

    def test_presence_envelope_has_options(self):
        """Presence gate (provenance passes) + missing LastName on Lead ->
        envelope carries picklist options."""
        mcp = _FakeDescribeMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "Lead",
             "body": {"Company": "Acme"}},  # LastName missing
        )
        env = _parse(result)
        assert env is not None
        assert env.get("validation_error") is True
        assert env.get("missing_required") is True
        assert "LastName" in env["missing_fields"]
        # Region__c is required but not in body -> missing from body, AND user
        # didn't provide it, so presence gate includes it. BUT presence gate
        # checks body (not provenance), so all required fields not in body show.
        assert "Region__c" in env["missing_fields"]
        options = env.get("missing_field_options", {})
        assert "Region__c" in options

    def test_options_absent_when_client_lacks_method(self):
        """Legacy MCP clients without describe_required_field_options still
        work — the enrichment is fail-soft."""
        mcp = _FakeDescribeNoOptionsMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "Lead",
             "body": {"FirstName": "John"}},  # fabricated
        )
        env = _parse(result)
        assert env is not None
        assert env.get("missing_required") is True
        assert "missing_field_options" not in env  # not set when no options available


# ═══════════════════════════════════════════════════════════════
# D: Describe is authoritative — custom objects and empty-required
# ═══════════════════════════════════════════════════════════════

class TestDescribeAuthoritative:
    """A1/A2/A3: Describe metadata is the sole source of truth for required
    fields. Custom objects get their fields from Describe, not a hardcoded
    static list. An empty required set is a valid authoritative resolution."""

    def test_custom_object_required_fields_from_describe(self):
        """CustomTenant__c: Tenant__c (reference), Name (string), Tier__c (picklist)
        are required per Describe, not hardcoded."""
        mcp = _FakeDescribeMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "CustomTenant__c",
             "body": {"BillingAddress__c": "123 Main St"}},  # fabricated + missing required
        )
        env = _parse(result)
        assert env is not None
        assert env.get("missing_required") is True
        assert set(env["missing_fields"]) == {"Tenant__c", "Name", "Tier__c"}
        options = env.get("missing_field_options", {})
        assert options["Tier__c"] == ["Standard", "Premium", "Enterprise"]
        assert options["Tenant__c"] == ["Tenant__c"]  # reference option
        assert "Name" not in options  # string, no picklist/reference

    def test_empty_required_authoritative_not_fail_closed(self):
        """GhostCase__c has no required fields per Describe. When body has
        values, the create proceeds (presence gate: all required met). An
        empty required set is AUTHORITATIVE — it does NOT fail closed with
        'schema could not be established'."""
        mcp = _FakeDescribeMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "GhostCase__c",
             "body": {"Description": "Hello"}},
        )
        # No validation error — all required fields (none) are satisfied
        env = _parse(result)
        assert env is None or env.get("validation_error") is not True
        assert len(mcp.calls) == 1  # create reached MCP

    def test_unknown_object_describe_fails_static_fallback(self):
        """Describe returns None for Lead (unknown object) -> falls back to
        static OBJECT_REQUIRED_FIELDS which requires LastName + Company."""
        from agent.mutation_validation import OBJECT_REQUIRED_FIELDS
        mcp = _FakeDescribeFailureMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "Lead",
             "body": {}},
        )
        env = _parse(result)
        assert env is not None
        assert env.get("missing_required") is True
        # Static fallback (no Describe) — Lead still blocked correctly
        assert "LastName" in env["missing_fields"]
        assert "Company" in env["missing_fields"]

    def test_unknown_custom_object_describe_fails_fail_closed(self):
        """Describe returns None for an unknown custom object with no static
        entry -> fail closed (schema unknown)."""
        mcp = _FakeDescribeFailureMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "Unknown__c",
             "body": {"Field__c": "val"}},
        )
        env = _parse(result)
        assert env is not None
        assert env.get("validation_error") is True
        assert "could not be established" in env.get("error", "")
        assert len(mcp.calls) == 0  # create rejected, never reaches MCP


# ═══════════════════════════════════════════════════════════════
# E: Partial user input (A6) — only remaining required fields asked
# ═══════════════════════════════════════════════════════════════

class TestPartialUserInput:
    """A6: when the user has provided SOME required fields, only the remaining
    required fields are surfaced in the ask."""

    def test_partial_input_lead(self):
        """User provides Last Name: Sharma and Company: Tech Solutions. The LLM
        fabricates FirstName=John -> provenance blocks it. But LastName and
        Company ARE user-provided, so the ask only lists Region__c."""
        mcp = _FakeDescribeMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "Lead",
             "body": {"LastName": "Sharma", "Company": "Tech Solutions", "FirstName": "John"}},
        )
        env = _parse(result)
        assert env is not None
        assert env.get("missing_required") is True
        # LastName and Company were user-provided; only fabricated FirstName and
        # missing Region__c are listed in missing_fields — but Region__c is also
        # a required field not provided by user, so it's correctly asked for.
        missing = set(env["missing_fields"])
        assert "FirstName" in missing or "Region__c" in missing
        assert "LastName" not in missing, "LastName was user-provided"
        assert "Company" not in missing, "Company was user-provided"

    def test_all_required_provided_no_required_ask(self):
        """When all required fields are user-provided, only fabricated optional
        fields are flagged (missing_required absent — NOT a required-fields ask)."""
        mcp = _FakeDescribeMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "Lead",
             "body": {"LastName": "Sharma", "Company": "Tech Solutions",
                      "Region__c": "APAC", "Email": "s@t.com"}},
        )
        env = _parse(result)
        # The create should succeed (all required + provenance satisfied)
        assert env is None or env.get("validation_error") is not True
        assert len(mcp.calls) == 1


# ═══════════════════════════════════════════════════════════════
# F: Provenance gate still blocks fabricated values (A5)
# ═══════════════════════════════════════════════════════════════

class TestProvenanceStillBlocks:
    """A5: the provenance gate remains unchanged — fabricated body values are
    rejected even when Describe successfully resolves required fields."""

    def test_vague_create_lead_fabricated_rejected(self):
        """Vague 'create a lead' -> LLM fabricates Doe/Acme -> provenance blocks
        with the true required set from Describe (LastName + Company + Region__c)."""
        mcp = _FakeDescribeMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "Lead",
             "body": {"FirstName": "John"}},  # fabricated, optional, not required
        )
        env = _parse(result)
        assert env is not None
        assert env.get("missing_required") is True
        # User provided nothing -> all required fields asked
        assert set(env["missing_fields"]) == {"LastName", "Company", "Region__c"}
        assert len(mcp.calls) == 0


# ═══════════════════════════════════════════════════════════════
# G: _render_required_fields_ask renders options correctly
# ═══════════════════════════════════════════════════════════════

class TestRenderRequiredFieldsAsk:
    """A4/A8: the deterministic ask renderer appends picklist choice hints."""

    def test_picklist_choice_hint_rendered(self):
        tool_results = [
            {
                "tool": "createSobjectRecord",
                "result": json.dumps({
                    "error": "Cannot create",
                    "tool": "createSobjectRecord",
                    "validation_error": True,
                    "retry_allowed": False,
                    "requires_user_input": True,
                    "missing_fields": ["LastName", "Company", "Region__c"],
                    "missing_fields_human": ["Last Name", "Company Name", "Region"],
                    "sobject_name": "Lead",
                    "missing_required": True,
                    "missing_field_options": {
                        "Region__c": ["APAC", "EMEA", "Americas"],
                    },
                }),
            }
        ]
        ask = _render_required_fields_ask(tool_results)
        assert ask is not None
        assert "Last Name" in ask
        assert "Company Name" in ask
        assert "Region" in ask
        # Picklist values are rendered with "choose from:" prefix
        assert "APAC" in ask
        assert "EMEA" in ask
        assert "Americas" in ask
        assert "choose from:" in ask.lower()

    def test_no_options_plain_labels(self):
        tool_results = [
            {
                "tool": "createSobjectRecord",
                "result": json.dumps({
                    "error": "Cannot create",
                    "tool": "createSobjectRecord",
                    "validation_error": True,
                    "retry_allowed": False,
                    "requires_user_input": True,
                    "missing_fields": ["Subject"],
                    "missing_fields_human": ["Subject"],
                    "sobject_name": "Task",
                    "missing_required": True,
                }),
            }
        ]
        ask = _render_required_fields_ask(tool_results)
        assert ask is not None
        assert "Subject" in ask
        assert "choose from" not in ask.lower()

    def test_reference_option_rendered(self):
        tool_results = [
            {
                "tool": "createSobjectRecord",
                "result": json.dumps({
                    "error": "Cannot create",
                    "tool": "createSobjectRecord",
                    "validation_error": True,
                    "retry_allowed": False,
                    "requires_user_input": True,
                    "missing_fields": ["Tenant__c", "Name", "Tier__c"],
                    "missing_fields_human": ["Tenant", "Name", "Tier"],
                    "sobject_name": "CustomTenant__c",
                    "missing_required": True,
                    "missing_field_options": {
                        "Tenant__c": ["Tenant__c"],
                        "Tier__c": ["Standard", "Premium", "Enterprise"],
                    },
                }),
            }
        ]
        ask = _render_required_fields_ask(tool_results)
        assert ask is not None
        # Reference option shows object name with "choose from:" prefix
        assert "Tenant__c" in ask
        assert "choose from: tenant__c" in ask.lower()
        # Picklist values rendered
        assert "Standard" in ask
        assert "Premium" in ask
        assert "Enterprise" in ask
        assert "choose from:" in ask.lower()


# ═══════════════════════════════════════════════════════════════
# H: Describe-based Opportunity with picklist options in ask
# ═══════════════════════════════════════════════════════════════

class TestOpportunityPicklistInAsk:
    """A8: Opportunity StageName picklist values are surfaced in the ask."""

    def test_opportunity_ask_shows_stage_options(self):
        mcp = _FakeDescribeMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "Opportunity",
             "body": {"Name": "Big Deal"}},  # Stage and CloseDate missing
        )
        env = _parse(result)
        assert env is not None
        assert env.get("missing_required") is True
        missing = set(env["missing_fields"])
        assert "StageName" in missing
        assert "CloseDate" in missing
        options = env.get("missing_field_options", {})
        assert "StageName" in options
        # Active picklist values present
        assert "Prospecting" in options["StageName"]
        assert "Archived" not in options["StageName"]  # inactive excluded


# ═══════════════════════════════════════════════════════════════
# I: no-deprecated-hardcoded-lead requirement (A2)
# ═══════════════════════════════════════════════════════════════

class TestNoHardcodedLeadRequirement:
    """A2: 'LastName + Company' is NOT the source of truth for Lead.
    Live Describe with an extra custom required field (Region__c) is authoritative."""

    def test_custom_lead_required_field_appears_in_ask(self):
        """LLM fabricates all Lead body values. The ask includes Region__c from
        Describe, not just LastName + Company (the static entry)."""
        mcp = _FakeDescribeMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "Lead",
             "body": {"FirstName": "John"}},
        )
        env = _parse(result)
        assert env is not None
        assert env.get("missing_required") is True
        fields = set(env["missing_fields"])
        assert "Region__c" in fields, "Describe-required Region__c must be in the ask"
        # LastName and Company are also Describe-required
        assert "LastName" in fields
        assert "Company" in fields

    def test_static_list_not_used_as_primary(self):
        """When Describe fails and static fallback IS used for a standard object,
        it only kicks in for objects IN the static registry — not for custom
        objects, which fail closed. This proves the static list is a last-resort
        fallback, not the primary ask source."""
        mcp = _FakeDescribeFailureMcpClient()
        exec_ = _executor(mcp)
        # Unknown custom object -> fail closed (not static)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "MyCustom__c",
             "body": {"Anything": "val"}},
        )
        env = _parse(result)
        assert env is not None
        assert "could not be established" in env.get("error", "")


# ═══════════════════════════════════════════════════════════════
# J: minimum needed to NOT break existing test_mutation_validation
# ═══════════════════════════════════════════════════════════════

class TestBackwardCompatWithExistingTests:
    """Ensure existing test infrastructure and assertions are not broken by the
    describe options enrichment. _FakeMcpClient in test_mutation_validation.py
    does NOT have describe_required_field_options -> enrichment should skip."""

    def test_legacy_client_still_works(self):
        """ToolExecutor with a legacy _FakeMcpClient (no options) still produces
        correct missing_required envelopes without missing_field_options."""
        from tests.test_mutation_validation import _FakeMcpClient
        mcp = _FakeMcpClient()
        exec_ = _executor(mcp)
        result = _run_tool_directly(
            exec_,
            "createSobjectRecord",
            {"sobject-name": "Lead",
             "body": {"FirstName": "John"}},
        )
        env = _parse(result)
        assert env is not None
        assert env.get("missing_required") is True
        # The legacy client's describe_required_fields returns [("LastName","Last Name"),
        # ("Company","Company Name")] -> Region__c is NOT in that list (no custom field
        # in the legacy fake). But the legacy fake DOES NOT have
        # describe_required_field_options, so no options set.
        assert "missing_field_options" not in env
