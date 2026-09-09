"""
Production Mutation Safety Validation.

Every mutating/destructive Salesforce tool call is required to pass a mandatory
validation gate before execution. The gate enforces FAIL-CLOSED semantics:

- Known standard objects use local static required-field rules (zero I/O).
- Unknown/custom objects resolve their required fields from live Salesforce
  Describe metadata (read-only path) supplied by the caller.
- If required fields cannot be established (unknown object AND Describe fails),
  the mutation is REJECTED — never guessed, never fabricated.
- No fabricated/default mutation values ("Unknown", "Individual", "New Account",
  etc.) are ever inserted. A field value may only be satisfied by data the user
  explicitly provided or that is deterministically derived from user data.

The executor remains the final hard mutation gate. This module only supplies
the validation logic; it never calls Salesforce itself.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Validation failure contract (mutations are rejected via these fields)
# ──────────────────────────────────────────────────────────────

# ─── Static required-field registry for known standard objects ───
# Maps lowercase sobject names to a list of (field_api_name, human_label).
# This registry is the ZERO-I/O FALLBACK used only when live Salesforce Describe
# metadata is unavailable. The values reflect the intrinsic API requirements of
# the standard objects and deliberately DO NOT include fields Salesforce can
# default on create:
#   - Case: no field is universally required on create; Subject is optional and
#     Status is defaulted (unresolved -> the describe path decides; an empty body
#     is still rejected below as a vague request).
#   - Task: Subject is required; Status defaulted on create (so NOT required).
OBJECT_REQUIRED_FIELDS: dict[str, list[tuple[str, str]]] = {
    "lead":        [("LastName", "Last Name"), ("Company", "Company Name")],
    "contact":     [("LastName", "Last Name")],
    "account":     [("Name", "Account Name")],
    "opportunity": [("Name", "Opportunity Name"), ("StageName", "Stage"),
                    ("CloseDate", "Close Date")],
    "case":        [],
    "task":        [("Subject", "Subject")],
}

# ─── Salesforce ID format ───
# Salesforce record IDs are case-sensitive base-62 encoded strings of length 15
# (non-entity-history) or 18 (with entity-history suffix).
_SF_ID_RE = re.compile(r"^[0-9A-Za-z]{15}$|^[0-9A-Za-z]{18}$")


def is_valid_salesforce_id(value: Any) -> bool:
    """Return True only for a well-formed Salesforce record ID (15 or 18 chars)."""
    if not isinstance(value, str):
        return False
    return bool(_SF_ID_RE.match(value))


def _is_present(value: Any) -> bool:
    """A field value satisfies a required field only if explicitly non-blank."""
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    return True


def _missing_required(required: list[tuple[str, str]], body: dict[str, Any]) -> list[tuple[str, str]]:
    """Return the subset of required (api_name, label) pairs missing/blank in body."""
    missing: list[tuple[str, str]] = []
    for api_name, label in required:
        if api_name not in body or not _is_present(body.get(api_name)):
            missing.append((api_name, label))
    return missing


def _build_validation_error(
    tool_name: str,
    sobject_name: str,
    missing: list[tuple[str, str]],
    reason: str,
    missing_required: bool = False,
) -> str:
    """Build the structured validation-error envelope (machine readable + human readable).

    ``missing_required=True`` marks the case where the ``missing`` list is the
    createable-and-required set the user must supply for a blocked CREATE (as
    opposed to a structural error with no field list)."""
    import json

    api_names = [m[0] for m in missing]
    human_labels = [m[1] for m in missing]
    human_list = ", ".join(human_labels) if human_labels else ""
    error = (
        f"Cannot {tool_name}: required fields are missing or blank. "
        f"Please provide: {human_list}." if human_list else reason
    )
    payload = {
        "error": error,
        "tool": tool_name,
        "validation_error": True,
        "retry_allowed": False,
        "requires_user_input": True,
        "missing_fields": api_names,
        "missing_fields_human": human_labels,
        "sobject_name": sobject_name,
        "suggestion": (
            "Provide the missing required fields in the request body before retrying. "
            "Do not retry automatically; ask the user to supply the values."
        ),
    }
    if missing_required:
        payload["missing_required"] = True
    return json.dumps(payload)


def validate_mutation_fields(
    tool_name: str,
    arguments: dict[str, Any],
    describe_resolver: Callable[[str], list[tuple[str, str]] | None] | None = None,
) -> str | None:
    """
    Validate a mutation tool call. Returns None when the mutation may proceed, or
    a structured validation-error JSON string (fail-closed) when it must be blocked.

    Args:
        tool_name: The mutation tool name.
        arguments: The raw tool arguments dict.
        describe_resolver: Optional callable resolving required fields for an object
            from live Salesforce Describe (read-only). It must return a list of
            (api_name, human_label) tuples, or None when the schema cannot be
            established. Used only for objects not in the static registry.
    """
    # sobject-name (present for create/update/delete; optional for related/upload)
    sobject_name = ""
    for key in ("sobject-name", "sobject_name", "sobject", "object", "sobjectName", "objectName"):
        if key in arguments and arguments[key]:
            sobject_name = str(arguments[key]).strip()
            break

    raw_id = (
        arguments.get("id")
        or arguments.get("record_id")
        or (arguments.get("arguments", {}) or {}).get("id")
        or ""
    )

# ── createSobjectRecord: required fields on the object ──
    if tool_name == "createSobjectRecord":
        if not sobject_name:
            return _build_validation_error(
                tool_name, sobject_name, [],
                "Cannot create a record: the Salesforce object name is missing.",
            )
        body = _extract_clean_body(arguments)
        if not isinstance(body, dict):
            return _build_validation_error(
                tool_name, sobject_name, [],
                f"Cannot create {sobject_name}: the body must be a JSON object of fields.",
            )

        # REQUIRED-FIELD METADATA RESOLUTION (fail-closed):
        # 1) Live Describe metadata when a resolver is available — authoritative
        #    for BOTH standard and custom objects, so per-org custom required
        #    fields are honored and nothing is blindly hard-coded (a field
        #    Salesforce can default is never required).
        # 2) Static registry (OBJECT_REQUIRED_FIELDS) as a zero-I/O fallback when
        #    Describe metadata is unavailable.
        # 3) Neither available (unknown object) -> FAIL CLOSED, never guessed.
        required: list[tuple[str, str]] | None = None
        describe_known = False
        if describe_resolver is not None:
            try:
                describe_fields = describe_resolver(sobject_name)
                describe_known = describe_fields is not None
            except Exception as exc:  # noqa: BLE001 - resolver failure fails closed
                logger.error(
                    f"[MUTATION-VALIDATION] Describe resolver failed for '{sobject_name}': {exc}"
                )
                describe_fields = None
                describe_known = False
            if describe_known:
                required = describe_fields
        if not describe_known:
            required = OBJECT_REQUIRED_FIELDS.get(sobject_name.lower())
            if required is None:
                return _fail_closed_unknown_schema(tool_name, sobject_name)

        required = [(api, lbl) for (api, lbl) in required if api and lbl]

        # A create with ZERO fields never reaches Salesforce — fail-closed for
        # vague requests even when the schema reports no required fields.
        if not body:
            if required:
                return _build_validation_error(tool_name, sobject_name, required,
                                               f"Cannot create {sobject_name}: required fields missing.",
                                               missing_required=True)
            return _build_validation_error(
                tool_name, sobject_name, [],
                f"Cannot create {sobject_name}: no fields were provided. Please provide "
                "the values to set on the new record.",
            )

        missing = _missing_required(required, body)
        if missing:
            return _build_validation_error(tool_name, sobject_name, missing,
                                           f"Cannot create {sobject_name}: required fields missing.",
                                           missing_required=True)
        return None

    if tool_name == "updateSobjectRecord":
        if not sobject_name:
            return _build_validation_error(
                tool_name, sobject_name, [],
                "Cannot update a record: the Salesforce object name is missing.",
            )
        if not is_valid_salesforce_id(raw_id):
            return _build_validation_error(
                tool_name, sobject_name, [],
                f"Cannot update {sobject_name}: the record id '{raw_id}' is not a valid "
                "Salesforce record ID.",
            )
        body = _extract_clean_body(arguments)
        if not isinstance(body, dict) or not body:
            return _build_validation_error(
                tool_name, sobject_name, [],
                f"Cannot update {sobject_name}: no fields were provided to update.",
            )
        return None

    if tool_name == "updateRelatedRecord":
        rel_path = arguments.get("relationship-path") or arguments.get("relationship_path") or ""
        if not is_valid_salesforce_id(raw_id):
            return _build_validation_error(
                tool_name, sobject_name or raw_id, [],
                "Cannot update the related record: the record id is not a valid "
                "Salesforce record ID.",
            )
        if not str(rel_path).strip():
            return _build_validation_error(
                tool_name, sobject_name or "record", [],
                "Cannot update the related record: the relationship path is missing.",
            )
        body = _extract_clean_body(arguments)
        if not isinstance(body, dict) or not body:
            return _build_validation_error(
                tool_name, sobject_name or "record", [],
                "Cannot update the related record: no fields were provided.",
            )
        return None

    if tool_name == "uploadRecordAttachment":
        if not is_valid_salesforce_id(raw_id):
            return _build_validation_error(
                tool_name, "Attachment", [],
                "Cannot upload the attachment: the record id is not a valid Salesforce record ID.",
            )
        file_name = arguments.get("file_name") or ""
        content = arguments.get("file_content_base64") or ""
        if not str(file_name).strip():
            return _build_validation_error(
                tool_name, "Attachment", [],
                "Cannot upload the attachment: the file name is missing.",
            )
        if not str(content).strip():
            return _build_validation_error(
                tool_name, "Attachment", [],
                "Cannot upload the attachment: the file content is missing.",
            )
        return None

    if tool_name == "deleteSobjectRecord":
        if not is_valid_salesforce_id(raw_id):
            return _build_validation_error(
                tool_name, sobject_name or "record", [],
                "Cannot delete the record: the record id is not a valid Salesforce record ID.",
            )
        return None

    if tool_name == "deleteRelatedRecord":
        rel_path = arguments.get("relationship-path") or arguments.get("relationship_path") or ""
        if not is_valid_salesforce_id(raw_id):
            return _build_validation_error(
                tool_name, sobject_name or "record", [],
                "Cannot delete the related record: the record id is not a valid "
                "Salesforce record ID.",
            )
        if not str(rel_path).strip():
            return _build_validation_error(
                tool_name, sobject_name or "record", [],
                "Cannot delete the related record: the relationship path is missing.",
            )
        return None

    # Non-mutation tool: no validation applied.
    return None


def _fail_closed_unknown_schema(tool_name: str, sobject_name: str) -> str:
    """
    Fail-closed envelope for an object whose required-field schema could not be
    established (Describe unresolved AND no static registry entry). The mutation
    is rejected — required fields are never guessed and values never fabricated.
    """
    import json
    return json.dumps({
        "error": (
            f"Cannot create {sobject_name}: the required-field schema for this "
            "object could not be established. Refusing to create without "
            "confirmed required fields."
        ),
        "tool": tool_name,
        "validation_error": True,
        "retry_allowed": False,
        "requires_user_input": True,
        "missing_fields": [],
        "missing_fields_human": [],
        "sobject_name": sobject_name,
        "suggestion": (
            "Ask the user which fields this object requires, or which values to "
            "set, before creating it. The schema could not be auto-resolved."
        ),
    })


def _extract_clean_body(arguments: dict[str, Any]) -> dict[str, Any]:
    """Extract the field body the same way the client does (no cross-import at module
    load to keep validation logic self-contained and import-order safe)."""
    if "body" in arguments and isinstance(arguments["body"], dict):
        return dict(arguments["body"])
    if "fields" in arguments and isinstance(arguments["fields"], dict):
        return dict(arguments["fields"])
    if "record" in arguments and isinstance(arguments["record"], dict):
        return dict(arguments["record"])
    ignore_keys = {
        "sobject-name", "sobject_name", "sobject", "object", "sobjectName", "objectName",
        "id", "record_id", "relationship-path", "relationship_path", "name",
        "file_name", "file_content_base64", "arguments",
    }
    return {k: v for k, v in arguments.items() if k not in ignore_keys}
