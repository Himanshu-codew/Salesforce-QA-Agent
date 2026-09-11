"""
Core Salesforce Agent — the main agent loop that ties together
the LLM (Qwen3), MCP executor, memory, and planner.
"""

import asyncio
import json
import logging
import os
import re
from typing import Any, AsyncGenerator

from llm.base import BaseLLM
from sfmcp.executor import ToolExecutor
from tools.salesforce import get_tool_definitions, is_read_only, is_mutating, is_destructive
from .memory import ConversationMemory
from .planner import TaskPlanner
from .prompts import SYSTEM_PROMPT, TOOL_CALLING_PROMPT, ERROR_MESSAGES
from .rag import ToolRAGRetriever

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Bounded-latency / resilience configuration (agent.agent path)
#
# These reuse the SAME environment variables and defaults as the Orchestrator
# (agent/multi_agent.py) so there is a single source of truth through the env.
# agent.agent cannot import them from multi_agent directly (that would create a
# circular import), so they are re-read here with identical names/values.
# ──────────────────────────────────────────────────────────────
AGENT_LLM_TIMEOUT = float(os.getenv("LLM_STAGE_TIMEOUT", "90.0"))
AGENT_EXECUTOR_TIMEOUT = float(os.getenv("EXECUTOR_TIMEOUT", "90.0"))
RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "120.0"))


class AgentTimeoutError(Exception):
    """A controlled timeout failure for the agent.agent (SalesforceAgent) path.

    Is raised only after an `asyncio.wait_for` bound fires. It is NOT a
    Salesforce result and MUST NOT be let through as successful synthesis input.
    """

    def __init__(self, stage: str):
        super().__init__(f"The '{stage}' step timed out. Please try again.")
        self.stage = stage


async def _bounded_call(awaitable: Any, timeout: float, stage: str) -> Any:
    """Run an awaitable under `asyncio.wait_for`, raising a controlled timeout.

    - On `asyncio.TimeoutError` -> raises `AgentTimeoutError(stage)`.
    - `asyncio.CancelledError` is a BaseException and is NOT caught here, so it
      propagates naturally to the caller / task cancellation.
    """
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except asyncio.TimeoutError:
        logger.error(f"[TIMEOUT] SalesforceAgent '{stage}' exceeded {timeout:.0f}s.")
        raise AgentTimeoutError(stage) from None


def _timeout_error_event(stage: str) -> dict[str, Any]:
    """Build the agent.agent structured error event for a timeout."""
    return {
        "type": "error",
        "code": "TIMEOUT",
        "message": f"The '{stage}' step timed out. Please try a simpler or more specific query.",
        "data": f"The '{stage}' step timed out. Please try again.",
    }


# ──────────────────────────────────────────────────────────────
# D1: Executor error-envelope detection.
#
# ToolExecutor.execute() converts every failure into a JSON string envelope
# (see sfmcp/executor.py), e.g.:
#   {"error": "..."}
#   {"error": "...", "tool": "..."}
#   {"error": "...", "tool": "...", "suggestion": "..."}
#
# A top-level non-empty string "error" key is exclusive to the executor's
# failure envelope — a normal successful Salesforce result never carries one.
# We detect it here (without requiring the "tool" field, because tool-not-found
# errors omit it) and surface a controlled SALESFORCE_FAILED event instead of
# letting the envelope reach memory / tool_results_fetched / synthesis as
# ordinary data.
# ──────────────────────────────────────────────────────────────
SALESFORCE_FAILED = "SALESFORCE_FAILED"


def _executor_error_message(result: Any) -> str | None:
    """
    Detect the executor's failure envelope ({"error": ..., ...}) and return a
    human-readable message, or None when the result is a normal tool result.

    Never throws on malformed/non-JSON input. Does NOT require the "tool"
    field (tool-not-found envelopes omit it).

    A VALIDATION failure ({"validation_error": true, ...}) is NOT treated as a
    fatal SALESFORCE_FAILED: it returns None so the result flows to memory /
    synthesis as a normal tool result. The LLM then reads
    requires_user_input / retry_allowed=false and asks the user for the missing
    fields instead of aborting the turn (see #8 stop-on-validation-failure).
    """
    if not isinstance(result, str):
        return None
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    if parsed.get("validation_error") is True:
        return None
    error = parsed.get("error")
    if not isinstance(error, str) or not error.strip():
        return None
    suggestion = parsed.get("suggestion")
    if isinstance(suggestion, str) and suggestion.strip():
        return f"{error} Suggestion: {suggestion}"
    return error


def _blocked_mutation_result(tc: dict) -> str:
    """Synthetic fail-closed result for a mutation that was NOT executed because
    ANOTHER mutation in the SAME response failed validation (#8 all-or-nothing).
    Carries the same validation_error envelope so it flows to synthesis and the
    agent asks the user for input; ZERO mutations from this response ran."""
    args = tc.get("arguments", {}) if isinstance(tc.get("arguments"), dict) else {}
    sobject_name = ""
    for key in ("sobject-name", "sobject_name", "sobject", "object", "sobjectName", "objectName"):
        if args.get(key):
            sobject_name = str(args[key]).strip()
            break
    return json.dumps({
        "error": (
            f"Tool '{tc['name']}' was NOT executed: a mutation in this response "
            "failed required-field validation, so ZERO mutations from this response "
            "were executed (all-or-nothing fail-closed)."
        ),
        "tool": tc["name"],
        "validation_error": True,
        "retry_allowed": False,
        "requires_user_input": True,
        "missing_fields": [],
        "missing_fields_human": [],
        "sobject_name": sobject_name,
        "suggestion": (
            "Ask the user for the missing required fields, then submit a corrected "
            "request. No record was created or modified."
        ),
    })


def _salesforce_failed_event(message: str) -> dict[str, Any]:
    """Build the agent.agent structured error event for an executor failure."""
    return {
        "type": "error",
        "code": SALESFORCE_FAILED,
        "message": message,
        "data": message,
    }


# ──────────────────────────────────────────────────────────────
# Intent-Aware Tool Filtering (read-only fast path)
# Shared write-intent keyword list. Reused by the loop's batch-mode
# detection so there is a single source of truth for read/write intent.
# READ_ONLY_MODE is NOT touched here — it remains the final execution gate.
# ──────────────────────────────────────────────────────────────
_WRITE_KEYWORDS = [
    "create", "add", "insert", "new", "make", "banao", "daalo",
    "update", "edit", "change", "modify", "badlo", "set",
    "delete", "remove", "hatao", "mitao", "drop",
]


def _has_write_intent(user_message: str) -> bool:
    """Return True when a message clearly requests a create/update/delete/upload action."""
    if not user_message:
        return False
    lowered = user_message.lower()
    return any(kw in lowered for kw in _WRITE_KEYWORDS)


# ──────────────────────────────────────────────────────────────
# Salesforce / data-intent detection.
# Reuses the same keyword set as the Turn-1 Data Query Interceptor so there is
# a single source of truth. Used by E5 to (a) decide RAG-empty/timeout fallback
# to the complete read-only tool registry and (b) guard against ungrounded
# answers for Salesforce-specific requests.
# ──────────────────────────────────────────────────────────────
_DATA_INTENT_KEYWORDS = [
    "account", "accounts", "lead", "leads", "contact", "contacts",
    "opportunity", "opportunities", "case", "cases", "task", "tasks",
    "event", "events", "user", "who am i", "schema", "fields",
    "profile", "identity", "my profile", "my email", "my role",
    "my username", "user info", "user information",
    "show", "list", "select", "find", "search", "count", "how many",
    "delete", "remove", "update", "edit", "create", "banao", "dikhao",
    "hatao", "badlo", "kitne", "saare",
]


def _has_salesforce_intent(user_message: str) -> bool:
    """Return True when a message clearly requests Salesforce data/action access."""
    if not user_message:
        return False
    lowered = user_message.lower()
    return any(kw in lowered for kw in _DATA_INTENT_KEYWORDS)


def filter_tools_for_query(tools: list[dict[str, Any]], user_message: str) -> list[dict[str, Any]]:
    """
    Intent-aware tool-schema filtering for the ACTIVE /chat path.

    For clearly read-only requests, mutation/destructive tools are removed from
    the schema passed to Qwen so the model cannot propose them (e.g. RAG returning
    createSobjectRecord for "How many Account records do we have?"). Genuine write
    or compound requests keep the full tool set, preserving mutation semantics.

    READ_ONLY_MODE is intentionally neither read nor modified here: planner.py and
    executor.py remain the authoritative final safety gate.
    """
    if _has_write_intent(user_message):
        return tools
    removed = [
        t["function"].get("name", "")
        for t in tools
        if not is_read_only(t.get("function", {}).get("name", ""))
    ]
    if removed:
        logger.info(
            f"[READ-ONLY FILTER] Pure-read query; removed mutation/destructive "
            f"tools from Qwen schema: {removed}"
        )
    return [t for t in tools if is_read_only(t.get("function", {}).get("name", ""))]


def _detect_direct_intent(user_message: str) -> tuple[str, dict[str, Any]] | None:
    """
    Zero-latency deterministic router for standard common queries.
    Bypasses LLM prefill delay on slow backends (e.g. 32B model on Kaggle)
    and directly dispatches the exact MCP tool.
    """
    if not user_message:
        return None
    cleaned = re.sub(r"[^\w\s]", " ", user_message.lower()).strip()
    words = cleaned.split()

    # 1. User info / profile ("who am i", "my user info", etc.)
    user_info_phrases = [
        "who am i", "my user info", "show my user info", "show user info",
        "user info", "my profile", "current user info", "show my salesforce user info",
    ]
    if any(phrase in cleaned for phrase in user_info_phrases):
        if not any(w in words for w in ["update", "delete", "create", "change", "set"]):
            return ("getUserInfo", {})

    # 2. Objects available / schema list
    # e.g., "What objects are available in my Salesforce org?", "list objects", "available objects"
    object_schema_triggers = [
        "what objects are available", "available objects", "list objects",
        "show objects", "all objects in my salesforce", "all objects in salesforce",
        "list all objects", "kya objects hai", "show available objects",
        "objects are available", "objects available", "what objects are in",
    ]
    if any(trigger in cleaned for trigger in object_schema_triggers):
        if not any(w in words for w in ["record", "records", "count", "where", "delete", "update", "create", "insert"]):
            return ("getObjectSchema", {})

    # 3. Recent records (recent accounts, recent leads, etc.)
    if "recent" in words:
        if "account" in words or "accounts" in words:
            return ("listRecentSobjectRecords", {"sobject-name": "Account"})
        if "lead" in words or "leads" in words:
            return ("listRecentSobjectRecords", {"sobject-name": "Lead"})
        if "contact" in words or "contacts" in words:
            return ("listRecentSobjectRecords", {"sobject-name": "Contact"})
        if "opportunity" in words or "opportunities" in words or "opp" in words or "opps" in words:
            return ("listRecentSobjectRecords", {"sobject-name": "Opportunity"})
        if "case" in words or "cases" in words:
            return ("listRecentSobjectRecords", {"sobject-name": "Case"})

    return None



# ──────────────────────────────────────────────────────────────
# Python-Side Salesforce Table Formatter
# Formats flat SOQL results and COUNT queries as clean markdown.
# Hierarchical/subquery results are left for the LLM to render as cards.
# ──────────────────────────────────────────────────────────────

# Fields to skip in auto-generated tables (internal Salesforce system fields)
_SKIP_FIELDS = {"attributes", "type", "url"}

# Preferred display order for common objects
_FIELD_ORDER = {
    "Lead":        ["Id", "FirstName", "LastName", "Name", "Company", "Email", "Phone", "Status", "LeadSource", "Industry", "Rating", "CreatedDate"],
    "Account":     ["Id", "Name", "Industry", "Phone", "Website", "AnnualRevenue", "NumberOfEmployees", "Type", "CreatedDate"],
    "Contact":     ["Id", "FirstName", "LastName", "Name", "Email", "Phone", "Title", "Department", "Account.Name", "CreatedDate"],
    "Opportunity": ["Id", "Name", "StageName", "Amount", "CloseDate", "Probability", "Account.Name", "Owner.Name", "CreatedDate"],
    "Case":        ["Id", "CaseNumber", "Subject", "Status", "Priority", "AccountId", "CreatedDate"],
    "Task":        ["Id", "Subject", "Status", "Priority", "ActivityDate", "CreatedDate", "Who.Name", "What.Name"],
    "Event":       ["Id", "Subject", "StartDateTime", "EndDateTime", "Who.Name", "What.Name"],
}


def _fmt_value(val: Any) -> str:
    """Format a single cell value for markdown display."""
    if val is None or val == "":
        return "-"
    if isinstance(val, dict):
        # Nested relationship object — extract Name or first string value
        return str(val.get("Name") or val.get("name") or next(
            (str(v) for v in val.values() if isinstance(v, str) and v), "-"
        ))
    s = str(val)
    # Format ISO timestamps → "18 Aug 2026, 11:56 AM"
    iso_match = re.match(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})", s)
    if iso_match:
        from datetime import datetime
        try:
            dt = datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            hour = dt.hour % 12 or 12
            ampm = "AM" if dt.hour < 12 else "PM"
            return f"{dt.day} {months[dt.month-1]} {dt.year}, {hour:02d}:{dt.minute:02d} {ampm}"
        except Exception:
            pass
    # Format date-only strings → "18 Aug 2026"
    date_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if date_match:
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        y, m, d = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
        return f"{d} {months[m-1]} {y}"
    return s


def _get_nested(record: dict, field: str) -> Any:
    """Get a (possibly nested) field value like 'Account.Name' from a record dict."""
    parts = field.split(".", 1)
    val = record.get(parts[0])
    if len(parts) == 2 and isinstance(val, dict):
        return val.get(parts[1])
    return val


def _detect_subquery_collections(record: dict) -> list[str]:
    """
    Detect subquery collection fields in a Salesforce record dict.
    These are fields whose value is a dict with 'totalSize' and 'records' keys,
    e.g. {"Opportunities": {"totalSize": 4, "records": [...]}, "Contacts": {...}}
    Returns a list of subquery field names in order.
    """
    collections = []
    for key, val in record.items():
        if key in _SKIP_FIELDS:
            continue
        if isinstance(val, dict) and "totalSize" in val and "records" in val:
            collections.append(key)
    return collections


def _fmt_currency(val: Any) -> str:
    """Format a numeric value as currency: $35,000."""
    if val is None or val == "":
        return "-"
    try:
        num = float(val)
        return f"${num:,.0f}"
    except (ValueError, TypeError):
        return str(val)


def _fmt_record_id(record: dict) -> str:
    """Extract the 18-char Salesforce ID from a record, or empty string."""
    raw = record.get("Id", "")
    return str(raw)[:18] if raw else ""


# ──────────────────────────────────────────────────────────────
# Per-type child record formatters
# Each receives a single child record dict and returns a one-line markdown string.
# ──────────────────────────────────────────────────────────────

def _fmt_child_opportunity(r: dict) -> str:
    """Format one Opportunity child record."""
    name = _fmt_value(r.get("Name", "Unnamed"))
    amount = _fmt_currency(r.get("Amount")) if r.get("Amount") else None
    stage = _fmt_value(r.get("StageName", "")) if r.get("StageName") else None
    close = _fmt_value(r.get("CloseDate", "")) if r.get("CloseDate") else None
    rid = _fmt_record_id(r)

    parts = [f"💰 **{name}**"]
    if amount:
        parts.append(f"— **{amount}**")
    meta = []
    if stage:
        meta.append(f"*Stage:* {stage}")
    if close:
        meta.append(f"*Close Date:* {close}")
    if meta:
        parts.append("| " + " | ".join(meta))
    if rid:
        parts.append(f"*(ID: {rid})*")
    return " ".join(parts)


def _fmt_child_contact(r: dict) -> str:
    """Format one Contact child record."""
    name = _fmt_value(r.get("Name", "Unknown"))
    rid = _fmt_record_id(r)
    email = _fmt_value(r.get("Email", "")) if r.get("Email") else None
    phone = _fmt_value(r.get("Phone", "")) if r.get("Phone") else None

    parts = [f"👤 **{name}**"]
    detail_parts = []
    if rid:
        detail_parts.append(f"ID: {rid}")
    if email:
        detail_parts.append(email)
    if phone:
        detail_parts.append(f"Phone: {phone}")
    if detail_parts:
        parts.append(f"*({' | '.join(detail_parts)})*")
    return " ".join(parts)


def _fmt_child_case(r: dict) -> str:
    """Format one Case child record."""
    case_num = _fmt_value(r.get("CaseNumber", ""))
    subject = _fmt_value(r.get("Subject", "No subject"))
    status = _fmt_value(r.get("Status", "")) if r.get("Status") else None
    priority = _fmt_value(r.get("Priority", "")) if r.get("Priority") else None

    parts = [f"🎫 **#{case_num}** — {subject}"]
    meta = []
    if status:
        meta.append(f"*Status:* {status}")
    if priority:
        meta.append(f"*Priority:* {priority}")
    if meta:
        parts.append("| " + " | ".join(meta))
    return " ".join(parts)


def _fmt_child_task(r: dict) -> str:
    """Format one Task child record."""
    subject = _fmt_value(r.get("Subject", "No subject"))
    status = _fmt_value(r.get("Status", "")) if r.get("Status") else None
    due = _fmt_value(r.get("ActivityDate", "")) if r.get("ActivityDate") else None

    parts = [f"✅ **{subject}**"]
    meta = []
    if status:
        meta.append(f"*Status:* {status}")
    if due:
        meta.append(f"*Due:* {due}")
    if meta:
        parts.append("| " + " | ".join(meta))
    return " ".join(parts)


def _fmt_child_lead(r: dict) -> str:
    """Format one Lead child record."""
    name = _fmt_value(r.get("Name", "Unknown"))
    company = _fmt_value(r.get("Company", "")) if r.get("Company") else None
    status = _fmt_value(r.get("Status", "")) if r.get("Status") else None

    parts = [f"📋 **{name}**"]
    if company:
        parts.append(f"— {company}")
    if status:
        parts.append(f"| *Status:* {status}")
    return " ".join(parts)


def _fmt_child_event(r: dict) -> str:
    """Format one Event child record."""
    subject = _fmt_value(r.get("Subject", "No subject"))
    start = _fmt_value(r.get("StartDateTime", "")) if r.get("StartDateTime") else None
    end = _fmt_value(r.get("EndDateTime", "")) if r.get("EndDateTime") else None

    parts = [f"📅 **{subject}**"]
    meta = []
    if start:
        meta.append(f"*Start:* {start}")
    if end:
        meta.append(f"*End:* {end}")
    if meta:
        parts.append("| " + " | ".join(meta))
    return " ".join(parts)


def _fmt_child_generic(r: dict) -> str:
    """Generic fallback: extract Name + key non-system fields cleanly."""
    name = r.get("Name") or r.get("Subject") or r.get("CaseNumber") or "Record"
    rid = _fmt_record_id(r)

    # Collect meaningful scalar fields (skip system metadata, nested objects, and duplicates of Name)
    skip = _SKIP_FIELDS | {"Name", "Subject", "CaseNumber", "Id"}
    detail_parts = []
    if rid:
        detail_parts.append(f"ID: {rid}")
    for k, v in r.items():
        if k in skip or isinstance(v, (dict, list)) or v is None or v == "":
            continue
        detail_parts.append(f"{k}: {_fmt_value(v)}")
        if len(detail_parts) >= 5:
            break

    parts = [f"📄 **{_fmt_value(name)}**"]
    if detail_parts:
        parts.append(f"*({' | '.join(detail_parts)})*")
    return " ".join(parts)


# Registry: relationship name → formatter (checked first), then child type → formatter
_CHILD_FORMATS_BY_REL: dict[str, callable] = {
    "Opportunities": _fmt_child_opportunity,
    "Contacts": _fmt_child_contact,
    "Cases": _fmt_child_case,
    "Tasks": _fmt_child_task,
    "Events": _fmt_child_event,
}

_CHILD_FORMATS_BY_TYPE: dict[str, callable] = {
    "Opportunity": _fmt_child_opportunity,
    "Contact": _fmt_child_contact,
    "Case": _fmt_child_case,
    "Task": _fmt_child_task,
    "Lead": _fmt_child_lead,
    "Event": _fmt_child_event,
}


def _format_subquery_child(
    child_records: list[dict],
    parent_obj_type: str,
    child_rel_name: str,
) -> str:
    """
    Format a list of child records from a SOQL subquery into markdown bullet lines.
    Uses relationship-name-based dispatch first, then child-type-based dispatch,
    then a clean generic fallback. No raw field dumps.

    Output examples:
      * 💰 **Edge SLA** — **$60,000** | *Stage:* Closed Won | *Close Date:* 08 Mar 2026
      * 👤 **Sean Forbes** *(ID: 003g500000NhEDmAAN | sean@edge.com)*
    """
    if not child_records:
        return ""

    # Determine child object type from first record's attributes
    child_type = ""
    first = child_records[0]
    if isinstance(first.get("attributes"), dict):
        child_type = first["attributes"].get("type", "")

    # Resolve formatter: relationship name takes priority over type
    formatter = _CHILD_FORMATS_BY_REL.get(child_rel_name) or _CHILD_FORMATS_BY_TYPE.get(child_type) or _fmt_child_generic

    lines = []
    for rec in child_records:
        line = formatter(rec)
        if line.strip():
            lines.append(f"  * {line.strip()}")

    return "\n".join(lines) if lines else ""


def _format_parent_with_children(record: dict, total_size: int) -> str:
    """
    Format a single parent record with its subquery children as a hierarchical card.

    Output format:
    ### 🏢 Edge Communications *(Electronics)*
    * **💰 Opportunities (4):**
      * 💰 **Edge Emergency Generator** — **$35,000** | *Stage:* Closed Won | *Close Date:* 21 Jun 2026
      * 💰 **Edge SLA** — **$60,000** | *Stage:* Closed Won | *Close Date:* 08 Mar 2026
    * **👤 Contacts (2):**
      * 👤 **Sean Forbes** *(ID: 003... | sean@edge.com)*
      * 👤 **Rose Gonzalez** *(ID: 003... | rose@edge.com)*

    Accounts with no children render a clean empty state:
    ### 🏢 ABC Technologies
    * *No linked Opportunities or Contacts*
    """
    # Determine object type
    obj_type = ""
    if isinstance(record.get("attributes"), dict):
        obj_type = record["attributes"].get("type", "")

    # Build parent card header
    parent_name = record.get("Name") or record.get("Subject") or record.get("Id", "Record")
    parent_subtitle_parts = []
    for hint_field in ["Industry", "Type", "Status", "StageName", "Rating"]:
        val = record.get(hint_field)
        if val:
            parent_subtitle_parts.append(_fmt_value(val))
    subtitle = f" *({' | '.join(parent_subtitle_parts)})*" if parent_subtitle_parts else ""

    card_lines = [f"### {_fmt_value(parent_name)}{subtitle}"]

    # Detect and format subquery collections
    collections = _detect_subquery_collections(record)

    # Icons for common relationship names
    _REL_ICONS = {
        "Opportunities": "💰",
        "Contacts": "👤",
        "Cases": "🎫",
        "Tasks": "✅",
        "Events": "📅",
        "Notes": "📝",
        "Attachments": "📎",
        "Quotes": "📋",
        "Orders": "📦",
    }

    if not collections:
        # No subquery children at all — clean empty state
        card_lines.append(f"  * *No linked Opportunities or Contacts*")
    else:
        for rel_name in collections:
            sub_data = record[rel_name]
            child_records = sub_data.get("records", [])
            child_count = sub_data.get("totalSize", len(child_records))
            icon = _REL_ICONS.get(rel_name, "📄")

            card_lines.append(f"* **{icon} {rel_name} ({child_count}):**")
            child_formatted = _format_subquery_child(child_records, obj_type, rel_name)
            if child_formatted:
                card_lines.append(child_formatted)
            else:
                card_lines.append(f"  * *No {rel_name.lower()} linked*")

    return "\n".join(card_lines)


def _is_soql_count(soql_query: str) -> bool:
    """Return True when the given SOQL is an aggregate COUNT query."""
    if not soql_query:
        return False
    return bool(re.search(r"\bCOUNT\s*\(", soql_query, re.IGNORECASE))


# Tools whose flat record-list results are rendered deterministically as a
# Markdown reference table. Adding a tool here makes its FLAT results feed the
# deterministic fast path (no second Qwen synthesis call). Hierarchical,
# schema, and narrative/related-record tools stay out so the LLM still formats
# those. listRecentSobjectRecords' primary SOQL result and its `/recent` REST
# fallback are both flat record lists, so recent-record queries ("show me my
# recent Accounts") skip the ~67s synthesis step that previously dominated the
# ~79s end-to-end latency.
_FLAT_LIST_TOOLS = {"soqlQuery", "listRecentSobjectRecords"}

# Pluralization for labeled count/section lines. Custom objects (``__c``) keep
# their returned type verbatim (Salesforce types are already plural-free names).
_PLURAL = {
    "Account": "Accounts",
    "Lead": "Leads",
    "Contact": "Contacts",
    "Opportunity": "Opportunities",
    "Case": "Cases",
    "Task": "Tasks",
    "Event": "Events",
    "User": "Users",
    "Note": "Notes",
    "Attachment": "Attachments",
    "Order": "Orders",
}


def _plural(label: str) -> str:
    if label.lower().endswith("__c"):
        return label
    plural = _PLURAL.get(label)
    if plural:
        return plural
    if label.endswith("s"):
        return label
    return label + "s"


def _soql_count_object(soql: str) -> str | None:
    """Object token from a COUNT aggregate's FROM clause ('' -> None)."""
    if not soql:
        return None
    m = re.search(r"\bFROM\s+([A-Za-z_][\w]*(?:__c)?)", soql, re.IGNORECASE)
    if not m:
        return None
    return m.group(1)


def _render_count_line(value: int, soql: str = "", obj_type: str | None = None) -> str:
    """Deterministic count line, attributed to its object when determinable.

    - ``**Total Leads: 66**`` when the object is known (FROM clause / attributes
      type that is not the AggregateResult container), so wording always
      identifies the business object instead of a generic "Total Count".
    - ``**Total Count:** 66`` only as a last resort when no object can be
      determined. The value is the aggregate business value (expr0/count or the
      totalSize of a genuinely empty COUNT) — NEVER the empty wrapper's row
      count ``{"totalSize": 1}``.
    """
    obj = _soql_count_object(soql)
    if not obj and obj_type and obj_type != "AggregateResult":
        obj = obj_type
    if obj:
        return f"**Total {_plural(obj)}: {value:,}**"
    return f"**Total Count:** {value:,}"

_AGGREGATE_FUNC_RE = re.compile(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*[\w.]+\s*\)", re.IGNORECASE)
_METRIC_FUNC_RE = re.compile(r"\b(SUM|AVG|MIN|MAX)\s*\(", re.IGNORECASE)


def _aggregate_metric_name(soql: str) -> str:
    """Return the SOQL aggregate function name (COUNT/SUM/AVG/MIN/MAX), or 'Count'."""
    m = _AGGREGATE_FUNC_RE.search(soql or "")
    return m.group(1).upper() if m else "Count"


def _aggregate_metric_label(soql: str) -> str:
    """Aggregate column label with metric AND field context, e.g. 'SUM(Amount)'.

    Falls back to the bare function name ('SUM') when the expression cannot be
    parsed from the SOQL, so legacy/no-SOQL callers still render correctly.
    """
    m = _AGGREGATE_FUNC_RE.search(soql or "")
    if m:
        return m.group(0)
    return "Count"


def _has_metric_aggregate(soql: str) -> bool:
    """True when the SOQL selects a non-Count aggregate function (SUM/AVG/MIN/MAX)."""
    return bool(_METRIC_FUNC_RE.search(soql or ""))


def _fmt_aggregate_cell(value) -> str:
    """Format one aggregate table cell (group value or metric)."""
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{int(value):,}" if value.is_integer() else f"{value:,}"
    if isinstance(value, (dict, list)):
        return ""
    return str(value)


def _aggregate_group_columns(first: dict) -> list[str]:
    """Group/selector columns of an expr0/count aggregate row.

    Derived from the RETURNED record keys (never a hardcoded object/field), so
    arbitrary GROUP BY selectors (Status, CloseDate, custom ``__c`` fields, ...)
    are preserved without fabricating columns.
    """
    return [
        k
        for k in first
        if k not in ("expr0", "count", "attributes")
        and not isinstance(first[k], (dict, list))
    ]


def _render_aggregate_markdown(records: list, soql: str = "") -> str:
    """Render a COUNT/SUM/GROUP-BY aggregate result as a deterministic table.

    Group columns come from the returned record keys (never a hardcoded object),
    and the metric column (``expr0``/``count``) is named by the SOQL aggregate
    function and right-aligned. Aggregates are summaries, NOT business records,
    so no '**Total: N records**' line is ever appended here.
    """
    if not records or not isinstance(records[0], dict):
        return ""
    first = records[0]
    col_names = _aggregate_group_columns(first)
    metric = _aggregate_metric_label(soql)
    ncols = len(col_names) + 1
    sep_cells = ["---"] * (ncols - 1) + ["---:"]
    header = "| " + " | ".join(col_names + [metric]) + " |"
    sep = "| " + " | ".join(sep_cells) + " |"
    lines = [header, sep]
    for rec in records:
        cells = [_fmt_aggregate_cell(rec.get(k)) for k in col_names]
        cells.append(_fmt_aggregate_cell(rec.get("expr0", rec.get("count", ""))))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _classify_expr_count(first: dict, records: list, soql: str = "") -> str | None:
    """Classify an ``expr0``/``count``-shaped result as ``count`` or ``aggregate``.

    This is THE single shared decision used by both the flat formatter and the
    section renderer so they can never diverge:

    - ``count``     single row, no GROUP BY group columns, no SUM/AVG/MIN/MAX
                    metric → a scalar business value.
    - ``aggregate`` one or more group columns OR multiple rows OR a non-Count
                    metric function → a deterministic aggregate table. It is
                    NEVER collapsed to the first row and NEVER a business row
                    count (len(records) is the number of groups, not a total).
    - ``None``      the record is not expr0/count-shaped at all.
    """
    if "expr0" not in first and "count" not in first:
        return None
    group_cols = _aggregate_group_columns(first)
    if group_cols or len(records) > 1 or _has_metric_aggregate(soql):
        return "aggregate"
    return "count"


def format_sf_records_as_markdown(
    result_json: str,
    tool_name: str = "soqlQuery",
    soql_query: str = "",
) -> str | None:
    """
    Parse a Salesforce flat record-list JSON result and return formatted markdown.

    Handles two cases:
    1. Aggregate/COUNT queries → clean count line
    2. Flat record lists → Standard markdown table

    Recognized for any tool in ``_FLAT_LIST_TOOLS`` (soqlQuery and
    listRecentSobjectRecords) and for both result shapes the REST API returns:
    the ``{"totalSize": N, "records": [...]}`` envelope from ``/query`` and a
    bare top-level JSON array (``/recent`` fallback for listRecentSobjectRecords).

    Hierarchical/subquery results return None — they are passed raw to the
    LLM which renders them as structured cards.

    Returns None if the result is not a parseable flat record list.
    """
    if tool_name not in _FLAT_LIST_TOOLS:
        return None
    try:
        data = json.loads(result_json)
    except Exception:
        return None

    # `listRecentSobjectRecords` can return a bare array (REST `/recent`
    # fallback) instead of the {totalSize, records} envelope from `/query`.
    # The Salesforce Platform MCP API may also use different key names
    # (recentItems, items, results) depending on the endpoint and SDK version.
    if isinstance(data, list):
        records = data
        total_size = len(records)
    elif isinstance(data, dict):
        # Try all known Salesforce record-list key shapes in priority order
        records = (
            data.get("records")
            or data.get("recentItems")   # /recent REST endpoint shape
            or data.get("items")         # some MCP structuredContent shapes
            or data.get("results")       # alternative MCP shapes
            or data.get("edges")         # GraphQL-style shapes
            or []
        )
        total_size = data.get("totalSize", data.get("total", len(records)))
    else:
        return None

    if not records:
        # A COUNT query can legitimately return {"totalSize": 0, "records": []}
        # (e.g. SELECT COUNT(Id) FROM Account with zero rows). Surface it as the
        # project's count line so the direct-response fast path triggers instead
        # of a second LLM synthesis call. Value comes straight from totalSize.
        if tool_name == "soqlQuery" and _is_soql_count(soql_query):
            # value comes straight from totalSize (a genuinely empty COUNT =
            # zero business value), attributed via the FROM-clause object.
            return _render_count_line(int(total_size), soql_query)
        return None

    first = records[0]

    # Detect object type from records (attributes may be stripped by the cleaner).
    # Fall back to the top-level type hint the MCP API sometimes injects.
    obj_type = None
    if isinstance(first.get("attributes"), dict):
        obj_type = first["attributes"].get("type")
    if not obj_type and isinstance(data, dict):
        # Some MCP responses include a top-level 'objectType' or 'sobjectType' hint
        obj_type = data.get("objectType") or data.get("sobjectType") or data.get("type")

    # Case 1: Aggregate/COUNT result. Semantic branching (shared classifier in
    # _classify_expr_count, so this never diverges from result_types.py):
    #
    #  - 'count'     plain single-value COUNT (expr0/count, no group columns, no
    #                SUM/AVG/MIN/MAX metric) → the scalar count line.
    #  - 'aggregate' GROUP BY COUNT (group columns) or multi-row or a non-Count
    #                metric → deterministic aggregate table. NEVER collapsed to
    #                the first row, NEVER labeled as a record count, NEVER using
    #                len(records) as the business count (it is the group count).
    #
    # totalSize is deliberately ignored here (for COUNT it is 1 row, not the
    # business value); the value always comes from expr0 / count.
    if records and isinstance(first, dict):
        agg_kind = _classify_expr_count(first, records, soql_query)
        if agg_kind == "count":
            count_val = first.get("expr0", first.get("count", 0))
            # Format with commas for readability: 1234567 → 1,234,567
            try:
                count_num = int(float(count_val)) if count_val is not None else 0
            except (ValueError, TypeError):
                count_num = 0
            # Attributed to the queried object (never "AggregateResult", never
            # totalSize=1 as the business value).
            return _render_count_line(count_num, soql_query, obj_type)
        if agg_kind == "aggregate":
            return _render_aggregate_markdown(records, soql_query)

    # Case 2: Subquery results → return None (let LLM handle hierarchical formatting)
    has_subqueries = any(
        len(_detect_subquery_collections(rec)) > 0 for rec in records
    )
    if has_subqueries:
        return None

    # Case 3: Flat record list → Standard markdown table
    all_keys = []
    for rec in records[:5]:
        for k, v in rec.items():
            if k in _SKIP_FIELDS:
                continue
            if isinstance(v, dict) and "attributes" not in v:
                for subk in v.keys():
                    if subk not in _SKIP_FIELDS:
                        composite = f"{k}.{subk}"
                        if composite not in all_keys:
                            all_keys.append(composite)
            elif k not in all_keys:
                all_keys.append(k)

    preferred = _FIELD_ORDER.get(obj_type, [])
    ordered = [f for f in preferred if f in all_keys]
    remaining = [f for f in all_keys if f not in ordered]
    cols = ordered + remaining
    cols = [c for c in cols if c.split(".")[0] not in _SKIP_FIELDS]

    if not cols:
        return None

    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = []
    for rec in records:
        cells = []
        for col in cols:
            cells.append(_fmt_value(_get_nested(rec, col)))
        rows.append("| " + " | ".join(cells) + " |")

    count = int(total_size)
    noun = "record" if count == 1 else "records"
    total_line = f"\n**Total: {count} {noun}**"
    return "\n".join([header, sep] + rows) + total_line


# ──────────────────────────────────────────────────────────────
# Output Sanitizer — strips internal artifacts before user delivery
# ──────────────────────────────────────────────────────────────
def sanitize_response_output(text: str) -> str:
    """
    Final output sanitizer that strips any leftover internal artifacts
    from the assistant response before delivering to the user.
    Removes: <think> tags, raw JSON tool call blocks, XML tool tags,
    stray system artifacts, and code blocks containing tool schemas.
    Preserves: Hierarchical cards (### headers with bullet children),
    Markdown tables, and natural language content.
    Strips: [reference_table] and [PRE-BUILT TABLE...] internal markers.
    """
    if not text:
        return text

    cleaned = text

    # 1. Strip <think>...</think> reasoning blocks (Qwen3 internal monologue)
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", cleaned, flags=re.DOTALL)

    # 2. Strip XML tool tags: <tools>...</tools>, <tool_call>...</tool_call>
    cleaned = re.sub(r"<(?:tools|tool_call|function_call)>[\s\S]*?</(?:tools|tool_call|function_call)>", "", cleaned, flags=re.IGNORECASE)

    # 3. Strip markdown code blocks containing JSON tool call schemas
    #    (```json { "name": "soqlQuery", ... } ```)
    cleaned = re.sub(
        r"```(?:json)?\s*\{[\s\S]*?\"(?:name|function|arguments)\"[\s\S]*?\}```",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # 4. Strip standalone JSON objects that look like tool calls
    #    { "name": "toolName", "arguments": { ... } }
    cleaned = re.sub(
        r"\{\s*\"name\"\s*:\s*\"(?:soqlQuery|find|getUserInfo|getObjectSchema|createSobjectRecord|updateSobjectRecord|deleteSobjectRecord|getRelatedRecords|listRecentSobjectRecords|updateRelatedRecord|deleteRelatedRecord|uploadRecordAttachment)\"[\s\S]*?\}",
        "",
        cleaned,
    )

    # 5. Strip stray code block markers and empty blocks
    #    But preserve markers that are part of hierarchical cards or tables
    lines = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        # Skip lines that are ONLY stray markers (not embedded in content)
        if stripped in ("{", "}", "]", "[", "```", "```json", "```tool_call", "}}", "}}}", "`]"):
            continue
        # Skip stray [PRE-BUILT TABLE...] and [reference_table] markers
        if stripped.startswith("[PRE-BUILT TABLE") or stripped.startswith("[PRE-BUILT"):
            continue
        if stripped == "[reference_table]":
            continue
        lines.append(line)
    cleaned = "\n".join(lines)

    # 6. Collapse multiple blank lines into max 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # 7. Final trim
    cleaned = cleaned.strip()

    # 8. If the result is empty or only whitespace/braces, return a generic fallback
    if not cleaned or re.fullmatch(r"[\{\}\[\]\`\s]*", cleaned):
        return ""

    return cleaned


# ──────────────────────────────────────────────────────────────
# LAST-MILE USER OUTPUT GUARD
# Guarantees the user-facing chatbot NEVER exposes raw tool-call
# JSON, tool schemas, XML function calls, debug logs, or parser
# output. Applied at the delivery boundary (app.py) as a hard
# security barrier — the final answer returned to the user is
# always clean natural language (or a clean fallback message).
# ──────────────────────────────────────────────────────────────

# Tool call JSON key signatures used to detect leaked raw tool payloads.
_CLEAN_LINE_ONLY_MARKERS = {"{", "}", "]", "[", "```", "```json", "```tool_call",
                            "}}", "}}}", "`]", "{}", "[]", "```xml", "<tool_call>"}

# Qwen/adapter native tool-call markers that must never reach the user.
_RAW_TOOL_MARKERS = [
    "tool_calls", "[TOOL_CALLS]", "<tool_call>", "</tool_call>",
    "<tools>", "</tools>", "<function_call>", "</function_call>",
    "function_calls", "[reference_table]", "[PRE-BUILT",
]


def _looks_like_tool_json(text: str) -> bool:
    """Return True if text contains a leaked raw tool/tech artifact."""
    if not text:
        return False
    lower = text.lower()
    # Debug / exception / internal markers can never be shown to the user.
    if any(k in lower for k in (
        "[rag debug", "[mcp]", "traceback", "\"error\":", "'error':",
        "raise runtimeerror", "theme error", "not found. available:", "parser output",
    )):
        return True
    # A JSON/XML block carrying a known tool name invocation.
    if any(kw in lower for kw in (
        '"name"', "'name'", '"arguments"', '"function"',
        '"parameters"', '"toolname"', 'soqlquery', 'getobjectschema',
        'createsobjectrecord', 'updatesobjectrecord', 'deletesobjectrecord',
        'getrelatedrecords', 'listrecentsobjectrecords', 'uploadrecordattachment',
        '<tool_call>', '<tools>', '[tool_calls]',
    )):
        # Only flag JSON-ish/XML-ish structures (contains braces or angle tags),
        # not natural sentences that merely word-match a tool name.
        if ("{" in text or "}" in text or "[" in text or "]" in text
                or "<" in text or ">" in text):
            return True
    return False


# ──────────────────────────────────────────────────────────────
# MARKDOWN HEALER
# Bulletproofs UI rendering against slightly-malformed LLM output.
# Runs at the delivery boundary (inside finalize_user_response) so the
# user ALWAYS receives valid GFM: matched asterisks, well-spaced bullets,
# and intact GFM tables with blank-line separation. Falls back to a
# strict GFM-table rebuilder for [reference_table] content that the LLM
# tried to rewrite with broken pipes.
# ──────────────────────────────────────────────────────────────

def _is_table_row(line: str) -> bool:
    """True if a line looks like a GFM table row (even if outer pipes were dropped)."""
    s = line.strip()
    if not s or "|" not in s:
        return False
    if _is_table_separator(s):
        return True
    body = s
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    cells = [c.strip() for c in body.split("|")]
    cells = [c for c in cells if c != ""]
    return len(cells) >= 2


def _heal_bullets(text: str) -> str:
    """Normalize `*`/`+` bullet markers to `- ` with proper spacing."""
    out = []
    for line in text.split("\n"):
        # A single `*` or `+` at line start is a bullet (exclude `**bold**`).
        m = re.match(r"^(\s*)[*+](?![\*\+])\s*(.*)$", line)
        if m and "|" not in line:
            indent, content = m.group(1), m.group(2)
            marker = "- "
            if content:
                out.append(f"{indent}{marker}{content}" if not content.startswith("*") else f"{indent}*{content}")
            else:
                out.append(f"{indent}{marker}".rstrip())
            continue
        out.append(line)
    return "\n".join(out)


def _heal_asterisks_line(line: str) -> str:
    """Balance asterisks in a single line so bold/italic never render raw."""
    if "*" not in line:
        return line
    # Protect code spans so we never touch asterisks inside `...`.
    protected = []

    def _proto(m):
        protected.append(m.group(0))
        return f"\u0000{len(protected) - 1}\u0000"

    line = re.sub(r"`[^`]*`", _proto, line)

    # Protect well-formed bold pairs **x**.
    def _proto_bold(m):
        protected.append(m.group(0))
        return f"\u0000{len(protected) - 1}\u0000"

    line = re.sub(r"\*\*[^*]+?\*\*", _proto_bold, line)

    # Collapse empty/mangled asterisk runs (****, ***) down to nothing.
    line = re.sub(r"\*{4,}", "", line)
    line = re.sub(r"\*{3}", "", line)

    # Any remaining `**` is an UNMATCHED bold marker (matched pairs were already
    # protected) — strip it so it never renders as literal `**`.
    line = re.sub(r"\*\*", "", line)

    # Drop trailing standalone asterisk(s) and a leading orphan asterisk.
    line = re.sub(r"\*+\s*$", "", line)
    line = re.sub(r"^\s*\*+(?![\s\*])", " ", line)

    # Balance any remaining single asterisks as italics: drop the odd count's
    # last one so a lone `*` never reaches the UI as literal text.
    scopes = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] == "*":
            j = i
            while j < n and line[j] == "*":
                j += 1
            if j - i == 1:
                scopes.append((i, j))
            i = j
        else:
            i += 1
    if len(scopes) % 2 == 1:
        idx, _ = scopes[-1]
        line = line[:idx] + line[idx + 1:]

    # Restore protected tokens.
    for i, token in enumerate(protected):
        line = line.replace(f"\u0000{i}\u0000", token)
    return line


def _is_table_separator(line: str) -> bool:
    line = line.strip()
    if not line or "|" not in line:
        return False
    cells = [c.strip() for c in line.strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c or "---") for c in cells)


def _cells_of(line: str) -> list:
    """Split a pipe table line into trimmed cells (works with/without outer pipes)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _rebuild_gfm_table(rows: list) -> str:
    """Rebuild strict GFM table from header + optional separator + body rows.

    The HEADER row defines the table's column schema. Every body row is
    normalized to exactly that many cells (short rows padded with empty cells,
    over-wide rows truncated). This guarantees every row has the same number of
    cells as the header, so fields like Id and Name always stay in their own
    aligned columns even when a body cell contains an embedded ``|`` or the
    synthesizer merged two cells into one.
    """
    if not rows:
        return ""
    table_cells = [_cells_of(r) for r in rows]
    # The header defines the column count, not the widest body row.
    ncols = len(table_cells[0])
    if ncols == 0:
        return ""
    # Ignore a separator row (all dashes) already present after the header.
    body = list(table_cells)
    if len(body) >= 2:
        if all(re.fullmatch(r":?-{2,}:?", (c or "---")) for c in body[1]):
            body.pop(1)
    header = body[0]
    data = body[1:]
    sep = ["---"] * ncols
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(sep) + " |"]
    for rec in data:
        normalized = (list(rec) + [""] * ncols)[:ncols]
        lines.append("| " + " | ".join(normalized) + " |")
    return "\n".join(lines)


def heal_markdown(text: str) -> str:
    """
    Bulletproof markdown for UI rendering.

    1. Guarantees contiguous GFM tables (from [reference_table] or raw LLM
       tables) are reconstructed with strict pipes and blank-line separation,
       even if the LLM dropped outer pipes or mangled the separator row.
    2. Normalizes `*`/`+` bullets to `- ` with proper spacing.
    3. Balances/removes unmatched asterisks so bold/italic never leak raw `*`.
    """
    if not text:
        return text

    lines = text.split("\n")
    result = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if _is_table_row(line):
            # Gather a contiguous table block (header/separator/body rows).
            block = []
            j = i
            while j < n:
                if _is_table_row(lines[j]):
                    block.append(lines[j])
                    j += 1
                else:
                    break
            # Only treat as a real table if the block has >=2 rows; otherwise
            # it's ambiguous prose (e.g. "usa | 10"). Preserve as-is.
            if len(block) >= 2:
                result.append("")
                rebuilt = _rebuild_gfm_table(block)
                if rebuilt:
                    result.append(rebuilt)
                result.append("")
            else:
                result.append(line)
            i = j
            continue
        result.append(line)
        i += 1

    assembled = "\n".join(result)

    # Bullet + asterisk healing on non-table content.
    out = []
    for line in assembled.split("\n"):
        if _is_table_row(line):
            out.append(line)
            continue
        out.append(_heal_asterisks_line(_heal_bullets(line)))
    healed = "\n".join(out)

    # Collapse extra blank lines / tidy.
    healed = re.sub(r"\n{3,}", "\n\n", healed)
    healed = healed.replace("\n \n", "\n\n")
    return healed.strip()


def finalize_user_response(text: str) -> str:
    """
    STRICT final barrier before any string is delivered to the user.

    Runs the existing sanitizer, then strips any residual raw tool-call
    JSON/XML/schema/debug artifacts. If the cleaned result is empty or
    still contains raw tool-call syntax, returns a clean natural-language
    fallback so the user NEVER sees parser output.

    Safe for Markdown tables / hierarchical cards / natural language.
    """
    if not text:
        return "I processed your request. How else can I assist you?"

    cleaned = sanitize_response_output(text)

    # 1. Remove stray coding-fence / structure lines that are NOT part of content.
    lines = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if stripped in _CLEAN_LINE_ONLY_MARKERS:
            continue
        # Skip lines that are pure JSON key:value pairs (tool schema / result dumps).
        if re.fullmatch(r'"?(name|type|function|arguments|parameters|properties|'
                        r'required|tool_calls|records|totalSize|attributes|id)"?\s*[:,]', stripped):
            continue
        lines.append(line)
    cleaned = "\n".join(lines).strip()

    # 2. Remove residual XML tool-call wrappers and their inner content.
    cleaned = re.sub(
        r"<(?:tool_call|tools|function_call|assistant_tool_call)>[\s\S]*?"
        r"</(?:tool_call|tools|function_call|assistant_tool_call)>",
        "", cleaned, flags=re.IGNORECASE,
    )
    # Strip any orphaned XML tool tags left behind (e.g. a lone closing `</tools>`).
    cleaned = re.sub(
        r"</?(?:tool_call|tools|function_call|assistant_tool_call|br|pre|code)[^>]*>",
        "", cleaned, flags=re.IGNORECASE,
    )
    # 3. Remove complete JSON arrays that are tool-call payloads
    #    e.g. [ { "name": "soqlQuery", "arguments": {...} } ]
    cleaned = re.sub(
        r"\[\s*\{(?:[^{}]*|\{[^{}]*\})*\}?\s*\]", "", cleaned, flags=re.DOTALL,
    )
    # 4. Remove a raw JSON object that carries tool-call keys.
    cleaned = re.sub(
        r"\{(?:[^{}]|\{[^{}]*\})*:\s*(?:[^{}]|\{[^{}]*\})*\}", "", cleaned,
        flags=re.DOTALL,
    )
    # 4b. Remove orphaned bracket residue left after the strips above
    #     (e.g. `[}]`, `[{`, `]}`, stray single braces). Strip any run of 2+.
    bracket_run = re.compile(r"[\[\]\{\}]{2,}")
    cleaned = bracket_run.sub("", cleaned)
    # Remove single brackets/braces that stand alone between spaces or newlines.
    cleaned = re.sub(r"(?:^|[\s])\[[\s\]]", " ", cleaned)
    cleaned = re.sub(r"(?:^|[\s])\][\s$]", " ", cleaned)

    # 5. Heal malformed markdown so tables/cards/bullets always render:
    #    fix unmatched asterisks, normalize bullet spacing, and guarantee
    #    GFM tables are preserved with blank-line separation.
    cleaned = heal_markdown(cleaned)

    # 6. Collapse blank lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    # 7. If raw tool-call syntax survived or nothing usable remains, fall back.
    if not cleaned or re.fullmatch(r"[\{\}\[\]\`\s]*", cleaned) or _looks_like_tool_json(cleaned):
        logger.warning("Last-mile guard replaced leaked tool-call content with a clean fallback.")
        return "I processed your request. How else can I assist you?"

    return cleaned


# ──────────────────────────────────────────────────────────────
# SOQL Error Auto-Correction Helper
# ──────────────────────────────────────────────────────────────
_SOQL_ERROR_PATTERNS = [
    # (regex_pattern, fix_description, replacement_suggestion)
    (r"\$\d[\d,]*", "Remove dollar signs and commas from numeric literals"),
    (r"'\d[\d,]*'", "Remove quotes around numeric values"),
    (r"AS\s+\w+", "Remove 'AS' keyword — SOQL uses implicit aliases"),
    (r"DATE\s*\(", "Replace DATE() with SOQL date literals (TODAY, THIS_WEEK, etc.)"),
    (r"DATEADD\s*\(", "Replace DATEADD() with SOQL date literals (LAST_N_DAYS:N, etc.)"),
    (r"NOW\s*\(\s*\)", "Replace NOW() with TODAY or use datetime literals"),
    (r"GETDATE\s*\(\s*\)", "Replace GETDATE() with TODAY"),
    (r"WHERE\s+\w+\s*=\s*\(\s*SELECT", "SOQL does not support correlated subqueries in WHERE clauses. Use parent-to-child subqueries in SELECT, or filter by Account.Name = 'X', or use a literal ID from a prior query."),
]

_SOQL_FIX_SUGGESTIONS = {
    "$": "Remove dollar signs ($) from SOQL numeric filters. Write Amount > 50000, NOT Amount > '$50,000'.",
    "AS ": "SOQL does not support the 'AS' keyword for aliases. Write SUM(Amount) total instead of SUM(Amount) AS total.",
    "DATE()": "SOQL does not have DATE() function. Use date literals like TODAY, THIS_WEEK, LAST_N_DAYS:7, THIS_YEAR.",
    "DATEADD()": "SOQL does not have DATEADD(). Use date literals like LAST_N_DAYS:N, LAST_7_DAYS, THIS_MONTH.",
    "malformed": "Check SOQL syntax: ensure SELECT, FROM, WHERE, and LIMIT clauses are correct.",
    "invalid_field": "Check field API names using getObjectSchema if unsure.",
    "group by": "SOQL does not allow GROUP BY inside semi-join subqueries. Query the child object directly.",
    "unexpected token": "SOQL does not support subqueries in WHERE clauses. Use parent-to-child subqueries in the SELECT clause instead (e.g., SELECT Id, Name, (SELECT Id, Name FROM Opportunities) FROM Account WHERE Name = 'X'), or filter by relationship name (e.g., WHERE Account.Name = 'X').",
    "MALFORMED_QUERY": "The SOQL query structure is invalid. If using a subquery in WHERE, replace with: (a) parent-to-child subquery in SELECT, or (b) filter by Account.Name = 'X', or (c) use a literal ID from a prior query.",
}


def get_soql_fix_suggestion(error_msg: str) -> str | None:
    """
    Analyze a SOQL error message and return a fix suggestion if pattern matches.
    Returns None if no known pattern matches.
    """
    error_lower = error_msg.lower()
    for pattern, suggestion in _SOQL_FIX_SUGGESTIONS.items():
        if pattern.lower() in error_lower:
            return suggestion
    return None


class SalesforceAgent:
    """
    The core agent that orchestrates:
    1. Receiving user messages
    2. Calling Qwen3 with conversation history + available tools
    3. Executing tool calls via MCP
    4. Feeding tool results back to LLM for final response
    5. Multi-step reasoning (up to MAX_ITERATIONS tool calls per turn)
    """

    def __init__(
        self,
        llm: BaseLLM,
        executor: ToolExecutor,
        max_iterations: int = 20,  # Raised: multi-query (6+ parts) needs 12+ tool calls
        max_history: int = 4,
    ):
        self.llm = llm
        self.executor = executor
        self.max_iterations = max_iterations
        self.planner = TaskPlanner()

        # Per-session memories: {session_id: ConversationMemory}
        self._memories: dict[str, ConversationMemory] = {}
        self._max_history = max_history
        self.rag_retriever = ToolRAGRetriever(default_top_k=6)

    def _get_memory(self, session_id: str) -> ConversationMemory:
        """Get or create conversation memory for a session."""
        # Bound memories dict to prevent unbounded growth on Render (512 MB).
        _MAX_MEMORIES = 25
        if session_id not in self._memories and len(self._memories) >= _MAX_MEMORIES:
            oldest = next(iter(self._memories))
            self._memories[oldest].clear()
            del self._memories[oldest]
        if session_id not in self._memories:
            self._memories[session_id] = ConversationMemory(
                max_messages=self._max_history
            )
        else:
            self._memories[session_id].max_messages = self._max_history
            self._memories[session_id]._trim()
        return self._memories[session_id]

    async def _get_effective_tools(self, user_message: str, session_id: str = "default") -> list[dict[str, Any]]:
        """E5: fetch tools via RAG with a bounded timeout, falling back safely.

        - RAG is run off the event loop and bounded by RAG_TIMEOUT so an
          embedding/vector hang can never block the loop indefinitely.
        - On RAG timeout / empty result / residual exception, a Salesforce/data
          request falls back to the COMPLETE read-only-safe tool registry
          (same authoritative source as ENABLE_RAG_TOOLS=false) and then passes
          through filter_tools_for_query so read-only queries never gain
          mutation tools and write queries keep their required tools.
        - Clearly general/non-Salesforce queries are left with RAG's result
          (possibly []) so the existing general-answer behavior is preserved.
        """
        if self.planner.has_pending_confirmation(session_id):
            return get_tool_definitions()
        try:
            tools = await asyncio.wait_for(
                asyncio.to_thread(self.rag_retriever.get_relevant_tools, user_message, 6),
                timeout=RAG_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(
                "[RAG] Timeout retrieving tools (agent.agent). "
                f"stage=semantic-rag-tool-retrieval timeout_s={RAG_TIMEOUT:.1f} "
                "mcp_reached=False rest_fallback=False "
                "root_cause='embedding model cold-load or vector query hung'"
            )
            tools = []
        except Exception as e:
            logger.error(f"[RAG] Retrieval raised unexpectedly (agent.agent): {e}. Returning no tools.")
            tools = []

        if not tools and _has_salesforce_intent(user_message):
            logger.warning(
                "[RAG] Empty/timeout/failed retrieval for a Salesforce/data query; "
                "falling back to the complete read-only-safe tool registry."
            )
            tools = get_tool_definitions()
        return filter_tools_for_query(tools, user_message)

    async def process_message(
        self,
        user_message: str,
        session_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Process a user message through the agent loop.

        Yields events as dicts with:
            - type: 'thinking' | 'tool_call' | 'tool_result' | 'response' | 'error' | 'confirmation'
            - data: Event-specific payload

        This generator pattern allows the WebSocket/UI to show
        real-time progress as tools execute.
        """
        memory = self._get_memory(session_id)
        memory.max_messages = self._max_history
        memory._trim()
        # E5: RAG Tool Retrieval with bounded timeout + safe Salesforce fallback.
        # filter_tools_for_query is applied inside _get_effective_tools. When a
        # pending confirmation exists it returns the full registry so compound
        # WRITE flows keep the full tool set; READ_ONLY_MODE still gates execution.
        tools = await self._get_effective_tools(user_message, session_id)

        # ── Check for pending confirmations ──
        if self.planner.has_pending_confirmation(session_id):
            pending = self.planner.process_confirmation(user_message, session_id)
            # F5: expiry must be handled BEFORE the truthy-confirmed branch so an
            # expired "yes"/"ok" can never reach the executor.
            if pending and pending.get("status") == "expired":
                msg = pending.get("message", "This confirmation has expired. No action was executed.")
                logger.info(f"[F5] Expired confirmation for session '{session_id}': {msg}")
                memory.add_user_message(user_message)
                memory.add_assistant_message(msg)
                yield {"type": "response", "data": msg}
                return
            if pending:
                # User confirmed — execute the pending destructive action
                yield {"type": "thinking", "data": "Executing confirmed operation..."}

                tool_name = pending["tool_name"]
                arguments = pending["arguments"]
                action_id = f"confirmed_{pending['tool_name']}"

                yield {
                    "type": "tool_call",
                    "data": {"name": tool_name, "arguments": arguments},
                }

                try:
                    result = await _bounded_call(
                        self.executor.execute(
                            tool_name, arguments, user_provenance={}
                        ),
                        AGENT_EXECUTOR_TIMEOUT,
                        "Salesforce confirmed-operation execution",
                    )
                except AgentTimeoutError:
                    error_event = _timeout_error_event("Salesforce confirmed-operation execution")
                    logger.error(f"[TIMEOUT] {error_event['message']}")
                    memory.add_assistant_message(error_event["message"])
                    yield error_event
                    return

                # ── D1: a confirmed operation that fails is NOT a successful result.
                # Never store it in memory as a tool result or let it reach synthesis. ──
                err_msg = _executor_error_message(result)
                if err_msg:
                    logger.error(f"[SALESFORCE_FAILED] Confirmed {tool_name} failed: {err_msg}")
                    memory.add_assistant_message(f"{tool_name} failed: {err_msg}")
                    yield _salesforce_failed_event(f"Salesforce call '{tool_name}' failed: {err_msg}")
                    return

                yield {
                    "type": "tool_result",
                    "data": {"name": tool_name, "result": result},
                }

                # Add to memory and continue turn execution so LLM can proceed with remaining steps (e.g. create account)
                memory.add_user_message(user_message)
                memory.add_assistant_tool_calls([{
                    "id": action_id,
                    "name": tool_name,
                    "arguments": arguments,
                }])
                memory.add_tool_result(action_id, tool_name, result)

                # Continue turn loop naturally below
            else:
                # User declined
                memory.add_user_message(user_message)
                decline_msg = "✅ Operation cancelled. No records were deleted."
                memory.add_assistant_message(decline_msg)
                yield {"type": "response", "data": decline_msg}
                return
        else:
            # ── Normal message processing ──
            logger.info(f"📩 [USER MESSAGE] ({session_id}): {user_message}")
            memory.add_user_message(user_message)
            yield {"type": "thinking", "data": "Analyzing your request..."}

        iteration = 0
        # ── PERMANENT ANTI-HALLUCINATION: Track only real tool-fetched data ──
        # Keys = tool names called, Values = their actual results from Salesforce
        # LLM is NEVER allowed to present data that isn't in this dict
        tool_results_fetched: dict[str, str] = {}

        # ── #8 ALL-OR-NOTHING: True when a mutation in this response failed the
        # mandatory fail-closed pre-flight validation and ZERO mutations ran. Lets
        # the final synthesis ask the user for the missing fields instead of being
        # treated as an ungrounded (E5) answer — the validation/blocked envelopes
        # are the grounding for an input request.
        validation_blocked = False

        # ── Python-built tables: tc_id → (tool_name, markdown) ──
        # Stores pre-formatted flat tables and count lines. Hierarchical/subquery
        # results are NOT stored here — they flow raw to the LLM for card formatting.
        python_tables: dict[str, tuple[str, str]] = {}  # tc_id → (tool_name, markdown)

        # ── Track zero-record results for conversational follow-up ──
        zero_record_results: list[dict] = []

        # ── Multi-step action tracking ──
        # Tracks sequential actions (read→update→delete) for proper dependency handling
        pending_sequential_actions: list[dict] = []

        # ── Detect multi-query upfront ──
        user_msg_lower = user_message.lower()
        query_lines = [l.strip() for l in user_message.strip().splitlines() if l.strip()]
        _compound_seps = [" and ", " & ", " also ", " along with ", " aur ", " plus "]
        _is_multi = (
            len(query_lines) >= 3
            or any(sep in user_msg_lower for sep in _compound_seps)
        )

        while iteration < self.max_iterations:
            iteration += 1

            # ── FAST ROUTER: Zero-latency direct dispatch for standard queries ──
            direct_handled = False
            if iteration == 1:
                direct_intent = _detect_direct_intent(user_message)
                if direct_intent:
                    tool_name, tool_args = direct_intent
                    logger.info(
                        f"⚡ [FAST ROUTER] Zero-latency direct matched tool '{tool_name}' "
                        f"with args {tool_args} for query: '{user_message}'"
                    )
                    llm_result = {
                        "content": "",
                        "tool_calls": [{"id": f"direct_tc_{iteration}", "name": tool_name, "arguments": tool_args}],
                        "finish_reason": "tool_calls",
                    }
                    direct_handled = True

            if not direct_handled:
                try:
                    messages = memory.get_messages_for_llm(TOOL_CALLING_PROMPT)

                    # ── SPEED FIX: On Turn 1 of multi-query, inject batch instruction ──
                    # ONLY for pure READ queries — create/update/delete are sequential by nature.
                    # Intent classification is shared with filter_tools_for_query (see `_write_keywords`).
                    _is_read_only = not _has_write_intent(user_msg_lower)

                    if iteration == 1 and _is_multi and _is_read_only:
                        batch_hint = (
                            "BATCH MODE — SPEED CRITICAL: The user has asked multiple independent READ questions. "
                            f"There are approximately {len(query_lines)} sub-queries. "
                            "You MUST return ALL required tool calls in THIS SINGLE RESPONSE right now. "
                            "Do NOT make one tool call and wait — output every tool call simultaneously. "
                            "This is mandatory for fast response."
                        )
                        messages = messages + [{"role": "user", "content": batch_hint}]

                    llm_result = await _bounded_call(
                        self.llm.chat_with_tools(
                            messages=messages,
                            tools=tools,
                            temperature=0.0,  # Zero temp = fastest, most deterministic
                        ),
                        AGENT_LLM_TIMEOUT,
                        "LLM tool-call generation",
                    )

                except AgentTimeoutError as e:
                    error_event = _timeout_error_event(e.stage)
                    logger.error(f"[TIMEOUT] {error_event['message']}")
                    memory.add_assistant_message(error_event["message"])
                    yield error_event
                    return
                except Exception as e:
                    error_msg = ERROR_MESSAGES["llm_error"].format(error=str(e))
                    logger.error(f"LLM error: {e}")
                    yield {"type": "error", "data": error_msg}
                    memory.add_assistant_message(error_msg)
                    return

            # --- INTERCEPTORS (Pre-process LLM result) ---

            # 1. Bare Tool Name Cleanup & Interception
            if llm_result.get("content"):
                content_clean = llm_result["content"].strip().strip("`'\" \n\r\t").rstrip("()")
                _known_tools = {
                    "getUserInfo", "soqlQuery", "find", "getObjectSchema",
                    "describeSObject", "listRecentRecords", "createSobjectRecord",
                    "updateSobjectRecord", "deleteSobjectRecord", "getGlobalDescribe",
                    "executeApex", "batchCreateRecords"
                }
                if content_clean in _known_tools:
                    logger.info(f"🔄 [BARE TOOL NAME CLEANUP]: {content_clean}")
                    if not llm_result.get("tool_calls") and tools:
                        llm_result["tool_calls"] = [{"id": f"intercepted_tc_{iteration}", "name": content_clean, "arguments": {}}]
                    llm_result["content"] = ""

            # 2. Mandatory Turn 1 Data Query Interceptor
            if not llm_result.get("tool_calls") and iteration == 1 and tools:
                user_msg_lower = user_message.lower()
                data_intent_keywords = _DATA_INTENT_KEYWORDS
                if any(kw in user_msg_lower for kw in data_intent_keywords):
                    logger.warning(
                        f"⚠️ [INTERCEPTOR] LLM returned text without tool call on Turn 1 for query: '{user_message[:40]}...'. "
                        "Forcing tool execution retry."
                    )
                    interceptor_messages = memory.get_messages_for_llm(TOOL_CALLING_PROMPT)
                    interceptor_messages.append({
                        "role": "user",
                        "content": (
                            "TOOL CALL MANDATORY: You must call an appropriate MCP tool (such as soqlQuery, find, or getObjectSchema) "
                            "to fetch real Salesforce data before answering. Do NOT reply with text or dummy data."
                        )
                    })
                    try:
                        retry_result = await _bounded_call(
                            self.llm.chat_with_tools(
                                messages=interceptor_messages,
                                tools=tools,
                                temperature=0.0,
                            ),
                            AGENT_LLM_TIMEOUT,
                            "LLM interceptor retry",
                        )
                        if retry_result.get("tool_calls"):
                            llm_result = retry_result
                    except AgentTimeoutError:
                        error_event = _timeout_error_event("LLM interceptor retry")
                        logger.error(f"[TIMEOUT] {error_event['message']}")
                        memory.add_assistant_message(error_event["message"])
                        yield error_event
                        return
                    except Exception as retry_err:
                        logger.error(f"Interceptor retry error: {retry_err}")

            # 3. Compound Query Completeness Interceptor
            if not llm_result.get("tool_calls") and iteration >= 2 and tools and llm_result.get("content"):
                user_msg_lower = user_message.lower()
                compound_separators = [
                    " and ", " & ", " also ", " along with ", " as well as ",
                    " plus ", " aur ", " tatha ", " evam ",
                ]
                if any(sep in user_msg_lower for sep in compound_separators):
                    response_text = llm_result["content"].strip()
                    headers = list(re.finditer(r"###\s+([^\n]+)", response_text))
                    truly_empty: list[str] = []
                    for i, match in enumerate(headers):
                        start = match.end()
                        end = headers[i + 1].start() if i + 1 < len(headers) else len(response_text)
                        section_content = response_text[start:end].strip()
                        if not section_content or section_content in ("", "-", "N/A"):
                            truly_empty.append(match.group(1).strip())

                    if truly_empty:
                        logger.warning(
                            f"⚠️ [COMPOUND INTERCEPTOR] Empty sections: {truly_empty}. "
                            "Forcing tool execution for missing parts."
                        )
                        interceptor_messages = memory.get_messages_for_llm(SYSTEM_PROMPT)
                        interceptor_messages.append({
                            "role": "user",
                            "content": (
                                "INCOMPLETE MULTI-QUERY RESPONSE: Your response has section headers "
                                f"({', '.join(truly_empty)}) with no data underneath. "
                                "You MUST execute the appropriate MCP tool calls (soqlQuery, find, etc.) "
                                "to fetch the missing data for ALL sections before providing the final answer. "
                                "Do NOT return a final answer until every section has real data from tool execution."
                            ),
                        })
                        try:
                            retry_result = await _bounded_call(
                                self.llm.chat_with_tools(
                                    messages=interceptor_messages,
                                    tools=tools,
                                    temperature=0.0,
                                ),
                                AGENT_LLM_TIMEOUT,
                                "LLM compound retry",
                            )
                            if retry_result.get("tool_calls"):
                                llm_result = retry_result
                        except AgentTimeoutError:
                            error_event = _timeout_error_event("LLM compound retry")
                            logger.error(f"[TIMEOUT] {error_event['message']}")
                            memory.add_assistant_message(error_event["message"])
                            yield error_event
                            return
                        except Exception as interceptor_err:
                            logger.error(f"Compound query interceptor error: {interceptor_err}")

            # ── Case 1: LLM wants to call tools ──
            if llm_result.get("tool_calls"):
                tool_calls = llm_result["tool_calls"]
                logger.info(f"🛠️ [LLM REQUESTED TOOL CALLS]: {[tc['name'] for tc in tool_calls]}")

                # Separate tool calls into safe (non-destructive) and destructive
                safe_calls = []
                destructive_calls = []

                for tc in tool_calls:
                    safety = self.planner.check_tool_safety(
                        tc["name"], tc["arguments"], session_id
                    )
                    if safety["requires_confirmation"]:
                        destructive_calls.append((tc, safety))
                    else:
                        safe_calls.append(tc)

                # ── PERMANENT SPEED FIX: Execute safe tool calls IN PARALLEL ──
                # Instead of waiting for each tool to finish before starting the next,
                # all independent safe calls run simultaneously via asyncio.gather().
                if safe_calls:
                    memory.add_assistant_tool_calls(safe_calls)

                    # Announce all tool calls first (for UI streaming)
                    for tc in safe_calls:
                        logger.info(f"🚀 [EXECUTING TOOL]: {tc['name']} with args: {tc['arguments']}")
                        yield {
                            "type": "tool_call",
                            "data": {"name": tc["name"], "arguments": tc["arguments"]},
                        }

                    # Run each tool under a bounded timeout, returning (tc, result).
                    async def _run_tool(tc: dict, user_provenance: dict) -> tuple[dict, str]:
                        try:
                            result = await _bounded_call(
                                self.executor.execute(
                                    tc["name"], tc["arguments"], user_provenance
                                ),
                                AGENT_EXECUTOR_TIMEOUT,
                                f"Salesforce tool '{tc['name']}' execution",
                            )
                            logger.info(f"✅ [TOOL FINISHED]: {tc['name']} (Result len: {len(result)} chars)")
                        except AgentTimeoutError:
                            # A timeout MUST NOT become a synthetic tool "result".
                            # Propagate so the enclosing handler aborts cleanly.
                            raise
                        except Exception as e:
                            result = json.dumps({"error": str(e), "tool": tc["name"]})
                            logger.error(f"❌ Tool execution error ({tc['name']}): {e}")
                        return tc, result

                    # ── MUTATION SAFETY (#8, ALL-OR-NOTHING): pre-flight validate
                    # EVERY mutation in this response BEFORE executing ANY of them.
                    # If ANY mutation fails the mandatory fail-closed validation,
                    # ZERO mutations from this response reach Salesforce — every
                    # mutation is returned to synthesis as a validation or
                    # blocked-sibling envelope, and only read-only calls execute.
                    # PROVENANCE: required fields must be traceable to the user's
                    # explicit message — LLM-fabricated values are rejected.
                    from sfmcp.executor import _extract_user_provided_fields
                    user_provided_fields = _extract_user_provided_fields(user_message)
                    try:
                        rejected_results: list[tuple[dict, str]] = []
                        validations: list[tuple[dict, str | None]] = []
                        any_rejected = False
                        for tc in safe_calls:
                            if not (is_mutating(tc["name"]) or is_destructive(tc["name"])):
                                continue
                            vres = await _bounded_call(
                                self.executor.validate_mutation(
                                    tc["name"], tc["arguments"], user_provided_fields
                                ),
                                AGENT_EXECUTOR_TIMEOUT,
                                f"Mutation pre-flight '{tc['name']}'",
                            )
                            validations.append((tc, vres))
                            if vres is not None:
                                any_rejected = True

                        if any_rejected:
                            # ZERO mutations execute; read-only calls still run. The
                            # failing mutation yields its own validation envelope;
                            # sibling mutations yield blocked envelopes so synthesis
                            # knows nothing was executed this turn.
                            validation_blocked = True
                            logger.warning(
                                f"[MUTATION-VALIDATION] A mutation in this response "
                                f"failed validation; executing ZERO mutations "
                                f"(all-or-nothing)."
                            )
                            for tc, vres in validations:
                                rejected_results.append(
                                    (tc, vres if vres is not None else _blocked_mutation_result(tc))
                                )
                            execution_calls = [
                                tc for tc in safe_calls
                                if not (is_mutating(tc["name"]) or is_destructive(tc["name"]))
                            ]
                        else:
                            execution_calls = safe_calls

                        parallel_results = await asyncio.gather(
                            *[_run_tool(tc, user_provided_fields)
                              for tc in execution_calls]
                        )
                    except AgentTimeoutError:
                        error_event = _timeout_error_event("Salesforce tool execution")
                        logger.error(f"[TIMEOUT] {error_event['message']}")
                        memory.add_assistant_message(error_event["message"])
                        yield error_event
                        return

                    # Map every announced call (executed + rejected/blocked) to its
                    # final result, processed in the ORIGINAL safe_calls order so
                    # memory and the streaming UI stay consistent.
                    result_by_id: dict[str, str] = {}
                    for tc, r in rejected_results:
                        result_by_id[tc["id"]] = r
                    for tc, r in parallel_results:
                        result_by_id[tc["id"]] = r

                    for tc in safe_calls:
                        result = result_by_id.get(tc["id"])
                        if result is None:
                            result = json.dumps({
                                "error": f"Tool '{tc['name']}' produced no result."
                            })
                        # ── D1: reject executor error envelopes before they can be
                        # formatted / stored / synthesized as normal Salesforce data ──
                        err_msg = _executor_error_message(result)
                        if err_msg:
                            logger.error(f"[SALESFORCE_FAILED] Tool '{tc['name']}' failed: {err_msg}")
                            memory.add_assistant_message(
                                f"Tool '{tc['name']}' failed: {err_msg}"
                            )
                            yield _salesforce_failed_event(f"Salesforce call '{tc['name']}' failed: {err_msg}")
                            return

                        # ── SOQL Error Auto-Correction (max 1 retry — keeps it fast) ──
                        if tc["name"] == "soqlQuery":
                            result_lower = result.lower()
                            is_soql_error = any(kw in result_lower for kw in [
                                "malformed", "syntax error", "invalid_field",
                                "unexpected token", "no such column",
                                "didn't understand", "parse_error",
                            ])
                            if is_soql_error:
                                fix_suggestion = get_soql_fix_suggestion(result)
                                if fix_suggestion:
                                    yield {
                                        "type": "thinking",
                                        "data": f"SOQL query needs correction. Auto-fixing...",
                                    }
                                    fix_messages = memory.get_messages_for_llm(SYSTEM_PROMPT)
                                    fix_messages.append({
                                        "role": "user",
                                        "content": (
                                            f"The SOQL query failed with error: {result}\n\n"
                                            f"Fix suggestion: {fix_suggestion}\n\n"
                                            f"Please provide a corrected SOQL query using the soqlQuery tool. "
                                            f"Output ONLY the corrected tool call."
                                        ),
                                    })
                                    try:
                                        fix_result = await _bounded_call(
                                            self.llm.chat_with_tools(
                                                messages=fix_messages,
                                                tools=tools,
                                                temperature=0.0,
                                            ),
                                            AGENT_LLM_TIMEOUT,
                                            "SOQL auto-fix LLM",
                                        )
                                        if fix_result.get("tool_calls"):
                                            fixed_tc = fix_result["tool_calls"][0]
                                            yield {
                                                "type": "tool_call",
                                                "data": {"name": fixed_tc["name"], "arguments": fixed_tc["arguments"]},
                                            }
                                            # ── SAFETY: the auto-fix LLM may return
                                            # ANY tool. As this path runs OUTSIDE the
                                            # mutation pre-flight/provenance block, a
                                            # mutating/destructive tool here is NEVER
                                            # executed — treated as a hard error. The
                                            # correction may only be a read-only call
                                            # (this whole path is a SOQL correction). ──
                                            if not is_read_only(fixed_tc["name"]):
                                                result = json.dumps({
                                                    "error": (
                                                        f"SOQL auto-fix refused to run "
                                                        f"'{fixed_tc['name']}': only read-only "
                                                        "tools are permitted in auto-corrected "
                                                        "executions."
                                                    ),
                                                    "tool": fixed_tc["name"],
                                                })
                                            else:
                                                result = await _bounded_call(
                                                    self.executor.execute(
                                                        fixed_tc["name"], fixed_tc["arguments"],
                                                        user_provenance={},
                                                    ),
                                                    AGENT_EXECUTOR_TIMEOUT,
                                                    "SOQL auto-fix execution",
                                                )
                                                memory.add_tool_result(tc["id"], tc["name"], result)
                                    except AgentTimeoutError:
                                        error_event = _timeout_error_event("SOQL auto-fix")
                                        logger.error(f"[TIMEOUT] {error_event['message']}")
                                        memory.add_assistant_message(error_event["message"])
                                        yield error_event
                                        return
                                    except Exception as retry_err:
                                        logger.error(f"SOQL retry error: {retry_err}")

                        # ── D1: after SOQL auto-fix retry, the final result must not be
                        # an executor error envelope masquerading as Salesforce data ──
                        err_msg = _executor_error_message(result)
                        if err_msg:
                            logger.error(f"[SALESFORCE_FAILED] Tool '{tc['name']}' failed (after retry): {err_msg}")
                            memory.add_assistant_message(
                                f"Tool '{tc['name']}' failed: {err_msg}"
                            )
                            yield _salesforce_failed_event(f"Salesforce call '{tc['name']}' failed: {err_msg}")
                            return

                        # ── PYTHON TABLE FORMATTING: Flat SOQL → markdown table, COUNT → count line ──
                        # Hierarchical/subquery results return None → raw JSON goes to LLM for card formatting.
                        py_table = format_sf_records_as_markdown(
                            result, tc["name"],
                            soql_query=tc.get("arguments", {}).get("q", tc.get("arguments", {}).get("query", "")),
                        )
                        if py_table:
                            # Store raw result for hallucination checking
                            tool_results_fetched[tc["id"]] = result
                            # Track the pre-built table for potential direct response bypass
                            python_tables[tc["id"]] = (tc["name"], py_table)
                            # Track zero-record results for conversational follow-up
                            try:
                                parsed = json.loads(result)
                                if parsed.get("totalSize", -1) == 0 or not parsed.get("records", []):
                                    zero_record_results.append({
                                        "tool_name": tc["name"],
                                        "query": tc.get("arguments", {}).get("q", tc.get("arguments", {}).get("query", "")),
                                        "tool_id": tc["id"],
                                    })
                            except Exception:
                                pass
                            # Pass formatted table to LLM as clean reference — no aggressive prefix
                            memory_result = (
                                f"[reference_table]\n\n{py_table}"
                            )
                            logger.info(f"📊 [PYTHON TABLE] Built {tc['name']} table "
                                        f"({py_table.count(chr(10))} rows)")
                        else:
                            # Non-tabular result (error, schema, count) — pass through as-is
                            if len(result) > 15000:
                                result = result[:15000] + "\n... [truncated, showing first 15000 chars]"
                            tool_results_fetched[tc["id"]] = result
                            memory_result = result

                        memory.add_tool_result(tc["id"], tc["name"], memory_result)

                        yield {
                            "type": "tool_result",
                            "data": {"name": tc["name"], "result": result},
                        }

                # ── PYTHON DIRECT RESPONSE: Skip LLM for fully-built flat queries ──
                # Only bypass LLM when ALL tool calls produced flat Python tables
                # (no hierarchical cards, no subqueries). Only action keywords that
                # signal a MUTATION force the LLM to narrate; compound read-only
                # phrasing ("and count", "and tell me", "also count") must NOT
                # disqualify a fully-table-built answer, otherwise a mixed list+count
                # query gets re-rendered by the LLM (which can fuse headers and
                # mislabel a COUNT as "1 record").
                _action_kw = [
                    "update", "edit", "change", "modify", "badlo", "set",
                    "delete", "remove", "hatao", "mitao", "drop",
                    "create", "add", "insert", "new", "make", "banao", "daalo",
                    "and delete", "and update", "and create",
                ]
                _has_action_or_complex = any(kw in user_msg_lower for kw in _action_kw)
                if python_tables and safe_calls and len(python_tables) == len(safe_calls) and not destructive_calls and not _has_action_or_complex:
                    sections = []
                    for tc in safe_calls:
                        if tc["id"] in python_tables:
                            _, table_md = python_tables[tc["id"]]
                            # Derive section header from SOQL query or tool name/args
                            soql = tc.get("arguments", {}).get("q", tc.get("arguments", {}).get("query", ""))
                            obj_match = re.search(r"FROM\s+(\w+)", soql, re.IGNORECASE)
                            if obj_match:
                                obj_name = obj_match.group(1)
                            elif tc["name"] == "listRecentSobjectRecords":
                                # listRecentSobjectRecords uses 'sobject-name' not SOQL
                                obj_name = tc.get("arguments", {}).get("sobject-name", "Records")
                            else:
                                obj_name = "Records"
                            # Friendly header
                            _headers = {
                                "Account": "### 🏢 Accounts Found",
                                "Lead": "### 📋 Leads Found",
                                "Contact": "### 👤 Contacts Found",
                                "Opportunity": "### 💰 Opportunities Found",
                                "Case": "### 🎫 Cases Found",
                                "Task": "### ✅ Tasks Found",
                                "Event": "### 📅 Events Found",
                                "User": "### 👥 Users Found",
                            }
                            header = _headers.get(obj_name, f"### {obj_name} Found")
                            sections.append(f"{header}\n\n{table_md}")

                    if sections:
                        direct_response = "\n\n---\n\n".join(sections)

                        # ── Conversational Zero-Records Enhancement ──
                        # When all queries return 0 records, produce ONE unified friendly card.
                        # NEVER expose raw SOQL syntax (WHERE, ORDER BY, >=, LIKE, etc.)
                        if zero_record_results and len(zero_record_results) == len(safe_calls):
                            # Extract clean, human-readable object names from each query
                            searched_objects = []
                            for zr in zero_record_results:
                                q = zr.get("query", "")
                                from_match = re.search(r"FROM\s+(\w+)", q, re.IGNORECASE)
                                if from_match:
                                    obj_name = from_match.group(1)
                                    # Title-case for display: "Event" not "event"
                                    friendly = obj_name.replace("_", " ").strip()
                                    friendly = " ".join(w.capitalize() for w in friendly.split())
                                    if friendly not in searched_objects:
                                        searched_objects.append(friendly)

                            if searched_objects:
                                if len(searched_objects) == 1:
                                    obj_list = searched_objects[0]
                                else:
                                    obj_list = ", ".join(searched_objects[:-1]) + f" and {searched_objects[-1]}"

                                direct_response += (
                                    f"\n\n---\n\n"
                                    f"🔍 **No matching records found** for {obj_list}.\n\n"
                                    f"This could mean:\n"
                                    f"- The records don't exist in your org yet\n"
                                    f"- The search filters didn't match any existing records\n"
                                    f"- The records may be named differently than expected\n\n"
                                    f"**What would you like to try?**\n"
                                    f"- Remove filters and show all {obj_list}\n"
                                    f"- Show recently modified {obj_list}\n"
                                    f"- Search for a specific record by name"
                                )

                        logger.info(f"⚡ [PYTHON DIRECT RESPONSE] Bypassing LLM formatter — "
                                    f"{len(sections)} table(s), {len(direct_response)} chars")
                        memory.add_assistant_message(direct_response)
                        yield {"type": "response", "data": direct_response}
                        return

                # If there are destructive tool calls, block execution of the first one and ask for confirmation
                if destructive_calls:
                    tc, safety = destructive_calls[0]
                    logger.warning(f"⚠️ [SAFETY BLOCK] Confirmation required for '{tc['name']}'")
                    memory.add_assistant_message(safety["confirmation_message"])
                    yield {
                        "type": "confirmation",
                        "data": safety["confirmation_message"],
                    }
                    return

                # Continue the loop — LLM will see tool results and decide next step

            # ── Case 2: LLM returns a final text response ──
            elif llm_result["content"] and llm_result["content"].strip():
                # ── E5 GROUNDED-ANSWER GUARD ──
                # A Salesforce/data request MUST NOT finish with a success=true
                # natural-language answer when no real Salesforce tool result was
                # fetched during the turn — otherwise Qwen could answer from its
                # own knowledge. Clearly general/non-Salesforce queries are
                # unaffected (no data intent => no guard). When a mutation was
                # blocked by fail-closed validation (#8), the validation/blocked
                # envelopes ARE the grounding: the answer asks the user for the
                # missing fields instead of guessing at data, so the guard does
                # not fire. Reuses the D1 controlled error shape (SALESFORCE_FAILED).
                if _has_salesforce_intent(user_message) and not tool_results_fetched and not validation_blocked:
                    logger.error(
                        "[SALESFORCE_FAILED] Salesforce/data request completed with no "
                        "Salesforce tool result; refusing to return an ungrounded answer."
                    )
                    msg = (
                        "I could not access Salesforce data for that request, so I won't "
                        "guess at an answer. Please try again."
                    )
                    memory.add_assistant_message(msg)
                    yield _salesforce_failed_event(msg)
                    return
                response = sanitize_response_output(llm_result["content"].strip())
                if not response:
                    # LLM returned only artifacts — ask for a proper summary
                    response = "I processed your request. How else can I assist you with your Salesforce data?"

                # ── PERMANENT ANTI-HALLUCINATION GUARD ──
                # Detect if LLM invented data not present in any real tool result.
                # Salesforce IDs always start with 3 alphanum chars + 'g5' or similar 18-char pattern.
                # If the response contains IDs that DON'T appear in any real tool result, it's hallucinated.
                invented_ids_found = False
                if tool_results_fetched:
                    # Extract all 18-char Salesforce-style IDs from response
                    response_ids = set(re.findall(r'\b[A-Za-z0-9]{15,18}\b', response))
                    # Collect all IDs that actually came from real tool results
                    all_real_content = " ".join(tool_results_fetched.values())
                    real_ids = set(re.findall(r'\b[A-Za-z0-9]{15,18}\b', all_real_content))
                    # Find IDs in response that never appeared in any tool result
                    ghost_ids = response_ids - real_ids
                    # Filter to only Salesforce-prefix IDs (start with 001/003/006/00Q/etc.)
                    sf_prefixes = ('001', '003', '006', '00Q', '00T', '00U', '500', '00P', '701')
                    ghost_sf_ids = {i for i in ghost_ids if i[:3] in sf_prefixes}
                    if ghost_sf_ids:
                        invented_ids_found = True
                        logger.error(
                            f"🚨 [HALLUCINATION BLOCKED] LLM invented {len(ghost_sf_ids)} fake SF IDs "
                            f"not returned by any tool: {list(ghost_sf_ids)[:5]}"
                        )

                if invented_ids_found:
                    # Strip hallucinated response, show only what was actually fetched
                    fetched_summary_lines = []
                    for tool_name, tool_result in tool_results_fetched.items():
                        fetched_summary_lines.append(f"**{tool_name}** result available (real data from Salesforce).")
                    honest_response = (
                        "⚠️ I detected that my response contained data not returned by Salesforce. "
                        "To prevent showing you incorrect information, here is only what was actually fetched:\n\n"
                        + "\n".join(fetched_summary_lines)
                        + "\n\nPlease try breaking your query into smaller parts (1-2 questions at a time) for accurate results."
                    )
                    logger.warning("🛡️ Hallucination guard activated — serving honest partial response instead.")
                    memory.add_assistant_message(honest_response)
                    yield {"type": "response", "data": honest_response}
                    return

                logger.info(f"🤖 [ASSISTANT RESPONSE]: {response[:150]}...")
                memory.add_assistant_message(response)
                yield {"type": "response", "data": response}
                return

            # ── Case 3: Empty content after tool calls or generation ──
            else:
                if iteration > 1:
                    try:
                        messages = memory.get_messages_for_llm(SYSTEM_PROMPT)
                        messages.append({
                            "role": "user",
                            "content": "Please provide a clear, concise natural language summary of the action or tool results completed above for the user. Format as clean Markdown with tables or bullet points. Do NOT output raw JSON."
                        })
                        summary = await _bounded_call(
                            self.llm.chat(messages),
                            AGENT_LLM_TIMEOUT,
                            "summary generation",
                        )
                        if summary and summary.strip():
                            fallback = sanitize_response_output(summary.strip())
                            if not fallback:
                                fallback = "✅ Operation completed successfully in Salesforce!"
                        else:
                            fallback = "✅ Operation completed successfully in Salesforce!"
                    except Exception:
                        fallback = "✅ Operation completed successfully in Salesforce!"
                else:
                    fallback = "I processed your request. How else can I assist you with your Salesforce data?"

                # E5 grounded-answer guard: a Salesforce/data request with no real
                # tool result must not end in a success=true ungrounded answer.
                # (#8) validation-blocked turns are exempt: the envelopes ground an
                # input request, not a data claim.
                if _has_salesforce_intent(user_message) and not tool_results_fetched and not validation_blocked:
                    logger.error(
                        "[SALESFORCE_FAILED] Salesforce/data request produced no tool "
                        "result; refusing an ungrounded fallback answer."
                    )
                    msg = (
                        "I could not access Salesforce data for that request, so I won't "
                        "guess at an answer. Please try again."
                    )
                    memory.add_assistant_message(msg)
                    yield _salesforce_failed_event(msg)
                    return

                logger.info(f"🤖 [ASSISTANT FALLBACK RESPONSE]: {fallback[:150]}...")
                memory.add_assistant_message(fallback)
                yield {"type": "response", "data": fallback}
                return

        # ── Max iterations reached ──
        # PERMANENT FIX: Never ask LLM to "summarize" here — it will hallucinate missing data.
        # Instead, show ONLY what was actually fetched from Salesforce via real tool calls.
        if tool_results_fetched:
            real_data_msg = (
                "⚠️ Your request had too many parts to complete in one go. "
                "Here is the data I actually fetched from Salesforce (100% real, no guesses):\n\n"
            )
            # Ask LLM to format ONLY the real fetched results — nothing else
            try:
                format_messages = memory.get_messages_for_llm(SYSTEM_PROMPT)
                format_messages.append({
                    "role": "user",
                    "content": (
                        "CRITICAL INSTRUCTION: Format ONLY the tool results already in this conversation into "
                        "clean Markdown tables. DO NOT add any data, records, IDs, names, or numbers that were "
                        "NOT returned by a tool call in this conversation. If some queries were not completed, "
                        "say so explicitly. Output ONLY verified data."
                    ),
                })
                summary = await _bounded_call(
                    self.llm.chat(format_messages),
                    AGENT_LLM_TIMEOUT,
                    "final-format summary generation",
                )
                sanitized = sanitize_response_output(summary.strip()) if summary else ""
                if sanitized:
                    real_data_msg += sanitized
                    real_data_msg += (
                        "\n\n---\n💡 **Tip:** Please send remaining queries separately for complete results."
                    )
                else:
                    real_data_msg = (
                        "⚠️ Request too large to complete in one go. "
                        "Please send queries one at a time (e.g., \"Show me all Accounts\" separately from \"Show all Leads\")."
                    )
            except Exception:
                real_data_msg = (
                    "⚠️ Request too large to complete in one go. "
                    "Please send queries one at a time for accurate results."
                )
        else:
            real_data_msg = (
                "⚠️ I wasn't able to fetch any data for this request. "
                "Please try again with a simpler query."
            )

        memory.add_assistant_message(real_data_msg)
        yield {"type": "response", "data": real_data_msg}

    def clear_session(self, session_id: str = "default") -> None:
        """Clear conversation history and pending confirmations for a session."""
        if session_id in self._memories:
            self._memories[session_id].clear()
        self.planner.clear_pending(session_id)
        logger.info(f"Session '{session_id}' cleared.")

    def get_session_info(self, session_id: str = "default") -> dict[str, Any]:
        """Get info about a session's state."""
        memory = self._get_memory(session_id)
        return {
            "session_id": session_id,
            "message_count": len(memory),
            "has_pending_confirmation": self.planner.has_pending_confirmation(session_id),
        }
