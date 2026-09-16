"""
Tool Executor — executes tool calls against the Salesforce MCP Server.
Provides validation, error handling, retry logic, and result formatting.
"""

import hashlib
import json
import logging
import os
import re
import time
from typing import Any

from .client import SalesforceMCPClient
from .registry import ToolRegistry
from tools.salesforce import is_mutating, is_destructive

logger = logging.getLogger(__name__)

# ── Mutation provenance (field-scoped, request-scoped, FAIL-CLOSED) ───────
# User provenance is NOT stored in any global/context state. The agent derives
# the field-scoped values the user authored in the CURRENT request message via
# _extract_user_provided_fields() and passes them EXPLICITLY to
# validate_mutation()/execute() as an immutable dict {api_field: frozenset}.
# There is no disable flag and no hidden context: a caller that provides no
# provenance at all is treated as having authored nothing and body-bearing
# mutations FAIL CLOSED.
#
# Body-bearing mutations are the ones with a field->value map that an LLM could
# fabricate. uploadRecordAttachment is deliberately excluded: it has no
# field->value body (the only field the REST fallback derives, ContentVersion.Title,
# comes deterministically from the file name — an approved transformation, not a
# fabricated value).
_BODY_BEARING_MUTATIONS: frozenset[str] = frozenset({
    "createSobjectRecord", "updateSobjectRecord", "updateRelatedRecord",
})

# Defense-in-depth: even if a caller bypasses the orchestrator safety planner,
# no mutating/destructive tool may run while READ_ONLY_MODE is enabled.
READ_ONLY_MODE = os.getenv("READ_ONLY_MODE", "false").lower() in ("true", "1", "yes", "on")

# ── Mutation idempotency (duplicate-record prevention) ───────────────────
# An identical mutating/destructive tool call (same tool name AND same
# arguments) repeated within this window returns the PREVIOUS result instead of
# executing a second time. This is the safety net for the LLM emitting the same
# create/update/delete twice, a planner re-issue, or a request replay — so one
# user submission can never create two Leads. Only MUTATIONS are deduplicated;
# read-only tools always re-execute. The store is bounded (TTL + size cap).
MUTATION_DEDUPE_TTL = float(os.getenv("MUTATION_DEDUPE_TTL", "90"))
_MAX_RECENT_MUTATIONS = 200
_recent_mutations: dict[str, tuple[str, float]] = {}


def _canonical_args(value: Any) -> Any:
    """Recursively canonicalize arguments so key order never changes the hash."""
    if isinstance(value, dict):
        return {k: _canonical_args(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [_canonical_args(v) for v in value]
    return value


def _mutation_key(tool_name: str, arguments: dict[str, Any]) -> str:
    payload = json.dumps(
        {"tool": tool_name, "arguments": _canonical_args(arguments)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _recent_mutation(key: str) -> str | None:
    prev, ts = _recent_mutations.get(key, (None, 0.0))
    if prev is not None and (time.monotonic() - ts) <= MUTATION_DEDUPE_TTL:
        return prev
    return None


def _remember_mutation(key: str, result: str) -> None:
    _recent_mutations[key] = (result, time.monotonic())
    now = time.monotonic()
    stale = [k for k, (_, ts) in _recent_mutations.items() if now - ts > MUTATION_DEDUPE_TTL]
    for k in stale:
        _recent_mutations.pop(k, None)
    while len(_recent_mutations) > _MAX_RECENT_MUTATIONS:
        _recent_mutations.pop(next(iter(_recent_mutations)))


# ── Mutation provenance enforcement ──────────────────────────────────────
# Maps common field name variants (spoken, written, labeled) to the API name
# used in Salesforce tool-call bodies.  Used by _extract_user_provided_fields
# to map natural-language field values from the user's message to the API
# fields we must verify.
_FIELD_LABEL_TO_API: dict[str, str] = {
    "last name": "LastName", "lastname": "LastName", "last-name": "LastName",
    "first name": "FirstName", "firstname": "FirstName", "first-name": "FirstName",
    "company": "Company", "company name": "Company", "company-name": "Company",
    "email": "Email", "email address": "Email", "email-address": "Email",
    "phone": "Phone", "phone number": "Phone", "phone-number": "Phone",
    "title": "Title", "lead source": "LeadSource", "leadsource": "LeadSource",
    "status": "Status", "lead-status": "Status", "leadstatus": "Status",
    "name": "Name", "account name": "Name", "account-name": "Name",
    "opportunity name": "Name", "opportunity-name": "Name",
    "stage": "StageName", "stage name": "StageName", "stagename": "StageName",
    "close date": "CloseDate", "closedate": "CloseDate", "close-date": "CloseDate",
    "subject": "Subject",
}

# Reverse map: Salesforce API field name -> human label, used in provenance /
# validation error messages so the user sees plain-language field names.
_API_TO_LABEL: dict[str, str] = {
    "LastName": "Last Name", "FirstName": "First Name", "Company": "Company Name",
    "Email": "Email", "Phone": "Phone", "Title": "Title", "LeadSource": "Lead Source",
    "Status": "Status", "Name": "Name", "StageName": "Stage", "CloseDate": "Close Date",
    "Subject": "Subject",
}


def _normalize_value(value: Any) -> str:
    """Normalize a single field value for FULL-VALUE provenance equality.

    - trims surrounding whitespace
    - strips surrounding matching single/double quotes
    - collapses internal whitespace runs to a single space
    - strips a single trailing period
    - casefolds

    Full normalized values are compared by equality with the field's allowed
    set — never by substring or token membership, so truncation and
    cross-field swaps cannot pass."""
    s = str(value).strip()
    if len(s) >= 2 and s[0] in ("'", '"') and s[-1] == s[0]:
        s = s[1:-1].strip()
    s = re.sub(r"\s+", " ", s).strip()
    if s.endswith("."):
        s = s[:-1].rstrip()
    return s.casefold()


# Map every user-facing label (and the lowercased API names) to the Salesforce
# API field name used in tool-call bodies.  Used by _extract_user_provided_fields
# to translate a user's words into the fields the gate must verify.
_TERM_TO_FIELD: dict[str, str] = {}
for _label, _api in _FIELD_LABEL_TO_API.items():
    _TERM_TO_FIELD.setdefault(_label.lower(), _api)
for _api in _FIELD_LABEL_TO_API.values():
    _TERM_TO_FIELD.setdefault(_api.lower(), _api)
# Drop one-shot loop variables left in the module namespace.
try:
    del _label, _api
except NameError:
    pass

# Words that terminate an UNQUOTED value binding (in addition to every field label).
_BINDING_STOP_WORDS: tuple[str, ...] = ("and", "or", "but", "please", "also", "then")

# Characters allowed inside a single unquoted value (e.g. emails, phones with
# hyphens/plus, apostrophes, ampersands, slashes, hashes, status strings with
# '-' like 'Working - Contacted').  Comma, period (sentence ends), newline and
# the stop words are boundaries, not value chars.
_VALUE_CHARS = r"0-9A-Za-z .'@#&+/_-"
_VALUE_START = r"0-9A-Za-z#+"


def _extract_user_provided_fields(user_message: str) -> dict[str, frozenset[str]]:
    """Extract the field-scoped set of NORMALIZED VALUES the user authored in
    the CURRENT message.

    Returns {api_field: frozenset(normalized_values)}.  Only deterministic
    explicit bindings count:

      - 'Last Name: Sharma'        (colon / 全角 colon)
      - 'Company = Tech Solutions' (=)
      - 'Company is Acme', 'Stage is Prospecting', 'Status are Working' (is/are/as)

    An unquoted value terminates at: the next field label (longest-match), a
    ';'/'.'/newline, a ',' (value-list separator), or a word-bounded
    'and'/'or'/'but' conjunction.  Quoted values ('Acme Corp' / "Research and
    Development") bind their full content and are exempt from those boundaries.

    NEVER bound (deterministically ambiguous -> the requesting side must repeat
    the value with an explicit label): bare comma-separated values, whitespace
    bindings ('Last Name Sharma'), and for/named/called/by-the-name-of captures.
    Mapping an unlabeled value to a field would be guessing, which is prohibited.
    """
    if not user_message:
        return {}
    terms = sorted(_TERM_TO_FIELD, key=len, reverse=True)
    stop_terms = sorted(set(terms) | set(_BINDING_STOP_WORDS), key=len, reverse=True)
    stop_alt = "|".join(re.escape(t) for t in stop_terms)

    result: dict[str, set[str]] = {}
    masked = user_message

    for term in terms:
        api = _TERM_TO_FIELD[term]
        escaped = re.escape(term)
        leading = r"(?:^|[\s,;，；。：（(])"
        # A value stops at a sentence boundary (comma/semicolon/newline, plus
        # full-width ，；。), a WORD-BOUNDED stop word / next field label
        # (longest-match), a period followed by whitespace or end (sentence
        # period — internal dots in emails/URLs stay part of the value), or the
        # end of the message.  Word boundaries prevent a stop word such as 'or'
        # from matching inside a longer word ('Acme Corp' must not truncate
        # after 'Acme C').
        boundary = r"(?:(?:\s*(?:,|;|，|；|。|\n|\b(?:" + stop_alt + r")\b)|\s*[.。](?=\s|$))|\s*$)"
        value_re = "([" + _VALUE_START + "][" + _VALUE_CHARS + "]*?)"

        mid_opt = r"(?:\s+of\s+[^,;:]+?)?"
        verb_sep = r"(?:is|are|as|to|set\s+to|change\s+to|equals|becomes)"

        pat_colon = re.compile(
            leading + escaped + r"\s*[:：=]\s*" + value_re +
            r"(?=" + boundary + r")",
            re.IGNORECASE,
        )
        pat_is = re.compile(
            leading + escaped + mid_opt + r"\s+(?:" + verb_sep + r")\s+" + value_re +
            r"(?=" + boundary + r")",
            re.IGNORECASE,
        )
        pat_colon_quoted = re.compile(
            leading + escaped + r"\s*[:：=]\s*(['\"])([^'\"]*)\1",
            re.IGNORECASE,
        )
        pat_is_quoted = re.compile(
            leading + escaped + mid_opt + r"\s+(?:" + verb_sep + r")\s+(['\"])([^'\"]*)\1",
            re.IGNORECASE,
        )
        # Label lookahead used ONLY to blank the label span (so a shorter term
        # like 'name' can never double-bind inside a longer label such as
        # 'last name').  It matches the label only when a binding separator
        # follows; it never consumes the separator or the value.
        label_re = re.compile(
            leading + escaped + mid_opt + r"(?=\s*[:：=]|\s+(?:" + verb_sep + r")\s+)",
            re.IGNORECASE,
        )

        label_spans: list[tuple[int, int]] = []
        for m in label_re.finditer(masked):
            label_spans.append((m.end() - len(term), m.end()))

        def _record(m: re.Match) -> None:
            raw = m.group(2) if m.lastindex == 2 else m.group(1)
            norm = _normalize_value(raw)
            if norm and norm not in ("none", "n/a", "unknown", "tbd", "na"):
                result.setdefault(api, set()).add(norm)

        for pat in (pat_colon, pat_is, pat_colon_quoted, pat_is_quoted):
            for m in pat.finditer(masked):
                _record(m)

        if label_spans:
            chars = list(masked)
            for ts, te in label_spans:
                for i in range(ts, te):
                    chars[i] = " "
            masked = "".join(chars)

    return {api: frozenset(vals) for api, vals in result.items()}


def _is_blank(value: Any) -> bool:
    """A body value is 'blank' when it is absent/None/whitespace-only.  Blank
    values are deferred to the presence gate, never treated as provenance
    fabrications."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _provenance_failure_envelope(
    tool_name: str,
    sobject_name: str,
    fields: list[tuple[str, str]],
    error: str,
    suggestion: str,
    missing_required: bool = False,
    missing_field_options: dict[str, list[str]] | None = None,
) -> str:
    """Structured fail-closed validation envelope (machine + human readable).

    ``missing_required=True`` marks the case where ``fields`` is the
    createable-and-required set the user must supply for a blocked CREATE
    (resolved from Describe metadata / static registry), as opposed to the
    list of body fields whose values were fabricated.

    ``missing_field_options`` (optional) carries the METADATA-DERIVED choice
    hints for those required fields — active picklist values for enum fields
    and referenced object names for lookup fields — so an ask can show what
    values are allowed. It is only populated with ENUM/lookup hints (never
    fabricated values), so no invented example values are ever presented.
    """
    payload: dict[str, Any] = {
        "error": error,
        "tool": tool_name,
        "validation_error": True,
        "retry_allowed": False,
        "requires_user_input": True,
        "missing_fields": [api for api, _ in fields],
        "missing_fields_human": [label for _, label in fields],
        "sobject_name": sobject_name,
        "suggestion": suggestion,
    }
    if missing_required:
        payload["missing_required"] = True
    if missing_field_options:
        payload["missing_field_options"] = {
            str(api): list(options) for api, options in missing_field_options.items() if options
        }
    return json.dumps(payload)


def _provenance_satisfied_fields(
    body: dict[str, Any],
    user_provenance: dict[str, frozenset[str]] | None,
) -> set[str]:
    """Return the body field APIs whose values are explicitly user-authored
    (full-value provenance satisfied) in the current request message."""
    prov = user_provenance if isinstance(user_provenance, dict) else {}
    satisfied: set[str] = set()
    for api, value in body.items():
        if _is_blank(value):
            continue
        if _normalize_value(value) in prov.get(str(api), frozenset()):
            satisfied.add(str(api))
    return satisfied


def _validate_provenance(
    tool_name: str,
    body: dict[str, Any],
    user_provenance: dict[str, frozenset[str]] | None,
    sobject_name: str,
) -> str | None:
    """Fail-closed provenance gate: returns an error JSON string when ANY
    non-blank body field value was NOT authored by the user for that SAME field
    in the current request; None when provenance holds.

    - `user_provenance is None` (a caller that provides no provenance at all)
      FAILS CLOSED for every non-blank body field — a mutation without
      user-authored values must never reach the transport.
    - Otherwise each body value is matched FULL-VALUE against its own field's
      allowed set.  Truncated values and cross-field swaps therefore fail.
    """
    if not isinstance(body, dict) or not body:
        return None
    fields = [(str(api), value) for api, value in body.items() if not _is_blank(value)]
    if not fields:
        return None

    def _label(api: str) -> str:
        return _API_TO_LABEL.get(api, api)

    if user_provenance is None:
        named = [(api, _label(api)) for api, _ in fields]
        labels = ", ".join(lbl for _, lbl in named)
        return _provenance_failure_envelope(
            tool_name, sobject_name, named,
            (f"Cannot {tool_name}: this mutation carried no user provenance. "
             f"The field value(s) ({labels}) were never provided by the user — "
             "executing would stamp un-authored data into Salesforce."),
            "Route the user's request through the agent so explicit per-field "
            "provenance is attached, or ask the user to provide the fields.",
        )

    fabricated: list[tuple[str, str]] = []
    for api, value in fields:
        allowed = user_provenance.get(api, frozenset())
        if _normalize_value(value) in allowed:
            continue
        fabricated.append((api, _label(api)))

    if not fabricated:
        return None

    labels = ", ".join(lbl for _, lbl in fabricated)
    return _provenance_failure_envelope(
        tool_name, sobject_name, fabricated,
        (f"Cannot {tool_name}: field value(s) ({labels}) were NOT provided by the "
         "user — the system must not fabricate field values; omit fields the user "
         "did not supply explicitly."),
        "Ask the user to provide the fields explicitly, then submit a corrected "
        "request with only user-supplied values.",
    )


class ToolExecutor:
    """
    Executes Salesforce MCP tool calls.

    Responsibilities:
    - Validate tool arguments against schemas
    - Execute tools via the MCP client
    - Handle errors gracefully with informative messages
    - Format results as strings for LLM consumption
    """

    def __init__(self, mcp_client: SalesforceMCPClient, registry: ToolRegistry):
        self.mcp_client = mcp_client
        self.registry = registry

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_provenance: dict[str, frozenset[str]] | None = None,
    ) -> str:
        """
        Execute a tool call and return the result as a formatted string.

        Args:
            tool_name: The name of the tool to execute.
            arguments: The arguments to pass to the tool.
            user_provenance: The field-scoped, request-scoped provenance map
                ({api_field -> frozenset(normalized values)}) the user authored
                in the CURRENT message. Passed through to validate_mutation.
                None fails closed for body-bearing mutations (never authorized).

        Returns:
            A JSON-formatted string with the tool result.
        """
        # Validate tool exists
        if not self.registry.has_tool(tool_name):
            available = ", ".join(self.registry.list_tool_names())
            error = f"Tool '{tool_name}' not found. Available: {available}"
            logger.error(error)
            return json.dumps({"error": error})

        if READ_ONLY_MODE and (is_mutating(tool_name) or is_destructive(tool_name)):
            logger.warning(
                f"Read-only mode blocked tool execution: {tool_name} "
                f"(args={_truncate_args(arguments)})"
            )
            return json.dumps({
                "error": (
                    f"Tool '{tool_name}' is blocked by READ_ONLY_MODE. "
                    "Create, update, upload, and delete operations are disabled "
                    "during this evaluation run."
                ),
                "tool": tool_name,
            })

        # Mutation field validation: a mandatory, FAIL-CLOSED gate for every
        # mutating/destructive tool. createSobjectRecord resolves required fields
        # from live Describe metadata (read-only schema path) for BOTH standard and
        # custom objects; the static registry is a zero-I/O fallback when Describe
        # is unavailable. If the required schema cannot be established, the mutation
        # is REJECTED (never guessed, never fabricated — no "Unknown"/"Individual"/
        # "New Account" defaults are ever inserted). update/delete/upload validate
        # structural fields (valid IDs, non-empty field bodies). The gate runs
        # BEFORE idempotency so a rejected mutation is never cached as a successful
        # execution, and BEFORE mcp_client.call_tool so a rejected mutation never
        # reaches Salesforce via MCP or REST.
        is_write = is_mutating(tool_name) or is_destructive(tool_name)
        if is_write:
            validation_error = await self.validate_mutation(
                tool_name, arguments, user_provenance
            )
            if validation_error is not None:
                logger.warning(
                    f"[MUTATION-VALIDATION] Tool '{tool_name}' rejected: {validation_error}"
                )
                return validation_error

        # Mutation idempotency: an identical mutating/destructive call within the
        # dedupe window reuses the previous result instead of creating a second
        # record. Reads are never deduplicated (re-list/query is harmless).
        dedupe_key = _mutation_key(tool_name, arguments) if is_write else None
        if dedupe_key:
            cached = _recent_mutation(dedupe_key)
            if cached is not None:
                logger.warning(
                    f"[IDEMPOTENCY] Skipping duplicate '{tool_name}' call "
                    f"(identical arguments within {MUTATION_DEDUPE_TTL:.0f}s window); "
                    "returning the previous result to avoid a duplicate record."
                )
                return cached

        # Sanitize SOQL query if LLM injected Apex-style bind variables (e.g. :$User.Id or :UserInfo.getUserId())
        if tool_name == "soqlQuery" and isinstance(arguments, dict):
            raw_q = arguments.get("q") or arguments.get("query")
            if raw_q:
                clean_q = await self._sanitize_soql_query(raw_q)
                arguments = dict(arguments)
                arguments["q"] = clean_q
                if "query" in arguments:
                    arguments["query"] = clean_q

        # Sanitize SOSL search query if tool is 'find' (enforce FIND {term} without quotes)
        if tool_name == "find" and isinstance(arguments, dict):
            raw_q = arguments.get("q") or arguments.get("query")
            if raw_q:
                clean_q = self._sanitize_sosl_query(raw_q)
                arguments = dict(arguments)
                arguments["q"] = clean_q
                if "query" in arguments:
                    arguments["query"] = clean_q

        logger.info(f"Executing tool: {tool_name} with args: {_truncate_args(arguments)}")

        try:
            result = await self.mcp_client.call_tool(tool_name, arguments)
            formatted = self._format_result(tool_name, result)
            logger.info(f"Tool {tool_name} executed successfully.")
            if dedupe_key:
                _remember_mutation(dedupe_key, formatted)
            return formatted

        except RuntimeError as e:
            error_msg = str(e)
            logger.error(f"Tool {tool_name} failed: {error_msg}")
            return json.dumps({
                "error": error_msg,
                "tool": tool_name,
                "suggestion": self._get_error_suggestion(tool_name, error_msg),
            })

        except Exception as e:
            error_msg = f"Unexpected error executing {tool_name}: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg, "tool": tool_name})

    async def _sanitize_soql_query(self, query: str) -> str:
        """
        Auto-correct common LLM SOQL syntax issues before sending to Salesforce:
        1. Replace Apex bind variables like :$User.Id or :UserInfo.getUserId()
           with the actual user ID literal '005...' (or drop the invalid WHERE clause).
        """
        if not query or not isinstance(query, str):
            return query

        cleaned = query.strip()

        # Check for Apex bind variables like :$User.Id, :UserInfo.getUserId(), :userId
        bind_match = re.search(r":(\$User\.Id|UserInfo\.getUserId\(\)|userId|currentUserId|[A-Za-z0-9_$.()]+)", cleaned, re.IGNORECASE)
        if bind_match:
            user_id = getattr(self.mcp_client, "_cached_user_id", None)
            if not user_id:
                try:
                    ui_res = await self.mcp_client.call_tool("getUserInfo", {})
                    if isinstance(ui_res, dict):
                        identity = ui_res.get("identity") or {}
                        user_id = identity.get("userId") or ui_res.get("userId") or ui_res.get("sub") or ui_res.get("user_id")
                    elif isinstance(ui_res, str):
                        try:
                            parsed = json.loads(ui_res)
                            if isinstance(parsed, dict):
                                identity = parsed.get("identity") or {}
                                user_id = identity.get("userId") or parsed.get("userId") or parsed.get("sub") or parsed.get("user_id")
                        except Exception:
                            pass
                    if user_id and str(user_id).startswith("005"):
                        self.mcp_client._cached_user_id = str(user_id)
                except Exception as e:
                    logger.warning(f"Could not auto-resolve user_id for bind variable: {e}")

            if user_id and str(user_id).startswith("005"):
                cleaned = re.sub(
                    r":(\$User\.Id|UserInfo\.getUserId\(\)|userId|currentUserId)",
                    f"'{user_id}'",
                    cleaned,
                    flags=re.IGNORECASE
                )
                logger.info(f"🔄 [SOQL AUTO-FIX] Inlined literal User ID '{user_id}' in place of Apex bind variable")
            else:
                # Strip the invalid WHERE clause so query succeeds cleanly
                cleaned = re.sub(
                    r"\s*WHERE\s+OwnerId\s*=\s*:[A-Za-z0-9_$.()]+\b",
                    "",
                    cleaned,
                    flags=re.IGNORECASE
                )
                cleaned = re.sub(
                    r"\s*AND\s+OwnerId\s*=\s*:[A-Za-z0-9_$.()]+\b",
                    "",
                    cleaned,
                    flags=re.IGNORECASE
                )
                logger.warning("⚠️ [SOQL AUTO-FIX] Removed invalid Apex bind variable from SOQL query")

        # Convert SQL subqueries like "WHERE AccountId = (SELECT Id FROM Account WHERE Name = 'X')"
        # into native Salesforce relationship traversal "WHERE Account.Name = 'X'"
        generic_subquery = re.search(
            r"WHERE\s+([A-Za-z0-9_]+)Id\s*(?:=|IN)\s*\(\s*SELECT\s+Id\s+FROM\s+([A-Za-z0-9_]+)\s+WHERE\s+(?:Name\s*=\s*|Name\s+LIKE\s*)('[^']+')\s*\)",
            cleaned,
            re.IGNORECASE
        )
        if generic_subquery:
            rel_field = generic_subquery.group(1)
            val = generic_subquery.group(3)
            cleaned = re.sub(
                r"WHERE\s+[A-Za-z0-9_]+Id\s*(?:=|IN)\s*\(\s*SELECT\s+Id\s+FROM\s+[A-Za-z0-9_]+\s+WHERE\s+(?:Name\s*=\s*|Name\s+LIKE\s*)'[^']+'\s*\)",
                f"WHERE {rel_field}.Name = {val}",
                cleaned,
                flags=re.IGNORECASE
            )
            logger.info(f"🔄 [SOQL AUTO-FIX] Converted SQL subquery into native relationship filter 'WHERE {rel_field}.Name = {val}'")

        # Auto-fix aggregate aliases in ORDER BY:
        # e.g. "SELECT OwnerId, SUM(Amount) totalSales ... ORDER BY totalSales DESC"
        # SOQL requires "ORDER BY SUM(Amount) DESC" because aliases cannot be used in ORDER BY
        alias_matches = re.findall(
            r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*([A-Za-z0-9_$.]+|\*)\s*\)\s+([A-Za-z0-9_]+)\b",
            cleaned,
            re.IGNORECASE,
        )
        if alias_matches:
            order_by_idx = cleaned.upper().find("ORDER BY")
            if order_by_idx != -1:
                before_part = cleaned[:order_by_idx]
                after_part = cleaned[order_by_idx:]
                for fn, arg, alias in alias_matches:
                    if re.search(r"\b" + re.escape(alias) + r"\b", after_part, re.IGNORECASE):
                        after_part = re.sub(
                            r"\b" + re.escape(alias) + r"\b",
                            f"{fn.upper()}({arg})",
                            after_part,
                            flags=re.IGNORECASE,
                        )
                        logger.info(f"🔄 [SOQL AUTO-FIX] Replaced invalid alias '{alias}' in ORDER BY with '{fn.upper()}({arg})'")
                cleaned = before_part + after_part

        return cleaned

    @staticmethod
    def _sanitize_sosl_query(query: str) -> str:
        """
        Universally sanitize SOSL search queries before sending to Salesforce.
        Salesforce SOSL requires FIND {term} without single or double quotes around the term.
        Handles:
          - FIND 'term' -> FIND {term}
          - FIND "term" -> FIND {term}
          - FIND {'term'} -> FIND {term}
          - FIND {"term"} -> FIND {term}
          - FIND term -> FIND {term}
          - Bare term e.g. "United" -> FIND {United} IN ALL FIELDS RETURNING Account(Id, Name), Contact(Id, Name, Email), Lead(Id, Name, Company)
        """
        if not query or not isinstance(query, str):
            return query
        cleaned = query.strip()

        # If LLM sent bare search term without FIND
        if not cleaned.upper().startswith("FIND"):
            term = cleaned.strip("'\"{} \t\r\n")
            return f"FIND {{{term}}} IN ALL FIELDS RETURNING Account(Id, Name), Contact(Id, Name, Email), Lead(Id, Name, Company)"

        pattern = r"FIND\s+(?:\{['\"]*([^}\"']+)['\"]*\}|['\"]([^'\"]+)['\"]|([^\s{'\"]+))"
        m = re.search(pattern, cleaned, re.IGNORECASE)
        if m:
            term = m.group(1) or m.group(2) or m.group(3)
            if term:
                term = term.strip("'\"{} \t\r\n")
                cleaned = re.sub(pattern, f"FIND {{{term}}}", cleaned, count=1, flags=re.IGNORECASE)
                logger.info(f"🔄 [SOSL AUTO-FIX] Sanitized SOSL query to: {cleaned}")

        return cleaned

    async def validate_mutation(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        user_provenance: dict[str, frozenset[str]] | None = None,
    ) -> str | None:
        """
        Run the mandatory fail-closed mutation validation. Returns an error JSON
        string when the mutation must be blocked, or None when it may proceed.

        user_provenance: the field-scoped set of normalized values the user
        authored in THIS request message ({api_field -> frozenset}). None means
        the caller carried NO provenance at all: for body-bearing mutations
        (create/update/related) that is FAIL-CLOSED — every non-blank body value
        is treated as un-authored.

        For a body-bearing mutation the checks run in order:
        1. PROVENANCE: every non-blank body field value must equal (full-value,
           normalized) a value the user bound to that SAME field in this request.
        2. PRESENCE: required fields must be present and non-blank.
        3. OTHER mutation safety (valid IDs, non-empty bodies, known schema).

        For createSobjectRecord, required fields are resolved from live Describe
        metadata (read-only schema path) for BOTH standard and custom objects so
        per-org custom required fields are honored and nothing is blindly
        hard-coded. The static registry is only a zero-I/O fallback when Describe
        is unavailable; unknown objects still fail closed.
        """
        from agent.mutation_validation import (
            validate_mutation_fields,
            _extract_clean_body,
        )

        sobject_name = ""
        for key in ("sobject-name", "sobject_name", "sobject", "object", "sobjectName", "objectName"):
            if key in arguments and arguments[key]:
                sobject_name = str(arguments[key]).strip()
                break

        # ── PROVENANCE GATE (deterministic, field-scoped, full-value): every
        # non-blank body field value must trace back to a value the user bound
        # to that SAME field in the current message. No global/context store, no
        # substring/token matching, no disable flag. A caller that passes no
        # provenance (None) FAILS CLOSED for every non-blank body field. ──
        if tool_name in _BODY_BEARING_MUTATIONS:
            body = _extract_clean_body(arguments)
            prov_err = _validate_provenance(
                tool_name, body, user_provenance, sobject_name
            )
            if prov_err is not None:
                # PROBLEM 1 (create UX): a vague "create a lead" is blocked
                # correctly by provenance, but the default envelope lists the
                # FABRICATED body fields. For a create, the user ask must instead
                # be the true createable-and-required set (Describe metadata
                # authoritative, static registry fallback) minus the fields the
                # user actually provided — so custom required fields are surfaced
                # and no fabricated labels leak into the ask. The gate itself is
                # unchanged: still blocked, still fail-closed, no record written.
                if tool_name == "createSobjectRecord":
                    prov_err = await self._enrich_create_provenance_envelope(
                        tool_name, body, user_provenance,
                        sobject_name, prov_err,
                    )
                logger.warning(
                    f"[MUTATION-PROVENANCE] '{tool_name}' rejected: body values "
                    "were not user-provided in this request."
                )
                return prov_err

        # ── PRESENCE GATE: required fields must exist and be non-blank. ──
        resolver = getattr(self.mcp_client, "describe_required_fields", None)
        needs_describe = (
            tool_name == "createSobjectRecord"
            and bool(sobject_name)
            and resolver is not None
        )
        if needs_describe:
            try:
                resolved = await resolver(sobject_name)
            except Exception as exc:  # noqa: BLE001 - resolver failure fails closed
                logger.error(
                    f"[MUTATION-VALIDATION] describe_required_fields failed for "
                    f"'{sobject_name}': {exc}"
                )
                resolved = None
            err = validate_mutation_fields(tool_name, arguments, lambda _s: resolved)
            return await self._enrich_presence_envelope_options(
                tool_name, sobject_name, err
            )
        err = validate_mutation_fields(tool_name, arguments, None)
        return await self._enrich_presence_envelope_options(
            tool_name, sobject_name, err
        )

    async def _enrich_presence_envelope_options(
        self,
        tool_name: str,
        sobject_name: str,
        envelope: str | None,
    ) -> str | None:
        """Attach METADATA-DERIVED choice hints to a PASSED-BACK required-fields
        envelope so the deterministic ask can list what values are allowed for
        enum/lookup fields. Only fills ``missing_field_options`` when the
        envelope is a required-create ask (``missing_required``), never for
        structural errors or fabricated-field provenance envelopes. Returns the
        envelope unchanged otherwise (safe on None / foreign envelopes)."""
        if tool_name != "createSobjectRecord" or not envelope:
            return envelope
        if not sobject_name:
            return envelope
        try:
            parsed = json.loads(envelope)
        except (json.JSONDecodeError, TypeError):
            return envelope
        if not isinstance(parsed, dict) or parsed.get("missing_required") is not True:
            return envelope
        fields = list(parsed.get("missing_fields") or [])
        if not fields:
            return envelope
        required_pairs = [
            (api, label)
            for api, label in zip(
                parsed.get("missing_fields") or [],
                parsed.get("missing_fields_human") or [],
            )
        ]
        required_with_options = await self._with_required_field_options(
            required_pairs, sobject_name
        )
        options = {
            str(api): list(opts)
            for api, _, opts in required_with_options
            if opts
        }
        if not options:
            return envelope
        parsed["missing_field_options"] = options
        return json.dumps(parsed)

    async def _resolve_create_required(
        self, sobject_name: str
    ) -> list[tuple[str, str]] | None:
        """Resolve the createable-and-required fields of an object for a create:
        live Describe metadata (read-only schema path) first, static registry as
        a zero-I/O fallback; None when neither is available.

        An authoritative EMPTY list (Describe resolved and the object genuinely
        has no create-required fields) is a VALID resolution and is returned as
        ``[]`` — it is distinct from None (schema unknown -> fail closed)."""
        from agent.mutation_validation import OBJECT_REQUIRED_FIELDS

        resolver = getattr(self.mcp_client, "describe_required_fields", None)
        if resolver is not None:
            try:
                resolved = await resolver(sobject_name)
            except Exception as exc:  # noqa: BLE001 - resolver failure fails closed
                logger.error(
                    f"[MUTATION-VALIDATION] describe_required_fields failed for "
                    f"'{sobject_name}' while resolving the required-fields ask: {exc}"
                )
                resolved = None
            if resolved is not None:
                return [(api, label) for api, label in resolved if api and label]
        static = OBJECT_REQUIRED_FIELDS.get(sobject_name.lower()) if sobject_name else None
        if static is None:
            return None
        return [(api, label) for api, label in static if api and label]

    async def _with_required_field_options(
        self,
        required: list[tuple[str, str]],
        sobject_name: str,
    ) -> list[tuple[str, str, list[str]]]:
        """Augment the required fields of an object with live choice hints
        (active picklist values for enum fields, referenced object names for
        lookup fields) so the deterministic required-fields ask can show what
        values are allowed. Hints come from Describe metadata (reusing the
        shared schema cache — no extra describe call difficulty); when Describe
        is unavailable the static fallback hints are empty (never guessed)."""
        options: dict[str, list[str]] = {}
        resolver = getattr(self.mcp_client, "describe_required_field_options", None)
        if resolver is not None:
            try:
                options = await resolver(sobject_name) or {}
            except Exception as exc:  # noqa: BLE001 - resolver failure fails closed
                logger.error(
                    f"[MUTATION-VALIDATION] describe_required_field_options failed for "
                    f"'{sobject_name}': {exc}"
                )
                options = {}
        return [
            (api, label, options.get(api) or [])
            for api, label in required
        ]

    async def _enrich_create_provenance_envelope(
        self,
        tool_name: str,
        body: dict[str, Any],
        user_provenance: dict[str, frozenset[str]] | None,
        sobject_name: str,
        prov_err: str,
    ) -> str:
        """For a CREATE blocked by the provenance gate, rebuild the envelope so
        ``missing_fields``/``missing_fields_human`` = the object's required set
        (minus the fields the user actually supplied) and mark it
        ``missing_required=True``. This lets the synthesizer ask deterministically
        for exactly the fields the user must provide, including custom required
        fields from Describe. Returns the original envelope unchanged when the
        required set cannot be established (fail-closed) or when every required
        field was already user-provided (e.g. only OPTIONAL fields were
        fabricated — those are NOT required and must not be asked for as such)."""
        required = await self._resolve_create_required(sobject_name)
        if not required:
            return prov_err
        provided = _provenance_satisfied_fields(body, user_provenance)
        missing = [(api, label) for api, label in required if api not in provided]
        if not missing:
            return prov_err
        required_with_options = await self._with_required_field_options(missing, sobject_name)
        missing_labels = ", ".join(label for _, label, _ in required_with_options)
        missing_option_map = {
            str(api): list(options) for api, _, options in required_with_options if options
        }
        return _provenance_failure_envelope(
            tool_name, sobject_name,
            [(api, label) for api, label, _ in required_with_options],
            (
                f"Cannot {tool_name}: the create was blocked because it did not "
                f"carry your values for required field(s) {missing_labels}. "
                "The system must not fabricate field values."
            ),
            "Provide the missing required fields with their values, then submit "
            "a corrected request. No record was created.",
            missing_required=True,
            missing_field_options=missing_option_map,
        )

    def _format_result(self, tool_name: str, result: Any) -> str:
        """Format tool result as a clean JSON string."""
        if isinstance(result, str):
            try:
                parsed = json.loads(result)
                if tool_name == "getObjectSchema":
                    return self._format_schema_table(tool_name, parsed)
                return json.dumps(parsed, indent=2, default=str)
            except (json.JSONDecodeError, TypeError):
                return result

        if isinstance(result, dict):
            cleaned = self._clean_salesforce_response(result)
            if tool_name == "getObjectSchema":
                return self._format_schema_table(tool_name, cleaned)
            return json.dumps(cleaned, indent=2, default=str)

        if isinstance(result, list):
            if tool_name == "getObjectSchema":
                return self._format_schema_table(tool_name, result)
            return json.dumps(result, indent=2, default=str)

        return str(result)

    @staticmethod
    def _format_schema_table(tool_name: str, data: Any) -> str:
        """Hard-code a GFM Markdown table for schema results so the LLM passes it through untouched.
        
        Extracts high-value, essential columns (Name, Label, Type, Required, Details)
        instead of dumping 50+ internal Salesforce metadata flags that bloat context.
        """
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return str(data)

        # Normalize: find the fields list regardless of nesting shape
        fields = []
        obj_name = ""
        if isinstance(data, dict):
            obj_name = data.get("name") or data.get("label") or data.get("object") or ""
            for key in ("fields", "fieldsList", "attributes", "properties"):
                if key in data and isinstance(data[key], list):
                    fields = data[key]
                    break
            if not fields:
                for key, value in data.items():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        fields = value
                        if not obj_name:
                            obj_name = key
                        break
        elif isinstance(data, list):
            fields = data

        if not fields:
            return json.dumps(data, indent=2, default=str)

        # Check if items look like Salesforce field definitions (have 'name' or 'label')
        first = fields[0] if isinstance(fields[0], dict) else {}
        is_field_def = any(k in first for k in ("name", "label", "type", "dataType", "soapType"))

        if is_field_def:
            headers = ["Field Name", "Label", "Type", "Required", "Details / Picklist"]
            header_str = "| " + " | ".join(headers) + " |"
            sep_str = "| " + " | ".join(["---"] * len(headers)) + " |"
            rows = []
            for f in fields:
                if not isinstance(f, dict):
                    continue
                name = str(f.get("name") or f.get("apiName") or "-")
                label = str(f.get("label") or "-")
                ftype = str(f.get("type") or f.get("dataType") or f.get("soapType") or "-")

                # Determine Required status
                is_nillable = f.get("nillable")
                is_createable = f.get("createable", True)
                is_required = f.get("required")
                if is_required is True or (
                    is_nillable is False
                    and is_createable is True
                    and name.lower() not in (
                        "id", "createddate", "createdbyid", "lastmodifieddate",
                        "lastmodifiedbyid", "systemmodstamp"
                    )
                ):
                    required_str = "Yes"
                else:
                    required_str = "No"

                # Picklist values or reference info
                details = []
                pv = f.get("picklistValues") or f.get("values")
                if isinstance(pv, list) and pv:
                    vals = []
                    for item in pv:
                        if isinstance(item, dict):
                            val = item.get("value") or item.get("label")
                            if val:
                                vals.append(str(val))
                        elif isinstance(item, str):
                            vals.append(item)
                    if vals:
                        if len(vals) > 6:
                            details.append(f"Values: {', '.join(vals[:6])}... ({len(vals)} total)")
                        else:
                            details.append(f"Values: {', '.join(vals)}")
                elif f.get("referenceTo"):
                    refs = f.get("referenceTo")
                    if isinstance(refs, list):
                        details.append(f"Ref: {', '.join(str(r) for r in refs)}")
                    else:
                        details.append(f"Ref: {refs}")
                elif f.get("length") and ftype.lower() in ("string", "textarea"):
                    details.append(f"Len: {f.get('length')}")

                detail_str = "; ".join(details) if details else "-"
                rows.append(f"| {name} | {label} | {ftype} | {required_str} | {detail_str} |")

            table = "\n".join([header_str, sep_str] + rows)
            title = f"### 📋 {obj_name} Fields\n\n" if obj_name else ""
            return f"[reference_table]\n{title}{table}"

        # Fallback for non-field schema (e.g. object describe lists)
        # Curate top 6 most relevant keys to avoid massive column explosion
        all_keys: list[str] = []
        seen_keys: set[str] = set()
        priority_keys = ["name", "label", "keyPrefix", "custom", "queryable", "createable", "updateable", "deletable"]
        for pk in priority_keys:
            if any(isinstance(item, dict) and pk in item for item in fields):
                all_keys.append(pk)
                seen_keys.add(pk)

        for item in fields:
            if isinstance(item, dict):
                for k in item:
                    if k not in seen_keys and len(all_keys) < 6:
                        all_keys.append(k)
                        seen_keys.add(k)

        if not all_keys:
            return json.dumps(data, indent=2, default=str)

        header = "| " + " | ".join(all_keys) + " |"
        separator = "| " + " | ".join(["---"] * len(all_keys)) + " |"
        rows = []
        for item in fields:
            if isinstance(item, dict):
                row_values = [str(item.get(k, "-")) if item.get(k) is not None else "-" for k in all_keys]
                rows.append("| " + " | ".join(row_values) + " |")

        table = "\n".join([header, separator] + rows)
        return f"[reference_table]\n{table}"

    @staticmethod
    def _format_datetime_value(val: Any) -> Any:
        if isinstance(val, str) and ("T" in val) and (val.endswith("+0000") or val.endswith("Z") or val.endswith("+00:00")):
            try:
                from datetime import datetime
                clean = val.replace("+0000", "+00:00").replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean)
                formatted_date = dt.strftime("%d %b %Y, %I:%M %p UTC")
                if formatted_date.startswith("0"):
                    formatted_date = formatted_date[1:]
                return formatted_date
            except Exception:
                return val
        return val

    @staticmethod
    def _clean_salesforce_response(data: dict) -> dict:
        """
        Clean up Salesforce API response by removing internal metadata
        fields and providing readable timezone conversions for timestamps.
        Injects explicit 'total_count' for aggregate/count queries.
        """
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if key == "attributes":
                    continue  # Skip Salesforce internal metadata
                elif isinstance(value, dict):
                    cleaned[key] = ToolExecutor._clean_salesforce_response(value)
                elif isinstance(value, list):
                    cleaned[key] = [
                        ToolExecutor._clean_salesforce_response(item)
                        if isinstance(item, dict) else ToolExecutor._format_datetime_value(item)
                        for item in value
                    ]
                else:
                    cleaned[key] = ToolExecutor._format_datetime_value(value)

            # Clean aggregate count results so LLM sees explicit total_count
            if "totalSize" in cleaned:
                records = cleaned.get("records", [])
                if len(records) == 1 and isinstance(records[0], dict) and "expr0" in records[0]:
                    cleaned["total_count"] = records[0]["expr0"]
                elif "totalSize" in cleaned:
                    cleaned["total_count"] = cleaned["totalSize"]

            return cleaned
        return data

    @staticmethod
    def _get_error_suggestion(tool_name: str, error: str) -> str:
        """Provide helpful suggestions based on common errors."""
        error_lower = error.lower()

        if "$" in error or "currency" in error_lower or ("unexpected token" in error_lower and "50" in error_lower):
            return "Do not include dollar signs ($) or commas (,) in SOQL numeric literals. Use raw numbers: Amount > 50000 instead of Amount > $50,000."
        elif "relationship" in error_lower and ("didn't understand" in error_lower or "subquery" in error_lower):
            return "When writing parent-to-child subqueries on Account, use PLURAL relationship names (e.g., (SELECT Id, Name FROM Opportunities), (SELECT Id, Name FROM Contacts))."
        elif "group by" in error_lower and ("subquery" in error_lower or "semi" in error_lower or "not supported" in error_lower or "malformed" in error_lower):
            return (
                "SOQL does not allow GROUP BY inside a subquery (WHERE Id IN (...)). "
                "Query the child object directly and group by parent (e.g., SELECT Account.Id, Account.Name, COUNT(Id) FROM Contact WHERE AccountId != null GROUP BY Account.Id, Account.Name HAVING COUNT(Id) > N)."
            )
        elif "company" in error_lower and "contact" in error_lower:
            return "Contact does not have a 'Company' field. Use 'Account.Name' to filter or query the Contact's company."
        elif "invalid_field" in error_lower or "no such column" in error_lower:
            return (
                "A field name may be incorrect. Use 'getObjectSchema' to check "
                "the correct field API names for this object."
            )
        elif "malformed query" in error_lower:
            return (
                "The SOQL/SOSL query has a syntax error. Check for proper "
                "SELECT, FROM, WHERE, and LIMIT clauses."
            )
        elif "insufficient_access" in error_lower or "permission" in error_lower:
            return (
                "You don't have permission to perform this operation. "
                "Contact your Salesforce admin."
            )
        elif "not_found" in error_lower or "404" in error_lower:
            return "The record ID may be incorrect or the record has been deleted."
        elif "duplicate" in error_lower:
            return "A record with this data already exists. Check for duplicates."
        elif "required" in error_lower:
            return (
                "Required fields are missing. Use 'getObjectSchema' to see "
                "which fields are required for this object."
            )
        elif "401" in error_lower or "unauthorized" in error_lower:
            return "Your session may have expired. Try reconnecting."
        else:
            return "Check the error message above for details."


def _truncate_args(args: dict[str, Any], max_length: int = 200) -> str:
    """Truncate arguments for logging."""
    s = str(args)
    return s[:max_length] + "..." if len(s) > max_length else s
