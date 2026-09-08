"""
Tool Executor — executes tool calls against the Salesforce MCP Server.
Provides validation, error handling, retry logic, and result formatting.
"""

import hashlib
import json
import logging
import os
import time
from typing import Any

from .client import SalesforceMCPClient
from .registry import ToolRegistry
from tools.salesforce import is_mutating, is_destructive

logger = logging.getLogger(__name__)

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

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute a tool call and return the result as a formatted string.

        Args:
            tool_name: The name of the tool to execute.
            arguments: The arguments to pass to the tool.

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
            validation_error = await self.validate_mutation(tool_name, arguments)
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

    async def validate_mutation(self, tool_name: str, arguments: dict[str, Any]) -> str | None:
        """
        Run the mandatory fail-closed mutation validation. Returns an error JSON
        string when the mutation must be blocked, or None when it may proceed.

        For createSobjectRecord, required fields are resolved from live Describe
        metadata (read-only schema path) for BOTH standard and custom objects so
        per-org custom required fields are honored and nothing is blindly
        hard-coded. The static registry is only a zero-I/O fallback when Describe
        is unavailable; unknown objects still fail closed.
        """
        from agent.mutation_validation import validate_mutation_fields

        sobject_name = ""
        for key in ("sobject-name", "sobject_name", "sobject", "object", "sobjectName", "objectName"):
            if key in arguments and arguments[key]:
                sobject_name = str(arguments[key]).strip()
                break

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
            return validate_mutation_fields(tool_name, arguments, lambda _s: resolved)
        return validate_mutation_fields(tool_name, arguments, None)

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
        """Hard-code a GFM Markdown table for schema results so the LLM passes it through untouched."""
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                return str(data)

        # Normalize: find the fields list regardless of nesting shape
        fields = []
        if isinstance(data, dict):
            for key in ("fields", "fieldsList", "attributes"):
                if key in data and isinstance(data[key], list):
                    fields = data[key]
                    break
            if not fields:
                for key, value in data.items():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        fields = value
                        break
        elif isinstance(data, list):
            fields = data

        if not fields:
            return json.dumps(data, indent=2, default=str)

        # Collect all unique keys across every field record for the header row
        all_keys: list[str] = []
        seen_keys: set[str] = set()
        for field in fields:
            if isinstance(field, dict):
                for k in field:
                    if k not in seen_keys:
                        all_keys.append(k)
                        seen_keys.add(k)

        if not all_keys:
            return json.dumps(data, indent=2, default=str)

        # Build the GFM table string
        header = "| " + " | ".join(all_keys) + " |"
        separator = "| " + " | ".join(["---"] * len(all_keys)) + " |"
        rows = []
        for field in fields:
            if isinstance(field, dict):
                row_values = [str(field.get(k, "-")) if field.get(k) is not None else "-" for k in all_keys]
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
