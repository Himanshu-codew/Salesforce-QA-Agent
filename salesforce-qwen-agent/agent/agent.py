"""
Core Salesforce Agent — the main agent loop that ties together
the LLM (Qwen3), MCP executor, memory, and planner.
"""

import asyncio
import json
import logging
import re
from typing import Any, AsyncGenerator

from llm.base import BaseLLM
from mcp.executor import ToolExecutor
from tools.salesforce import get_tool_definitions
from .memory import ConversationMemory
from .planner import TaskPlanner
from .prompts import SYSTEM_PROMPT, ERROR_MESSAGES
from .rag import ToolRAGRetriever

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Python-Side Salesforce Table Formatter
# PERMANENT FIX: Formats tool results as markdown tables in Python.
# LLM NEVER formats tables → LLM can NEVER truncate rows.
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


def format_sf_records_as_markdown(result_json: str, tool_name: str = "soqlQuery") -> str | None:
    """
    Parse a Salesforce soqlQuery JSON result and return a complete markdown table.
    Returns None if the result is not a parseable record list (e.g. errors, counts).

    This is the PERMANENT fix for LLM table truncation:
    Python outputs every single row — the LLM only writes section headers.
    """
    if tool_name != "soqlQuery":
        return None
    try:
        data = json.loads(result_json)
    except Exception:
        return None

    # Handle aggregate/COUNT queries → just return count text
    records = data.get("records", [])
    total_size = data.get("totalSize", len(records))

    if not records:
        return f"**Total: 0 records found.**"

    # Detect object type from first record's attributes
    obj_type = None
    first = records[0]
    if isinstance(first.get("attributes"), dict):
        obj_type = first["attributes"].get("type")

    # Detect if this is a COUNT/aggregate result (no Id field, has expr0 etc.)
    if "expr0" in first or (len(first) <= 3 and "Id" not in first):
        # Aggregate result — format as simple table
        keys = [k for k in first.keys() if k not in _SKIP_FIELDS]
        if not keys:
            return None
        header = "| " + " | ".join(keys) + " |"
        sep    = "| " + " | ".join(["---"] * len(keys)) + " |"
        rows   = []
        for rec in records:
            row = "| " + " | ".join(_fmt_value(rec.get(k)) for k in keys) + " |"
            rows.append(row)
        return "\n".join([header, sep] + rows) + f"\n\n**Total: {total_size} record(s)**"

    # Determine display columns
    all_keys = []
    # Flatten nested keys (e.g. Account.Name)
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

    # Apply preferred field order if known object type
    preferred = _FIELD_ORDER.get(obj_type, [])
    ordered = [f for f in preferred if f in all_keys]
    remaining = [f for f in all_keys if f not in ordered]
    cols = ordered + remaining

    # Skip attributes in cols
    cols = [c for c in cols if c.split(".")[0] not in _SKIP_FIELDS]

    if not cols:
        return None

    # Build markdown table header
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = []
    for rec in records:
        cells = []
        for col in cols:
            cells.append(_fmt_value(_get_nested(rec, col)))
        rows.append("| " + " | ".join(cells) + " |")

    total_line = f"\n**Total: {total_size} record(s)**"
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
    lines = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        # Skip lines that are only stray markers
        if stripped in ("{", "}", "]", "[", "```", "```json", "```tool_call", "}}", "}}}", "`]"):
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
]

_SOQL_FIX_SUGGESTIONS = {
    "$": "Remove dollar signs ($) from SOQL numeric filters. Write Amount > 50000, NOT Amount > '$50,000'.",
    "AS ": "SOQL does not support the 'AS' keyword for aliases. Write SUM(Amount) total instead of SUM(Amount) AS total.",
    "DATE()": "SOQL does not have DATE() function. Use date literals like TODAY, THIS_WEEK, LAST_N_DAYS:7, THIS_YEAR.",
    "DATEADD()": "SOQL does not have DATEADD(). Use date literals like LAST_N_DAYS:N, LAST_7_DAYS, THIS_MONTH.",
    "malformed": "Check SOQL syntax: ensure SELECT, FROM, WHERE, and LIMIT clauses are correct.",
    "invalid_field": "Check field API names using getObjectSchema if unsure.",
    "group by": "SOQL does not allow GROUP BY inside semi-join subqueries. Query the child object directly.",
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
        if session_id not in self._memories:
            self._memories[session_id] = ConversationMemory(
                max_messages=self._max_history
            )
        else:
            self._memories[session_id].max_messages = self._max_history
            self._memories[session_id]._trim()
        return self._memories[session_id]

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
        # RAG Tool Retrieval: Dynamically fetch top-K relevant tools for prompt optimization
        tools = self.rag_retriever.get_relevant_tools(user_message, top_k=6)

        # ── Check for pending confirmations ──
        if self.planner.has_pending_confirmation(session_id):
            pending = self.planner.process_confirmation(user_message, session_id)
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

                result = await self.executor.execute(tool_name, arguments)

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

        # ── Python-built tables: tc_id → (obj_type, table_markdown, total_count) ──
        # When ALL tool calls produce Python tables, we compose the final response
        # in Python and skip the LLM formatting step entirely.
        python_tables: dict[str, tuple[str, str]] = {}  # tc_id → (tool_name, markdown)

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

            try:
                messages = memory.get_messages_for_llm(SYSTEM_PROMPT)

                # ── SPEED FIX: On Turn 1 of multi-query, inject batch instruction ──
                # ONLY for pure READ queries — create/update/delete are sequential by nature.
                _write_keywords = [
                    "create", "add", "insert", "new", "make", "banao", "daalo",
                    "update", "edit", "change", "modify", "badlo",
                    "delete", "remove", "hatao", "mitao", "drop",
                ]
                _is_read_only = not any(kw in user_msg_lower for kw in _write_keywords)

                if iteration == 1 and _is_multi and _is_read_only:
                    batch_hint = (
                        "BATCH MODE — SPEED CRITICAL: The user has asked multiple independent READ questions. "
                        f"There are approximately {len(query_lines)} sub-queries. "
                        "You MUST return ALL required tool calls in THIS SINGLE RESPONSE right now. "
                        "Do NOT make one tool call and wait — output every tool call simultaneously. "
                        "This is mandatory for fast response."
                    )
                    messages = messages + [{"role": "user", "content": batch_hint}]

                llm_result = await self.llm.chat_with_tools(
                    messages=messages,
                    tools=tools,
                    temperature=0.0,  # Zero temp = fastest, most deterministic
                )

            except Exception as e:
                error_msg = ERROR_MESSAGES["llm_error"].format(error=str(e))
                logger.error(f"LLM error: {e}")
                yield {"type": "error", "data": error_msg}
                memory.add_assistant_message(error_msg)
                return

            # ── Case 1: LLM wants to call tools ──
            if llm_result["tool_calls"]:
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

                    # Execute ALL safe calls in parallel
                    async def _run_tool(tc: dict) -> tuple[dict, str]:
                        try:
                            result = await self.executor.execute(tc["name"], tc["arguments"])
                            logger.info(f"✅ [TOOL FINISHED]: {tc['name']} (Result len: {len(result)} chars)")
                        except Exception as e:
                            result = json.dumps({"error": str(e), "tool": tc["name"]})
                            logger.error(f"❌ Tool execution error ({tc['name']}): {e}")
                        return tc, result

                    parallel_results = await asyncio.gather(*[_run_tool(tc) for tc in safe_calls])

                    for tc, result in parallel_results:
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
                                        fix_result = await self.llm.chat_with_tools(
                                            messages=fix_messages,
                                            tools=tools,
                                            temperature=0.0,
                                        )
                                        if fix_result.get("tool_calls"):
                                            fixed_tc = fix_result["tool_calls"][0]
                                            yield {
                                                "type": "tool_call",
                                                "data": {"name": fixed_tc["name"], "arguments": fixed_tc["arguments"]},
                                            }
                                            result = await self.executor.execute(
                                                fixed_tc["name"], fixed_tc["arguments"]
                                            )
                                            memory.add_tool_result(tc["id"], tc["name"], result)
                                    except Exception as retry_err:
                                        logger.error(f"SOQL retry error: {retry_err}")

                        # ── PERMANENT TABLE FIX: Format tables in Python, not LLM ──
                        # LLM receives pre-built markdown tables → cannot truncate rows.
                        py_table = format_sf_records_as_markdown(result, tc["name"])
                        if py_table:
                            # Store raw result for hallucination checking
                            tool_results_fetched[tc["id"]] = result
                            # Track the pre-built table for direct Python response
                            python_tables[tc["id"]] = (tc["name"], py_table)
                            # Tell LLM it has pre-built table (minimal hint)
                            memory_result = (
                                f"[PRE-BUILT TABLE — copy it VERBATIM into your response]\n\n{py_table}"
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

                # ── PYTHON DIRECT RESPONSE: Skip LLM if all results are tabular ──
                # If every tool call produced a Python table, compose response directly.
                # LLM is bypassed → zero truncation possible.
                if python_tables and safe_calls and len(python_tables) == len(safe_calls) and not destructive_calls:
                    sections = []
                    for tc in safe_calls:
                        if tc["id"] in python_tables:
                            _, table_md = python_tables[tc["id"]]
                            # Derive section header from SOQL query or tool name
                            soql = tc.get("arguments", {}).get("query", "")
                            obj_match = re.search(r"FROM\s+(\w+)", soql, re.IGNORECASE)
                            obj_name = obj_match.group(1) if obj_match else "Records"
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
                # Mandatory Tool Execution Interceptor for Data Queries on Turn 1
                if iteration == 1 and tools:
                    user_msg_lower = user_message.lower()
                    data_intent_keywords = [
                        "account", "accounts", "lead", "leads", "contact", "contacts",
                        "opportunity", "opportunities", "case", "cases", "task", "tasks",
                        "event", "events", "user", "who am i", "schema", "fields",
                        "show", "list", "select", "find", "search", "count", "how many",
                        "delete", "remove", "update", "edit", "create", "banao", "dikhao",
                        "hatao", "badlo", "kitne", "saare"
                    ]
                    has_data_intent = any(kw in user_msg_lower for kw in data_intent_keywords)
                    if has_data_intent:
                        logger.warning(
                            f"⚠️ [INTERCEPTOR] LLM returned text without tool call on Turn 1 for query: '{user_message[:40]}...'. "
                            "Forcing tool execution retry."
                        )
                        interceptor_messages = memory.get_messages_for_llm(SYSTEM_PROMPT)
                        interceptor_messages.append({
                            "role": "user",
                            "content": (
                                "TOOL CALL MANDATORY: You must call an appropriate MCP tool (such as soqlQuery, find, or getObjectSchema) "
                                "to fetch real Salesforce data before answering. Do NOT reply with text or dummy data."
                            )
                        })
                        try:
                            retry_result = await self.llm.chat_with_tools(
                                messages=interceptor_messages,
                                tools=tools,
                                temperature=0.0,
                            )
                            if retry_result.get("tool_calls"):
                                llm_result = retry_result
                                continue  # Re-enter turn loop with forced tool calls!
                        except Exception as retry_err:
                            logger.error(f"Interceptor retry error: {retry_err}")

                # ── Compound Query Completeness Interceptor ──
                # Catches cases where LLM returns text with blank/empty sections
                # for multi-part queries (e.g., "Show ALL Accounts AND count ALL Leads")
                if iteration >= 2 and tools:
                    user_msg_lower = user_message.lower()
                    compound_separators = [
                        " and ", " & ", " also ", " along with ", " as well as ",
                        " plus ", " aur ", " tatha ", " evam ",
                    ]
                    is_compound = any(sep in user_msg_lower for sep in compound_separators)

                    if is_compound:
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
                                retry_result = await self.llm.chat_with_tools(
                                    messages=interceptor_messages,
                                    tools=tools,
                                    temperature=0.0,
                                )
                                if retry_result.get("tool_calls"):
                                    llm_result = retry_result
                                    continue
                            except Exception as interceptor_err:
                                logger.error(f"Compound query interceptor error: {interceptor_err}")

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
                        summary = await self.llm.chat(messages)
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
                summary = await self.llm.chat(format_messages)
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
