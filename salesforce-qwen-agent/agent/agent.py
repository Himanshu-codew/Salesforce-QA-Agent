"""
Core Salesforce Agent — the main agent loop that ties together
the LLM (Qwen3), MCP executor, memory, and planner.
"""

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
        max_iterations: int = 10,
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
        while iteration < self.max_iterations:
            iteration += 1

            try:
                messages = memory.get_messages_for_llm(SYSTEM_PROMPT)
                llm_result = await self.llm.chat_with_tools(
                    messages=messages,
                    tools=tools,
                    temperature=0.1,  # Low temp for fast, deterministic tool selection
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

                # Execute all safe calls first
                if safe_calls:
                    memory.add_assistant_tool_calls(safe_calls)
                    for tc in safe_calls:
                        logger.info(f"🚀 [EXECUTING TOOL]: {tc['name']} with args: {tc['arguments']}")
                        yield {
                            "type": "tool_call",
                            "data": {"name": tc["name"], "arguments": tc["arguments"]},
                        }

                        try:
                            result = await self.executor.execute(
                                tc["name"], tc["arguments"]
                            )
                            logger.info(f"✅ [TOOL FINISHED]: {tc['name']} (Result len: {len(result)} chars)")
                        except Exception as e:
                            result = json.dumps({
                                "error": str(e),
                                "tool": tc["name"],
                            })
                            logger.error(f"❌ Tool execution error ({tc['name']}): {e}")

                        # ── SOQL Error Auto-Correction (max 2 retries) ──
                        if tc["name"] == "soqlQuery":
                            retry_count = 0
                            max_soql_retries = 2
                            while retry_count < max_soql_retries:
                                result_lower = result.lower()
                                is_soql_error = any(kw in result_lower for kw in [
                                    "malformed", "syntax error", "invalid_field",
                                    "unexpected token", "no such column", "invalid",
                                    "didn't understand", "error", "parse_error",
                                ])
                                if not is_soql_error:
                                    break

                                fix_suggestion = get_soql_fix_suggestion(result)
                                if not fix_suggestion:
                                    break

                                retry_count += 1
                                logger.info(
                                    f"🔧 [SOQL AUTO-CORRECT] Attempt {retry_count}/{max_soql_retries}: "
                                    f"{fix_suggestion}"
                                )
                                yield {
                                    "type": "thinking",
                                    "data": f"SOQL query needs correction. Retrying... (attempt {retry_count}/{max_soql_retries})",
                                }

                                # Ask the LLM to fix the SOQL query
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
                                        logger.info(
                                            f"✅ [SOQL RETRY {retry_count} RESULT]: {tc['name']} "
                                            f"(Result len: {len(result)} chars)"
                                        )
                                        # Update the tool call in memory with the corrected one
                                        memory.add_tool_result(tc["id"], tc["name"], result)
                                    else:
                                        # LLM didn't produce a tool call, break out
                                        break
                                except Exception as retry_err:
                                    logger.error(f"SOQL retry error: {retry_err}")
                                    break

                        # Truncate very large results to avoid context overflow
                        if len(result) > 15000:
                            result = result[:15000] + "\n... [truncated, showing first 15000 chars]"

                        memory.add_tool_result(tc["id"], tc["name"], result)

                        yield {
                            "type": "tool_result",
                            "data": {"name": tc["name"], "result": result},
                        }

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
        max_iter_msg = ERROR_MESSAGES["max_iterations"]
        try:
            # Ask LLM to summarize what it's found so far
            messages = memory.get_messages_for_llm(SYSTEM_PROMPT)
            messages.append({
                "role": "user",
                "content": "Please summarize the results you've gathered so far in clean Markdown format (tables and bullet points). Do NOT output raw JSON.",
            })
            summary = await self.llm.chat(messages)
            sanitized = sanitize_response_output(summary.strip()) if summary else ""
            final_msg = f"{max_iter_msg}\n\n{sanitized}" if sanitized else max_iter_msg
        except Exception:
            final_msg = max_iter_msg

        memory.add_assistant_message(final_msg)
        yield {"type": "response", "data": final_msg}

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
