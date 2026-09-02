"""
Multi-Agent Orchestrator
Coordinates Planner, DataAgent, ActionAgent, and Synthesizer for robust execution.

Hardened for production reliability:
- Every external stage (planner / RAG / Qwen / executor / synthesizer) is bounded
  by an explicit timeout and traced with `[TRACE] name: X ms` logs.
- The whole `process_message` stream is bounded by OVERALL_PROCESS_TIMEOUT so a
  request can never hang indefinitely.
- Multi-step / multiple sequential tool calls are supported but bounded by
  MAX_TOOL_ITERATIONS to prevent infinite loops.
- Tool-call arguments are validated and normalized (raw-string arguments are
  JSON-parsed; calls without a valid name are rejected).
- MCP / Salesforce / LLM / synthesis failures surface as structured terminal
  errors with a stable `code` + `message` instead of raw JSON or a fabricated
  string.
- No query-specific routing rules: the planner decides general-vs-Salesforce and
  semantic RAG selects tools, so arbitrary new queries need no code changes.
"""
import asyncio
import json
import logging
import os
import re
import time
from typing import Any, AsyncGenerator

from llm.base import BaseLLM
from sfmcp.executor import ToolExecutor
from .multi_agent_prompts import (
    PLANNER_PROMPT,
    DATA_AGENT_PROMPT,
    ACTION_AGENT_PROMPT,
    SYNTHESIZER_PROMPT,
    GENERAL_ANSWER_PROMPT,
)
from .memory import ConversationMemory
from .planner import TaskPlanner
from .rag import ToolRAGRetriever
from .agent import (
    filter_tools_for_query,
    format_sf_records_as_markdown,
    _is_soql_count,
    _has_salesforce_intent,
)
from tools.salesforce import get_tool_definitions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounded-latency / resilience configuration (env-overridable, safe defaults)
#
# These are per-stage bounds for a streaming qwen2.5-coder:32b endpoint. They
# are deliberately generous enough to survive cold starts / network spikes for
# the FIRST request (embedding model download, MCP handshake) but still reject
# genuinely hung stages. All values are overridable via environment variables.
# ---------------------------------------------------------------------------
# Planner / intent LLM stage (single chat completion + JSON parse)
LLM_STAGE_TIMEOUT = float(os.getenv("LLM_STAGE_TIMEOUT", "90.0"))
# Salesforce tool-call execution through MCP (initialize + call_tool)
EXECUTOR_TIMEOUT = float(os.getenv("EXECUTOR_TIMEOUT", "90.0"))
# Final-answer (synthesizer) generation is the LAST LLM call after a successful
# tool result. It must have a generous timeout so a successful Salesforce/MCP
# result is not wasted because the final Qwen generation was slow.
SYNTHESIZER_TIMEOUT = float(os.getenv("SYNTHESIZER_TIMEOUT", "120.0"))
# Semantic RAG tool retrieval (embedding load + vector query). Raised so a
# cold-start model download does not abort the first request. The embedding
# model is also warmed at startup so warmed requests complete in well under 1s.
RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "120.0"))
# Final-answer synthesis bounds. Truncating the raw tool-result context and
# capping generated tokens prevent a single oversized schema/record dump from
# making the final Qwen generation exceed SYNTHESIZER_TIMEOUT. Normal results
# are well under the cap and are passed through untouched.
SYNTHESIZER_MAX_TOKENS = int(os.getenv("SYNTHESIZER_MAX_TOKENS", "1500"))
SYNTHESIZER_CONTEXT_CHARS = int(os.getenv("SYNTHESIZER_CONTEXT_CHARS", "24000"))
# Hard ceiling for the whole request. Bounded but permissive enough for a
# multi-stage pipeline against a 32B-coder model.
OVERALL_PROCESS_TIMEOUT = float(os.getenv("OVERALL_PROCESS_TIMEOUT", "300.0"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "6"))

# Which stage is currently being awaited (for detailed timeout diagnostics).
_ACTIVE_STAGE = {"name": None}


def _set_stage(name: str | None) -> None:
    _ACTIVE_STAGE["name"] = name


# ---------------------------------------------------------------------------
# Stable error codes (surfaced to the client as {code, message})
# ---------------------------------------------------------------------------
ERR_TIMEOUT = "TIMEOUT"
ERR_PLANNER = "PLANNER_FAILED"
ERR_LLM = "LLM_FAILED"
ERR_GENERAL = "GENERAL_ANSWER_FAILED"
ERR_RAG = "RAG_FAILED"
ERR_TOOL = "TOOL_FAILED"
ERR_MCP_SALESFORCE = "SALESFORCE_FAILED"
ERR_SYNTHESIS = "SYNTHESIS_FAILED"
ERR_INVALID_TOOL = "INVALID_TOOL_CALL"
ERR_TOO_MANY_STEPS = "TOO_MANY_STEPS"
ERR_INTERNAL = "INTERNAL_ERROR"


class AgentError(Exception):
    """A controlled, user-facing failure with a stable error code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _error_event(code: str, message: str) -> dict[str, Any]:
    return {"type": "error", "data": message, "code": code, "message": message}


def _trace(name: str, start: float) -> None:
    logger.info(f"[TRACE] {name}: {(time.monotonic() - start) * 1000:.0f} ms")


def _normalize_tool_call(tc: dict[str, Any]) -> tuple[str | None, dict[str, Any], str | None]:
    """
    Validate and normalize a tool call returned by the LLM.

    Returns (name, arguments, error). On error, name/arguments are None and the
    error string explains the problem. Raw-string arguments are JSON-parsed.
    """
    name = tc.get("name")
    if not name or not isinstance(name, str) or not name.strip():
        return None, {}, "Model returned a tool call without a valid name."
    name = name.strip()

    args = tc.get("arguments", {})
    if isinstance(args, str):
        stripped = args.strip()
        if not stripped:
            args = {}
        else:
            try:
                args = json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                return None, {}, (
                    f"Tool call '{name}' has malformed arguments that could not be parsed."
                )
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None, {}, f"Tool call '{name}' arguments must be a JSON object."

    return name, args, None


def _executor_error_message(result: str, tool_name: str) -> str | None:
    """
    Detect the executor's failure envelope ({"error": ..., "tool": ...}) and
    return the human-readable message, or None when the result is a normal
    tool result. This keeps MCP / Salesforce failures from being silently
    swallowed as ordinary tool results.
    """
    if not isinstance(result, str):
        return None
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    error = parsed.get("error")
    if isinstance(error, str) and error.strip() and "tool" in parsed:
        suggestion = parsed.get("suggestion")
        if isinstance(suggestion, str) and suggestion.strip():
            return f"{error} Suggestion: {suggestion}"
        return error
    return None


_COUNT_INTENT_PATTERN = re.compile(
    r"\b(how many|count|total count|total number|number of|total records)\b",
    re.IGNORECASE,
)


def _has_count_intent(user_text: str) -> bool:
    """Return True when the USER's request asks for a record count/total.

    This is a general intent classifier (not a special-case on any concrete
    phrasing). It uses word-boundary matching so "count" does NOT match the
    substring inside "accounts"/"counting". It is used to avoid an UNREQUESTED
    aggregate COUNT tool call when the user only asked to list records: the
    Salesforce tool schema advertises COUNT support, so a worker LLM can
    autonomously add ``SELECT COUNT(...)`` on top of a plain list query, which
    then surfaces as a duplicate ``Total`` block in the synthesized answer.
    Genuine count requests and explicit list+count requests continue to run
    their COUNT unchanged.
    """
    if not isinstance(user_text, str):
        return False
    return _COUNT_INTENT_PATTERN.search(user_text) is not None


def _has_non_count_query(tool_calls: list[dict[str, Any]]) -> bool:
    """Return True if the task's tool-call list includes a non-Count soqlQuery.

    Used to determine whether an aggregate COUNT call is truly redundant with a
    list/select query the same task already requested. Arguments are read
    tolerantly (dict or raw-string) so malformed calls never raise here.
    """
    for tc in tool_calls or []:
        try:
            name, args, _err = _normalize_tool_call(tc)
        except Exception:
            continue
        if name != "soqlQuery":
            continue
        soql = args.get("q", "") if isinstance(args, dict) else ""
        if soql and not _is_soql_count(str(soql)):
            return True
    return False


def _normalize_zero_count_result(tool_name: str, result: str, arguments: dict[str, Any]) -> str:
    """
    Fix B (Orchestrator): Preserve an explicitly-returned COUNT of zero so the
    synthesizer can state "**Total Count:** 0" instead of treating a genuinely
    empty COUNT result as missing/unknown data.

    Only normalizes when ALL of these hold (so genuine zeros are surfaced while
    every other case is left byte-for-byte identical):
      1. The tool is a soqlQuery.
      2. The SOQL in `arguments` is a real aggregate COUNT query.
      3. Salesforce explicitly returned totalSize == 0 (records empty).

    Any non-COUNT empty result, non-zero COUNT result, or COUNT with records is
    returned unchanged. Formatting reuses the shared helper from agent.py so no
    count-line logic is duplicated here. Does NOT shortcut synthesis; the
    normal planner → worker → executor → synthesizer flow is preserved.
    """
    if tool_name != "soqlQuery" or not isinstance(result, str):
        return result
    soql = (arguments.get("q") or arguments.get("query") or "") if isinstance(arguments, dict) else ""
    if not _is_soql_count(soql):
        return result
    try:
        parsed = json.loads(result)
    except (json.JSONDecodeError, TypeError):
        return result
    if not isinstance(parsed, dict):
        return result
    if parsed.get("totalSize") != 0:
        return result
    if parsed.get("records", []):
        return result
    normalized = format_sf_records_as_markdown(result, soql_query=soql)
    if not normalized:
        return result
    logger.info(
        "[FIX-B] COUNT query returned totalSize=0; normalized zero-result for synthesis "
        f"-> {normalized!r}"
    )
    return normalized


def _split_reference_results(
    tool_results: list[dict],
) -> tuple[list[str], list[dict]]:
    """Deterministically split tool results into:

    1. ``ref_tables`` — flat record lists rendered VERBATIM in Python by
       ``format_sf_records_as_markdown`` (Id/Name stay separate cells, every row
       matches the header, and Salesforce values / order / count are preserved
       byte-for-byte instead of being regenerated by the synthesizer LLM).
    2. ``raw_remainder`` — everything the backend cannot render as a flat table
       (hierarchical/subquery JSON, object schemas, error lines, or zero-count
       results that Fix B already turned into a ``**Total Count:**`` line) which
       still needs LLM synthesis.

    ``data_fidelity`` is satisfied deterministically here, never by the LLM.
    """
    ref_tables: list[str] = []
    raw_remainder: list[dict] = []
    for item in tool_results or []:
        if not isinstance(item, dict):
            continue
        tool = item.get("tool", "soqlQuery")
        result = item.get("result", "")
        if not isinstance(result, str):
            continue
        table = format_sf_records_as_markdown(result, tool_name=tool)
        if table:
            ref_tables.append(table)
        else:
            raw_remainder.append(item)
    return ref_tables, raw_remainder


class Orchestrator:
    """
    Orchestrates the multi-agent workflow:
    1. Planner breaks down the query.
    2. Workers (Data/Action) execute sub-tasks.
    3. Synthesizer formats final response.
    """

    def __init__(
        self,
        llm: BaseLLM,
        executor: ToolExecutor,
        max_iterations: int = 20,
        max_history: int = 4,
    ):
        self.llm = llm
        self.executor = executor
        self.safety_planner = TaskPlanner()
        self._memories: dict[str, ConversationMemory] = {}
        self._max_history = max_history
        # The configurable cap for the tool-iteration loop. Kept in sync with the
        # module default so tests can override predictably.
        self._max_tool_iterations = int(os.getenv("MAX_TOOL_ITERATIONS", str(MAX_TOOL_ITERATIONS)))
        self.rag_retriever = ToolRAGRetriever(default_top_k=6)

    def _get_memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self._memories:
            self._memories[session_id] = ConversationMemory(max_messages=self._max_history)
        return self._memories[session_id]

    # ------------------------------------------------------------------
    # Public entry: a bounded async generator over the whole pipeline.
    # ------------------------------------------------------------------
    async def process_message(
        self,
        user_message: str,
        session_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        Run the full pipeline with an overall latency bound. No request can hang
        indefinitely, even if a downstream stage misbehaves.
        """
        queue: asyncio.Queue = asyncio.Queue()

        async def _drive() -> None:
            try:
                async for event in self._stream_impl(user_message, session_id):
                    await queue.put(event)
            except AgentError as e:
                logger.error(f"[ERROR] {e.code}: {e.message}")
                await queue.put(_error_event(e.code, e.message))
            except asyncio.TimeoutError:
                await queue.put(
                    _error_event(ERR_TIMEOUT,
                                 "This request took too long and was safely stopped. Please try again.")
                )
            except Exception as e:  # noqa: BLE001 - last-resort safety net
                logger.exception(f"[ERROR] unexpected failure in agent pipeline: {e}")
                await queue.put(
                    _error_event(ERR_INTERNAL,
                                 "I ran into an unexpected issue while processing your request. Please try again.")
                )
            finally:
                await queue.put(None)

        driver = asyncio.create_task(_drive())
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=OVERALL_PROCESS_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.error(
                        f"[TIMEOUT] process_message exceeded {OVERALL_PROCESS_TIMEOUT:.0f}s "
                        f"overall bound; cancelling work for session '{session_id}'."
                    )
                    yield _error_event(
                        ERR_TIMEOUT,
                        "This request took too long to complete and was safely stopped. "
                        "Please try a simpler or more specific query.",
                    )
                    break
                if event is None:
                    break
                yield event
        finally:
            if not driver.done():
                driver.cancel()
            try:
                await driver
            except (asyncio.CancelledError, Exception):
                pass

    # ------------------------------------------------------------------
    # Inner implementation (collects per-stage traces and structured errors)
    # ------------------------------------------------------------------
    async def _stream_impl(
        self,
        user_message: str,
        session_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        memory = self._get_memory(session_id)
        memory.max_messages = self._max_history
        start_total = time.monotonic()

        # ── Per-request observability (latency / tokens / transport) ──
        _zero_usage = lambda: {  # noqa: E731
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0,
        }
        usage_start = getattr(self.llm, "usage_snapshot", _zero_usage)()
        stage_times: dict[str, float] = {}
        request_tool_calls: list[dict[str, Any]] = []

        def _record_stage(name: str, start: float) -> None:
            _trace(name, start)
            stage_times[name] = stage_times.get(name, 0.0) + (time.monotonic() - start) * 1000.0

        def _metrics_event() -> dict[str, Any]:
            usage_now = getattr(self.llm, "usage_snapshot", _zero_usage)()
            transports = sorted({tc.get("transport", "unknown") for tc in request_tool_calls})
            return {
                "type": "metrics",
                "data": {
                    "model": getattr(self.llm, "model", ""),
                    "latency_ms": {k: round(v, 1) for k, v in stage_times.items()},
                    "total_ms": round((time.monotonic() - start_total) * 1000.0, 1),
                    "tokens": {
                        "prompt": max(0, usage_now["prompt_tokens"] - usage_start["prompt_tokens"]),
                        "completion": max(0, usage_now["completion_tokens"] - usage_start["completion_tokens"]),
                        "total": max(0, usage_now["total_tokens"] - usage_start["total_tokens"]),
                    },
                    "tool_calls": request_tool_calls,
                    "transport": transports[-1] if transports else "",
                },
            }

        def _record_tool_call(tc_name: str, tc_args: dict[str, Any]) -> None:
            transport = ""
            client = getattr(self.executor, "mcp_client", None)
            if client is not None:
                transport = getattr(client, "mcp_transport", "") or "REST"
            request_tool_calls.append({
                "name": tc_name,
                "arguments": tc_args,
                "transport": transport,
            })

        # Confirmation handling (destructive actions)
        if self.safety_planner.has_pending_confirmation(session_id):
            pending = self.safety_planner.process_confirmation(user_message, session_id)
            # F5: expiry must be handled BEFORE the truthy-confirmed branch so an
            # expired "yes"/"ok" can never reach the executor.
            if pending and pending.get("status") == "expired":
                msg = pending.get("message", "This confirmation has expired. No action was executed.")
                logger.info(f"[F5] Expired confirmation for session '{session_id}': {msg}")
                memory.add_user_message(user_message)
                memory.add_assistant_message(msg)
                yield {"type": "response", "data": msg}
                _trace("total", start_total)
                yield _metrics_event()
                return
            if pending:
                yield {"type": "thinking", "data": "Executing confirmed operation..."}
                tool_name = pending["tool_name"]
                arguments = pending["arguments"]
                yield {"type": "tool_call", "data": {"name": tool_name, "arguments": arguments}}
                _record_tool_call(tool_name, arguments)
                t_exec = time.monotonic()
                try:
                    result = await asyncio.wait_for(
                        self.executor.execute(tool_name, arguments), timeout=EXECUTOR_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    raise AgentError(
                        ERR_TIMEOUT, f"Salesforce call for '{tool_name}' timed out."
                    )
                _record_stage("executor", t_exec)
                yield {"type": "tool_result", "data": {"name": tool_name, "result": result}}

                exec_err = _executor_error_message(result, tool_name)
                if exec_err:
                    raise AgentError(ERR_MCP_SALESFORCE, f"Salesforce call '{tool_name}' failed: {exec_err}")

                t_synth = time.monotonic()
                synth_response = await self._synthesize_response(
                    user_message, [{"tool": tool_name, "result": result}]
                )
                _record_stage("synthesis", t_synth)
                memory.add_user_message(user_message)
                memory.add_assistant_message(synth_response)
                yield {"type": "response", "data": synth_response}
                _trace("total", start_total)
                yield _metrics_event()
                return
            else:
                decline_msg = "✅ Operation cancelled. No records were deleted."
                memory.add_user_message(user_message)
                memory.add_assistant_message(decline_msg)
                yield {"type": "response", "data": decline_msg}
                _trace("total", start_total)
                yield _metrics_event()
                return

        logger.info(f"📩 [USER MESSAGE] ({session_id}): {user_message}")
        memory.add_user_message(user_message)
        logger.info(f"[ORCHESTRATOR] Original query: {user_message}")

        # 1. PLANNER STAGE
        yield {"type": "thinking", "data": "[Planner] Decomposing your request..."}
        t_planner = time.monotonic()
        plan = await self._generate_plan_bounded(user_message, memory)
        _record_stage("planner", t_planner)

        # Empty plan means no Salesforce task — route via semantic RAG.
        if not plan:
            t_rag = time.monotonic()
            tools = await self._get_relevant_tools_or_fallback(user_message)
            _record_stage("rag", t_rag)
            if not tools:
                logger.info(
                    "[PLANNER] Decision: General — no Salesforce tools above RAG threshold. "
                    "Routing original query to Qwen."
                )
                t_qwen = time.monotonic()
                response = await self._answer_general(user_message, memory)
                _record_stage("general", t_qwen)
                memory.add_assistant_message(response)
                yield {"type": "response", "data": response}
                _trace("total", start_total)
                yield _metrics_event()
                return
            logger.info(
                "[PLANNER] Empty plan, but Salesforce intent detected via RAG. "
                "Using single implicit task."
            )
            plan = [
                {
                    "task_id": 1,
                    "description": user_message,
                    "agent": "DataAgent",
                    "depends_on": [],
                }
            ]

        yield {"type": "thinking", "data": f"[Planner] Generated {len(plan)} sub-tasks."}

        # 2. EXECUTION STAGE
        t_rag = time.monotonic()
        tools = await self._get_relevant_tools_or_fallback(user_message)
        _record_stage("rag", t_rag)
        # Fix A: intent-aware read-only filter. For a pure READ_REQUEST, mutation/
        # destructive tool schemas are removed BEFORE the planner/worker tool-selection
        # step so Qwen cannot be offered them. Explicit write/compound requests keep the
        # full set. READ_ONLY_MODE planner/executor gates are left untouched.
        tools = filter_tools_for_query(tools, user_message)
        logger.info(f"[RAG] Selected tools: {[t['function']['name'] for t in tools]}")
        all_results = []
        task_outputs = {}
        step_count = 0

        for task in plan:
            task_id = task.get("task_id", 0)
            desc = task.get("description", "")
            agent_type = task.get("agent", "DataAgent")

            yield {"type": "thinking", "data": f"[{agent_type}] Executing task: {desc}"}

            context = ""
            deps = task.get("depends_on", [])
            for dep_id in deps:
                if dep_id in task_outputs:
                    context += f"\\nResult of Task {dep_id}: {task_outputs[dep_id]}"

            task_prompt = f"Task: {desc}\\nContext: {context}\\nOutput JSON tool calls only."
            agent_msgs = [
                {"role": "system",
                 "content": DATA_AGENT_PROMPT if agent_type == "DataAgent" else ACTION_AGENT_PROMPT},
                {"role": "user", "content": task_prompt},
            ]

            t_qwen = time.monotonic()
            _set_stage("qwen-native-tool-call")
            try:
                llm_result = await asyncio.wait_for(
                    self.llm.chat_with_tools(messages=agent_msgs, tools=tools, temperature=0.0),
                    timeout=LLM_STAGE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "[TIMEOUT] Qwen native tool-call generation timed out. "
                    f"stage=qwen-native-tool-call timeout_s={LLM_STAGE_TIMEOUT:.1f} "
                    f"model={os.getenv('QWEN_MODEL', '')} endpoint={os.getenv('QWEN_BASE_URL', '')} "
                    f"mcp_reached=False rest_fallback=False"
                )
                raise AgentError(
                    ERR_TIMEOUT, "The model took too long to decide on the next step. Please try again."
                )
            finally:
                _set_stage(None)
            _record_stage("qwen_tool", t_qwen)
            # Mark per-task LLM step against the loop cap.
            step_count += 1

            tool_calls = llm_result.get("tool_calls", [])
            task_res = []
            for tc in tool_calls:
                if step_count >= self._max_tool_iterations:
                    raise AgentError(
                        ERR_TOO_MANY_STEPS,
                        "This request required too many steps to complete safely. "
                        "Please break it into smaller queries.",
                    )

                tc_name, tc_args, tc_error = _normalize_tool_call(tc)
                if tc_error:
                    raise AgentError(ERR_INVALID_TOOL, tc_error)

                # Fidelity: never run an aggregate COUNT tool call that the user did
                # not ask for AND that is redundant with a list/select query the same
                # task already requested. A plain list request ("list...", "show...")
                # carries no count intent, but the worker LLM can autonomously add a
                # SELECT COUNT(...) on top of its list query (the soqlQuery schema
                # advertises COUNT support), which then surfaces as a duplicate "Total"
                # block in the synthesized answer. Skip such a redundant call so only
                # the requested list query executes. Standalone COUNT calls (genuine
                # count requests, or a COUNT with no sibling list query), and explicit
                # list+count requests (count intent detected), are left untouched.
                soql = tc_args.get("q", "") if isinstance(tc_args, dict) else ""
                if (
                    tc_name == "soqlQuery"
                    and _is_soql_count(str(soql))
                    and not _has_count_intent(user_message)
                    and _has_non_count_query(tool_calls)
                ):
                    logger.info(
                        f"[DUPLICATE-COUNT-GUARD] Skipping redundant COUNT tool call for "
                        f"session '{session_id}': query='{soql}'. User request had no count "
                        f"intent and a list/select query is already requested."
                    )
                    continue

                safety = self.safety_planner.check_tool_safety(tc_name, tc_args, session_id)
                if safety.get("requires_confirmation"):
                    yield {"type": "response", "data": safety["confirmation_message"]}
                    _trace("total", start_total)
                    yield _metrics_event()
                    return
                if not safety.get("safe"):
                    logger.info(
                        f"Safety guard blocked tool '{tc_name}' for session '{session_id}'."
                    )
                    blocked = safety.get(
                        "blocked_message",
                        "This operation is not allowed in the current mode.",
                    )
                    yield {"type": "response", "data": blocked}
                    _trace("total", start_total)
                    yield _metrics_event()
                    return

                yield {"type": "tool_call", "data": {"name": tc_name, "arguments": tc_args}}
                _record_tool_call(tc_name, tc_args)
                t_exec = time.monotonic()
                _set_stage(f"salesforce-mcp-{tc_name}")
                try:
                    res = await asyncio.wait_for(
                        self.executor.execute(tc_name, tc_args), timeout=EXECUTOR_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        f"[TIMEOUT] Salesforce call for '{tc_name}' timed out. "
                        f"stage=salesforce-mcp-{tc_name} timeout_s={EXECUTOR_TIMEOUT:.1f} "
                        f"mcp_reached=true rest_fallback=maybe "
                        f"model={os.getenv('QWEN_MODEL', '')} endpoint={os.getenv('QWEN_BASE_URL', '')}"
                    )
                    raise AgentError(
                        ERR_TIMEOUT, f"Salesforce call for '{tc_name}' timed out."
                    )
                finally:
                    _set_stage(None)
                _record_stage("executor", t_exec)
                yield {"type": "tool_result", "data": {"name": tc_name, "result": res}}

                exec_err = _executor_error_message(res, tc_name)
                if exec_err:
                    raise AgentError(
                        ERR_MCP_SALESFORCE, f"Salesforce call '{tc_name}' failed: {exec_err}"
                    )

                step_count += 1
                # Fix B: normalize an explicitly-returned COUNT of zero so the
                # synthesizer preserves totalSize==0 instead of treating an empty
                # records array as missing data. No-op for every other result.
                normalized = _normalize_zero_count_result(tc_name, res, tc_args)
                task_res.append({"tool": tc_name, "result": normalized})
                all_results.append({"tool": tc_name, "result": normalized})

            task_outputs[task_id] = json.dumps(task_res)

        # 3. SYNTHESIZER STAGE
        yield {"type": "thinking", "data": "[Synthesizer] Formatting final response..."}
        t_synth = time.monotonic()
        synth_response = await self._synthesize_response(user_message, all_results)
        _record_stage("synthesis", t_synth)

        memory.add_assistant_message(synth_response)
        yield {"type": "response", "data": synth_response}
        _trace("total", start_total)
        yield _metrics_event()

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------
    async def _generate_plan_bounded(self, user_query: str, memory: ConversationMemory) -> list[dict]:
        try:
            return await asyncio.wait_for(
                self._generate_plan(user_query, memory), timeout=LLM_STAGE_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(
                "[PLANNER] Timeout generating plan. "
                f"stage=intent-planner timeout_s={LLM_STAGE_TIMEOUT:.1f} "
                f"model={os.getenv('QWEN_MODEL', '')} endpoint={os.getenv('QWEN_BASE_URL', '')} "
                "mcp_reached=False rest_fallback=False"
            )
            return []

    async def _get_relevant_tools_bounded(self, user_query: str) -> list[dict[str, Any]]:
        """Run the sync RAG retrieval off the event loop with a bounded timeout."""
        _set_stage("semantic-rag-tool-retrieval")
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.rag_retriever.get_relevant_tools, user_query, 6),
                timeout=RAG_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error(
                "[RAG] Timeout retrieving tools. "
                f"stage=semantic-rag-tool-retrieval "
                f"timeout_s={RAG_TIMEOUT:.1f} "
                f"embedding_model={os.getenv('RAG_EMBEDDING_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2')} "
                f"mcp_reached=False rest_fallback=False "
                f"root_cause='embedding model cold-load or vector query hung'"
            )
            raise AgentError(ERR_TIMEOUT, "Tool selection timed out during RAG warm-up. Please try again.")
        finally:
            _set_stage(None)

    async def _get_relevant_tools_or_fallback(self, user_query: str) -> list[dict[str, Any]]:
        """E5: bounded RAG with a safe Salesforce fallback.

        Preserves the existing RAG timeout behavior. When RAG returns [] for a
        Salesforce/data query, the COMPLETE Salesforce tool registry (same
        authoritative source as ENABLE_RAG_TOOLS=false) is used, then passed
        through filter_tools_for_query (Fix A) so read-only queries never gain
        mutation tools. Clearly general/non-Salesforce queries keep RAG's result
        (possibly []) so the existing general-answer path is preserved.
        """
        tools = await self._get_relevant_tools_bounded(user_query)
        if not tools and _has_salesforce_intent(user_query):
            logger.warning(
                "[RAG] Empty/timeout/failed retrieval for a Salesforce/data query; "
                "falling back to the complete read-only-safe tool registry."
            )
            tools = get_tool_definitions()
        return filter_tools_for_query(tools, user_query)

    async def _generate_plan(self, user_query: str, memory: ConversationMemory) -> list[dict]:
        msgs = [{"role": "system", "content": PLANNER_PROMPT}]
        hist = memory.get_messages_for_llm("")
        msgs.extend([m for m in hist if m["role"] != "system"])
        msgs.append({"role": "user", "content": f"Create an execution plan for: {user_query}"})

        t_qwen = time.monotonic()
        try:
            res = await self.llm.chat(messages=msgs, temperature=0.0)
            logger.info(f"🤔 [PLANNER RAW OUTPUT]:\n{res}")
        except Exception as e:
            logger.error(f"❌ Planning LLM call failed! Exception: {e}")
            return []
        finally:
            _trace("qwen/planner", t_qwen)

        content = res.strip()
        start_idx = content.find("[")
        end_idx = content.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
            try:
                json_str = content[start_idx:end_idx + 1]
                plan = json.loads(json_str)
                if isinstance(plan, list):
                    return plan
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse extracted JSON array: {e}")

        logger.warning("⚠️ Planner did not return a valid JSON list.")
        return []

    async def _synthesize_response(self, user_query: str, tool_results: list[dict]) -> str:
        msgs = [{"role": "system", "content": SYNTHESIZER_PROMPT}]

        # Deterministic data fidelity: flat tables are built in Python and handed
        # to the synthesizer as authoritative verbatim `[reference_table]` blocks,
        # so Id/Name stay separate cells and Salesforce values/order/count are
        # never regenerated or altered by the LLM. These same tables are the
        # deterministic fallback when the LLM output is incomplete or fails.
        #
        # The raw JSON tool results are still included in context (so downstream
        # behavior — e.g. Fix B's totalSize-preserving mix for COUNT-with-records —
        # is unchanged), but the verbatim reference tables take precedence and the
        # deterministic fallback is used instead of a half/rewritten answer.
        ref_tables, _ = _split_reference_results(tool_results)
        deterministic_tables = "\n\n".join(ref_tables) if ref_tables else None
        ref_blocks = [f"[reference_table]\n\n{t}" for t in ref_tables]

        parts: list[str] = [json.dumps(tool_results, indent=2)]
        if ref_blocks:
            parts.append(
                "The following pre-built Markdown tables are FINAL and authoritative. "
                "Present them VERBATIM — do NOT reformat, truncate, reorder, rename "
                "columns, or change any value, and show ALL rows.\n\n"
                + "\n\n".join(ref_blocks)
            )
        data_context = "\n\n".join(parts)

        # Bound the context passed to the synthesizer. Huge raw tool dumps
        # (e.g. a full object schema) make the 32B model slow to respond and can
        # blow SYNTHESIZER_TIMEOUT; trimming to a sane budget keeps the final
        # answer prompt fast without losing the information most queries need.
        if len(data_context) > SYNTHESIZER_CONTEXT_CHARS:
            data_context = (
                data_context[:SYNTHESIZER_CONTEXT_CHARS]
                + "\n... [tool results truncated for length]"
            )
        user_msg = (
            f"Original Query: {user_query}\\n\\nTool Results:\\n{data_context}"
            f"\\n\\nPlease provide the final Markdown response."
        )
        msgs.append({"role": "user", "content": user_msg})

        def _fallback_or_raise(reason: str) -> str:
            if deterministic_tables:
                logger.warning(
                    "[SYNTH] Synthesis %s; returning the deterministic reference "
                    "table instead of an incomplete/hallucinated answer.", reason
                )
                return deterministic_tables
            raise AgentError(
                ERR_SYNTHESIS, "I couldn't summarize the results. Please try your request again."
            )

        t_qwen = time.monotonic()
        _set_stage("qwen-final-answer")
        try:
            res = await asyncio.wait_for(
                self.llm.chat(
                    messages=msgs,
                    temperature=0.3,
                    max_tokens=SYNTHESIZER_MAX_TOKENS,
                ),
                timeout=SYNTHESIZER_TIMEOUT,
            )
            # Truncation detection: a token-cap-cut generation (finish_reason ==
            # "length") would otherwise be delivered as a silently incomplete
            # "half" reply. Detect it here and retry ONCE with a larger token
            # budget (still bounded by the existing SYNTHESIZER_TIMEOUT — no new
            # unbounded wait and no timeout increase). If it still truncates, fall
            # back to the deterministic reference table (never a half answer).
            if getattr(self.llm, "last_finish_reason", None) == "length":
                logger.warning(
                    "[SYNTH] Final answer hit the token cap; retrying once with a larger budget."
                )
                res = await asyncio.wait_for(
                    self.llm.chat(
                        messages=msgs,
                        temperature=0.3,
                        max_tokens=SYNTHESIZER_MAX_TOKENS * 2,
                    ),
                    timeout=SYNTHESIZER_TIMEOUT,
                )
                if getattr(self.llm, "last_finish_reason", None) == "length":
                    return _fallback_or_raise("was truncated at the token limit")
            return res.strip()
        except asyncio.TimeoutError:
            logger.error(
                "[TIMEOUT] Final answer generation timed out. "
                f"stage=qwen-final-answer timeout_s={SYNTHESIZER_TIMEOUT:.1f} "
                f"model={os.getenv('QWEN_MODEL', '')} endpoint={os.getenv('QWEN_BASE_URL', '')} "
                "mcp_reached=true rest_fallback=false "
                f"root_cause='Salesforce/MCP succeeded but final Qwen generation was too slow'"
            )
            # P0: a bound firing is never replaced by a synthetic "result"; but a
            # verified deterministic table is real Salesforce data, so it is a safe
            # (and strictly more informative) response than a raw timeout error.
            if deterministic_tables:
                logger.warning(
                    "[SYNTH] Synthesis timed out; returning the deterministic reference table."
                )
                return deterministic_tables
            raise AgentError(
                ERR_TIMEOUT, "The final response could not be produced in time. Please try again."
            )
        except Exception as e:
            logger.error(f"Synthesizer failed: {e}")
            return _fallback_or_raise("failed")
        finally:
            _set_stage(None)
            _trace("qwen/synthesis", t_qwen)

    async def _answer_general(self, user_query: str, memory: ConversationMemory) -> str:
        """
        Answer a general / non-Salesforce query directly with the LLM.
        Preserves the ORIGINAL user query. Raises a structured error if the
        LLM is unavailable so the client gets a controlled failure.
        """
        logger.info(f"[QWEN] General response request sent for query: {user_query}")
        _set_stage("qwen-general-answer")
        try:
            msgs = [{"role": "system", "content": GENERAL_ANSWER_PROMPT}]
            msgs.append({"role": "user", "content": user_query})
            res = await asyncio.wait_for(
                self.llm.chat(messages=msgs, temperature=0.3), timeout=LLM_STAGE_TIMEOUT
            )
            answer = res.strip()
            logger.info(f"[QWEN] General response received ({len(answer)} chars).")
            return answer
        except asyncio.TimeoutError:
            logger.error(
                "[TIMEOUT] General answer generation timed out. "
                f"stage=qwen-general-answer timeout_s={LLM_STAGE_TIMEOUT:.1f} "
                f"model={os.getenv('QWEN_MODEL', '')} endpoint={os.getenv('QWEN_BASE_URL', '')} "
                "mcp_reached=False rest_fallback=False"
            )
            raise AgentError(
                ERR_TIMEOUT, "The assistant did not respond in time. Please try again."
            )
        except Exception as e:
            logger.error(f"[QWEN] General response failed: {e}")
            raise AgentError(
                ERR_GENERAL,
                "I couldn't reach the assistant model to answer that. Please try again shortly.",
            ) from e
        finally:
            _set_stage(None)
