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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounded-latency / resilience configuration (env-overridable, safe defaults)
# ---------------------------------------------------------------------------
LLM_STAGE_TIMEOUT = float(os.getenv("LLM_STAGE_TIMEOUT", "60.0"))
EXECUTOR_TIMEOUT = float(os.getenv("EXECUTOR_TIMEOUT", "60.0"))
RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "20.0"))
OVERALL_PROCESS_TIMEOUT = float(os.getenv("OVERALL_PROCESS_TIMEOUT", "180.0"))
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "6"))


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

        # Confirmation handling (destructive actions)
        if self.safety_planner.has_pending_confirmation(session_id):
            pending = self.safety_planner.process_confirmation(user_message, session_id)
            if pending:
                yield {"type": "thinking", "data": "Executing confirmed operation..."}
                tool_name = pending["tool_name"]
                arguments = pending["arguments"]
                yield {"type": "tool_call", "data": {"name": tool_name, "arguments": arguments}}
                t_exec = time.monotonic()
                try:
                    result = await asyncio.wait_for(
                        self.executor.execute(tool_name, arguments), timeout=EXECUTOR_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    raise AgentError(
                        ERR_TIMEOUT, f"Salesforce call for '{tool_name}' timed out."
                    )
                _trace("mcp/salesforce", t_exec)
                yield {"type": "tool_result", "data": {"name": tool_name, "result": result}}

                exec_err = _executor_error_message(result, tool_name)
                if exec_err:
                    raise AgentError(ERR_MCP_SALESFORCE, f"Salesforce call '{tool_name}' failed: {exec_err}")

                t_synth = time.monotonic()
                synth_response = await self._synthesize_response(
                    user_message, [{"tool": tool_name, "result": result}]
                )
                _trace("synthesis", t_synth)
                memory.add_user_message(user_message)
                memory.add_assistant_message(synth_response)
                yield {"type": "response", "data": synth_response}
                _trace("total", start_total)
                return
            else:
                decline_msg = "✅ Operation cancelled. No records were deleted."
                memory.add_user_message(user_message)
                memory.add_assistant_message(decline_msg)
                yield {"type": "response", "data": decline_msg}
                _trace("total", start_total)
                return

        logger.info(f"📩 [USER MESSAGE] ({session_id}): {user_message}")
        memory.add_user_message(user_message)
        logger.info(f"[ORCHESTRATOR] Original query: {user_message}")

        # 1. PLANNER STAGE
        yield {"type": "thinking", "data": "[Planner] Decomposing your request..."}
        t_planner = time.monotonic()
        plan = await self._generate_plan_bounded(user_message, memory)
        _trace("planner", t_planner)

        # Empty plan means no Salesforce task — route via semantic RAG.
        if not plan:
            t_rag = time.monotonic()
            tools = await self._get_relevant_tools_bounded(user_message)
            _trace("rag", t_rag)
            if not tools:
                logger.info(
                    "[PLANNER] Decision: General — no Salesforce tools above RAG threshold. "
                    "Routing original query to Qwen."
                )
                t_qwen = time.monotonic()
                response = await self._answer_general(user_message, memory)
                _trace("general/qwen", t_qwen)
                memory.add_assistant_message(response)
                yield {"type": "response", "data": response}
                _trace("total", start_total)
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
        tools = await self._get_relevant_tools_bounded(user_message)
        _trace("rag", t_rag)
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
            try:
                llm_result = await asyncio.wait_for(
                    self.llm.chat_with_tools(messages=agent_msgs, tools=tools, temperature=0.0),
                    timeout=LLM_STAGE_TIMEOUT,
                )
            except asyncio.TimeoutError:
                raise AgentError(
                    ERR_TIMEOUT, "The model took too long to decide on the next step. Please try again."
                )
            _trace("qwen/tool-selection", t_qwen)
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

                safety = self.safety_planner.check_tool_safety(tc_name, tc_args, session_id)
                if safety.get("requires_confirmation"):
                    yield {"type": "response", "data": safety["confirmation_message"]}
                    _trace("total", start_total)
                    return

                yield {"type": "tool_call", "data": {"name": tc_name, "arguments": tc_args}}
                t_exec = time.monotonic()
                try:
                    res = await asyncio.wait_for(
                        self.executor.execute(tc_name, tc_args), timeout=EXECUTOR_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    raise AgentError(
                        ERR_TIMEOUT, f"Salesforce call for '{tc_name}' timed out."
                    )
                _trace("mcp/salesforce", t_exec)
                yield {"type": "tool_result", "data": {"name": tc_name, "result": res}}

                exec_err = _executor_error_message(res, tc_name)
                if exec_err:
                    raise AgentError(
                        ERR_MCP_SALESFORCE, f"Salesforce call '{tc_name}' failed: {exec_err}"
                    )

                step_count += 1
                task_res.append({"tool": tc_name, "result": res})
                all_results.append({"tool": tc_name, "result": res})

            task_outputs[task_id] = json.dumps(task_res)

        # 3. SYNTHESIZER STAGE
        yield {"type": "thinking", "data": "[Synthesizer] Formatting final response..."}
        t_synth = time.monotonic()
        synth_response = await self._synthesize_response(user_message, all_results)
        _trace("synthesis", t_synth)

        memory.add_assistant_message(synth_response)
        yield {"type": "response", "data": synth_response}
        _trace("total", start_total)

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------
    async def _generate_plan_bounded(self, user_query: str, memory: ConversationMemory) -> list[dict]:
        try:
            return await asyncio.wait_for(
                self._generate_plan(user_query, memory), timeout=LLM_STAGE_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error("[PLANNER] Timeout generating plan.")
            return []

    async def _get_relevant_tools_bounded(self, user_query: str) -> list[dict[str, Any]]:
        """Run the sync RAG retrieval off the event loop with a bounded timeout."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self.rag_retriever.get_relevant_tools, user_query, 6),
                timeout=RAG_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.error("[RAG] Timeout retrieving tools.")
            raise AgentError(ERR_TIMEOUT, "Tool selection timed out. Please try again.")

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
        data_context = json.dumps(tool_results, indent=2)
        user_msg = (
            f"Original Query: {user_query}\\n\\nRaw Tool Results:\\n{data_context}"
            f"\\n\\nPlease provide the final Markdown response."
        )
        msgs.append({"role": "user", "content": user_msg})

        t_qwen = time.monotonic()
        try:
            res = await asyncio.wait_for(
                self.llm.chat(messages=msgs, temperature=0.3), timeout=LLM_STAGE_TIMEOUT
            )
            return res.strip()
        except asyncio.TimeoutError:
            raise AgentError(
                ERR_TIMEOUT, "The final response could not be produced in time. Please try again."
            )
        except Exception as e:
            logger.error(f"Synthesizer failed: {e}")
            raise AgentError(
                ERR_SYNTHESIS, "I couldn't summarize the results. Please try your request again."
            ) from e
        finally:
            _trace("qwen/synthesis", t_qwen)

    async def _answer_general(self, user_query: str, memory: ConversationMemory) -> str:
        """
        Answer a general / non-Salesforce query directly with the LLM.
        Preserves the ORIGINAL user query. Raises a structured error if the
        LLM is unavailable so the client gets a controlled failure.
        """
        logger.info(f"[QWEN] General response request sent for query: {user_query}")
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
            raise AgentError(
                ERR_TIMEOUT, "The assistant did not respond in time. Please try again."
            )
        except Exception as e:
            logger.error(f"[QWEN] General response failed: {e}")
            raise AgentError(
                ERR_GENERAL,
                "I couldn't reach the assistant model to answer that. Please try again shortly.",
            ) from e
