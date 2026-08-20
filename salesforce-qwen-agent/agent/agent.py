"""
Core Salesforce Agent — the main agent loop that ties together
the LLM (Qwen3), MCP executor, memory, and planner.
"""

import json
import logging
from typing import Any, AsyncGenerator

from llm.base import BaseLLM
from mcp.executor import ToolExecutor
from tools.salesforce import get_tool_definitions
from .memory import ConversationMemory
from .planner import TaskPlanner
from .prompts import SYSTEM_PROMPT, ERROR_MESSAGES
from .rag import ToolRAGRetriever

logger = logging.getLogger(__name__)


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

                yield {
                    "type": "tool_call",
                    "data": {"name": tool_name, "arguments": arguments},
                }

                result = await self.executor.execute(tool_name, arguments)

                yield {
                    "type": "tool_result",
                    "data": {"name": tool_name, "result": result},
                }

                # Add to memory and get LLM summary
                memory.add_user_message(user_message)
                memory.add_assistant_tool_calls([{
                    "id": "confirmed_action",
                    "name": tool_name,
                    "arguments": arguments,
                }])
                memory.add_tool_result("confirmed_action", tool_name, result)

                # Get LLM to summarize the result
                messages = memory.get_messages_for_llm(SYSTEM_PROMPT)
                llm_response = await self.llm.chat(messages)
                memory.add_assistant_message(llm_response)

                yield {"type": "response", "data": llm_response}
                return
            else:
                # User declined
                memory.add_user_message(user_message)
                decline_msg = "✅ Operation cancelled. No records were deleted."
                memory.add_assistant_message(decline_msg)
                yield {"type": "response", "data": decline_msg}
                return

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

                # Check safety for each tool call
                for tc in tool_calls:
                    safety = self.planner.check_tool_safety(
                        tc["name"], tc["arguments"], session_id
                    )

                    if safety["requires_confirmation"]:
                        # Block execution and ask for confirmation
                        logger.warning(f"⚠️ [SAFETY BLOCK] Confirmation required for '{tc['name']}'")
                        memory.add_assistant_message(safety["confirmation_message"])
                        yield {
                            "type": "confirmation",
                            "data": safety["confirmation_message"],
                        }
                        return

                # All tool calls are safe — execute them
                memory.add_assistant_tool_calls(tool_calls)

                for tc in tool_calls:
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

                    # Truncate very large results to avoid context overflow
                    if len(result) > 15000:
                        result = result[:15000] + "\n... [truncated, showing first 15000 chars]"

                    memory.add_tool_result(tc["id"], tc["name"], result)

                    yield {
                        "type": "tool_result",
                        "data": {"name": tc["name"], "result": result},
                    }

                # Continue the loop — LLM will see tool results and decide next step

            # ── Case 2: LLM returns a final text response ──
            elif llm_result["content"]:
                response = llm_result["content"]
                logger.info(f"🤖 [ASSISTANT RESPONSE]: {response[:150]}...")
                memory.add_assistant_message(response)
                yield {"type": "response", "data": response}
                return

            # ── Case 3: Empty response (shouldn't happen) ──
            else:
                fallback = "I processed your request but didn't generate a response. Could you rephrase?"
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
                "content": "Please summarize the results you've gathered so far.",
            })
            summary = await self.llm.chat(messages)
            final_msg = f"{max_iter_msg}\n\n{summary}"
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
