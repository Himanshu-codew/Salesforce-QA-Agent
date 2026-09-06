"""
Conversation memory management for the Salesforce Agent.
Maintains a sliding window of conversation history with support
for user messages, assistant responses, and tool call results.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    Manages conversation history with a sliding window to stay
    within LLM context limits.
    """

    def __init__(self, max_messages: int = 6):
        """
        Args:
            max_messages: Maximum number of messages to retain.
                          Oldest messages are dropped when exceeded to keep inference ultra-fast.
        """
        self.max_messages = max_messages
        self._messages: list[dict[str, Any]] = []

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return current conversation history."""
        return list(self._messages)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the history."""
        self._messages.append({"role": "user", "content": content})
        self._trim()
        logger.debug(f"Added user message, history size: {len(self._messages)}")

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant text response to the history."""
        self._messages.append({"role": "assistant", "content": content})
        self._trim()
        logger.debug(f"Added assistant message, history size: {len(self._messages)}")

    def add_assistant_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        """
        Add an assistant message that contains tool calls.
        Stored in OpenAI-compatible format for subsequent LLM calls.
        """
        formatted_tool_calls = []
        for tc in tool_calls:
            import json
            args = tc["arguments"]
            if isinstance(args, dict):
                args = json.dumps(args)
            formatted_tool_calls.append({
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": args,
                },
            })

        self._messages.append({
            "role": "assistant",
            "content": "",
            "tool_calls": formatted_tool_calls,
        })
        logger.debug(
            f"Added assistant tool calls: "
            f"{[tc['name'] for tc in tool_calls]}"
        )

    def add_tool_result(self, tool_call_id: str, tool_name: str, result: str) -> None:
        """
        Add a tool execution result to the history.
        Must follow an assistant message containing the corresponding tool call.
        Truncated at 10000 chars — balances context fidelity against memory pressure.
        On Render (512 MB), retaining 25 KB per tool result across 20 messages can
        consume ~500 KB per session; 10 KB keeps the same window under ~200 KB.
        """
        clean_result = result
        if len(clean_result) > 10000:
            # Cut at line boundary to preserve last complete record row
            truncated_pos = clean_result.rfind("\n", 0, 10000)
            if truncated_pos > 3000:
                clean_result = clean_result[:truncated_pos] + "\n... [truncated at 10000 chars — query returned large dataset]"
            else:
                clean_result = clean_result[:10000] + "\n... [truncated at 10000 chars]"

        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": clean_result,
        })
        logger.debug(f"Added tool result for {tool_name} (id={tool_call_id}, len={len(clean_result)})")


    def clear(self) -> None:
        """Clear all conversation history."""
        self._messages.clear()
        logger.info("Conversation memory cleared.")

    def _trim(self) -> None:
        """
        Trim the history to stay within max_messages.
        Preserves the integrity of tool call chains —
        never splits a tool call from its result.
        """
        while len(self._messages) > self.max_messages:
            # Check if removing the first message would orphan a tool result
            if len(self._messages) > 1:
                first = self._messages[0]
                second = self._messages[1]

                # If the first message has tool_calls and the second is a tool result,
                # remove both to maintain chain integrity
                if (
                    first.get("role") == "assistant"
                    and first.get("tool_calls")
                    and second.get("role") == "tool"
                ):
                    # Count how many tool results follow this assistant message
                    tool_result_count = 0
                    for msg in self._messages[1:]:
                        if msg.get("role") == "tool":
                            tool_result_count += 1
                        else:
                            break
                    # Remove the assistant message + all its tool results
                    self._messages = self._messages[1 + tool_result_count:]
                else:
                    self._messages.pop(0)
            else:
                self._messages.pop(0)

    def get_messages_for_llm(self, system_prompt: str) -> list[dict[str, Any]]:
        """
        Build the full message list for the LLM, including the system prompt.

        Args:
            system_prompt: The system prompt to prepend.

        Returns:
            List of messages with system prompt + conversation history.
        """
        return [{"role": "system", "content": system_prompt}] + self.messages

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ConversationMemory(messages={len(self._messages)}, max={self.max_messages})"
