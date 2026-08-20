"""
Abstract base class for LLM providers.
Defines the interface that all LLM implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """Abstract base class for Language Model providers."""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """
        Send a chat completion request without tool calling.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            temperature: Sampling temperature (0.0 - 1.0).
            max_tokens: Maximum tokens in the response.

        Returns:
            The assistant's response text.
        """
        ...

    @abstractmethod
    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """
        Send a chat completion request with tool/function calling support.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            tools: List of tool definitions in OpenAI function calling format.
            temperature: Sampling temperature (0.0 - 1.0).
            max_tokens: Maximum tokens in the response.

        Returns:
            A dict containing:
                - 'content': The assistant's text response (may be None if tool call).
                - 'tool_calls': List of tool call dicts, each with:
                    - 'id': Tool call ID.
                    - 'name': Function/tool name.
                    - 'arguments': Dict of arguments.
                - 'finish_reason': Why the model stopped ('stop', 'tool_calls', etc.).
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (e.g., close HTTP clients)."""
        ...
