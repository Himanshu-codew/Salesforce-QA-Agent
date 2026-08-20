"""
Tool Registry — discovers, caches, and converts MCP tool schemas
to OpenAI function calling format for use with Qwen3.
"""

import logging
from typing import Any

from tools.salesforce import SALESFORCE_TOOLS, get_tool_definitions

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Manages the set of available Salesforce MCP tools.

    Responsibilities:
    - Discover tools from the MCP server on startup
    - Cache tool schemas
    - Convert MCP tool format to OpenAI function calling format
    - Provide lookup by tool name
    """

    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}
        self._openai_format: list[dict[str, Any]] = []
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self, mcp_client=None) -> None:
        """
        Initialize the registry by discovering tools.
        Tries MCP server first, falls back to local definitions.
        """
        if mcp_client:
            try:
                mcp_tools = await mcp_client.list_tools()
                self._load_mcp_tools(mcp_tools)
                logger.info(
                    f"Registry initialized with {len(self._tools)} tools from MCP."
                )
                self._initialized = True
                return
            except Exception as e:
                logger.warning(f"MCP tool discovery failed: {e}")

        # Fallback: use local tool definitions
        self._load_local_tools()
        logger.info(
            f"Registry initialized with {len(self._tools)} local tool definitions."
        )
        self._initialized = True

    def _load_mcp_tools(self, mcp_tools: list[dict[str, Any]]) -> None:
        """Load tools from MCP server response and convert to OpenAI format."""
        for tool in mcp_tools:
            # MCP tools might come in MCP format or already in OpenAI format
            if "function" in tool:
                # Already in OpenAI format
                name = tool["function"]["name"]
                self._tools[name] = tool
                self._openai_format.append(tool)
            elif "name" in tool:
                # MCP native format — convert
                name = tool["name"]
                openai_tool = self._mcp_to_openai(tool)
                self._tools[name] = openai_tool
                self._openai_format.append(openai_tool)

    def _load_local_tools(self) -> None:
        """Load tools from local SALESFORCE_TOOLS definitions."""
        for tool in get_tool_definitions():
            name = tool["function"]["name"]
            self._tools[name] = tool
            self._openai_format.append(tool)

    @staticmethod
    def _mcp_to_openai(mcp_tool: dict[str, Any]) -> dict[str, Any]:
        """
        Convert an MCP tool definition to OpenAI function calling format.

        MCP format:
            {"name": "...", "description": "...", "inputSchema": {...}}

        OpenAI format:
            {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        """
        return {
            "type": "function",
            "function": {
                "name": mcp_tool["name"],
                "description": mcp_tool.get("description", ""),
                "parameters": mcp_tool.get("inputSchema", {
                    "type": "object",
                    "properties": {},
                    "required": [],
                }),
            },
        }

    def get_tools_openai_format(self) -> list[dict[str, Any]]:
        """Return all tools in OpenAI function calling format."""
        return self._openai_format

    def get_tool(self, name: str) -> dict[str, Any] | None:
        """Look up a tool by name."""
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool exists in the registry."""
        return name in self._tools

    def list_tool_names(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"
