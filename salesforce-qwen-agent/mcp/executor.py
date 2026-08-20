"""
Tool Executor — executes tool calls against the Salesforce MCP Server.
Provides validation, error handling, retry logic, and result formatting.
"""

import json
import logging
from typing import Any

from .client import SalesforceMCPClient
from .registry import ToolRegistry

logger = logging.getLogger(__name__)


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

        logger.info(f"Executing tool: {tool_name} with args: {_truncate_args(arguments)}")

        try:
            result = await self.mcp_client.call_tool(tool_name, arguments)
            formatted = self._format_result(tool_name, result)
            logger.info(f"Tool {tool_name} executed successfully.")
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

    def _format_result(self, tool_name: str, result: Any) -> str:
        """Format tool result as a clean JSON string."""
        if isinstance(result, str):
            try:
                # Try to parse as JSON for pretty formatting
                parsed = json.loads(result)
                return json.dumps(parsed, indent=2, default=str)
            except (json.JSONDecodeError, TypeError):
                return result

        if isinstance(result, dict):
            # Clean up Salesforce REST API response attributes
            cleaned = self._clean_salesforce_response(result)
            return json.dumps(cleaned, indent=2, default=str)

        if isinstance(result, list):
            return json.dumps(result, indent=2, default=str)

        return str(result)

    @staticmethod
    def _format_datetime_value(val: Any) -> Any:
        if isinstance(val, str) and ("T" in val) and (val.endswith("+0000") or val.endswith("Z") or val.endswith("+00:00")):
            try:
                from datetime import datetime
                import zoneinfo
                clean = val.replace("+0000", "+00:00").replace("Z", "+00:00")
                dt = datetime.fromisoformat(clean)
                tz = zoneinfo.ZoneInfo("America/Los_Angeles")
                local_dt = dt.astimezone(tz)
                local_formatted = local_dt.strftime("%m/%d/%Y, %I:%M %p %Z")
                if local_formatted.startswith("0"):
                    local_formatted = local_formatted[1:]
                return f"{val} ({dt.strftime('%I:%M:%S %p UTC')} | {local_formatted} Salesforce UI Time)"
            except Exception:
                return val
        return val

    @staticmethod
    def _clean_salesforce_response(data: dict) -> dict:
        """
        Clean up Salesforce API response by removing internal metadata
        fields and providing readable timezone conversions for timestamps.
        """
        # Remove 'attributes' keys from records (internal SF metadata)
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
            return cleaned
        return data

    @staticmethod
    def _get_error_suggestion(tool_name: str, error: str) -> str:
        """Provide helpful suggestions based on common errors."""
        error_lower = error.lower()

        if "group by" in error_lower and ("subquery" in error_lower or "semi" in error_lower or "not supported" in error_lower or "malformed" in error_lower):
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
