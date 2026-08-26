"""
Task Planner for the Salesforce Agent.
Handles intent classification, safety checks for destructive operations,
and multi-step task decomposition.
"""

import logging
from typing import Any

from tools.salesforce import is_destructive, is_mutating, DESTRUCTIVE_TOOLS
from .prompts import DELETE_CONFIRMATION_PROMPT

logger = logging.getLogger(__name__)


class TaskPlanner:
    """
    Pre-processes tool calls from the LLM to enforce safety guardrails
    and manage multi-step task execution.
    """

    def __init__(self):
        # Track pending confirmations: {session_id: pending_action}
        self._pending_confirmations: dict[str, dict[str, Any]] = {}

    def check_tool_safety(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str = "default",
    ) -> dict[str, Any]:
        """
        Check if a tool call requires user confirmation before execution.

        Returns:
            A dict with:
                - 'safe': bool — True if the tool can execute immediately.
                - 'requires_confirmation': bool — True if user must confirm.
                - 'confirmation_message': str — Message to show user (if confirmation needed).
                - 'pending_action': dict — The action awaiting confirmation (if any).
        """
        result = {
            "safe": True,
            "requires_confirmation": False,
            "confirmation_message": "",
            "pending_action": None,
        }

        if is_destructive(tool_name):
            # Destructive operations ALWAYS require confirmation
            sobject_name = arguments.get("sobject-name", "Unknown")
            record_id = arguments.get("id", "Unknown")

            confirmation_msg = DELETE_CONFIRMATION_PROMPT.format(
                sobject_name=sobject_name,
                record_id=record_id,
            )

            pending = {
                "tool_name": tool_name,
                "arguments": arguments,
                "type": "delete",
            }
            self._pending_confirmations[session_id] = pending

            result["safe"] = False
            result["requires_confirmation"] = True
            result["confirmation_message"] = confirmation_msg
            result["pending_action"] = pending

            logger.info(
                f"Destructive operation blocked for confirmation: "
                f"{tool_name}({sobject_name}, {record_id})"
            )

        return result

    def process_confirmation(
        self,
        user_response: str,
        session_id: str = "default",
    ) -> dict[str, Any] | None:
        """
        Process a user's confirmation response for a pending destructive action.

        Args:
            user_response: The user's text response.
            session_id: Session identifier.

        Returns:
            The pending action dict if confirmed, None if rejected or no pending action.
        """
        pending = self._pending_confirmations.get(session_id)
        if not pending:
            return None

        # Check if user confirmed
        response_lower = user_response.strip().lower()
        confirmed = response_lower in {
            "yes", "y", "yeah", "yep", "confirm", "confirmed", "proceed",
            "do it", "go ahead", "ok", "okay", "sure", "aye", "absolutely",
        }

        if confirmed:
            # Remove from pending and return the action for execution
            del self._pending_confirmations[session_id]
            logger.info(f"User confirmed destructive operation: {pending['tool_name']}")
            return pending
        else:
            # User declined — clear the pending action
            del self._pending_confirmations[session_id]
            logger.info(f"User declined destructive operation: {pending['tool_name']}")
            return None

    def has_pending_confirmation(self, session_id: str = "default") -> bool:
        """Check if there's a pending confirmation for this session."""
        return session_id in self._pending_confirmations

    def get_pending_confirmation(self, session_id: str = "default") -> dict[str, Any] | None:
        """Get the pending confirmation details without clearing it."""
        return self._pending_confirmations.get(session_id)

    def clear_pending(self, session_id: str = "default") -> None:
        """Clear any pending confirmation for a session."""
        self._pending_confirmations.pop(session_id, None)

    def classify_intent(self, tool_calls: list[dict[str, Any]]) -> str:
        """
        Classify the overall intent of a set of tool calls.

        Returns:
            'read_only', 'mutating', or 'destructive'
        """
        has_destructive = any(
            is_destructive(tc["name"]) for tc in tool_calls
        )
        has_mutating = any(
            is_mutating(tc["name"]) for tc in tool_calls
        )

        if has_destructive:
            return "destructive"
        elif has_mutating:
            return "mutating"
        else:
            return "read_only"
