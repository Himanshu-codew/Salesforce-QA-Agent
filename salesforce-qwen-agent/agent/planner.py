"""
Task Planner for the Salesforce Agent.
Handles intent classification, safety checks for destructive operations,
and multi-step task decomposition.
"""

import logging
import os
import time
from typing import Any

from tools.salesforce import is_destructive, is_mutating, DESTRUCTIVE_TOOLS
from .prompts import DELETE_CONFIRMATION_PROMPT

logger = logging.getLogger(__name__)

# Hard safety gate for evaluation/baseline runs. When enabled, NO create,
# update, upload, or delete tool may execute — the orchestrator blocks the call
# before it reaches Salesforce. Controlled by READ_ONLY_MODE=true.
READ_ONLY_MODE = os.getenv("READ_ONLY_MODE", "false").lower() in ("true", "1", "yes", "on")

# F5: configurable TTL for pending destructive-action confirmations. Uses the
# SAME env-convention as the rest of the agent (monotonic clock, safe default).
# A stale "yes" received after this window expires is never executed.
PENDING_CONFIRMATION_TTL = float(os.getenv("PENDING_CONFIRMATION_TTL", "300.0"))


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
            "blocked_message": "",
        }

        if READ_ONLY_MODE and (is_mutating(tool_name) or is_destructive(tool_name)):
            # Evaluation / baseline mode: refuse every mutation attempt outright.
            result["safe"] = False
            result["blocked_message"] = (
                "I'm running in read-only mode for this evaluation, so I can't "
                "create, update, upload, or delete any records. I can still run "
                "search and read queries, or help you plan the change."
            )
            logger.info(
                f"Read-only mode blocked tool call: {tool_name} "
                f"(args={str(arguments)[:160]})"
            )
            return result

        if is_destructive(tool_name):
            # Destructive operations ALWAYS require confirmation
            sobject_name = arguments.get("sobject-name", "Unknown")
            record_id = arguments.get("id", "Unknown")

            confirmation_msg = DELETE_CONFIRMATION_PROMPT.format(
                sobject_name=sobject_name,
                record_id=record_id,
            )

            # F5 exact-action binding: the FIRST destructive action that becomes
            # pending must remain the action awaiting confirmation. Additional
            # destructive calls in the same turn must NOT overwrite it.
            existing = self._pending_confirmations.get(session_id)
            if existing is not None:
                pending = existing
                confirmation_msg = pending.get("confirmation_message") or confirmation_msg
                result["safe"] = False
                result["requires_confirmation"] = True
                result["confirmation_message"] = confirmation_msg
                result["pending_action"] = pending
                logger.info(
                    f"Additional destructive call {tool_name}({sobject_name}, {record_id}) "
                    f"ignored for confirmation; already pending "
                    f"{pending.get('tool_name')}({pending.get('arguments', {}).get('sobject-name')}, "
                    f"{pending.get('arguments', {}).get('id')}) for session '{session_id}'"
                )
                return result

            pending = {
                "tool_name": tool_name,
                "arguments": arguments,
                "type": "delete",
                "created_at": time.monotonic(),
                "confirmation_message": confirmation_msg,
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

        # F5 TTL: a stale confirmation must never be executed. Missing
        # created_at is treated as not-yet-expired (backward-compatible with
        # any legacy in-memory state) rather than crashing.
        created_at = pending.get("created_at")
        if created_at is not None and (time.monotonic() - created_at) >= PENDING_CONFIRMATION_TTL:
            del self._pending_confirmations[session_id]
            logger.info(
                f"Pending destructive confirmation for '{pending.get('tool_name')}' "
                f"expired for session '{session_id}'; no action executed."
            )
            return {
                "status": "expired",
                "tool_name": pending.get("tool_name", ""),
                "message": "This confirmation has expired. No action was executed.",
            }

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
