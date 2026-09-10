"""
Tool "RAG" retriever (passthrough).

Tool subset selection by retrieval is intentionally DISABLED: the model (Qwen)
receives the complete Salesforce tool registry and decides which tool to call.

    User Query
      -> ToolRAGRetriever.get_relevant_tools() returns the full registry
      -> model (Qwen) selects the correct tool + arguments from the schema
      -> filter_tools_for_query (agent.agent / agent.multi_agent) strips
         mutation/destructive tools from clearly read-only requests so the
         model can never pick one for a read-only ask

Why was retrieval removed?
    Retrieval-as-gating was the original design, but it caused real failures:
      * a greeting could surface a mutation tool and let the model run it;
      * a valid SOQL query could fail to surface soqlQuery, breaking the task;
      * a single RAG_MIN_SCORE threshold could not both accept real queries
        and reject greetings/flattery.

    Routing decisions are now deterministic and model-time:
      1. _has_salesforce_intent() decides general-chat vs Salesforce.
      2. The planner decomposes Salesforce requests into tool tasks.
      3. The model picks the tool from the full schema set.
      4. filter_tools_for_query() is the read-only safety net.

Memory note: this module imports NO torch / sentence-transformers / chromadb,
so it stays far below Render's 512 MB budget. warm_up() is a cheap no-op kept
for startup-call compatibility.

Configuration (environment variables):
    RAG_TOP_K        accepted for backwards compat (ignored; model decides)
    RAG_MIN_SCORE    accepted for backwards compat (ignored; model decides)
    ENABLE_RAG_TOOLS accepted for backwards compat (ignored; always full set)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from tools.salesforce import get_tool_definitions

logger = logging.getLogger(__name__)


def warm_up():
    """
    No-op warm-up kept for startup compatibility (app.py, SDK).

    There is no index to precompute and no model to preload: retrieval was
    disabled in favor of full-registry passthrough, so warm-up is trivially
    cheap and never imports heavy backends.
    """
    try:
        get_tool_definitions()
        logger.info("[RAG] Passthrough retriever ready: all tools exposed to the model.")
        return True
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"[RAG] Warm-up failed (will retry on first request): {e}")
        return False


class ToolRAGRetriever:
    """
    Passthrough tool retriever: returns the complete Salesforce tool registry.

    Model-driven tool selection is intentional. Read-only safety is enforced
    downstream by ``filter_tools_for_query`` (agent.agent / agent.multi_agent),
    which removes mutation/destructive tools before they reach the model.
    """

    def __init__(self, default_top_k: int = 5, min_confidence: float = 0.18):
        self.all_tools: list[dict[str, Any]] = get_tool_definitions()
        self.tool_map: dict[str, Any] = {
            t["function"]["name"]: t for t in self.all_tools
        }
        # Kept for API/config backward compatibility; selection is model-time.
        self.default_top_k = int(os.getenv("RAG_TOP_K", str(default_top_k)))
        self.min_score = float(os.getenv("RAG_MIN_SCORE", str(min_confidence)))

    @staticmethod
    def _extract_actual_query(user_query: str) -> str:
        """Isolate the real user instruction when a file attachment payload is present."""
        if "[Attached File:" not in user_query:
            return user_query.strip()
        if "User Message:" in user_query:
            return user_query.split("User Message:")[-1].strip()
        parts = user_query.split("]\n")
        if len(parts) > 1:
            return parts[-1].strip()
        return user_query.strip()

    def get_relevant_tools(self, user_query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Return the complete tool registry so the model decides which tool to call.

        Returns an empty list only for trivially short/empty queries so the
        orchestrator can route them to the general-chat path.
        """
        _ = top_k or self.default_top_k  # API compat: selection is delegated to the model.

        actual_query = self._extract_actual_query(user_query)
        if not actual_query or len(actual_query.strip()) < 2:
            logger.debug("[RAG] Query too short; returning no tools.")
            return []

        logger.debug(
            "RAG passthrough: returning all %d tools to the model; "
            "the model selects the tool (RAG subsetting disabled).",
            len(self.all_tools),
        )
        return self.all_tools