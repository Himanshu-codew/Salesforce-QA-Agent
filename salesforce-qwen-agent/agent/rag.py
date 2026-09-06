"""
RAG (Retrieval-Augmented Generation) Tool Retriever.

PRODUCTION-SAFE lightweight tool-intent retrieval pipeline:

    User Query
      -> Tokenize + normalize (stopwords, lightweight singular stemming)
      -> Rare-term-weighted lexical overlap vs curated tool trigger documents
      -> Coverage score per tool (no embedding model, no vector DB)
      -> Top-K relevant tool definitions (deduplicated by tool)
      -> Relevant Salesforce tool definitions (OpenAI function-calling format)

Why not an embedding model?
    The previous pipeline used sentence-transformers (paraphrase-multilingual-
    MiniLM-L12-v2) + ChromaDB. Measuring fresh in Python (ctypes WorkingSet):

        baseline                         ~  16 MB
        + import torch                   ~ 202 MB
        + import sentence_transformers   ~ 454 MB
        + create model                   ~ 769 MB
        + encode 1 query                 ~ 856 MB
        + encode 15 doc chunks           ~ 928 MB
        + add to ChromaDB                ~ 961 MB
        after gc.collect (model kept)    ~ 593 MB   <-- still above 512 MB

    That exceeds Render's 512 MB memory limit even after garbage collection, and
    requirements.txt does not even ship torch / sentence-transformers / chromadb.
    Fine-tuning a smaller English-only model was rejected earlier because no single
    RAG_MIN_SCORE threshold can both include the required tool for a real SOQL query
    AND exclude greetings like "hi"/"help me" (quality regressed both ways).

    This module therefore uses a DETERMINISTIC, dependency-free lexical tool-intent
    retriever built on the same curated trigger documents (see _chunks_for_tool).
    It keeps every RAG safety property (threshold, top-k, read-only bias, easy
    fallback) and adds zero heavy imports, so the first real RAG request stays far
    below the 512 MB budget.

Configuration (environment variables, no secrets):
    RAG_TOP_K          number of candidate tools to retrieve (default 5)
    RAG_MIN_SCORE      minimum relative coverage score for selection (default 0.18)
    ENABLE_RAG_TOOLS   set 'false' to disable retrieval (returns all tools)
    RAG_WARMUP_ON_STARTUP  accepted for backward compat; warm-up is now trivially cheap

Selection is purely lexical-semantic across the curated trigger documents. If
nothing meets the RAG_MIN_SCORE threshold, no tools are returned.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
from typing import Any

from tools.salesforce import (
    get_tool_definitions,
    is_read_only,
    is_mutating,
    is_destructive,
    READ_ONLY_TOOLS as _READ_ONLY_TOOLS,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Lightweight lexical scoring (deterministic, no heavy deps)
# ──────────────────────────────────────────────────────────────

# A tool must contribute at least this much weighted term mass to be selected.
# This filters out single-common-word noise (e.g. `createSobjectRecord` matching
# only "account" on a read-only "show me accounts" query) while still letting a
# rare, decisive term (e.g. "soql", "update", "delete", "recent") select a tool.
_MIN_MATCHED_WEIGHT = 2.0

# When a query contains at least one DECISIVE signal term (a rare/informative
# word, weight >= _DECISIVE_WEIGHT), the matched-weight bar is raised so tools
# riding purely on common words (e.g. "show"+"account") cannot crowd out the
# decisive match. Ambiguous common-word-only queries (e.g. "show my accounts")
# keep the lenient floor so the correct listing tools still surface.
_DECISIVE_WEIGHT = 2.0
_MIN_MATCHED_WEIGHT_FOR_DECISIVE_QUERY = 3.0

_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "am", "an", "and", "any",
    "are", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "during", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "i", "if", "in", "into", "is", "it",
    "its", "itself", "me", "might", "more", "most", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "so", "some", "such", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would",
    "you", "your", "yours", "yourself", "yourselves",
})


def _stem_lite(word: str) -> str:
    """Very light singular-plural folding. Keeps proper nouns and odd words intact."""
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ss") or word.endswith("us") or word.endswith("is"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _tokenize(text: str) -> list[str]:
    """Tokenize + normalize. Pure digits and short/noise tokens are dropped."""
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    out: list[str] = []
    for token in tokens:
        if token in _STOPWORDS:
            continue
        if len(token) < 3:
            continue
        if token.isdigit():
            continue
        out.append(_stem_lite(token))
    return out


def _term_weights(index: dict[str, set[str]]) -> dict[str, float]:
    """Rare terms get more weight (IDF-flavoured), so specific signals dominate."""
    n_tools = max(len(index), 1)
    df: dict[str, int] = {}
    for terms in index.values():
        for term in terms:
            df[term] = df.get(term, 0) + 1
    return {
        term: 1.0 + math.log((n_tools + 1) / (1.0 + count))
        for term, count in df.items()
    }


# ──────────────────────────────────────────────────────────────
# Module-level shared resources: built once per process, reused.
# ──────────────────────────────────────────────────────────────

# Cached tool definitions: built once, shared by all ToolRAGRetriever instances
# so N sessions do not each hold their own copy.
_cached_tool_definitions: list[dict[str, Any]] | None = None
_cached_tool_map: dict[str, Any] | None = None

_signal_lock = threading.Lock()
_signal_cache: dict[str, Any] = {"sig": None, "index": None, "weights": None}


def _build_signal_index(tool_defs: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, float]]:
    """Build tool -> normalized-term set + term weights from curated trigger docs."""
    index: dict[str, set[str]] = {}
    for doc in _build_documents(tool_defs):
        tool = doc["tool"]
        terms = set(_tokenize(doc["document"]))
        index.setdefault(tool, set()).update(terms)
    return index, _term_weights(index)


def _ensure_signal_index(
    tool_defs: list[dict[str, Any]],
) -> tuple[dict[str, set[str]], dict[str, float]]:
    sig = tuple(t["function"]["name"] for t in tool_defs)
    with _signal_lock:
        if _signal_cache["sig"] != sig:
            index, weights = _build_signal_index(tool_defs)
            _signal_cache.update(sig=sig, index=index, weights=weights)
        return _signal_cache["index"], _signal_cache["weights"]


# ──────────────────────────────────────────────────────────────
# Knowledge base: meaningful searchable documents per tool
# ──────────────────────────────────────────────────────────────

def _chunks_for_tool(definition: dict[str, Any]) -> list[tuple[str, str]]:
    """Build searchable (chunk_label, text) documents for one tool definition."""
    fn = definition["function"]
    name = fn["name"]

    chunks: list[tuple[str, str]] = []

    if name == "soqlQuery":
        chunks.extend([
            ("soql", (
                "soqlQuery executes a Salesforce SOQL SELECT query to read, filter, "
                "search or count Salesforce records from objects like Account, Opportunity, "
                "Lead, Contact, Case, Task, Event. Use it when the user wants to query, "
                "show, select, list, search, count, filter, find, or see ALL records of an "
                "object, such as: show me all Accounts, list all Accounts, display every "
                "Opportunity, get all Leads, show me Accounts, tell me about Accounts. "
                "Example: SELECT Id, Name, Industry, Type FROM Account. Supports WHERE, "
                "ORDER BY, LIMIT, COUNT, GROUP BY, HAVING. Raw numbers only, no $ or commas."
            )),
            ("filter", (
                "soqlQuery filters records in Salesforce using a WHERE clause, for example "
                "select records where amount exceeds 50000, where status equals Closed Won, "
                "where created date is after a specific date, or where a field is not null. "
                "Use for queries with numeric comparisons, text matching, null checks, date filters."
            )),
            ("count", (
                "soqlQuery counts Salesforce records using COUNT(Id) or COUNT(), for example "
                "how many leads, how many accounts, total opportunities. Use for aggregate "
                "counting and summary queries like how many, total number, count of."
            )),
            ("all-records", (
                "soqlQuery lists or shows ALL records of a Salesforce object (read-only). "
                "When the user wants to see every record or all records of an object such as "
                "Account without specifying criteria, use soqlQuery with a query like "
                "SELECT Id, Name, Industry, Type FROM Account LIMIT 200. This is the correct "
                "read-only tool for: show all Accounts, list all accounts, every account, "
                "all opportunities, all leads, all cases, all contacts, see the accounts."
            )),
        ])
    elif name == "getRelatedRecords":
        chunks.append(("related", (
            "getRelatedRecords retrieves child records related to a parent Salesforce record "
            "by relationship name. Use for: list the Contacts under an Account, list the "
            "Opportunities under an Account, list Cases for a Contact, show related child "
            "records of a parent object, get all linked records of an entity. Requires the "
            "parent object name, parent record ID, and relationship path."
        )))
    elif name == "listRecentSobjectRecords":
        chunks.append(("recent", (
            "listRecentSobjectRecords returns recently viewed or recently modified records of "
            "a Salesforce object like Account, Lead, Contact, Opportunity, Case. Use when the "
            "user asks: show me my recent records, what did I view last, recent accounts, "
            "most recently modified leads, last viewed opportunities, meri pichli records."
        )))
    elif name == "find":
        chunks.append(("find", (
            "find executes a Salesforce SOSL full-text search across multiple objects in all "
            "fields. Use for: find Acme company, search for a person name across contacts and "
            "leads, lookup a term, find a product, search text across the org. Returns matching "
            "records from all searchable objects."
        )))
    elif name == "getUserInfo":
        chunks.extend([
            ("user", (
                "getUserInfo returns the current logged-in Salesforce user profile including "
                "user ID, display name, email, username, role, and identity URL. Use when the "
                "user asks: who am I, what is my profile, my email, my role, my username, who "
                "is logged in, am I an admin, show me my account details, what is my identity, "
                "what is my user information, get my user info, what is my account information, "
                "current Salesforce user profile."
            )),
        ])
    elif name == "getObjectSchema":
        chunks.append(("schema", (
            "getObjectSchema returns the schema and metadata of a Salesforce object: all "
            "available fields, their data types, whether they are required, picklist values, "
            "and field relationships. Use when: describe an object's fields, show me the schema "
            "of Account, what fields does Lead have, which fields are mandatory, list required "
            "fields, what data types are used, show me the picklist values."
        )))
    elif name == "createSobjectRecord":
        chunks.extend([
            ("create", (
                "createSobjectRecord creates a new Salesforce record for an object like Account, "
                "Contact, Lead, Opportunity, Case, Task, or Campaign. Use when the user wants to "
                "create, add, insert, make, generate, start, or new record. Requires object name "
                "and a body with field values. Company is mandatory for Leads. Account lookup for "
                "Contacts is via AccountId."
            )),
        ])
    elif name == "updateSobjectRecord":
        chunks.append(("update", (
            "updateSobjectRecord updates an existing Salesforce record by its ID with new field "
            "values. Use when the user wants to update, edit, change, modify, set, patch, "
            "or save new values to a record: change a contact's phone, update a lead status, "
            "modify an opportunity stage, set a priority, update a task status."
        )))
    elif name == "updateRelatedRecord":
        chunks.append(("update-related", (
            "updateRelatedRecord updates a child record by navigating from a parent record "
            "through a relationship path. Use for editing child records that are linked to a "
            "parent: update a Contact belonging to an Account, change the status of a Case "
            "under an Account, modify an Opportunity under a parent Account."
        )))
    elif name == "deleteSobjectRecord":
        chunks.append(("delete", (
            "deleteSobjectRecord permanently deletes a Salesforce record by its object name "
            "and ID. Use when the user wants to delete, remove, erase, drop, or destroy a "
            "record: delete this lead, remove that account, erase the oldest case, discard "
            "the record, mitao that lead."
        )))
    elif name == "deleteRelatedRecord":
        chunks.append(("delete-related", (
            "deleteRelatedRecord deletes a child record that is related to a parent record "
            "through a relationship path. Use for deleting child records: delete contacts "
            "under an account, remove related cases, delete opportunities under a parent."
        )))
    elif name == "uploadRecordAttachment":
        chunks.append(("upload", (
            "uploadRecordAttachment uploads and attaches a file or document to a Salesforce "
            "record via ContentVersion. Use when: attach a file to this record, upload this "
            "document to an Account, add a file attachment to a Lead, upload a PDF to an "
            "Opportunity, attach this content to a case."
        )))

    return chunks


def _build_documents(tool_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Turn tool definitions into a flat list of searchable chunk documents."""
    docs: list[dict[str, Any]] = []
    for definition in tool_defs:
        name = definition["function"]["name"]
        for label, text in _chunks_for_tool(definition):
            if not text or not text.strip():
                continue
            doc_id = f"{name}::{label}"
            docs.append({
                "id": doc_id,
                "tool": name,
                "chunk": label,
                "document": text,
                "definition": definition,
            })
    return docs


# ──────────────────────────────────────────────────────────────
# Warm-up (cold-start prevention)
# ──────────────────────────────────────────────────────────────

def warm_up():
    """
    Pre-compute the lightweight tool-intent index (pure Python; no model).

    Memory-safe: builds a small term -> tool index from the curated tool
    trigger documents (a few KB). There is NO embedding model to preload; the
    heavyweight sentence-transformers/torch stack is intentionally NOT used in
    production because it spiked RSS to ~970 MB, far above Render's 512 MB
    budget. This runs in microseconds and absorbs the build at startup so the
    first real RAG request never pays a cold-load cost.
    """
    try:
        tool_defs = get_tool_definitions()
        _ensure_signal_index(tool_defs)
        logger.info(
            "[RAG] Lightweight tool-intent index ready: "
            f"({len(tool_defs)} tools, no embedding model loaded; production-safe RSS)."
        )
        return True
    except Exception as e:
        logger.warning(f"[RAG] Warm-up failed (will retry on first request): {e}")
        return False


# ──────────────────────────────────────────────────────────────
# Public retriever
# ──────────────────────────────────────────────────────────────

class ToolRAGRetriever:
    """
    Lightweight tool-intent retriever (no embedding model, no vector DB).

    Retrieves the top-K relevant Salesforce tool definitions for a user query
    using rare-term-weighted lexical overlap against the curated trigger
    documents, subject to a configurable coverage threshold. If nothing is
    sufficiently relevant, returns no tools.
    """

    def __init__(self, default_top_k: int = 5, min_confidence: float = 0.18):
        global _cached_tool_definitions, _cached_tool_map
        # Share a single copy of tool definitions across all retriever instances
        # to avoid duplicating ~25-30 tool schemas per session.
        if _cached_tool_definitions is None:
            _cached_tool_definitions = get_tool_definitions()
            _cached_tool_map = {t["function"]["name"]: t for t in _cached_tool_definitions}
        self.all_tools = _cached_tool_definitions
        self.tool_map = _cached_tool_map
        self.default_top_k = int(os.getenv("RAG_TOP_K", str(default_top_k)))
        self.min_score = float(os.getenv("RAG_MIN_SCORE", str(min_confidence)))

    # -- helpers ------------------------------------------------------------

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

    # -- retrieval ------------------------------------------------------------

    def get_relevant_tools(self, user_query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Retrieve top-K relevant tool definitions matching the user query.

        Returns a list of tool definitions in OpenAI function-calling format.
        Returns an empty list when no retrieved tool meets the RAG_MIN_SCORE
        coverage threshold (or its minimum-weight floor).
        """
        total_k = top_k or self.default_top_k

        if os.getenv("ENABLE_RAG_TOOLS", "true").lower() in ("false", "0", "no"):
            logger.debug("[RAG DEBUG] RAG disabled via ENABLE_RAG_TOOLS; returning all tools.")
            return self.all_tools

        actual_query = self._extract_actual_query(user_query)
        if not actual_query or len(actual_query.strip()) < 2:
            logger.debug("[RAG DEBUG] Query too short; returning no tools.")
            return []

        try:
            index, weights = _ensure_signal_index(self.all_tools)

            # Known terms: query tokens that appear in at least one tool document.
            qset = [t for t in set(_tokenize(actual_query)) if t in weights]
            if not qset:
                logger.debug(f"[RAG DEBUG] Query: {actual_query!r}")
                logger.debug("[RAG DEBUG] No known signal terms; returning no tools.")
                logger.debug("[RAG DEBUG] Selection source: lightweight_lexical_tool_retrieval")
                logger.info("🔍 RAG Retriever: Selected tools: []")
                return []

            denominator = sum(weights[t] for t in qset)

            # Weighted term-overlap per tool (denominator shared across tools).
            ranked: list[tuple[str, float]] = []
            for tool, terms in index.items():
                matched = sum(weights[t] for t in terms & set(qset))
                if matched > 0:
                    ranked.append((tool, matched))
            ranked.sort(key=lambda kv: (-kv[1], kv[0]))

            # Apply coverage threshold + a dynamic matched-weight floor, then cap.
            # If the query has a decisive signal term, demand stronger evidence so
            # common-word hangers-on cannot crowd out the decisive match.
            decisive_query = any(weights[t] >= _DECISIVE_WEIGHT for t in qset)
            floor = (
                _MIN_MATCHED_WEIGHT_FOR_DECISIVE_QUERY if decisive_query
                else _MIN_MATCHED_WEIGHT
            )
            filtered = [
                (tool, matched)
                for tool, matched in ranked
                if (matched / denominator) >= self.min_score and matched >= floor
            ]
            # The natural (score-first) top-K before any read-only preference is
            # applied. Used so the bias can re-order without ever evicting a tool
            # that the pure-score ranking genuinely chose.
            selected_names = [tool for tool, _ in filtered[:total_k]]
            selected = [self.tool_map[n] for n in selected_names if n in self.tool_map]

            # ── Read-only safety bias ──
            # For clearly read-only queries (e.g. "show me all Accounts"), never
            # let create/update tools crowd out the correct read-only query tool.
            # This only re-ranks the ALREADY-relevant top-K candidates by giving
            # read-only tools priority; it never manufactures tools or forces a
            # mutating request onto a read tool. Re-ordering happens within the
            # natural top-K, so a compound request like "show ... then update ...
            # then delete ..." keeps its mutation tools in the set.
            actual_lower = actual_query.lower()
            read_only_intent = any(
                kw in actual_lower for kw in (
                    "show", "list", "display", "get ", "see ", "tell me about",
                    "view", "find ", "search", "fetch", "retrieve", "how many",
                    "all accounts", "all leads", "all opportunities", "all cases",
                    "all contacts", "which accounts", "what accounts",
                )
            )
            if read_only_intent and any(
                is_mutating(t) or is_destructive(t) for t in selected_names
            ):
                logger.debug("[RAG DEBUG] Read-only intent detected; re-ranking to prefer read-only tools.")
                ordered = sorted(
                    filtered[:total_k],
                    key=lambda kv: (
                        kv[0] not in _READ_ONLY_TOOLS,  # read-only first
                        -kv[1],                          # then by descending weight
                        kv[0],
                    ),
                )
                selected_names = [tool for tool, _ in ordered]
                selected = [self.tool_map[n] for n in selected_names if n in self.tool_map]
                logger.debug(
                    f"[RAG DEBUG] After read-only bias, selected tools: "
                    f"{[t['function']['name'] for t in selected]}"
                )

            # ── Debug logging (never logs secrets/tokens) ──
            logger.debug(f"[RAG DEBUG] Query: {actual_query!r}")
            logger.debug(f"[RAG DEBUG] Normalized known query terms: {qset}")
            logger.debug(
                "[RAG DEBUG] Retrieval: rare-term-weighted lexical overlap "
                "(no embedding model loaded; production-safe under 512MB)"
            )
            logger.debug("[RAG DEBUG] Scored tools:")
            for tool, matched in ranked:
                rel = matched / denominator if denominator else 0.0
                flag = "SELECTED" if (tool, matched) in filtered else "below-threshold"
                logger.debug(f"[RAG DEBUG]   {tool} weight={matched:.3f} rel={rel:.4f} [{flag}]")
            logger.debug(f"[RAG DEBUG] top_k (distinct tools): {total_k}")
            logger.debug(f"[RAG DEBUG] min_score threshold: {self.min_score}")
            logger.debug(f"[RAG DEBUG] Selected tools: {selected_names}")
            logger.debug("[RAG DEBUG] Selection source: lightweight_lexical_tool_retrieval")
            if not selected:
                logger.debug("[RAG DEBUG] No tools above threshold; returning [].")

            logger.info(f"🔍 RAG Retriever: Selected tools: {selected_names}")
            return selected

        except Exception as e:
            logger.error(
                f"[RAG ERROR] Tool-intent retrieval failed: {e}. Returning no tools."
            )
            return []