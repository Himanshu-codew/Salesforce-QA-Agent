"""
RAG (Retrieval-Augmented Generation) Tool Retriever.

REAL semantic retrieval pipeline:

    User Query
      -> Embedding Model (sentence-transformers)
      -> Vector Database  (ChromaDB, in-memory, cosine)
      -> Semantic Similarity Search
      -> Top-K relevant tool documents  (deduplicated by tool)
      -> Relevant Salesforce tool definitions (OpenAI function-calling format)

The tool definitions in `tools/salesforce.py` form the knowledge base. Each tool
is described by rich, searchable documents (purpose, when to use, important
arguments, examples). Embeddings are computed once at process start and stored in
an in-memory ChromaDB collection so they are NOT recomputed on every user request.

Configuration (environment variables, no secrets):
    RAG_EMBEDDING_MODEL   sentence-transformers model id (default multilingual MiniLM)
    RAG_TOP_K             number of candidate documents to retrieve (default 5)
    RAG_MIN_SCORE         minimum cosine similarity for a tool to be selected (default 0.18)
    ENABLE_RAG_TOOLS      set 'false' to disable retrieval (returns all tools)
    RAG_WARMUP_ON_STARTUP set 'true' to preload the model at startup (default: lazy-load)

Selection is purely semantic. No hardcoded keyword rules override relevance.
If nothing meets the `RAG_MIN_SCORE` threshold, no tools are returned.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
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

# DO NOT switch this to a smaller English-only embedding model. We tested
# all-MiniLM-L6-v2, all-MiniLM-L12-v2, and BAAI/bge-small-en-v1.5 as drop-in
# replacements and all of them compress semantic similarity badly for the
# tool-retrieval task: no single RAG_MIN_SCORE threshold can both include the
# required tool for a real SOQL query AND exclude greetings like "hi"/"help me",
# so RAG quality regresses in both directions. The multilingual model below has
# the separation this pipeline needs. To stay within Render's memory budget we
# instead lazy-load it (see _get_embedder) rather than using a degraded model.
_EMBEDDING_MODEL_DEFAULT = "paraphrase-multilingual-MiniLM-L12-v2"
_COLLECTION_NAME = "salesforce_tool_retrieval"

# Module-level shared resources: built once per process, reused across calls.
_embedder = None
_embedder_lock = threading.Lock()
_index_lock = threading.Lock()
_index = None

# Cached tool definitions: built once, shared by all ToolRAGRetriever instances
# so N sessions do not each hold their own copy.
_cached_tool_definitions: list[dict[str, Any]] | None = None
_cached_tool_map: dict[str, Any] | None = None


def _tool_documents_hash(tool_defs: list[dict[str, Any]]) -> str:
    blob = json.dumps(tool_defs, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _get_embedder():
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            model_id = os.getenv("RAG_EMBEDDING_MODEL", _EMBEDDING_MODEL_DEFAULT)
            logger.info(f"[RAG] Loading embedding model: {model_id}")
            from sentence_transformers import SentenceTransformer
            _embedder = SentenceTransformer(model_id)
            logger.info(f"[RAG] Embedding model loaded successfully.")
        return _embedder


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
                "is logged in, am I an admin, show me my account details, what is my identity."
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
    """Turn tool definitions into a flat list of embeddable chunk documents."""
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
    Optionally load the embedding model + build the vector index at startup.

    BY DEFAULT this is a NO-OP: the embedding model (~470MB in torch with the
    multilingual model it needs for good tool-retrieval separation) is NOT loaded
    during cold start. Render's 512MB budget is too tight to reserve that memory
    up front, and doing so was a primary contributor to full-process OOM kills.
    Instead the model + ChromaDB index are built lazily on the FIRST semantic
    retrieval (_get_embedder / _ensure_vector_index). The orchestrator already
    wraps RAG in a bounded timeout and falls back to the complete read-only-safe
    tool registry if that cold load is slow on the first request.

    Set RAG_WARMUP_ON_STARTUP=true to force eager preloading (best when running
    on a host with spare memory and you want to absorb the cold-load on boot
    rather than on the first user request).
    """
    if os.getenv("RAG_WARMUP_ON_STARTUP", "false").lower() not in ("true", "1", "yes"):
        logger.info(
            "[RAG] Lazy-load mode: skipping embedding model at startup "
            "(set RAG_WARMUP_ON_STARTUP=true to preload). Model loads on first RAG use."
        )
        return True
    if os.getenv("ENABLE_RAG_TOOLS", "true").lower() in ("false", "0", "no"):
        logger.info("[RAG] ENABLE_RAG_TOOLS is false; skipping embedding model warm-up.")
        return True
    try:
        tool_defs = get_tool_definitions()
        from utils.memory_diag import log_memory
        log_memory("before RAG embedder load")
        collection, dimension = _ensure_vector_index(tool_defs)
        log_memory("after RAG embedder load")
        logger.info(
            f"[RAG] Warm-up complete: embedding_model='{os.getenv('RAG_EMBEDDING_MODEL', _EMBEDDING_MODEL_DEFAULT)}', "
            f"chroma_collection_count={collection.count()}, embedding_dim={dimension}."
        )
        return True
    except Exception as e:
        logger.warning(f"[RAG] Warm-up failed (will retry on first request): {e}")
        return False


# ──────────────────────────────────────────────────────────────
# Vector database (ChromaDB) initialization
# ──────────────────────────────────────────────────────────────

def _ensure_vector_index(tool_defs: list[dict[str, Any]]) -> tuple[Any, int]:
    """
    Ensure the in-memory ChromaDB index exists and is populated for the current
    tool definitions. Built at most once per process and reused.

    Returns (chroma_collection, embedding_dimension).
    """
    global _index
    with _index_lock:
        tool_hash = _tool_documents_hash(tool_defs)
        if _index is not None and _index.get("hash") == tool_hash:
            return _index["collection"], _index["dimension"]

        import chromadb

        docs = _build_documents(tool_defs)
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        embedder = _get_embedder()
        texts = [d["document"] for d in docs]
        vectors = embedder.encode(texts, show_progress_bar=False).tolist()
        dimension = len(vectors[0]) if vectors else 384

        collection.add(
            ids=[d["id"] for d in docs],
            documents=[d["document"] for d in docs],
            embeddings=vectors,
            metadatas=[{"tool": d["tool"], "chunk": d["chunk"]} for d in docs],
        )

        logger.info(
            f"[RAG] Built ChromaDB index: {len(docs)} documents across {len(tool_defs)} tools, "
            f"embedding_dim={dimension}, collection_count={collection.count()}."
        )

        _index = {
            "collection": collection,
            "dimension": dimension,
            "hash": tool_hash,
            "docs": docs,
        }
        return collection, dimension


# ──────────────────────────────────────────────────────────────
# Public retriever
# ──────────────────────────────────────────────────────────────

class ToolRAGRetriever:
    """
    Real semantic RAG tool retriever.

    Retrieves the top-K relevant Salesforce tool definitions for a user query
    using an embedding model + ChromaDB vector search, subject to a configurable
    similarity threshold. If nothing is sufficiently relevant, returns no tools.
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

    @staticmethod
    def _to_similarity(distances: list[float]) -> list[float]:
        """Convert Chroma cosine distances in [0, 2] to cosine similarities in [-1, 1]."""
        return [1.0 - d for d in distances]

    # -- retrieval ------------------------------------------------------------

    def get_relevant_tools(self, user_query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Retrieve top-K relevant tool definitions matching the user query.

        Returns a list of tool definitions in OpenAI function-calling format.
        Returns an empty list when no retrieved document meets the RAG_MIN_SCORE threshold.
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
            collection, dimension = _ensure_vector_index(self.all_tools)
            docs = _index["docs"] if _index else []

            embedder = _get_embedder()
            query_vec = embedder.encode([actual_query], show_progress_bar=False).tolist()[0]

            # Retrieve a superset of candidate documents so multiple intents in a
            # compound query can surface, then deduplicate by tool.
            n_results = max(1, min(total_k * 3, collection.count()))
            result = collection.query(
                query_embeddings=[query_vec],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )

            ids = result.get("ids", [[]])[0] or []
            distances = result.get("distances", [[]])[0] or []
            metadatas = result.get("metadatas", [[]])[0] or []
            scores = self._to_similarity(distances)

            # Deduplicate by tool, keeping the highest score per tool.
            best: dict[str, tuple[float, str]] = {}
            for i, doc_id in enumerate(ids):
                meta = metadatas[i] if i < len(metadatas) else {}
                tool = (meta or {}).get("tool") or (doc_id.split("::", 1)[0] if doc_id else "")
                score = scores[i] if i < len(scores) else 0.0
                prev = best.get(tool)
                if prev is None or score > prev[0]:
                    best[tool] = (score, doc_id)

            # Rank by score, enforce TOP-K distinct tools, then apply threshold.
            ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
            ranked = ranked[: max(1, total_k)]
            selected_names = [tool for tool, _ in ranked if _[0] >= self.min_score]
            selected = [self.tool_map[n] for n in selected_names if n in self.tool_map]

            # ── Read-only safety bias ──
            # For clearly read-only queries (e.g. "show me all Accounts"), never
            # let create/update tools crowd out the correct read-only query tool.
            # This only re-ranks the ALREADY-relevant top-K candidates by giving
            # read-only tools priority; it never manufactures tools or forces a
            # mutating request onto a read tool.
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
                is_mutating(t["function"]["name"]) or is_destructive(t["function"]["name"])
                for t in selected
            ):
                logger.debug("[RAG DEBUG] Read-only intent detected; re-ranking to prefer read-only tools.")
                # Rank all top-K candidates (already threshold-filtered) read-only first.
                ordered = sorted(
                    ranked,
                    key=lambda kv: (
                        kv[0] not in _READ_ONLY_TOOLS,  # read-only first
                        -kv[1][0],                        # then by descending score
                    ),
                )
                ordered_names = [tool for tool, _ in ordered]
                deduped = list(dict.fromkeys(ordered_names))
                selected = [self.tool_map[n] for n in deduped]
                logger.debug(f"[RAG DEBUG] After read-only bias, selected tools: {[t['function']['name'] for t in selected]}")
                selected_names = [t["function"]["name"] for t in selected]

            # ── Debug logging (never logs secrets/tokens) ──
            logger.debug(f"[RAG DEBUG] Query: {actual_query!r}")
            logger.debug(
                f"[RAG DEBUG] Embedding model: "
                f"{os.getenv('RAG_EMBEDDING_MODEL', _EMBEDDING_MODEL_DEFAULT)}"
            )
            logger.debug(f"[RAG DEBUG] Embedding dimension: {len(query_vec)}")
            logger.debug(f"[RAG DEBUG] Candidate documents: {collection.count()}")
            logger.debug("[RAG DEBUG] Retrieved results:")
            for tool, (score, doc) in best.items():
                logger.debug(f"[RAG DEBUG]   {tool} score={score:.4f}")
            logger.debug(f"[RAG DEBUG] top_k (distinct tools): {total_k}")
            logger.debug(f"[RAG DEBUG] min_score threshold: {self.min_score}")
            logger.debug(f"[RAG DEBUG] Selected tools: {selected_names}")
            logger.debug("[RAG DEBUG] Selection source: semantic_vector_retrieval")
            if not selected:
                logger.debug("[RAG DEBUG] No tools above threshold; returning [].")

            logger.info(f"🔍 RAG Retriever: Selected tools: {selected_names}")
            return selected

        except Exception as e:
            logger.error(
                f"[RAG ERROR] Semantic retrieval failed: {e}. Returning no tools."
            )
            return []