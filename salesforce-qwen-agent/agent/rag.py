"""
RAG (Retrieval-Augmented Generation) Tool Retriever.
Fast, offline vector similarity search over Salesforce MCP tool definitions.
Selects top-K relevant tools dynamically based on user intent with confidence thresholding.
"""

import logging
import math
import os
import re
from typing import Any

from tools.salesforce import get_tool_definitions

logger = logging.getLogger(__name__)


TOOL_SYNONYMS = {
    "createSobjectRecord": "create add insert new make generate lead account contact opportunity case task record banao daalo naya new record",
    "updateSobjectRecord": "update edit change modify patch save record badlo change karo set status stage",
    "deleteSobjectRecord": "delete remove erase drop destroy record hatao delete karo mitao discard purge",
    "listRecentSobjectRecords": "recent recently viewed accounts leads contacts opportunities cases show my list last meri aakhri pichle",
    "soqlQuery": "select query find search get list show records soql count how many kitne dikhao top highest lowest order group filter where closed won all saare naye",
    "getObjectSchema": "schema fields describe metadata mandatory required picklist type datatype structure konse fields column columns",
    "getRelatedRecords": "related child parent relationships contacts cases opportunities notes tasks under of ke saare linked associated",
    "find": "find search text lookup sosl dhundo khojo across everywhere all objects phrase term",
    "getUserInfo": "user info whoami who am i me myself my profile email identity role username organization org logged login admin mera kaun meri details user details",
    "updateRelatedRecord": "update edit change modify related child parent relationship update contact of update case of",
    "deleteRelatedRecord": "delete remove related child parent relationship hatao delete contacts of delete cases of",
    "uploadRecordAttachment": "upload attach file document pdf image content version add attachment upload file to record jod do daal do",
}


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, splitting camelCase identifiers."""
    # Split camelCase e.g., createSobjectRecord -> create Sobject Record
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"\w+", text.lower())


def compute_vector(tokens: list[str], vocab: dict[str, int]) -> list[float]:
    """Compute TF vector for a list of tokens given a vocabulary."""
    vec = [0.0] * len(vocab)
    for t in tokens:
        if t in vocab:
            vec[vocab[t]] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two normalized vectors."""
    return sum(a * b for a, b in zip(vec1, vec2))


class ToolRAGRetriever:
    """
    RAG Tool Retriever with Confidence Thresholding & Fallback.
    Serves tools based on query intent. If confidence is low or if query is conversational
    (e.g., 'hi', 'how are you'), falls back to serving all tools.
    """

    def __init__(self, default_top_k: int = 5, min_confidence: float = 0.12):
        self.all_tools = get_tool_definitions()
        self.tool_map = {t["function"]["name"]: t for t in self.all_tools}
        self.default_top_k = default_top_k
        self.min_confidence = min_confidence
        self._index_tools()

    def _index_tools(self) -> None:
        """Build vector vocabulary and tool embeddings."""
        self.documents = {}
        all_words = set()

        for t in self.all_tools:
            name = t["function"]["name"]
            desc = t["function"].get("description", "")
            params = " ".join(t["function"].get("parameters", {}).get("properties", {}).keys())
            synonyms = TOOL_SYNONYMS.get(name, "")
            doc_text = f"{name} {desc} {params} {synonyms}"
            tokens = tokenize(doc_text)
            self.documents[name] = tokens
            all_words.update(tokens)

        self.vocab = {word: i for i, word in enumerate(sorted(all_words))}
        self.vectors = {
            name: compute_vector(tokens, self.vocab)
            for name, tokens in self.documents.items()
        }
        logger.info(f"✅ RAG ToolRetriever: Indexed {len(self.all_tools)} Salesforce tools.")

    def get_relevant_tools(self, user_query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Retrieve top-K relevant tool definitions matching the user query.
        Falls back to all tools if query is conversational or confidence is low.
        """
        # Enable RAG by default unless explicitly set to false
        if os.getenv("ENABLE_RAG_TOOLS", "true").lower() in ("false", "0", "no"):
            return self.all_tools

        k = top_k or self.default_top_k
        if not user_query or len(user_query.strip()) < 2:
            return []

        # When a file is attached, isolate the actual user instruction/question
        actual_query = user_query
        if "[Attached File:" in user_query:
            if "User Message:" in user_query:
                actual_query = user_query.split("User Message:")[-1].strip()
            else:
                parts = user_query.split("]\n")
                if len(parts) > 1:
                    actual_query = parts[-1].strip()

        target_text = actual_query if actual_query else user_query
        query_tokens = tokenize(target_text)
        query_vec = compute_vector(query_tokens, self.vocab)

        # Calculate similarity scores
        scores = []
        for name, doc_vec in self.vectors.items():
            sim = cosine_similarity(query_vec, doc_vec)
            scores.append((sim, name))

        # Sort by highest similarity
        scores.sort(key=lambda x: x[0], reverse=True)

        top_score = scores[0][0] if scores else 0.0

        # Check for explicit Salesforce intent keywords in the actual query
        q_lower = target_text.lower()
        sf_keywords = {
            "salesforce", "sf", "soql", "sosl", "sobject", "record", "records", "object", "schema", "field", "fields",
            "account", "accounts", "contact", "contacts", "lead", "leads", "opportunity", "opportunities", "opp", "case", "cases", "user",
            "task", "tasks", "event", "events", "campaign", "campaigns",
            "query", "select", "find", "search", "list", "recent", "describe", "get", "show", "view",
            "create", "insert", "add", "new", "update", "edit", "modify", "delete", "remove", "related", "child",
            "who", "am", "i", "my", "me", "profile", "role", "email", "username", "logged", "identity", "admin",
            "banao", "dikhao", "hatao", "daalo", "nikalo", "badlo", "mera", "meri", "kaun", "konse", "kitne", "saare", "sab", "pichle"
        }
        has_sf_intent = any(kw in q_lower for kw in sf_keywords)

        # Explicit check for file attachment action intent (e.g. attach to record, upload to salesforce)
        is_attach_request = any(phrase in q_lower for phrase in [
            "attach to", "attach this file", "attach file to", "upload to record", "upload to account",
            "upload to lead", "upload to opportunity", "upload to case", "upload to contact",
            "salesforce me attach", "salesforce me upload", "record pe attach", "record me attach",
            "attachment banao", "contentversion"
        ])

        # High confidence check: If top score is below threshold AND no Salesforce intent, return 0 tools for ultra-fast response
        if top_score < self.min_confidence and not has_sf_intent and not is_attach_request:
            logger.info(f"⚡ RAG Retriever: General/conversational/document query detected for '{target_text[:30]}...'. Serving 0 tools for max speed.")
            return []

        # Select top-K tool names above confidence
        top_names = [name for sim, name in scores[:k] if sim > 0.05]
        if not top_names:
            top_names = [name for _, name in scores[:k]]

        # ── Intent-Based Safety Guarantees for all 11 MCP Tools ──
        # 1. Create
        if any(w in q_lower for w in ["create", "add", "insert", "new", "make", "generate", "banao", "daalo"]):
            if "createSobjectRecord" not in top_names:
                top_names.append("createSobjectRecord")

        # 2. Update (direct & related)
        if any(w in q_lower for w in ["update", "edit", "change", "modify", "set", "badlo"]):
            if any(w in q_lower for w in ["related", "child", "contact of", "case of", "opportunities under", "under account"]):
                if "updateRelatedRecord" not in top_names:
                    top_names.append("updateRelatedRecord")
            if "updateSobjectRecord" not in top_names:
                top_names.append("updateSobjectRecord")

        # 3. Delete (direct & related)
        if any(w in q_lower for w in ["delete", "remove", "drop", "erase", "hatao", "destroy", "mitao"]):
            if any(w in q_lower for w in ["related", "child", "contact of", "contacts under", "cases under", "opportunities under", "under account"]):
                if "deleteRelatedRecord" not in top_names:
                    top_names.append("deleteRelatedRecord")
            if "deleteSobjectRecord" not in top_names:
                top_names.append("deleteSobjectRecord")

        # 4. Recent
        if any(w in q_lower for w in ["recent", "recently", "viewed", "last viewed", "last contacts", "pichle", "aakhri"]):
            if "listRecentSobjectRecords" not in top_names:
                top_names.append("listRecentSobjectRecords")

        # 5. Search / SOSL
        if any(w in q_lower for w in ["search", "find", "lookup", "sosl", "dhundo", "khojo"]):
            if "find" not in top_names:
                top_names.append("find")

        # 6. User Info
        if any(w in q_lower for w in ["who am i", "my profile", "my email", "my role", "my username", "logged in", "mera account", "meri details", "kaun", "am i admin"]):
            if "getUserInfo" not in top_names:
                top_names.append("getUserInfo")

        # 7. Schema / Describe
        if any(w in q_lower for w in ["schema", "fields", "describe", "metadata", "required", "mandatory", "picklist", "data type", "datatype", "type of", "konse fields"]):
            if "getObjectSchema" not in top_names:
                top_names.append("getObjectSchema")

        # 8. Related Records
        if any(w in q_lower for w in ["related", "child", "contacts of", "cases of", "opportunities of", "under account", "ke saare", "linked to", "associated"]):
            if "getRelatedRecords" not in top_names:
                top_names.append("getRelatedRecords")

        # 9. SOQL Query (show, select, count, how many, all, etc.)
        if any(w in q_lower for w in ["select", "show", "query", "how many", "count", "list", "top", "highest", "closed won", "filter", "where", "dikhao", "saare leads", "saare accounts"]):
            if "soqlQuery" not in top_names:
                top_names.append("soqlQuery")

        # 10. File Upload / Attachment (only when explicit attachment command)
        if is_attach_request:
            if "uploadRecordAttachment" not in top_names:
                top_names.append("uploadRecordAttachment")

        retrieved = [self.tool_map[name] for name in top_names if name in self.tool_map]
        logger.info(f"🔍 RAG Retriever: Selected tools: {[t['function']['name'] for t in retrieved]}")
        return retrieved
