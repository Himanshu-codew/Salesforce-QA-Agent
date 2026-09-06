"""
Lightweight RAG production-safety tests.

Guarantees the tool-intent retriever behaves correctly WITHOUT any heavy
dependencies: no torch / sentence-transformers / chromadb may be imported by
the RAG path, greeting/ambiguous queries stay empty, read-only queries never
pick up create/update tools, and results are deterministic.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import agent.rag as rag
from agent.rag import ToolRAGRetriever, warm_up

_HEAVY_MODULES = ("torch", "sentence_transformers", "chromadb")


def _names(tools):
    return [t["function"]["name"] for t in tools]


def _retriever():
    return ToolRAGRetriever(default_top_k=5, min_confidence=0.18)


def test_warm_up_does_not_import_heavy_models():
    ok = warm_up()
    assert ok is True
    for heavy in _HEAVY_MODULES:
        assert heavy not in sys.modules, f"{heavy} must never be imported by the RAG warm-up"


def test_retrieval_does_not_import_heavy_models():
    names = _names(_retriever().get_relevant_tools("Show me my recent Accounts.", top_k=5))
    assert "listRecentSobjectRecords" in names
    for heavy in _HEAVY_MODULES:
        assert heavy not in sys.modules, f"{heavy} must never be imported by tool retrieval"


def test_read_only_query_never_selects_mutation_tools():
    tools = _retriever().get_relevant_tools("Show me all Accounts", top_k=6)
    names = _names(tools)
    assert "soqlQuery" in names
    for mutating in ("createSobjectRecord", "updateSobjectRecord", "updateRelatedRecord",
                     "deleteSobjectRecord", "deleteRelatedRecord"):
        assert mutating not in names, f"read-only query must not select {mutating}"


def test_owner_filtered_query_is_clean_and_read_only():
    # The "show my accounts" style query that previously leaked a mutation tool.
    names = _names(_retriever().get_relevant_tools("show my accounts", top_k=6))
    assert "soqlQuery" in names
    assert "getUserInfo" in names
    for mutating in ("createSobjectRecord", "updateSobjectRecord", "deleteSobjectRecord"):
        assert mutating not in names


def test_compound_mutation_query_keeps_required_mutating_tools():
    names = _names(_retriever().get_relevant_tools(
        "Find the newest Lead and delete it, then create a new Account for whatever company it was from",
        top_k=6,
    ))
    assert "deleteSobjectRecord" in names
    assert "createSobjectRecord" in names


def test_results_are_deterministic():
    query = "Show me Contacts at John Doe, update the phone on the first one to 555-1111, and then delete the oldest Lead"
    first = _names(_retriever().get_relevant_tools(query, top_k=5))
    second = _names(_retriever().get_relevant_tools(query, top_k=5))
    assert first == second and first, "identical inputs must produce identical tool selections"


def test_signal_index_is_cached_and_shared():
    r1 = _retriever()
    r2 = _retriever()
    before = rag._signal_cache["sig"]
    _ = r1.get_relevant_tools("What is my Salesforce user information?", top_k=5)
    _ = r2.get_relevant_tools("What is my Salesforce user information?", top_k=5)
    assert rag._signal_cache["sig"] == before, "signal index must be built exactly once"


def test_query_too_short_returns_empty():
    assert _retriever().get_relevant_tools("x", top_k=5) == []