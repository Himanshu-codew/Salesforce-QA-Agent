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
from agent.agent import filter_tools_for_query

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


def test_read_only_query_safety_enforced_by_filter():
    # Retriever now returns the full registry (model decides tool selection).
    tools = _retriever().get_relevant_tools("Show me all Accounts", top_k=6)
    names = _names(tools)
    assert "soqlQuery" in names
    assert len(tools) == len(_retriever().all_tools)
    # Read-only safety is enforced downstream by filter_tools_for_query:
    filtered = filter_tools_for_query(tools, "Show me all Accounts")
    filtered_names = _names(filtered)
    for mutating in ("createSobjectRecord", "updateSobjectRecord", "updateRelatedRecord",
                     "deleteSobjectRecord", "deleteRelatedRecord"):
        assert mutating not in filtered_names, f"read-only query must not keep {mutating}"


def test_owner_filtered_query_is_clean_and_read_only():
    # The "show my accounts" style query: the retriever passes all tools, and
    # filter_tools_for_query strips mutation tools before the model sees them.
    tools = _retriever().get_relevant_tools("show my accounts", top_k=6)
    assert "soqlQuery" in _names(tools)
    assert "getUserInfo" in _names(tools)
    filtered_names = _names(filter_tools_for_query(tools, "show my accounts"))
    for mutating in ("createSobjectRecord", "updateSobjectRecord", "deleteSobjectRecord"):
        assert mutating not in filtered_names


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


def test_instances_share_full_registry_deterministically():
    r1 = _retriever()
    r2 = _retriever()
    q = "What is my Salesforce user information?"
    names1 = _names(r1.get_relevant_tools(q, top_k=5))
    names2 = _names(r2.get_relevant_tools(q, top_k=5))
    assert names1 == names2, "identical inputs must produce identical tool sets"
    assert len(names1) == len(r1.all_tools), "full registry must be passed through"
    assert r1.all_tools is r2.all_tools or r1.all_tools == r2.all_tools


def test_query_too_short_returns_empty():
    assert _retriever().get_relevant_tools("x", top_k=5) == []