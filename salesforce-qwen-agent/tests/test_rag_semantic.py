"""RAG passthrough acceptance tests.

Tool subset selection is intentionally disabled: ToolRAGRetriever returns the
complete Salesforce tool registry so the model (Qwen) decides which tool to
call. Read-only safety is enforced separately by filter_tools_for_query in
agent.agent / agent.multi_agent. These tests pin that behavior (no heavy
imports, full registry returned, trivially short queries still return []).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.rag import ToolRAGRetriever


def _names(tools):
    return [t["function"]["name"] for t in tools]


def _retriever():
    # Fresh instance per test so config/env changes do not leak.
    return ToolRAGRetriever(default_top_k=5, min_confidence=0.18)


def test_all_tools_returned_for_user_info_query():
    tools = _retriever().get_relevant_tools("What is my Salesforce user information?", top_k=5)
    names = _names(tools)
    assert names, "expected the full registry"
    assert "getUserInfo" in names
    assert len(names) == len(_retriever().all_tools)


def test_all_tools_returned_for_recent_records_query():
    names = _names(_retriever().get_relevant_tools("Show me my recent Accounts.", top_k=5))
    assert names, "expected the full registry"
    assert "listRecentSobjectRecords" in names
    assert len(names) == len(_retriever().all_tools)


def test_soqlquery_present_in_custom_query():
    names = _names(_retriever().get_relevant_tools("Find Opportunities where Amount is greater than 50000.", top_k=5))
    assert "soqlQuery" in names


def test_full_registry_returned_for_unrelated_query():
    names = _names(_retriever().get_relevant_tools("What is the weather in London?", top_k=5))
    assert len(names) == len(_retriever().all_tools)


def test_full_registry_returned_for_ambiguous_query():
    names = _names(_retriever().get_relevant_tools("Help me with something.", top_k=5))
    assert len(names) == len(_retriever().all_tools)


def test_trivially_short_query_returns_empty():
    names = _names(_retriever().get_relevant_tools("x", top_k=5))
    assert names == []
