"""Real semantic RAG acceptance tests.

Verifies ToolRAGRetriever returns relevant tool definitions via embedding +
vector retrieval, and returns [] for unrelated/ambiguous queries.
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


def test_user_info_query_selects_getuserinfo():
    names = _names(_retriever().get_relevant_tools("What is my Salesforce user information?", top_k=5))
    assert names, "expected at least one tool"
    assert names[0] == "getUserInfo"


def test_recent_records_query_selects_listrecent():
    names = _names(_retriever().get_relevant_tools("Show me my recent Accounts.", top_k=5))
    assert names, "expected at least one tool"
    assert names[0] == "listRecentSobjectRecords"


def test_custom_soql_query_selects_soqlquery():
    names = _names(_retriever().get_relevant_tools("Find Opportunities where Amount is greater than 50000.", top_k=5))
    assert "soqlQuery" in names


def test_unrelated_query_returns_empty():
    names = _names(_retriever().get_relevant_tools("What is the weather in London?", top_k=5))
    assert names == []


def test_ambiguous_query_returns_empty():
    names = _names(_retriever().get_relevant_tools("Help me with something.", top_k=5))
    assert names == []


def test_greeting_returns_empty():
    names = _names(_retriever().get_relevant_tools("hi", top_k=5))
    assert names == []
