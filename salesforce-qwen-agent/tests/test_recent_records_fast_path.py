"""
Regression tests: "recent records" deterministic fast path (Issue 1/6).

"Show me my recent Accounts" previously spent ~67s in a second (synthesizer)
Qwen call because listRecentSobjectRecords flat results never became reference
tables: format_sf_records_as_markdown only recognized soqlQuery. These tests pin
the fix that routes flat listRecentSobjectRecords results to the same
deterministic fast path as soqlQuery (both the shared formatter and the
Orchestrator's reference-table splitter/synthesizer).

All tests are pure unit tests (no live LLM / Salesforce / MCP).
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import format_sf_records_as_markdown
from agent.multi_agent import Orchestrator, _split_reference_results

_SOQL_SHAPE = {
    "totalSize": 2,
    "records": [
        {"attributes": {"type": "Account"}, "Id": "001a", "Name": "Acme"},
        {"attributes": {"type": "Account"}, "Id": "001b", "Name": "Globex"},
    ],
}

_RECENT_ARRAY_SHAPE = [
    {"attributes": {"type": "Account"}, "Id": "001a", "Name": "Acme"},
    {"attributes": {"type": "Account"}, "Id": "001b", "Name": "Globex"},
]


def test_list_recent_soql_shape_formats_as_markdown():
    md = format_sf_records_as_markdown(
        json.dumps(_SOQL_SHAPE), tool_name="listRecentSobjectRecords"
    )
    assert md is not None
    assert "Acme" in md and "Globex" in md
    assert "| Id | Name |" in md
    assert "**Total: 2 records**" in md


def test_list_recent_top_level_array_formats_as_markdown():
    # The REST /recent fallback returns a bare JSON array, not {totalSize, records}.
    md = format_sf_records_as_markdown(
        json.dumps(_RECENT_ARRAY_SHAPE), tool_name="listRecentSobjectRecords"
    )
    assert md is not None
    assert "Acme" in md
    assert "**Total: 2 records**" in md


def test_list_recent_empty_falls_back_to_synthesis():
    assert (
        format_sf_records_as_markdown(
            json.dumps({"totalSize": 0, "records": []}),
            tool_name="listRecentSobjectRecords",
        )
        is None
    )


def test_list_recent_hierarchical_subquery_is_not_flat():
    hierarchical = {
        "totalSize": 1,
        "records": [{
            "attributes": {"type": "Account"},
            "Id": "001a",
            "Contacts": {"totalSize": 1, "records": [{"Id": "003x"}]},
        }],
    }
    assert (
        format_sf_records_as_markdown(
            json.dumps(hierarchical), tool_name="listRecentSobjectRecords"
        )
        is None
    )


def test_soql_still_formats_and_find_stays_on_synthesis():
    assert (
        format_sf_records_as_markdown(json.dumps(_SOQL_SHAPE), tool_name="soqlQuery")
        is not None
    )
    # find / SOSL stays on Qwen synthesis (not in the flat-list tool set).
    assert (
        format_sf_records_as_markdown(
            json.dumps(_RECENT_ARRAY_SHAPE), tool_name="find"
        )
        is None
    )


def test_split_reference_results_routes_recent_records_to_ref_tables():
    ref_tables, raw_remainder = _split_reference_results([
        {"tool": "listRecentSobjectRecords", "result": json.dumps(_SOQL_SHAPE)},
    ])
    assert len(ref_tables) == 1
    assert raw_remainder == []


def test_orchestrator_synthesis_skips_qwen_for_recent_records_only():
    llm = MagicMock()
    llm.last_finish_reason = None
    orch = Orchestrator(llm=llm, executor=object())
    result = asyncio.run(orch._synthesize_response(
        "Show me my recent Accounts",
        [{"tool": "listRecentSobjectRecords", "result": json.dumps(_SOQL_SHAPE)}],
    ))
    deterministic = format_sf_records_as_markdown(
        json.dumps(_SOQL_SHAPE), tool_name="listRecentSobjectRecords"
    )
    assert result == deterministic
    llm.chat.assert_not_called()


def test_orchestrator_synthesis_still_uses_qwen_when_a_data_tool_remains():
    # A related-record result in the same turn (no flat table) must NOT bypass
    # synthesis — the deterministic reference table alone is not a complete answer.
    llm = MagicMock()
    llm.last_finish_reason = None
    llm.chat = AsyncMock(return_value="Final synthesized answer.")
    orch = Orchestrator(llm=llm, executor=object())
    result = asyncio.run(orch._synthesize_response(
        "Show my recent Accounts with contacts",
        [
            {"tool": "listRecentSobjectRecords", "result": json.dumps(_SOQL_SHAPE)},
            {"tool": "getRelatedRecords", "result": json.dumps({
                "totalSize": 1,
                "records": [{"attributes": {"type": "Contact"}, "Id": "003x", "Name": "Jane"}],
            })},
        ],
    ))
    assert result == "Final synthesized answer."
    llm.chat.assert_called_once()