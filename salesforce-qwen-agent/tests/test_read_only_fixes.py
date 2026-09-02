"""
Focused unit tests for the audit-mandated fixes (OFFLINE — no Salesforce/live calls).

Fix A — filter_tools_for_query: for purely read-only requests, mutation/destructive
        tools are removed from the tool schema passed to Qwen; soqlQuery and other
        read tools remain; genuine write/compound requests keep the full tool set;
        READ_ONLY_MODE module flags are not touched.

Fix B — format_sf_records_as_markdown: a COUNT query that returns
        {"totalSize": 0, "records": []} yields the project count line (non-None)
        so the direct-response fast path triggers and no second LLM call is made.
        Non-zero COUNT, flat list tables, hierarchical results, and errors keep
        their existing behavior.
"""

import json

import agent.agent as agent_module
import agent.planner as planner_module
import sfmcp.executor as executor_module
from agent.agent import format_sf_records_as_markdown, filter_tools_for_query
from tools.salesforce import DESTRUCTIVE_TOOLS, MUTATING_TOOLS, READ_ONLY_TOOLS


def _tool(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


_READ_TOOLS = ["soqlQuery", "listRecentSobjectRecords", "getRelatedRecords", "getUserInfo"]
_MUTATION_TOOLS = ["createSobjectRecord", "updateRelatedRecord", "deleteSobjectRecord"]


def _tool_names(tools: list[dict]) -> list[str]:
    return [t["function"]["name"] for t in tools]


class TestFilterToolsForQuery:
    def setup_method(self):
        self.mock_rag_tools = [_tool(n) for n in _READ_TOOLS + _MUTATION_TOOLS]

    def test_mock_tool_names_match_real_categories(self):
        assert MUTATING_TOOLS >= {"createSobjectRecord", "updateRelatedRecord"}
        assert DESTRUCTIVE_TOOLS >= {"deleteSobjectRecord"}
        assert set(_READ_TOOLS) <= READ_ONLY_TOOLS

    def test_pure_read_query_removes_mutation_tools(self):
        filtered = filter_tools_for_query(
            self.mock_rag_tools, "How many Account records do we have?"
        )
        names = _tool_names(filtered)
        assert "soqlQuery" in names
        assert "listRecentSobjectRecords" in names
        assert "getRelatedRecords" in names
        assert "getUserInfo" in names
        assert "createSobjectRecord" not in names
        assert "updateRelatedRecord" not in names
        assert "deleteSobjectRecord" not in names

    def test_simple_show_or_list_query_is_filtered(self):
        for msg in ["Show me all Accounts", "List 10 Latest Contacts"]:
            assert not any(
                n in MUTATING_TOOLS or n in DESTRUCTIVE_TOOLS
                for n in _tool_names(filter_tools_for_query(self.mock_rag_tools, msg))
            )

    def test_compound_write_request_keeps_mutation_tools(self):
        filtered = filter_tools_for_query(
            self.mock_rag_tools, "Create a new Account and show all Contacts"
        )
        assert "createSobjectRecord" in _tool_names(filtered)

    def test_update_intent_keeps_mutation_tools(self):
        filtered = filter_tools_for_query(
            self.mock_rag_tools, "Update the biggest Opportunity and list Cases"
        )
        assert "updateRelatedRecord" in _tool_names(filtered)

    def test_delete_or_remove_intent_keeps_mutation_tools(self):
        filtered = filter_tools_for_query(self.mock_rag_tools, "Delete 5 Leads")
        assert "deleteSobjectRecord" in _tool_names(filtered)

    def test_show_not_superseded_by_read_word(self):
        # "show" must NOT override a write intent in the same message.
        filtered = filter_tools_for_query(
            self.mock_rag_tools, "Show all accounts, then remove the oldest one"
        )
        assert "deleteSobjectRecord" in _tool_names(filtered)

    def test_existing_intent_helper_reused(self):
        # The filter uses the agent's own write-intent classifier — a pure read
        # message must be classified read-only by that same helper.
        assert not agent_module._has_write_intent("How many Account records do we have?")
        assert agent_module._has_write_intent("Create an Account")

    def test_read_only_mode_unchanged(self):
        planner_before = planner_module.READ_ONLY_MODE
        executor_before = executor_module.READ_ONLY_MODE
        filter_tools_for_query(
            self.mock_rag_tools, "How many Account records do we have?"
        )
        assert planner_module.READ_ONLY_MODE == planner_before
        assert executor_module.READ_ONLY_MODE == executor_before


class TestZeroCountFastPath:
    def test_zero_count_produces_project_count_line(self):
        md = format_sf_records_as_markdown(
            json.dumps({"totalSize": 0, "records": []}),
            "soqlQuery",
            soql_query="SELECT COUNT(Id) FROM Account",
        )
        assert md == "**Total Count:** 0"

    def test_zero_count_not_none_enables_direct_path(self):
        md = format_sf_records_as_markdown(
            '{"totalSize":0,"records":[]}',
            "soqlQuery",
            soql_query="SELECT COUNT() FROM Account",
        )
        assert md is not None

    def test_nonzero_count_regression(self):
        md = format_sf_records_as_markdown(
            json.dumps({
                "totalSize": 1,
                "records": [{"attributes": {"type": "AggregateResult"}, "expr0": 60}],
            }),
            "soqlQuery",
            soql_query="SELECT COUNT(Id) FROM Account",
        )
        assert md == "**Total Count:** 60"

    def test_nonzero_count_works_without_soql_arg(self):
        # Legacy callers (no soql_query arg) keep working via expr0.
        assert format_sf_records_as_markdown(
            json.dumps({"totalSize": 1, "records": [{"expr0": 7}]}), "soqlQuery"
        ) == "**Total Count:** 7"

    def test_flat_zero_list_non_count_returns_none(self):
        md = format_sf_records_as_markdown(
            json.dumps({"totalSize": 0, "records": []}),
            "soqlQuery",
            soql_query="SELECT Id, Name FROM Account LIMIT 10",
        )
        assert md is None

    def test_flat_zero_list_without_query_returns_none(self):
        # Legacy zero-list behavior is preserved when the query is unknown.
        assert format_sf_records_as_markdown(
            '{"totalSize":0,"records":[]}', "soqlQuery"
        ) is None

    def test_nonzero_flat_list_returns_table(self):
        flat = json.dumps({
            "totalSize": 1,
            "records": [{"attributes": {"type": "Account"}, "Id": "001x", "Name": "Acme"}],
        })
        md = format_sf_records_as_markdown(
            flat, "soqlQuery", soql_query="SELECT Id, Name FROM Account LIMIT 10"
        )
        assert md is not None
        assert "Acme" in md

    def test_hierarchical_result_returns_none(self):
        sub = json.dumps({
            "totalSize": 1,
            "records": [{
                "attributes": {"type": "Account", "url": "/services/data/v60.0/sobjects/Account/001x"},
                "Id": "001x",
                "Contacts": {"totalSize": 2, "records": []},
            }],
        })
        assert format_sf_records_as_markdown(sub, "soqlQuery") is None

    def test_non_soql_tool_returns_none(self):
        assert format_sf_records_as_markdown(
            '{"totalSize":0,"records":[]}', "getUserInfo"
        ) is None

    def test_invalid_json_returns_none(self):
        assert format_sf_records_as_markdown("not json", "soqlQuery") is None