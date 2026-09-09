"""
Response-fidelity regression tests for multi-result read-only queries.

The reported defect: a compound query such as "Show me all Accounts AND tell me
how many Leads I have" could be handed to the synthesizer LLM, which re-rendered
the tables from raw JSON — producing a fused "| IdNameIndustry |" header and
labeling a COUNT of 66 as "Total: 1 record".

The fix (agent/result_types.py + orchestrator wiring) classifies every data
result into a semantic type (record list / single record / count / aggregate /
empty) and renders ALL sections deterministically when every data result is
classifiable, so:
- COUNT/aggregate values are NEVER presented as a record count,
- counts are attributed to their object ("**Total Leads: 66**"),
- tables keep proper separate columns,
- single-result and metadata-only turns keep the existing byte-for-byte output,
- non-classifiable results (related records, schemas, errors) keep the LLM path.

All tests are offline (mocks / direct `_synthesize_response` calls) — no live
LLM / Salesforce / MCP.
"""

import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import format_sf_records_as_markdown
from agent.multi_agent import Orchestrator, _METADATA_ONLY_TOOLS
from agent.result_types import ResultInfo, classify_result, render_sections

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ACCOUNTS = {
    "totalSize": 22,
    "records": [
        {"attributes": {"type": "Account"}, "Id": "001a", "Name": "Edge Communications", "Industry": "Electronics"},
        {"attributes": {"type": "Account"}, "Id": "001b", "Name": "Globex", "Industry": "Software"},
    ],
}
_ACCOUNTS_JSON = json.dumps(_ACCOUNTS)

_LEADS_EXPR0 = {"totalSize": 1, "records": [{"attributes": {"type": "Lead"}, "expr0": 66}], "total_count": 66}
_LEADS_JSON = json.dumps(_LEADS_EXPR0)

_LEADS_COUNT_KEY = {"totalSize": 1, "records": [{"attributes": {"type": "Lead"}, "count": 66}]}
_LEADS_TOTAL_COUNT_ONLY = {"totalSize": 1, "records": [{"attributes": {"type": "Lead"}}], "total_count": 66}

_EMPTY_ACCOUNTS = {"totalSize": 0, "records": []}
_SINGLE_CONTACT = {
    "totalSize": 1,
    "records": [{"attributes": {"type": "Contact"}, "Id": "003x", "Name": "Jane"}],
}
_LEADS_BY_STATUS = {
    "totalSize": 2,
    "records": [
        {"attributes": {"type": "Lead"}, "Status": "Open", "expr0": 5},
        {"attributes": {"type": "Lead"}, "Status": "Contacted", "expr0": 7},
    ],
}
_CUSTOM_TENANT = {
    "totalSize": 1,
    "records": [
        {"attributes": {"type": "CustomTenant__c"}, "Id": "a01x", "Name": "Tenant A", "Lease_End_Date__c": "2026-12-31"}
    ],
}
_SUBQUERY_ACCOUNTS = {
    "totalSize": 1,
    "records": [
        {
            "attributes": {"type": "Account"},
            "Id": "001a",
            "Name": "Acme",
            "Contacts": {"totalSize": 2, "records": [{"Id": "003x"}]},
        }
    ],
}


# ---------------------------------------------------------------------------
# Direct _synthesize_response harness
# ---------------------------------------------------------------------------


def _orch(llm=None):
    llm = llm if llm is not None else MagicMock()
    llm.last_finish_reason = None
    return Orchestrator(llm=llm, executor=object())


def _synth(orch, items, message="Show me all Accounts AND tell me how many Leads I have"):
    return asyncio.run(orch._synthesize_response(message, items))


def _item(result, query="", tool="soqlQuery"):
    item = {"tool": tool, "result": result}
    if query:
        item["query"] = query
    return item


# ---------------------------------------------------------------------------
# Case A: list + count in one turn
# ---------------------------------------------------------------------------


def test_mixed_list_and_count_renders_labeled_sections():
    llm = MagicMock()
    llm.last_finish_reason = None
    orch = _orch(llm)
    res = _synth(orch, [
        _item(_ACCOUNTS_JSON, "SELECT Id, Name FROM Account"),
        _item(_LEADS_JSON, "SELECT COUNT(Id) FROM Lead"),
    ])

    assert "### Accounts Found" in res
    assert "**Total Leads: 66**" in res
    assert "Total: 1 record" not in res
    assert "**Total Count:**" not in res
    assert "IdName" not in res
    assert "| Id | Name | Industry |" in res
    assert "**Total: 22 records**" in res
    llm.chat.assert_not_called()


# ---------------------------------------------------------------------------
# Case B: two record lists
# ---------------------------------------------------------------------------

_CONTACTS = {
    "totalSize": 4,
    "records": [
        {"attributes": {"type": "Contact"}, "Id": "003a", "Name": "Jane"},
        {"attributes": {"type": "Contact"}, "Id": "003b", "Name": "John"},
    ],
}


def test_two_lists_rendered_as_two_labeled_sections():
    res = _synth(_orch(), [
        _item(_ACCOUNTS_JSON, "SELECT Id, Name FROM Account"),
        _item(json.dumps(_CONTACTS), "SELECT Id, Name FROM Contact"),
    ])

    assert "### Accounts Found" in res
    assert "### Contacts Found" in res
    assert res.index("### Accounts Found") < res.index("### Contacts Found")
    assert "**Total: 22 records**" in res
    assert "**Total: 4 records**" in res


# ---------------------------------------------------------------------------
# Case C: two counts must stay attributable
# ---------------------------------------------------------------------------


def test_two_counts_are_labeled_separately():
    res = _synth(_orch(), [
        _item(json.dumps({"totalSize": 1, "records": [{"attributes": {"type": "Account"}, "expr0": 22}]}),
              "SELECT COUNT(Id) FROM Account"),
        _item(_LEADS_JSON, "SELECT COUNT(Id) FROM Lead"),
    ])

    assert "**Total Accounts: 22**" in res
    assert "**Total Leads: 66**" in res


# ---------------------------------------------------------------------------
# Case J: count is rendered from any count shape, never as a record count
# ---------------------------------------------------------------------------


def test_count_key_shape_renders_as_count():
    res = _synth(_orch(), [
        _item(_ACCOUNTS_JSON, "SELECT Id, Name FROM Account"),
        _item(json.dumps(_LEADS_COUNT_KEY), "SELECT COUNT(Id) FROM Lead"),
    ])
    assert "**Total Leads: 66**" in res
    assert "Total: 1 record" not in res


def test_total_count_only_shape_renders_as_count():
    res = _synth(_orch(), [
        _item(_ACCOUNTS_JSON, "SELECT Id, Name FROM Account"),
        _item(json.dumps(_LEADS_TOTAL_COUNT_ONLY), "SELECT COUNT(Id) FROM Lead"),
    ])
    assert "**Total Leads: 66**" in res
    assert "Total: 1 record" not in res


# ---------------------------------------------------------------------------
# Case E: grouped aggregate must render rows, never "N records"
# ---------------------------------------------------------------------------


def test_single_grouped_aggregate_renders_deterministic_table_case_a():
    # Requirement A: "Count Leads by Status" (SELECT Status, COUNT(Id) FROM Lead
    # GROUP BY Status) — a SINGLE aggregate result must never collapse into the
    # flat formatter's count line or a record count.
    res = _synth(_orch(), [
        _item(json.dumps(_LEADS_BY_STATUS), "SELECT Status, COUNT(Id) FROM Lead GROUP BY Status"),
    ])
    assert "### Leads by Status" in res
    assert "| Status | COUNT(Id) |" in res
    assert "| --- | ---: |" in res
    assert "| Open | 5 |" in res
    assert "| Contacted | 7 |" in res
    assert "records" not in res
    assert "**Total Count:**" not in res
    assert "Total: 2" not in res


def test_single_metric_aggregate_sum_never_called_count_case_e_scalar():
    # A scalar SUM (single row, expr0-only) must render as an aggregate, not as
    # "**Total Count:** N" and never as a record count. The metric header carries
    # metric + field context (SUM(Amount)), and the title carries the object.
    _SUMMED = {"totalSize": 1, "records": [{"attributes": {"type": "Opportunity"}, "expr0": 250000}]}
    res = _synth(_orch(), [
        _item(json.dumps(_SUMMED), "SELECT SUM(Amount) FROM Opportunity"),
    ])
    assert "### Opportunities Summary" in res
    assert "| SUM(Amount) |" in res
    assert "| 250,000 |" in res
    assert "records" not in res
    assert "**Total Count:**" not in res
    # A plain COUNT keeps its count-line semantics, attributed to its object
    # (never a record count, never a generic "Total Count").
    single_count = _synth(_orch(), [_item(_LEADS_JSON, "SELECT COUNT(Id) FROM Lead")])
    assert single_count == "**Total Leads: 66**"


def test_single_metric_aggregate_avg_and_minmax_never_called_count_cases_d_e():
    # AVG, MIN, and MAX are all metrics: single-row expr0 results must render as
    # deterministic aggregate tables (metric column labeled metric+field), never
    # the count line and never a record count.
    _AVG = {"totalSize": 1, "records": [{"attributes": {"type": "Opportunity"}, "expr0": 12345.67}]}
    _MIN = {"totalSize": 1, "records": [{"attributes": {"type": "Opportunity"}, "expr0": 1000}]}
    _MAX = {"totalSize": 1, "records": [{"attributes": {"type": "Opportunity"}, "expr0": 9999}]}

    avg_res = _synth(_orch(), [_item(json.dumps(_AVG), "SELECT AVG(Amount) FROM Opportunity")])
    assert "### Opportunities Summary" in avg_res
    assert "| AVG(Amount) |" in avg_res
    assert "| 12,345.67 |" in avg_res
    assert "records" not in avg_res
    assert "**Total Count:**" not in avg_res

    min_res = _synth(_orch(), [_item(json.dumps(_MIN), "SELECT MIN(Amount) FROM Opportunity")])
    assert "### Opportunities Summary" in min_res
    assert "| MIN(Amount) |" in min_res
    assert "| 1,000 |" in min_res
    assert "records" not in min_res

    max_res = _synth(_orch(), [_item(json.dumps(_MAX), "SELECT MAX(Amount) FROM Opportunity")])
    assert "### Opportunities Summary" in max_res
    assert "| MAX(Amount) |" in max_res
    assert "| 9,999 |" in max_res
    assert "records" not in max_res


def test_grouped_aggregate_section_in_multi_result():
    res = _synth(_orch(), [
        _item(json.dumps(_LEADS_BY_STATUS), "SELECT Status, COUNT(Id) FROM Lead GROUP BY Status"),
        _item(_ACCOUNTS_JSON, "SELECT Id, Name FROM Account"),
    ])
    assert "### Accounts Found" in res
    assert "**Total: 22 records**" in res
    assert "### Leads by Status" in res
    assert "| Status | COUNT(Id) |" in res
    assert "| --- | ---: |" in res
    assert "| Open | 5 |" in res
    assert "| Contacted | 7 |" in res
    agg_part = res.split("### Leads by Status")[1].split("### Accounts")[0]
    assert "records" not in agg_part
    assert "Total: 2" not in agg_part


# ---------------------------------------------------------------------------
# Case F: single record
# ---------------------------------------------------------------------------


def test_single_record_section_keeps_singular_semantics():
    res = _synth(_orch(), [
        _item(json.dumps(_SINGLE_CONTACT), "SELECT Id, Name FROM Contact"),
        _item(_ACCOUNTS_JSON, "SELECT Id, Name FROM Account"),
    ])
    assert "### Contacts Found" in res
    assert "**Total: 1 record**" in res
    assert "**Total: 22 records**" in res


# ---------------------------------------------------------------------------
# Case G: empty result
# ---------------------------------------------------------------------------


def test_empty_list_renders_no_records_found():
    res = _synth(_orch(), [
        _item(_ACCOUNTS_JSON, "SELECT Id, Name FROM Account"),
        _item(json.dumps(_EMPTY_ACCOUNTS), "SELECT Id, Name FROM Account LIMIT 10"),
    ])
    assert "### Accounts Found" in res
    assert "**No Accounts found**" in res


# ---------------------------------------------------------------------------
# Case H: section order follows tool-result order
# ---------------------------------------------------------------------------


def test_section_order_preserved_when_count_comes_first():
    res = _synth(_orch(), [
        _item(_LEADS_JSON, "SELECT COUNT(Id) FROM Lead"),
        _item(_ACCOUNTS_JSON, "SELECT Id, Name FROM Account"),
    ])
    assert res.index("**Total Leads: 66**") < res.index("### Accounts Found")


# ---------------------------------------------------------------------------
# Case I: custom object with arbitrary fields — no fabricated columns
# ---------------------------------------------------------------------------


def test_custom_object_arbitrary_fields_build_table_from_returned_keys():
    res = _synth(_orch(), [
        _item(json.dumps(_CUSTOM_TENANT), "SELECT Id, Name, Lease_End_Date__c FROM CustomTenant__c"),
        _item(_ACCOUNTS_JSON, "SELECT Id, Name FROM Account"),
    ])
    assert "### CustomTenant__c Found" in res
    assert "| Id | Name | Lease_End_Date__c |" in res
    assert "| Tenant A |" in res
    assert "IdName" not in res


# ---------------------------------------------------------------------------
# Case K: classifier units
# ---------------------------------------------------------------------------


def test_classifier_result_types():
    assert classify_result("soqlQuery", _ACCOUNTS_JSON, "SELECT Id, Name FROM Account").result_type == "record_list"
    assert classify_result("soqlQuery", json.dumps(_SINGLE_CONTACT), "SELECT Id, Name FROM Contact").result_type == "single_record"
    assert classify_result("soqlQuery", _LEADS_JSON, "SELECT COUNT(Id) FROM Lead").result_type == "count"
    assert classify_result("soqlQuery", json.dumps(_LEADS_BY_STATUS), "SELECT Status, COUNT(Id) FROM Lead GROUP BY Status").result_type == "aggregate"
    assert classify_result("soqlQuery",
                           json.dumps({"totalSize": 1, "records": [{"attributes": {"type": "Opportunity"}, "expr0": 250000}]}),
                           "SELECT SUM(Amount) FROM Opportunity").result_type == "aggregate"
    assert classify_result("soqlQuery", json.dumps(_EMPTY_ACCOUNTS), "SELECT Id, Name FROM Account").result_type == "empty"
    assert classify_result("listRecentSobjectRecords", json.dumps([_ACCOUNTS["records"][0]]), "").result_type == "single_record"


def test_classifier_rejects_non_eligible_and_unparseable():
    # Related-records / schema / error / getUserInfo shapes are NOT eligible.
    assert classify_result("getRelatedRecords", _ACCOUNTS_JSON, "") is None
    assert classify_result("getObjectSchema", json.dumps({"fields": []}), "") is None
    assert classify_result("soqlQuery", "not json", "") is None
    assert classify_result("soqlQuery", json.dumps({"error": "results.my_soql_error", "message": "boom"}), "") is None
    # Hierarchical subquery results are not flat-rendered.
    assert classify_result("soqlQuery", json.dumps(_SUBQUERY_ACCOUNTS), "SELECT Id, Name FROM Account") is None


def test_classifier_rejects_flat_shaped_non_listing_tool():
    # A getRelatedRecords result that LOOKS flat must stay on the LLM path.
    assert classify_result("getRelatedRecords", json.dumps(_CONTACTS), "") is None


# ---------------------------------------------------------------------------
# Stability: single-result and metadata-only turns stay byte-for-byte
# ---------------------------------------------------------------------------


def test_single_flat_result_fast_path_unchanged():
    llm = MagicMock()
    llm.last_finish_reason = None
    orch = _orch(llm)
    res = _synth(orch, [_item(_ACCOUNTS_JSON, "SELECT Id, Name FROM Account")])
    assert res == format_sf_records_as_markdown(_ACCOUNTS_JSON, tool_name="soqlQuery")
    assert "### Accounts Found" not in res
    llm.chat.assert_not_called()


def test_single_count_fast_path_unchanged():
    # Worded without knowing the business object from only the count line itself,
    # the single COUNT must STILL identify the object (Leads) — never the generic
    # "Total Count" and never the totalSize=1 container as the value.
    res = _synth(_orch(), [_item(_LEADS_JSON, "SELECT COUNT(Id) FROM Lead")])
    assert res == "**Total Leads: 66**"


def test_single_custom_object_fast_path_keeps_returned_columns_case_c():
    # A single custom-object record list stays on the byte-for-byte flat path;
    # dynamic (arbitrary) returned columns must never be collapsed.
    res = _synth(_orch(), [_item(json.dumps(_CUSTOM_TENANT), "SELECT Id, Name, Lease_End_Date__c FROM CustomTenant__c")])
    assert "| Id | Name | Lease_End_Date__c |" in res
    assert "| Tenant A |" in res
    assert "### CustomTenant__c Found" not in res
    assert res == format_sf_records_as_markdown(json.dumps(_CUSTOM_TENANT), tool_name="soqlQuery")


def test_single_empty_non_count_keeps_llm_path_case_g():
    # A single empty (non-Count) result has no deterministic flat rendering, so
    # it must keep the LLM path ("No X found" semantics) — never a fabricated
    # "**No X found**" section and never a count line.
    llm = MagicMock()
    llm.last_finish_reason = None
    llm.chat = AsyncMock(return_value="No active Accounts found.")
    res = _synth(_orch(llm), [_item(json.dumps(_EMPTY_ACCOUNTS), "SELECT Id, Name FROM Account WHERE IsActive = true")])
    assert res == "No active Accounts found."
    llm.chat.assert_called_once()


def test_single_empty_count_shows_zero_case_g_count():
    # A COUNT query returning totalSize 0 is normalized by Fix B (multi_agent
    # `_normalize_zero_count_result`) BEFORE synthesis into a non-JSON count
    # line, so it keeps the LLM path and never surfaces a record count. The
    # normalization is object-attributed. Pinned by test_orchestrator_fix_b.py.
    normalized = format_sf_records_as_markdown(json.dumps(_EMPTY_ACCOUNTS), soql_query="SELECT COUNT(Id) FROM Account")
    assert normalized == "**Total Accounts: 0**"


# ---------------------------------------------------------------------------
# Direct flat-formatter guards: the formatter itself must never collapse a
# grouped/metric aggregate into the first-row count line (control-flow guard).
# ---------------------------------------------------------------------------


def test_format_flat_plain_count_stays_scalar_line():
    res = format_sf_records_as_markdown(_LEADS_JSON, tool_name="soqlQuery", soql_query="SELECT COUNT(Id) FROM Lead")
    assert res == "**Total Leads: 66**"


def test_format_flat_groupby_never_collapses_to_first_row():
    res = format_sf_records_as_markdown(json.dumps(_LEADS_BY_STATUS), tool_name="soqlQuery",
                                        soql_query="SELECT Status, COUNT(Id) FROM Lead GROUP BY Status")
    assert "| Status | COUNT(Id) |" in res
    assert "| --- | ---: |" in res
    assert "| Open | 5 |" in res
    assert "| Contacted | 7 |" in res
    assert "**Total Count:**" not in res
    assert "Total: 2" not in res
    assert "records" not in res


def test_format_flat_scalar_sum_never_count_line():
    res = format_sf_records_as_markdown(
        json.dumps({"totalSize": 1, "records": [{"attributes": {"type": "Opportunity"}, "expr0": 250000}]}),
        tool_name="soqlQuery", soql_query="SELECT SUM(Amount) FROM Opportunity")
    assert "| SUM(Amount) |" in res
    assert "| 250,000 |" in res
    assert "**Total Count:**" not in res
    assert "records" not in res


def test_metadata_only_plus_flat_soql_fast_path_unchanged():
    res = _synth(_orch(), [
        {"tool": "getUserInfo", "result": '{"identity":{"userId":"005g5000009G1fiAAC"}}'},
        _item(_ACCOUNTS_JSON, "SELECT Id, Name FROM Account"),
    ])
    assert res == format_sf_records_as_markdown(_ACCOUNTS_JSON, tool_name="soqlQuery")
    assert "### Accounts Found" not in res


def test_metadata_only_tool_set_constant():
    assert _METADATA_ONLY_TOOLS == {"getUserInfo", "getObjectSchema"}


# ---------------------------------------------------------------------------
# Safety: non-classifiable result keeps LLM synthesis
# ---------------------------------------------------------------------------


def test_non_classifiable_result_keeps_llm_path():
    llm = MagicMock()
    llm.last_finish_reason = None
    llm.chat = AsyncMock(return_value="Final synthesized answer.")
    orch = _orch(llm)
    res = _synth(orch, [
        _item(_ACCOUNTS_JSON, "SELECT Id, Name FROM Account"),
        {"tool": "getRelatedRecords", "result": json.dumps({"totalSize": 1, "records": [{"Id": "003x", "Name": "Jane"}]})},
    ])
    assert res == "Final synthesized answer."
    llm.chat.assert_called_once()


# ---------------------------------------------------------------------------
# End-to-end: the exact reported query through process_message
# ---------------------------------------------------------------------------

SAFE = {"safe": True, "requires_confirmation": False, "confirmation_message": "", "pending_action": None}


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": f"Tool {name}", "parameters": {"type": "object", "properties": {}}}}


class _RunLLM:
    def __init__(self, tool_calls, chat_result="[final answer]"):
        self._tool_calls = tool_calls
        self._chat_result = chat_result
        self.chat_calls = 0
        self.synthesis_user_msg = None

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        self.chat_calls += 1
        for m in (messages or []):
            if m.get("role") == "user":
                self.synthesis_user_msg = m.get("content", "")
        return self._chat_result

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        return {"content": "", "tool_calls": list(self._tool_calls), "finish_reason": "tool_calls"}


class _ScriptedExec:
    def __init__(self, mapping):
        self.mapping = mapping
        self.executed = []

    async def execute(self, name, arguments, user_provenance=None):
        self.executed.append((name, arguments))
        q = arguments.get("q") if isinstance(arguments, dict) else ""
        return self.mapping.get(q, json.dumps({"totalSize": 0, "records": []}))


class _Planner:
    def has_pending_confirmation(self, session_id):
        return False

    def check_tool_safety(self, tool_name, arguments, session_id="default"):
        return dict(SAFE)


def _build_e2e(mapping, tool_calls, message, chat_result="[final answer]"):
    llm = _RunLLM(tool_calls, chat_result=chat_result)
    orch = Orchestrator(llm=llm, executor=_ScriptedExec(mapping), max_iterations=5, max_history=4)
    orch.safety_planner = _Planner()

    rag = MagicMock()
    rag.get_relevant_tools = MagicMock(return_value=[_tool("soqlQuery")])
    orch.rag_retriever = rag

    orch._generate_plan = AsyncMock(return_value=[{
        "task_id": 1, "description": "compound query", "agent": "DataAgent", "depends_on": [],
    }])
    events = []

    async def _go():
        async for ev in orch.process_message(message, "default"):
            events.append(ev)
        return events

    asyncio.run(_go())
    return orch, llm, events


def test_e2e_show_all_accounts_and_tell_me_how_many_leads():
    mapping = {
        "SELECT Id, Name FROM Account": _ACCOUNTS_JSON,
        "SELECT COUNT(Id) FROM Lead": _LEADS_JSON,
    }
    tool_calls = [
        {"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id, Name FROM Account"}},
        {"id": "t2", "name": "soqlQuery", "arguments": {"q": "SELECT COUNT(Id) FROM Lead"}},
    ]
    orch, llm, events = _build_e2e(mapping, tool_calls, "Show me all Accounts AND tell me how many Leads I have")

    responses = " ".join(str(e.get("data")) for e in events if e.get("type") == "response")
    assert responses
    assert "### Accounts Found" in responses
    assert "| Id | Name | Industry |" in responses
    assert "**Total Leads: 66**" in responses
    assert "Total: 1 record" not in responses
    assert "**Total Count:**" not in responses
    assert "IdName" not in responses
    assert llm.chat_calls == 0, "deterministic sections — synthesizer skipped"


# ---------------------------------------------------------------------------
# End-to-end scenario matrix (real orchestrator path: plan -> execute -> classify
# -> deterministic render -> _synthesize_response)
# --------------------------------------------------------------------------


def test_e2e_show_all_accounts_scenario_1():
    # Single record list. Response is the byte-for-byte flat table; the record
    # total refers to Accounts only.
    mapping = {"SELECT Id, Name FROM Account": _ACCOUNTS_JSON}
    tool_calls = [{"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id, Name FROM Account"}}]
    orch, llm, events = _build_e2e(mapping, tool_calls, "Show me all Accounts")
    responses = " ".join(str(e.get("data")) for e in events if e.get("type") == "response")
    assert responses == format_sf_records_as_markdown(_ACCOUNTS_JSON, tool_name="soqlQuery")
    assert "| Id | Name | Industry |" in responses
    assert "**Total: 22 records**" in responses
    assert "Lead" not in responses
    assert llm.chat_calls == 0


def test_e2e_how_many_leads_scenario_2():
    # Single COUNT. Value MUST come from expr0 (66), never totalSize=1, never the
    # generic "Total Count" wording, never "1 record".
    mapping = {"SELECT COUNT(Id) FROM Lead": _LEADS_JSON}
    tool_calls = [{"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT COUNT(Id) FROM Lead"}}]
    orch, llm, events = _build_e2e(mapping, tool_calls, "How many Leads do I have?")
    responses = " ".join(str(e.get("data")) for e in events if e.get("type") == "response")
    assert responses == "**Total Leads: 66**"
    assert "Total: 1 record" not in responses
    assert "**Total Count:**" not in responses
    assert llm.chat_calls == 0


def test_e2e_leads_grouped_by_status_scenario_4():
    # Single GROUP BY aggregate: EVERY group renders, never collapsed to the
    # first row, never len(records) as a business count.
    mapping = {"SELECT Status, COUNT(Id) FROM Lead GROUP BY Status": json.dumps(_LEADS_BY_STATUS)}
    tool_calls = [{"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Status, COUNT(Id) FROM Lead GROUP BY Status"}}]
    orch, llm, events = _build_e2e(mapping, tool_calls, "Show Leads grouped by Status")
    responses = " ".join(str(e.get("data")) for e in events if e.get("type") == "response")
    assert "### Leads by Status" in responses
    assert "| Status | COUNT(Id) |" in responses
    assert "| Open | 5 |" in responses
    assert "| Contacted | 7 |" in responses
    assert "**Total Count:**" not in responses
    assert "records" not in responses
    assert llm.chat_calls == 0


def test_e2e_two_independent_aggregates_scenario_6():
    # COUNT LeADS + SUM Opportunity amount in one request: each keeps its own
    # object/metric identity; sections are separated and never merged.
    _OPP_SUM = {"totalSize": 1, "records": [{"attributes": {"type": "Opportunity"}, "expr0": 250000}]}
    mapping = {
        "SELECT COUNT(Id) FROM Lead": _LEADS_JSON,
        "SELECT SUM(Amount) FROM Opportunity": json.dumps(_OPP_SUM),
    }
    tool_calls = [
        {"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT COUNT(Id) FROM Lead"}},
        {"id": "t2", "name": "soqlQuery", "arguments": {"q": "SELECT SUM(Amount) FROM Opportunity"}},
    ]
    orch, llm, events = _build_e2e(mapping, tool_calls, "Count my Leads and show me the total Opportunity amount")
    responses = " ".join(str(e.get("data")) for e in events if e.get("type") == "response")
    assert "**Total Leads: 66**" in responses
    assert "### Opportunities Summary" in responses
    assert "| SUM(Amount) |" in responses
    assert "| 250,000 |" in responses
    assert responses.index("**Total Leads: 66**") < responses.index("### Opportunities Summary")
    assert "**Total Count:**" not in responses
    assert "records" not in responses
    assert llm.chat_calls == 0


def test_e2e_empty_non_count_result_scenario_8():
    # Empty (non-Count) single result keeps the LLM synthesis path; no fabricated
    # "1 record", no count line.
    mapping = {"SELECT Id FROM Account WHERE IsActive = true": json.dumps(_EMPTY_ACCOUNTS)}
    tool_calls = [{"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Account WHERE IsActive = true"}}]
    orch, llm, events = _build_e2e(
        mapping, tool_calls, "Show me active Accounts", chat_result="No active Accounts were found."
    )
    responses = " ".join(str(e.get("data")) for e in events if e.get("type") == "response")
    assert "No active Accounts were found." in responses
    assert "1 record" not in responses
    assert "**Total Count:**" not in responses
    assert llm.chat_calls == 1, "non-classifiable empty non-count -> synthesizer"


def test_e2e_custom_object_and_custom_groupby_scenario_9():
    # No Account/Lead hardcoding: arbitrary custom-object fields and custom GROUP
    # BY columns come from the returned Salesforce records.
    _custom_by_lease = {
        "totalSize": 2,
        "records": [
            {"attributes": {"type": "CustomTenant__c"}, "Lease_End_Date__c": "2026-12-31", "expr0": 1},
            {"attributes": {"type": "CustomTenant__c"}, "Lease_End_Date__c": "2027-06-01", "expr0": 3},
        ],
    }
    mapping = {
        "SELECT Id, Name, Lease_End_Date__c FROM CustomTenant__c": json.dumps(_CUSTOM_TENANT),
        "SELECT Lease_End_Date__c, COUNT(Id) FROM CustomTenant__c GROUP BY Lease_End_Date__c": json.dumps(_custom_by_lease),
    }
    tool_calls = [
        {"id": "t1", "name": "soqlQuery", "arguments": {"q": "SELECT Id, Name, Lease_End_Date__c FROM CustomTenant__c"}},
        {"id": "t2", "name": "soqlQuery", "arguments": {"q": "SELECT Lease_End_Date__c, COUNT(Id) FROM CustomTenant__c GROUP BY Lease_End_Date__c"}},
    ]
    orch, llm, events = _build_e2e(
        mapping, tool_calls,
        "Show me tenant records AND how many end each lease date",
    )
    responses = " ".join(str(e.get("data")) for e in events if e.get("type") == "response")
    assert "### CustomTenant__c Found" in responses
    assert "| Id | Name | Lease_End_Date__c |" in responses
    assert "**Total: 1 record**" in responses
    assert "### CustomTenant__c by Lease_End_Date__c" in responses
    assert "| Lease_End_Date__c | COUNT(Id) |" in responses
    assert "| 2026-12-31 | 1 |" in responses
    assert "| 2027-06-01 | 3 |" in responses
    agg_part = responses.split("### CustomTenant__c by Lease_End_Date__c")[1]
    assert "records" not in agg_part
    assert llm.chat_calls == 0


# ---------------------------------------------------------------------------
# render_sections direct units
# ---------------------------------------------------------------------------


def test_render_sections_returns_none_for_unhandled_type():
    infos = [ResultInfo("hierarchy", "Account", "{}")]
    assert render_sections(infos) is None


if __name__ == "__main__":
    import unittest
    unittest.main(module=__name__)