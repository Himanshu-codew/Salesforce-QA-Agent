"""
Deterministic "my <object>" owner pre-resolution tests.

Root cause fixed: the worker LLM used to generate e.g.
    SELECT Id, Name, Email, Phone FROM Contact WHERE OwnerId = '005...' LIMIT 10
with a User/Owner id it invented/truncated/masked. Salesforce rejected the literal
with INVALID_QUERY_FILTER_OPERATOR / "invalid ID field" because the id was not the
user's real id.

Fix: for "my <object>" ownership reads the application resolves the current
Salesforce User id via getUserInfo (bounded, validated) and DETERMINISTICALLY
rebuilds the soqlQuery with WHERE OwnerId = '<real_id>'. The id always comes from
the tool result — never from Qwen.

Verified here:
  1. "Show my Contacts" -> getUserInfo first -> soqlQuery with the exact resolved id
  2. getUserInfo returns an invalid/missing id -> soqlQuery NOT called, SALESFORCE_FAILED
  3. "Show all Contacts" -> normal behavior unchanged (no getUserInfo)
  4. "How many Contacts are there?" -> COUNT unchanged
  5. "Show my Opportunities" -> same owner-resolution mechanism
  6. unit checks for the helpers (detect / validate / extract / build)

All tests use mocks - no live LLM / Salesforce / embedding model.
"""

import os
import sys
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.multi_agent import (
    Orchestrator,
    _detect_ownership_object,
    _validate_salesforce_user_id,
    _extract_user_id,
    _build_owner_soql,
)

SAFE = {"safe": True, "requires_confirmation": False, "confirmation_message": "", "pending_action": None}


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": f"Tool {name}", "parameters": {"type": "object", "properties": {}}}}


_CLOCK = 0


def _tc(name, q=None, args=None):
    global _CLOCK
    _CLOCK += 1
    if q is not None:
        args = {"q": q}
    return {"id": f"t{_CLOCK}", "name": name, "arguments": args or {}}


class _RunLLM:
    def __init__(self, tool_calls):
        self._tool_calls = list(tool_calls)

    async def chat(self, messages=None, temperature=0.0, max_tokens=8192):
        return "[general answer]"

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=8192):
        return {"content": "", "tool_calls": list(self._tool_calls), "finish_reason": "tool_calls"}


class _Exec:
    """getUserInfo returns a VALID id by default."""

    def __init__(self, user_id=None):
        self.executed = []
        self.user_id = user_id if user_id else "005000000000000001"

    async def execute(self, name, arguments):
        self.executed.append((name, arguments))
        if name == "getUserInfo":
            # Real Salesforce MCP getUserInfo shape: {"identity": {"userId": "005..."}}.
            return json.dumps({"identity": {"userId": self.user_id, "displayName": "Himanshu"}})
        q = arguments.get("q", "") if isinstance(arguments, dict) else ""
        if q.startswith("SELECT COUNT"):
            return json.dumps({"totalSize": 1, "records": [{"expr0": 7}]})
        return json.dumps({"totalSize": 2, "records": [{"Id": "003A", "Name": "Ada"}, {"Id": "003B", "Name": "Bob"}], "done": True})


class _Planner:
    def __init__(self, safety=SAFE):
        self._safety = safety

    def has_pending_confirmation(self, session_id):
        return False

    def check_tool_safety(self, tool_name, arguments, session_id="default"):
        return dict(self._safety)


def _build(llm, exec_, safety=SAFE):
    orch = Orchestrator(llm=llm, executor=exec_, max_iterations=5, max_history=4)
    orch.safety_planner = _Planner(safety)
    rag = MagicMock()
    rag.get_relevant_tools = MagicMock(return_value=[_tool("soqlQuery"), _tool("getUserInfo")])
    orch.rag_retriever = rag
    orch._generate_plan = AsyncMock(return_value=[])
    return orch


def _run(orch, message, session_id="default"):
    async def _go():
        events = []
        async for ev in orch.process_message(message, session_id):
            events.append(ev)
        return events
    return asyncio.run(_go())


def _executed_queries(exec_):
    return [a.get("q") for n, a in exec_.executed if n == "soqlQuery"]


# ---------------------------------------------------------------------------
# 1. "Show my Contacts" - deterministic owner resolution
# ---------------------------------------------------------------------------


def test_show_my_contacts_getuserinfo_first_then_deterministic_soql():
    llm = _RunLLM([_tc("soqlQuery", "SELECT Id, Name FROM Contact LIMIT 200")])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    events = _run(orch, "Show my Contacts.")
    names = [n for n, _ in exec_.executed]
    assert names[0] == "getUserInfo", names
    assert "soqlQuery" in names
    qs = _executed_queries(exec_)
    assert len(qs) == 1
    assert qs[0] == "SELECT Id, Name, Email, Phone FROM Contact WHERE OwnerId = '005000000000000001' LIMIT 10"
    # The id must come from getUserInfo, never a Qwen placeholder or binding syntax.
    assert ":$User" not in qs[0]
    assert ":id" not in qs[0]
    assert "${" not in qs[0]
    assert "005000000000000001" in qs[0]


def test_live_qwen_placeholder_owner_id_is_rewritten():
    # The exact production failure: Qwen emitted
    #   SELECT Id, Name, Email, Phone FROM Contact WHERE OwnerId = '005...' LIMIT 100
    # (the prompt's literal example + invented/placeholder id) which Salesforce
    # rejected with INVALID_QUERY_FILTER_OPERATOR / "invalid ID field: 005...".
    # The deterministic interception MUST rewrite it to a validated owner id.
    bad = "SELECT Id, Name, Email, Phone FROM Contact WHERE OwnerId = '005...' LIMIT 100"
    llm = _RunLLM([_tc("soqlQuery", bad)])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    _run(orch, "Show my Contacts.")
    qs = _executed_queries(exec_)
    assert len(qs) == 1
    assert qs[0] == "SELECT Id, Name, Email, Phone FROM Contact WHERE OwnerId = '005000000000000001' LIMIT 10"
    assert "'005...'" not in qs[0]  # placeholder must never survive
    assert "005000000000000001" in qs[0]


def test_show_my_contacts_no_redundant_getuserinfo_call():
    # getUserInfo is resolved once and cached for the request.
    llm = _RunLLM([_tc("soqlQuery", "SELECT Id, Name FROM Contact LIMIT 200")])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    _run(orch, "Show my Contacts.")
    getuserinfo_calls = [a for n, a in exec_.executed if n == "getUserInfo"]
    assert len(getuserinfo_calls) == 1


# ---------------------------------------------------------------------------
# 2. getUserInfo returns invalid / missing id -> no soqlQuery, SALESFORCE_FAILED
# ---------------------------------------------------------------------------


def test_invalid_user_id_aborts_ownership_query():
    llm = _RunLLM([_tc("soqlQuery", "SELECT Id, Name FROM Contact LIMIT 200")])

    class _ExecBad(_Exec):
        async def execute(self, name, arguments):
            self.executed.append((name, arguments))
            if name == "getUserInfo":
                return json.dumps({"display_name": "Himanshu", "email": "h@example.com"})
            return json.dumps({"totalSize": 1, "records": [{"Id": "003A"}]})

    exec_ = _ExecBad()
    orch = _build(llm, exec_)
    events = _run(orch, "Show my Contacts.")
    # getUserInfo ran but resolved no valid id -> no soqlQuery, error surfaced.
    assert [n for n, _ in exec_.executed] == ["getUserInfo"]
    types = [e.get("type") for e in events]
    assert "tool_call" not in types  # soqlQuery never executed
    err = next((e for e in events if e.get("type") == "error"), None)
    assert err and err.get("code") == "SALESFORCE_FAILED"


def test_truncated_user_id_aborts_ownership_query():
    # A 005... id cut short (or a non-005 id) must be rejected.
    llm = _RunLLM([_tc("soqlQuery", "SELECT Id, Name FROM Contact LIMIT 200")])
    exec_ = _Exec(user_id="005")
    orch = _build(llm, exec_)
    _run(orch, "Show my Contacts.")
    assert [n for n, _ in exec_.executed] == ["getUserInfo"]


def test_empty_user_id_result_aborts_ownership_query():
    llm = _RunLLM([_tc("soqlQuery", "SELECT Id, Name FROM Contact LIMIT 200")])

    class _ExecEmpty(_Exec):
        async def execute(self, name, arguments):
            self.executed.append((name, arguments))
            if name == "getUserInfo":
                return "{}"
            return json.dumps({"totalSize": 1, "records": [{"Id": "003A"}]})

    exec_ = _ExecEmpty()
    orch = _build(llm, exec_)
    events = _run(orch, "Show my Contacts.")
    assert [n for n, _ in exec_.executed] == ["getUserInfo"]
    assert "tool_call" not in [e.get("type") for e in events]
    assert any(e.get("type") == "error" and e.get("code") == "SALESFORCE_FAILED" for e in events)


# ---------------------------------------------------------------------------
# 3. "Show all Contacts" - normal behavior unchanged
# ---------------------------------------------------------------------------


def test_show_all_contacts_normal_no_getuserinfo():
    llm = _RunLLM([_tc("soqlQuery", "SELECT Id, Name FROM Contact LIMIT 200")])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    _run(orch, "Show all Contacts")
    names = [n for n, _ in exec_.executed]
    assert names == ["soqlQuery"]
    assert "getUserInfo" not in names


def test_show_contacts_named_rahul_normal_no_getuserinfo():
    llm = _RunLLM([_tc("soqlQuery", "SELECT Id, Name FROM Contact WHERE Name LIKE '%Rahul%' LIMIT 200")])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    _run(orch, "Show Contacts named Rahul")
    assert "getUserInfo" not in [n for n, _ in exec_.executed]


# ---------------------------------------------------------------------------
# 4. "How many Contacts are there?" - COUNT unchanged
# ---------------------------------------------------------------------------


def test_how_many_contacts_count_unchanged():
    llm = _RunLLM([_tc("soqlQuery", "SELECT COUNT(Id) FROM Contact")])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    _run(orch, "How many Contacts are there?")
    qs = _executed_queries(exec_)
    assert qs == ["SELECT COUNT(Id) FROM Contact"]
    assert "getUserInfo" not in [n for n, _ in exec_.executed]


# ---------------------------------------------------------------------------
# 5. "Show my Opportunities" - same owner-resolution mechanism
# ---------------------------------------------------------------------------


def test_show_my_opportunities_owner_resolution():
    llm = _RunLLM([_tc("soqlQuery", "SELECT Id, Name FROM Opportunity LIMIT 200")])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    _run(orch, "Show my Opportunities")
    names = [n for n, _ in exec_.executed]
    assert names[0] == "getUserInfo"
    qs = _executed_queries(exec_)
    assert qs == ["SELECT Id, Name, Amount, StageName FROM Opportunity WHERE OwnerId = '005000000000000001' LIMIT 10"]


def test_show_my_leads_owner_resolution():
    llm = _RunLLM([_tc("soqlQuery", "SELECT Id, Name FROM Lead LIMIT 200")])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    _run(orch, "Show my Leads")
    assert [n for n, _ in exec_.executed][0] == "getUserInfo"
    assert "FROM Lead WHERE OwnerId = '005000000000000001' LIMIT 10" in _executed_queries(exec_)[0]


# ---------------------------------------------------------------------------
# 6. Unit helpers
# ---------------------------------------------------------------------------


def test_detect_ownership_object_only_for_possessive_my_object():
    assert _detect_ownership_object("Show my Contacts.") == "Contact"
    assert _detect_ownership_object("List my Opportunities") == "Opportunity"
    assert _detect_ownership_object("Show my Cases") == "Case"
    assert _detect_ownership_object("Show my Accounts") == "Account"
    assert _detect_ownership_object("Show all Contacts") is None
    assert _detect_ownership_object("How many Contacts are there?") is None
    assert _detect_ownership_object("Show Contacts named Rahul") is None
    assert _detect_ownership_object("Show Contacts owned by Bob") is None
    assert _detect_ownership_object("my Contact records") is None  # singular, not in scope
    assert _detect_ownership_object("") is None
    assert _detect_ownership_object(None) is None


def test_validate_salesforce_user_id():
    assert _validate_salesforce_user_id("005000000000000001") == "005000000000000001"
    assert _validate_salesforce_user_id("005000000000000") == "005000000000000"  # 15 chars
    assert _validate_salesforce_user_id("003A000000000001") is None  # not a User id prefix
    assert _validate_salesforce_user_id("005") is None  # too short
    assert _validate_salesforce_user_id("") is None
    assert _validate_salesforce_user_id(None) is None
    assert _validate_salesforce_user_id(123) is None


def test_extract_user_id_handles_userinfo_and_soql_shapes():
    # OAuth /userinfo shape
    assert _extract_user_id(json.dumps({"sub": "005000000000000001", "name": "H"})) == "005000000000000001"
    assert _extract_user_id(json.dumps({"user_id": "005000000000000001"})) == "005000000000000001"
    # SOQL fallback shape
    assert _extract_user_id(
        json.dumps({"totalSize": 1, "records": [{"Id": "005000000000000001"}]})
    ) == "005000000000000001"
    # invalid -> None
    assert _extract_user_id(json.dumps({"sub": "003A000000000001"})) is None
    assert _extract_user_id(json.dumps({"display_name": "H"})) is None
    assert _extract_user_id("not-json") is None
    assert _extract_user_id(None) is None


def test_extract_user_id_handles_salesforce_mcp_identity_shape():
    # Exact shape returned by the Salesforce MCP getUserInfo tool
    # (confirmed live: identity.userId is the only 005-prefixed id present).
    mcp_response = json.dumps({
        "identity": {
            "companyName": "Learning",
            "displayName": "Himanshu Swami",
            "email": "himanshuswami898@gmail.com",
            "profileId": "00eg50000063GzFAAU",
            "userId": "005g5000009G1fiAAC",
            "username": "himanshuswami898.e86b4be632fc@agentforce.com",
        },
        "userTimeAndLocale": {
            "humanReadableTime": "Sunday, 11:24 AM",
            "localTimeIso": "6 Sep 2026, 06:24 PM UTC",
            "localeCode": "en_US",
            "timeZoneIana": "America/Los_Angeles",
        },
    })
    assert _extract_user_id(mcp_response) == "005g5000009G1fiAAC"
    # Defense-in-depth: an id nested under a non-standard key is still found.
    assert (
        _extract_user_id(
            json.dumps({"wrapper": {"user": {"user_id": "005000000000000001"}}})
        )
        == "005000000000000001"
    )
    # profileId (00e prefix) must never be mistaken for the User id.
    assert _extract_user_id(json.dumps({"identity": {"profileId": "00eg50000063GzFAAU"}})) is None


def test_build_owner_soql():
    assert _build_owner_soql("Contact", "005000000000000001") == (
        "SELECT Id, Name, Email, Phone FROM Contact WHERE OwnerId = '005000000000000001' LIMIT 10"
    )


if __name__ == "__main__":
    import unittest

    unittest.main(module=__name__)