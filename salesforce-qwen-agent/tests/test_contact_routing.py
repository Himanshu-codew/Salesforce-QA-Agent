"""
Regression tests for Contact-record routing (previously mis-routed to getUserInfo).

Root cause: agent/multi_agent_prompts.py DATA_AGENT_PROMPT contained an
over-broad rule: "If the user asks for 'my' records, you must call getUserInfo
first...". That forced the worker LLM to call getUserInfo for queries like
"Show my Contacts.", returning the logged-in user's PROFILE instead of Contact
records.

Fix (Option A, prompt-only): getUserInfo is now restricted to identity/profile
questions ("Who am I?", "What is my profile?"). Record-list/query requests for
any object (Contact, Account, Lead, Opportunity, Case, Task, Event) MUST use
soqlQuery/find. getUserInfo may only be a SUPPORTING step for ownership-filtered
reads (to resolve OwnerId/CreatedById) and must then be followed by the object
query; it is never a valid standalone answer for a record request.

These tests verify:
  - the prompt contains the corrected rules and no longer forces getUserInfo
    for "my" record requests
  - a record-request message routes to soqlQuery/find (never getUserInfo as the
    answer)
  - COUNT queries preserve count intent
  - list queries do not produce a redundant COUNT
  - identity questions ("Who am I?", "What is my profile?") still use getUserInfo

All tests use mocks - no live LLM / Salesforce / embedding model.
"""

import os
import sys
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.multi_agent import Orchestrator
import agent.multi_agent_prompts as prompts

SAFE = {"safe": True, "requires_confirmation": False, "confirmation_message": "", "pending_action": None}


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": f"Tool {name}", "parameters": {"type": "object", "properties": {}}}}

CLOCK = 0


def _tc(name, q):
    global CLOCK
    CLOCK += 1
    return {"id": f"t{CLOCK}", "name": name, "arguments": {"q": q}}


class _RunLLM:
    def __init__(self, tool_calls):
        self._tool_calls = list(tool_calls)
        self.chat_with_tools_calls = []

    async def chat(self, messages=None, temperature=0.0, max_tokens=8192):
        return "[general answer]"

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=8192):
        self.chat_with_tools_calls.append((messages, tools))
        return {"content": "", "tool_calls": list(self._tool_calls), "finish_reason": "tool_calls"}


class _Exec:
    def __init__(self):
        self.executed = []

    async def execute(self, name, arguments):
        self.executed.append((name, arguments))
        if name == "getUserInfo":
            return json.dumps({"id": "005000000000000001", "display_name": "Himanshu"})
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
    return [a.get("q") for n, a in exec_.executed]


# ---------------------------------------------------------------------------
# Prompt content checks (the actual fix)
# ---------------------------------------------------------------------------


def test_prompt_no_longer_forces_getuserinfo_for_my_records():
    assert 'must call `getUserInfo` first if' not in prompts.DATA_AGENT_PROMPT
    # The buggy unconditional rule must be gone.
    assert "you must call `getUserInfo` first" not in prompts.DATA_AGENT_PROMPT


def test_prompt_restricts_getuserinfo_to_identity():
    assert "ONLY the current logged-in user's identity" in prompts.DATA_AGENT_PROMPT
    assert "NEVER returns or retrieves any record" in prompts.DATA_AGENT_PROMPT


def test_prompt_routes_record_requests_to_soql_find():
    assert "use `soqlQuery` or `find` for that object" in prompts.DATA_AGENT_PROMPT
    assert "Show my Contacts" in prompts.DATA_AGENT_PROMPT
    assert "List my Contacts" in prompts.DATA_AGENT_PROMPT


def test_prompt_allows_getuserinfo_but_requires_followup_query():
    assert "SUPPORTING step" in prompts.DATA_AGENT_PROMPT
    # getUserInfo may only be a support step; the record query must follow with a
    # literal inlined owner ID (never binding syntax), and getUserInfo alone is never
    # a valid answer. (Phrasing updated with the binding-syntax fix.)
    assert "inline" in prompts.DATA_AGENT_PROMPT
    assert "`getUserInfo` alone is never a valid final answer" in prompts.DATA_AGENT_PROMPT


def test_prompt_keeps_identity_questions_on_getuserinfo():
    assert "Who am I?" in prompts.DATA_AGENT_PROMPT
    assert "What is my profile?" in prompts.DATA_AGENT_PROMPT


# ---------------------------------------------------------------------------
# Contact routing: soqlQuery/find, NEVER getUserInfo as the answer
# ---------------------------------------------------------------------------


def test_show_my_contacts_routes_to_contact_query():
    # Deterministic owner resolution: getUserInfo is called first, then soqlQuery
    # is rebuilt with the validated OwnerId (never Qwen's invented id).
    llm = _RunLLM([_tc("soqlQuery", "SELECT Id, Name FROM Contact LIMIT 200")])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    events = _run(orch, "Show my Contacts.")
    # Fast path (planner skipped) and deterministic two-step owner flow.
    assert orch._generate_plan.call_count == 0
    assert any(e.get("type") == "tool_call" for e in events)
    names = [n for n, _ in exec_.executed]
    assert names[0] == "getUserInfo"
    assert "soqlQuery" in names
    assert "FROM Contact" in _executed_queries(exec_)[-1]
    # OwnerId must be the validated id resolved from getUserInfo, not a placeholder.
    assert "WHERE OwnerId = '005000000000000001'" in _executed_queries(exec_)[-1]


def test_list_my_contacts_routes_to_contact_query():
    llm = _RunLLM([_tc("soqlQuery", "SELECT Id, Name FROM Contact LIMIT 200")])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    _run(orch, "List my Contacts")
    names = [n for n, _ in exec_.executed]
    assert names[0] == "getUserInfo"
    assert "soqlQuery" in names
    assert "WHERE OwnerId = '005000000000000001'" in _executed_queries(exec_)[-1]


def test_show_all_contacts_no_redundant_count():
    # Worker returns list + a redundant COUNT; no count intent -> guard drops COUNT.
    llm = _RunLLM([
        _tc("soqlQuery", "SELECT Id, Name FROM Contact LIMIT 200"),
        _tc("soqlQuery", "SELECT COUNT(Id) FROM Contact"),
    ])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    _run(orch, "Show all Contacts")
    executed_qs = _executed_queries(exec_)
    assert any("FROM Contact" in q for q in executed_qs)
    assert all("COUNT" not in q for q in executed_qs)
    assert "getUserInfo" not in [n for n, _ in exec_.executed]


def test_find_my_contacts_routes_to_contact_query():
    llm = _RunLLM([_tc("find", "FIND {Contact} IN ALL FIELDS RETURNING Contact(Id, Name)")])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    _run(orch, "Find my Contacts")
    assert [n for n, _ in exec_.executed][0] == "find"
    assert "getUserInfo" not in [n for n, _ in exec_.executed]


def test_give_me_my_contact_records_routes_to_contact_query():
    llm = _RunLLM([_tc("soqlQuery", "SELECT Id, Name FROM Contact LIMIT 200")])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    _run(orch, "Give me my Contact records")
    assert [n for n, _ in exec_.executed] == ["soqlQuery"]
    assert "getUserInfo" not in [n for n, _ in exec_.executed]
    assert "FROM Contact" in _executed_queries(exec_)[0]


def test_how_many_contacts_preserves_count():
    llm = _RunLLM([_tc("soqlQuery", "SELECT COUNT(Id) FROM Contact")])
    exec_ = _Exec()
    orch = _build(llm, exec_)
    _run(orch, "How many Contacts do I have?")
    assert "SELECT COUNT(Id) FROM Contact" in _executed_queries(exec_)
    assert "getUserInfo" not in [n for n, _ in exec_.executed]


# ---------------------------------------------------------------------------
# Identity questions must STILL use getUserInfo
# ---------------------------------------------------------------------------


def test_who_am_i_uses_getuserinfo():
    llm = _RunLLM([{"id": "t99", "name": "getUserInfo", "arguments": {}}])

    class _ExecUser(_Exec):
        async def execute(self, name, arguments):
            self.executed.append((name, arguments))
            return json.dumps({"display_name": "Himanshu", "email": "himanshu@example.com"})

    exec_ = _ExecUser()
    orch = _build(llm, exec_)
    events = _run(orch, "Who am I?")
    assert [n for n, _ in exec_.executed] == ["getUserInfo"]
    assert any(e.get("type") == "tool_call" for e in events)


def test_what_is_my_profile_uses_getuserinfo():
    llm = _RunLLM([{"id": "t98", "name": "getUserInfo", "arguments": {}}])

    class _ExecUser(_Exec):
        async def execute(self, name, arguments):
            self.executed.append((name, arguments))
            return json.dumps({"display_name": "Himanshu", "role": "Standard"})

    exec_ = _ExecUser()
    orch = _build(llm, exec_)
    _run(orch, "What is my profile?")
    assert [n for n, _ in exec_.executed] == ["getUserInfo"]


# ---------------------------------------------------------------------------
# SOQL binding-syntax guard (raw soqlQuery has no bind-variable substitution)
# ---------------------------------------------------------------------------


def test_prompt_forbids_binding_syntax():
    assert "variable binding syntax" in prompts.DATA_AGENT_PROMPT
    assert ":$User.Id" in prompts.DATA_AGENT_PROMPT
    assert ":param" in prompts.DATA_AGENT_PROMPT
    assert "${...}" in prompts.DATA_AGENT_PROMPT
    assert "does NOT substitute bind variables" in prompts.DATA_AGENT_PROMPT
    assert "MALFORMED_QUERY" in prompts.DATA_AGENT_PROMPT


def test_prompt_requires_literal_id_for_ownership_reads():
    assert "getUserInfo" in prompts.DATA_AGENT_PROMPT
    assert "LITERAL Salesforce ID" in prompts.DATA_AGENT_PROMPT
    assert "WHERE OwnerId = '005..." in prompts.DATA_AGENT_PROMPT
    # Raw-soql section forbids bind vars; it must not invite `:$User.Id` as an option.
    soql_section = prompts.DATA_AGENT_PROMPT.split("TOOL PURPOSE RULES")[0]
    assert "you must call `getUserInfo` first" not in soql_section


def test_ownership_filtered_read_inlines_literal_user_id():
    # A correct ownership read resolves the user via getUserInfo (supporting step)
    # then issues soqlQuery with the literal OwnerId — never binding syntax.
    llm = _RunLLM([
        {"id": "g1", "name": "getUserInfo", "arguments": {}},
        _tc("soqlQuery", "SELECT Id, Name FROM Contact WHERE OwnerId = '005000000000000001' LIMIT 200"),
    ])

    class _ExecOwned(_Exec):
        async def execute(self, name, arguments):
            self.executed.append((name, arguments))
            if name == "getUserInfo":
                return json.dumps({"id": "005000000000000001", "display_name": "Himanshu"})
            return json.dumps({"totalSize": 2, "records": [{"Id": "003A", "Name": "Ada"}], "done": True})

    exec_ = _ExecOwned()
    orch = _build(llm, exec_)
    events = _run(orch, "Show my Contacts.")
    assert orch._generate_plan.call_count == 0
    names = [n for n, _ in exec_.executed]
    assert "getUserInfo" in names
    assert "soqlQuery" in names
    executed_qs = [a.get("q") for n, a in exec_.executed if n == "soqlQuery"]
    assert any("WHERE OwnerId = '005000000000000001'" in q for q in executed_qs)
    # No binding syntax may be present in any emitted query.
    for q in executed_qs:
        assert ":$User" not in q
        assert ":param" not in q
        assert "${" not in q


def test_prompt_steers_contact_reads_off_binding_syntax():
    assert "NEVER use `:$User.Id`" in prompts.DATA_AGENT_PROMPT
    assert ":id" in prompts.DATA_AGENT_PROMPT


if __name__ == "__main__":
    import unittest
    unittest.main(module=__name__)