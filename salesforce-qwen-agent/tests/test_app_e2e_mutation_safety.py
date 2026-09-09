"""
Local-application end-to-end mutation safety tests (#12).

Drives the REAL FastAPI app (/chat endpoint, session_manager wiring,
Orchestrator, real ToolExecutor + validation gate) with a recording stub MCP
client and a scripted LLM — no live Salesforce / MCP / Qwen required.

Proves at the application layer:
- "create a lead" is BLOCKED and the app ASKS for Last Name + Company Name;
  nothing reaches the MCP/REST client.
- "Create a lead. Last Name: Sharma, Company: Tech Solutions." executes EXACTLY
  once and forwards the exact user-provided body.
"""

import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module  # noqa: E402
from agent.multi_agent import Orchestrator  # noqa: E402
from sfmcp.executor import ToolExecutor  # noqa: E402
from sfmcp.registry import ToolRegistry  # noqa: E402


# ─────────────────────────────────────────────────────────────
# Stubs
# ─────────────────────────────────────────────────────────────

_MCP_DESCRIBE = {
    "Lead":        [("LastName", "Last Name"), ("Company", "Company Name")],
    "Account":     [("Name", "Account Name")],
    "Opportunity": [("Name", "Opportunity Name"), ("StageName", "Stage"),
                    ("CloseDate", "Close Date")],
    "Case":        [],
    "Task":        [("Subject", "Subject")],
}


class _RecordingMcpClient:
    """Records EVERY call_tool invocation — a single write proves the create
    reached the Salesforce transport layer."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, tool_name, arguments):
        self.calls.append((tool_name, dict(arguments)))
        return json.dumps({"id": "00Qe2e"})

    async def describe_required_fields(self, sobject_name):
        return _MCP_DESCRIBE.get(sobject_name)


class _ScriptedLLM:
    """Emits the scripted tool calls once, then synthesizes from the (no-tool)
    final call — mirroring the production Qwen worker + synthesizer."""

    def __init__(self, tool_calls, ask_for_fields=False):
        self.tool_calls = tool_calls
        self.ask_for_fields = ask_for_fields
        self.tool_llm_calls = 0
        self.chat_calls = 0

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        self.tool_llm_calls += 1
        return {"content": "", "tool_calls": list(self.tool_calls), "finish_reason": "tool_calls"}

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        self.chat_calls += 1
        if self.ask_for_fields:
            return (
                "To create the lead I need the required fields: "
                "Last Name and Company Name. Please provide them."
            )
        return "Created the lead for Sharma at Tech Solutions."


class _SafePlanner:
    def has_pending_confirmation(self, session_id="default"):
        return False

    def check_tool_safety(self, tool_name, arguments, session_id="default"):
        return {"safe": True, "requires_confirmation": False, "confirmation_message": "",
                "pending_action": None, "blocked_message": ""}


class _AsyncMockCall:
    def __init__(self, return_value=None):
        self.return_value = return_value

    async def __call__(self, *args, **kwargs):
        return self.return_value


def _build_agent(llm: _ScriptedLLM, mcp: _RecordingMcpClient) -> Orchestrator:
    registry = ToolRegistry()
    registry._load_local_tools()
    executor = ToolExecutor(mcp, registry)
    orch = Orchestrator(llm=llm, executor=executor, max_iterations=5, max_history=4)
    orch.safety_planner = _SafePlanner()
    orch._generate_plan = _AsyncMockCall(return_value=[{
        "task_id": 1, "description": "create records", "agent": "ActionAgent", "depends_on": [],
    }])
    orch._get_relevant_tools_or_fallback = _AsyncMockCall(return_value=[])
    return orch


class _FakeSessionManager:
    def __init__(self, agent):
        self._agent = agent

    async def get_or_create_agent(self, session_id="default"):
        return self._agent


async def _chat(agent: Orchestrator, message: str) -> dict:
    old = app_module.session_manager
    app_module.session_manager = _FakeSessionManager(agent)
    try:
        transport = httpx.ASGITransport(app=app_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/chat", json={"message": message, "session_id": "e2e"})
        return resp.json()
    finally:
        app_module.session_manager = old


# ─────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────

def test_e2e_create_a_lead_blocked_and_asks_for_fields():
    """Plain 'create a lead' through the running app: NO create reaches the
    Salesforce client and the user is asked for Last Name + Company Name."""
    mcp = _RecordingMcpClient()
    llm = _ScriptedLLM(
        tool_calls=[{
            "id": "t1", "name": "createSobjectRecord",
            "arguments": {"sobject-name": "Lead", "body": {}},
        }],
        ask_for_fields=True,
    )
    agent = _build_agent(llm, mcp)

    response = asyncio_run(_chat(agent, "create a lead"))

    assert response.get("success") is True
    assert mcp.calls == [], "a vague 'create a lead' must never reach Salesforce"
    # The assistant asks the user for the missing required fields.
    answer = (response.get("answer") or "").lower()
    assert "last name" in answer and "company" in answer, f"must ask for fields: {answer}"
    # The app surfaced the (blocked) tool call in metadata, but it never executed.
    tool_calls = response.get("metadata", {}).get("tool_calls") or []
    assert any(tc.get("name") == "createSobjectRecord" for tc in tool_calls)
    # The tool-calling model was not re-invoked (no automatic retry).
    assert llm.tool_llm_calls == 1


def test_e2e_create_a_lead_with_fabricated_values_asks_for_required_fields():
    """PROBLEM 1: the tool-calling LLM HALLUCINATES a Lead body (Doe/Acme Corp/
    j.doe@acme.com) for a vague 'create a lead'. The running app must reject the
    fabricated create deterministically, never reach the Salesforce transport,
    and ask ONLY for the true CREATEABLE-AND-REQUIRED fields (Last Name +
    Company Name) — Email is optional for Lead and must NOT be demanded, and no
    invented value may leak into the ask."""
    mcp = _RecordingMcpClient()
    llm = _ScriptedLLM(
        tool_calls=[{
            "id": "t1", "name": "createSobjectRecord",
            "arguments": {"sobject-name": "Lead",
                          "body": {"LastName": "Doe", "Company": "Acme Corp",
                                   "Email": "j.doe@acme.com"}},
        }],
        ask_for_fields=True,
    )
    agent = _build_agent(llm, mcp)

    response = asyncio_run(_chat(agent, "create a lead"))

    assert response.get("success") is True
    assert mcp.calls == [], "a fabricated create must never reach the Salesforce client"
    answer = response.get("answer") or ""
    lower = answer.lower()
    assert "last name" in lower and "company" in lower, \
        f"must ask for the true required fields: {answer}"
    # Email was invented by the LLM and is OPTIONAL for Lead — not demanded.
    assert "email" not in lower, "the ask must not demand fabricated optional fields"
    # No invented value may leak into the ask.
    for invented in ("Doe", "Acme Corp", "j.doe@acme.com", "Acme"):
        assert invented not in answer, f"fabricated value {invented!r} leaked into the ask"
    # The ask is deterministic — the synthesizer LLM is never invoked.
    assert llm.chat_calls == 0
    # The tool-calling model was not re-invoked (no automatic retry).
    assert llm.tool_llm_calls == 1


def test_e2e_valid_create_executes_exactly_once():
    """'Create a lead. Last Name: Sharma, Company: Tech Solutions.' forwards the
    exact user-provided body to Salesforce exactly once."""
    mcp = _RecordingMcpClient()
    llm = _ScriptedLLM(
        tool_calls=[{
            "id": "t1", "name": "createSobjectRecord",
            "arguments": {"sobject-name": "Lead",
                          "body": {"LastName": "Sharma", "Company": "Tech Solutions"}},
        }],
        ask_for_fields=False,
    )
    agent = _build_agent(llm, mcp)

    response = asyncio_run(_chat(agent, "Create a lead. Last Name: Sharma, Company: Tech Solutions."))

    assert response.get("success") is True
    assert len(mcp.calls) == 1, "valid create must reach Salesforce exactly once"
    name, args = mcp.calls[0]
    assert name == "createSobjectRecord"
    assert args["body"] == {"LastName": "Sharma", "Company": "Tech Solutions"}
    assert "sharma" in (response.get("answer") or "").lower()


def test_e2e_batch_any_failure_blocks_zero_mutations():
    """REGRESSION (#8 all-or-nothing) at the app layer: one response emits a
    VALID Account create first and a vague/invalid Lead create second. The valid
    Account create must NOT execute at all — ANY validation failure means ZERO
    mutations from that response reach Salesforce."""
    mcp = _RecordingMcpClient()
    llm = _ScriptedLLM(
        tool_calls=[
            {
                "id": "t1", "name": "createSobjectRecord",
                "arguments": {"sobject-name": "Account", "body": {"Name": "Acme"}},
            },
            {
                "id": "t2", "name": "createSobjectRecord",
                "arguments": {"sobject-name": "Lead", "body": {}},
            },
        ],
        ask_for_fields=True,
    )
    agent = _build_agent(llm, mcp)

    response = asyncio_run(_chat(agent, "Create an account named Acme and a lead"))

    assert response.get("success") is True
    assert mcp.calls == [], \
        "a validation failure in the batch must block even the earlier valid Account create"
    answer = (response.get("answer") or "").lower()
    assert "last name" in answer and "company" in answer, f"must ask for fields: {answer}"
    # Both tool calls were announced but NEITHER executed (mcp.calls is empty).
    tool_calls = response.get("metadata", {}).get("tool_calls") or []
    assert len(tool_calls) == 2, "both mutations must be announced"
    # The tool-calling model was invoked exactly once — no automatic retry.
    assert llm.tool_llm_calls == 1


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))