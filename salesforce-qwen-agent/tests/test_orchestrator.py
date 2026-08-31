"""
Orchestrator (multi_agent) acceptance tests.

Verifies the critical production behavior:
  EMPTY PLAN != ERROR

For non-Salesforce queries (greetings, thanks, general knowledge) the Planner
returns [] and the Orchestrator MUST route the ORIGINAL user query to Qwen for
a natural-language response — it must NOT return
"I couldn't understand how to break down your request."

For Salesforce queries the existing multi-step workflow must be preserved.

All tests use mocks (no live LLM / Salesforce / embedding model).
"""

import os
import sys
import asyncio
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.multi_agent import Orchestrator

SAFE = {"safe": True, "requires_confirmation": False, "confirmation_message": "", "pending_action": None}


def _make_orchestrator(plan=None, rag_tools=None, chat_return="Hi! How can I help you?"):
    """Build an Orchestrator with fully mocked LLM, executor, planner, and RAG."""
    llm = MagicMockLLM(chat_return)
    executor = MagicMockExecutor()

    orch = Orchestrator(llm=llm, executor=executor, max_iterations=5, max_history=4)

    planner = MagicMockPlanner()
    orch.safety_planner = planner

    rag = MagicMock()
    rag.get_relevant_tools = MagicMock(return_value=rag_tools or [])
    orch.rag_retriever = rag

    orch._generate_plan = AsyncMock(return_value=plan if plan is not None else [])

    return orch


class MagicMockLLM:
    """A minimal async LLM double capturing calls."""
    def __init__(self, chat_return):
        self._chat_return = chat_return
        self.chat_calls = []
        self.last_tool_payload = None

    async def chat(self, messages=None, temperature=0.0, max_tokens=4096):
        self.chat_calls.append(dict(messages=messages, temperature=temperature))
        return self._chat_return

    async def chat_with_tools(self, messages=None, tools=None, temperature=0.0, max_tokens=4096):
        self.last_tool_payload = None
        return {"content": "", "tool_calls": [], "finish_reason": "stop"}


class MagicMockExecutor:
    def __init__(self):
        self.executed = []
        self.result = "[]"

    async def execute(self, name, arguments):
        self.executed.append((name, arguments))
        return self.result


class MagicMockPlanner:
    def has_pending_confirmation(self, session_id):
        return False

    def check_tool_safety(self, tool_name, arguments, session_id="default"):
        return dict(SAFE)


def _last_user_message(llm):
    """Return the last user-message content from the most recent chat() call."""
    if not llm.chat_calls:
        return None
    msgs = llm.chat_calls[-1]["messages"]
    for m in reversed(msgs):
        if m["role"] == "user":
            return m["content"]
    return None


async def _process(orch, message, session_id="default"):
    events = []
    async for event in orch.process_message(message, session_id):
        events.append(event)
    return events


def _responses(events):
    return [e["data"] for e in events if e["type"] == "response"]


def test_empty_plan_routes_to_general_qwen():
    """Empty plan + no RAG tools -> general Qwen answer, original query preserved."""
    orch = _make_orchestrator(plan=[], rag_tools=[], chat_return="Hi! How can I help you today?")
    events = asyncio.run(_process(orch, "hi"))

    responses = _responses(events)
    assert responses, "expected a response event"
    assert "couldn't understand" not in responses[0]
    # Original user query was routed to the LLM unchanged.
    assert _last_user_message(orch.llm) == "hi"


def test_empty_plan_greeting_hello():
    orch = _make_orchestrator(plan=[], rag_tools=[])
    events = asyncio.run(_process(orch, "hello"))
    responses = _responses(events)
    assert responses
    assert "couldn't understand" not in responses[0]


def test_empty_plan_thanks():
    orch = _make_orchestrator(plan=[], rag_tools=[])
    events = asyncio.run(_process(orch, "thanks"))
    responses = _responses(events)
    assert responses
    assert "couldn't understand" not in responses[0]


def test_general_knowledge_question_bypasses_salesforce():
    """What is Python? -> no tools -> general answer, no tool execution."""
    orch = _make_orchestrator(plan=[], rag_tools=[])
    events = asyncio.run(_process(orch, "what is Python?"))
    responses = _responses(events)
    assert responses
    assert "couldn't understand" not in responses[0]
    assert not any(e["type"] == "tool_call" for e in events)


def test_salesforce_query_uses_tools():
    """Show my recent Accounts -> RAG selects tools -> tool execution path."""
    soql = {"type": "function", "function": {"name": "soqlQuery"}}
    orch = _make_orchestrator(plan=[
        {"task_id": 1, "description": "query accounts", "agent": "DataAgent", "depends_on": []}
    ], rag_tools=[soql])
    orch.llm.chat_with_tools = _ChatWithTools([{"id": "t1", "name": "soqlQuery",
                                                "arguments": {"q": "SELECT Id FROM Account"}}])
    events = asyncio.run(_process(orch, "Show my recent Accounts"))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert any(tc["data"]["name"] == "soqlQuery" for tc in tool_calls)


def test_empty_plan_but_salesforce_intent_falls_through():
    """Empty plan but RAG selects tools -> single implicit Salesforce task runs."""
    soql = {"type": "function", "function": {"name": "soqlQuery"}}
    orch = _make_orchestrator(plan=[], rag_tools=[soql])
    orch.llm.chat_with_tools = _ChatWithTools([
        {"id": "t1", "name": "soqlQuery",
         "arguments": {"q": "SELECT Id FROM Account LIMIT 5"}}
    ])
    events = asyncio.run(_process(orch, "Accounts above 50000"))
    tool_calls = [e for e in events if e["type"] == "tool_call"]
    assert any(tc["data"]["name"] == "soqlQuery" for tc in tool_calls), \
        "Salesforce tool should still be executed when RAG detects intent"


class _ChatWithTools:
    """Async double for llm.chat_with_tools returning a fixed tool call set."""
    def __init__(self, tool_calls):
        self._tool_calls = tool_calls

    def __call__(self, *args, **kwargs):
        return _Awaiter(self._tool_calls)


class _Awaiter:
    def __init__(self, tool_calls):
        self._tool_calls = tool_calls

    def __await__(self):
        async def _inner():
            return {"content": "", "tool_calls": list(self._tool_calls),
                    "finish_reason": "tool_calls"}
        return _inner().__await__()


if __name__ == "__main__":
    import unittest
    unittest.main(module=__name__)
