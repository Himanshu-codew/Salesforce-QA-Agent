"""
End-to-end output-cleanliness test.

Verifies that a Salesforce query results in a clean, natural-language
chatbot response and that the user-facing delivery NEVER contains raw
tool-call JSON, XML function calls, tool schemas, debug logs, or parser
output — even when Qwen or the agent yields leaked/invalid tool-call content.

The last-mile guard (`finalize_user_response`, applied in app.py at the
delivery boundary) is the authoritative guarantee under test here.
"""

import asyncio
import json
import pytest

from agent.agent import finalize_user_response
from agent.multi_agent import Orchestrator
from agent.rag import ToolRAGRetriever

# Markers / syntax that must NEVER appear in user-facing output.
_FORBIDDEN = [
    '"name"', '"arguments"', '"function"', '"parameters"',
    '<tool_call>', '</tool_call>', '<tools>', '</tools>',
    '[TOOL_CALLS]', 'tool_calls', 'function_calls',
    '[RAG DEBUG]', '[MCP]', 'Traceback',
    '```json', '```xml', 'soqlQuery', 'getObjectSchema',
]


def _assert_clean(text: str) -> None:
    """Assert delivered user text is clean natural language / markdown."""
    assert text, "response must not be empty"
    lower = text.lower()
    for marker in _FORBIDDEN:
        assert marker.lower() not in lower, f"raw artifact leaked: {marker!r} in {text!r}"


class _MockLLM:
    """Drives the multi-agent flow deterministically."""

    def __init__(self, plan, worker_llm_result, synth_response, synth_raises=False):
        self.plan = plan
        self.worker_llm_result = worker_llm_result
        self.synth_response = synth_response
        self.synth_raises = synth_raises
        self.chat_calls = 0

    async def chat(self, messages, temperature=0.0, max_tokens=8192):
        self.chat_calls += 1
        # Planner call (first) returns the plan; synthesizer call returns text.
        if self.chat_calls == 1:
            return self.plan
        if self.synth_raises:
            raise RuntimeError("synthesizer exploded")
        return self.synth_response

    async def chat_with_tools(self, messages, tools, temperature=0.0, max_tokens=8192):
        return self.worker_llm_result

    async def close(self):
        pass


class _MockExecutor:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def execute(self, tool_name, arguments):
        self.calls.append((tool_name, arguments))
        return self.result


_PLAN = '[{"task_id": 0, "description": "List recent Accounts", "agent": "DataAgent", "depends_on": []}]'
_SOQL_RESULT = json.dumps({
    "totalSize": 1,
    "records": [{"attributes": {"type": "Account"}, "Id": "001g500000V9LDcAAN", "Name": "Acme Corp"}],
})


def _run(agent, message="Show me my recent Accounts"):
    events = []

    async def _collect():
        async for event in agent.process_message(message, session_id="e2e"):
            events.append(event)

    asyncio.run(_collect())
    return events


def test_salesforce_query_yields_clean_natural_response():
    llm = _MockLLM(
        plan=_PLAN,
        worker_llm_result={
            "content": "",
            "tool_calls": [{"id": "tc1", "name": "soqlQuery",
                            "arguments": {"q": "SELECT Id, Name FROM Account LIMIT 5"}}],
            "finish_reason": "tool_calls",
        },
        synth_response="### Accounts Found\n\n| Name |\n| --- |\n| Acme Corp |",
    )
    executor = _MockExecutor(_SOQL_RESULT)
    agent = Orchestrator(llm=llm, executor=executor)

    events = _run(agent)
    responses = [e["data"] for e in events if e["type"] == "response"]

    assert responses, "expected at least one response event"
    # The tool actually executed against the (mock) Salesforce result.
    assert executor.calls and executor.calls[0][0] == "soqlQuery"
    for text in responses:
        _assert_clean(text)


def test_raw_tool_json_from_qwen_never_reaches_user():
    # Simulate the worst case: the agent/synthesizer yields raw tool-call JSON
    # that would otherwise leak to the user. The last-mile guard must strip it.
    leaked = (
        '[{"name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Account"}}]\n'
        'Here are your accounts.'
    )
    cleaned = finalize_user_response(leaked)
    _assert_clean(cleaned)
    assert "Here are your accounts." in cleaned


def test_xml_tool_call_never_reaches_user():
    leaked = '<tools><tool_call>{ "name": "getUserInfo", "arguments": {} }</tool_call></tools>'
    cleaned = finalize_user_response(leaked)
    _assert_clean(cleaned)


def test_malformed_tool_json_returns_clean_user_facing_error():
    # If Qwen emits malformed/invalid tool-call JSON it must be handled
    # internally; the user only ever sees a clean message.
    malformed = '{"name": "soqlQuery", "arguments": { "q": "SELECT FROM" }'  # unterminated
    cleaned = finalize_user_response(malformed)
    _assert_clean(cleaned)


def test_debug_logs_never_reach_user():
    leaked = '[RAG DEBUG] Selected tools: soqlQuery\n[TOOL CALLS] finished\nYour data is ready.'
    cleaned = finalize_user_response(leaked)
    _assert_clean(cleaned)


def test_clean_markdown_table_is_preserved():
    markdown = "### Accounts Found\n\n| Name | Industry |\n| --- | --- |\n| Acme | Tech |"
    cleaned = finalize_user_response(markdown)
    assert cleaned == markdown
    _assert_clean(cleaned)
