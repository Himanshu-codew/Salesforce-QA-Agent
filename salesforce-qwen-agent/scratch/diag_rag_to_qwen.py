"""
Temporary diagnostic: Verify the selected tools from RAG actually reach Qwen
(chat_with_tools). Mocks the LLM and executor, uses the REAL Orchestrator /
SalesforceAgent and REAL ToolRAGRetriever. Confirms the full flow:
RAG -> selected tools -> chat_with_tools(tools=...)
"""

import asyncio
import logging
import sys
import os
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["ENABLE_RAG_TOOLS"] = "true"

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

from agent.agent import SalesforceAgent  # noqa: E402
from sfmcp.executor import ToolExecutor  # noqa: E402


class FakeLLM:
    """Records the tools passed into chat_with_tools."""

    def __init__(self):
        self.calls = []

    async def chat(self, messages, temperature=0.0, max_tokens=8192):
        return "Mock final response"

    async def chat_with_tools(self, messages, tools, temperature=0.0, max_tokens=8192):
        self.calls.append({"messages": messages, "tools": tools})
        selected = [t["function"]["name"] for t in tools]
        # Simulate an LLM deciding to call soqlQuery when provided
        if "soqlQuery" in selected and "always-return-tool" in str(messages):
            return {
                "content": "",
                "tool_calls": [{"id": "x1", "name": "soqlQuery", "arguments": {"q": "SELECT Id FROM Account LIMIT 1"}}],
                "finish_reason": "tool_calls",
            }
        return {"content": "Mock answer", "tool_calls": [], "finish_reason": "stop"}

    async def close(self):
        pass


async def main():
    llm = FakeLLM()
    # Mock executor so no real Salesforce call is made
    executor = MagicMock(spec=ToolExecutor)
    executor.execute = AsyncMock(return_value='{"totalSize": 1, "records": []}')

    agent = SalesforceAgent(llm=llm, executor=executor, max_iterations=1)

    queries = [
        ("Who am I? Show my profile", "user-info"),
        ("Show my recent Accounts", "recent-records"),
        ("What is the weather today?", "weather"),
    ]

    for q, label in queries:
        print("\n########## FLOW TEST:", label, "|", q, "##########")
        events = []
        async for ev in agent.process_message(q, session_id=f"diag_{label}"):
            events.append(ev)
        # Inspect the tools that reached chat_with_tools
        if llm.calls:
            tools_sent = [t["function"]["name"] for t in llm.calls[-1]["tools"]]
            print("=> TOOLS REACHING QWEN chat_with_tools:", tools_sent)
        else:
            print("=> No chat_with_tools call occurred")
        print("=> Event types emitted:", [e["type"] for e in events])


if __name__ == "__main__":
    asyncio.run(main())
