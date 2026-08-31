"""
Temporary diagnostic: Is the Salesforce MCP server actually being called,
and does its tool list match the tool names the RAG system selects?

Verifies:
1. MCP client can connect (Streamable HTTP SDK session)
2. What tools the MCP server exposes (names)
3. Whether RAG-selected names (soqlQuery, getUserInfo, listRecentSobjectRecords) are real MCP names
4. Whether a real call_tool(soqlQuery) hits the MCP SDK session or the REST fallback
"""

import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

from sfmcp.client import SalesforceMCPClient  # noqa: E402
from sfmcp.registry import ToolRegistry  # noqa: E402
from sfmcp.executor import ToolExecutor  # noqa: E402
from sfmcp.crypto.envelope import token_vault  # noqa: E402
from tools.salesforce import get_tool_definitions  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)


async def main():
    print("=" * 70)
    print("DIAGNOSTIC: Is Salesforce MCP actually being called?")
    print("=" * 70)

    # Show what RAG selects as tool names
    rag_names = [t["function"]["name"] for t in get_tool_definitions()]
    print(f"\n[RAG] Local tool names the retriever selects:")
    for n in rag_names:
        print(f"    - {n}")

    # Build the MCP client from env just like app.py does
    client = SalesforceMCPClient(
        mcp_url=os.getenv("SALESFORCE_MCP_URL", ""),
        instance_url=os.getenv("SALESFORCE_INSTANCE_URL", ""),
        client_id=os.getenv("SALESFORCE_CLIENT_ID", ""),
        client_secret=os.getenv("SALESFORCE_CLIENT_SECRET", ""),
        username=os.getenv("SALESFORCE_USERNAME", ""),
        password=os.getenv("SALESFORCE_PASSWORD", ""),
        security_token=os.getenv("SALESFORCE_SECURITY_TOKEN", ""),
        domain=os.getenv("SALESFORCE_DOMAIN", "login"),
        oauth_scope=os.getenv("SALESFORCE_OAUTH_SCOPE"),
        token_vault=token_vault,
    )

    print(f"\n[MCP] Target URL: {client.mcp_url}")
    print(f"[MCP] Instance: {client.instance_url}")
    print(f"[MCP] is_connected (before): {client.is_connected}")

    # 1. Try to connect
    print("\n[STEP 1] Calling mcp_client.connect()...")
    try:
        await client.connect()
        print(f"[MCP] is_connected (after connect): {client.is_connected}")
        print(f"[MCP] has session: {client._session is not None}")
    except Exception as e:
        print(f"[MCP] connect() RAISED: {type(e).__name__}: {e}")

    # 2. List tools from MCP
    print("\n[STEP 2] Calling registry.initialize(mcp_client) / list_tools...")
    registry = ToolRegistry()
    try:
        await registry.initialize(client)
    except Exception as e:
        print(f"[MCP] initialize RAISED: {type(e).__name__}: {e}")

    print(f"[MCP] Registry {len(registry)} tools:")
    mcp_tool_names = registry.list_tool_names()
    for n in mcp_tool_names:
        print(f"    - {n}")

    # 3. Check overlap between MCP names and RAG-selected names
    rag_set = set(rag_names)
    overlap = rag_set & set(mcp_tool_names)
    missing = rag_set - set(mcp_tool_names)
    print(f"\n[MATCH] RAG tool names present in registry: {sorted(overlap)}")
    print(f"[MATCH] RAG tool names MISSING from registry: {sorted(missing)}")

    # 4. Try executing soqlQuery through the FULL executor path
    print("\n[STEP 3] Executing soqlQuery via ToolExecutor (full pipeline)...")
    executor = ToolExecutor(client, registry)
    result = await executor.execute("soqlQuery", {"q": "SELECT Id, Name FROM Account LIMIT 1"})
    # Don't dump full sensitive data — just show whether it succeeded
    print(f"[EXEC] soqlQuery result (first 400 chars): {str(result)[:400]}")

    # 5. Show whether the MCP SDK session was actually used for the call
    print(f"\n[VERDICT] MCP session object present: {client._session is not None}")
    print(f"[VERDICT] is_connected: {client.is_connected}")

    await client.disconnect()


if __name__ == "__main__":
    # Set a short total timeout to avoid hanging on network
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=90))
    except asyncio.TimeoutError:
        print("\n[ERROR] Diagnostic timed out after 90s (network hang).")
