"""Temporary probe: try to open the hosted MCP session using the sfap:mcp-scoped
token already stored in the token vault (from a prior web OAuth login).

Prints only metadata + outcome; never the token value.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)
from sfmcp.crypto.envelope import token_vault  # noqa: E402
from sfmcp.client import SalesforceMCPClient  # noqa: E402


async def main():
    # Find the first vault session that carries an sfap:mcp-scoped token
    scope_sid = None
    access_token = None
    inst = None
    for sid in token_vault.sessions():
        rec = token_vault.get(sid)
        if rec and rec.get("oauth_scope") and "sfap:mcp" in rec.get("oauth_scope", ""):
            scope_sid = sid
            access_token = rec.get("access_token") or ""
            inst = rec.get("instance_url") or ""
            break

    if not access_token:
        print("[VAULT] No sfap:mcp-scoped token found in vault. Cannot probe MCP.")
        return

    print(f"[VAULT] Using scoped token from session '{scope_sid}'")
    print(f"[VAULT] scope has sfap:mcp: {True}")
    print(f"[VAULT] instance: {inst}")
    print(f"[VAULT] token length: {len(access_token)}  (appears to be an OAuth token: {access_token[:4]}...)")

    client = SalesforceMCPClient(
        mcp_url=os.getenv("SALESFORCE_MCP_URL", ""),
        instance_url=inst or os.getenv("SALESFORCE_INSTANCE_URL", ""),
        client_id=os.getenv("SALESFORCE_CLIENT_ID", ""),
        client_secret=os.getenv("SALESFORCE_CLIENT_SECRET", ""),
        username="", password="", security_token="",
        domain=os.getenv("SALESFORCE_DOMAIN", "login"),
        access_token=access_token,
        oauth_scope=os.getenv("SALESFORCE_OAUTH_SCOPE", ""),
    )

    print("\n[MCP] Target URL:", client.mcp_url)
    try:
        await client._ensure_connected()
    except Exception as e:
        print(f"[MCP] _ensure_connected RAISED: {type(e).__name__}: {e}")

    print(f"[MCP] is_connected: {client.is_connected}")
    if client._session is not None:
        print("[MCP] MCP SDK session OPEN (Streamable HTTP) — token accepted!")
        tools = await client.list_tools()
        print(f"[MCP] list_tools returned {len(tools)} tool(s):")
        for t in tools[:40]:
            name = t.get("name") or t.get("function", {}).get("name") or "?"
            print(f"    - {name}")
        await client.disconnect()
    else:
        print("[MCP] MCP session could NOT be opened with the scoped token.")


if __name__ == "__main__":
    try:
        asyncio.run(asyncio.wait_for(main(), timeout=90))
    except asyncio.TimeoutError:
        print("\n[ERROR] Timed out after 90s (network hang).")
