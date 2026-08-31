"""Temporary probe: raw HTTP against the hosted MCP endpoint using the scoped token.
Shows the actual status code + body so we can tell 401 (auth) from a protocol error.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)

import httpx2  # noqa: E402
from sfmcp.crypto.envelope import token_vault  # noqa: E402


async def main():
    scope_sid = None
    access_token = None
    for sid in token_vault.sessions():
        rec = token_vault.get(sid)
        if rec and rec.get("oauth_scope") and "sfap:mcp" in rec.get("oauth_scope", ""):
            scope_sid = sid
            access_token = rec.get("access_token") or ""
            break

    if not access_token:
        print("[VAULT] No scoped token found.")
        return

    url = os.getenv("SALESFORCE_MCP_URL", "")
    print(f"[MCP] URL: {url}")
    print(f"[MCP] token scoped: True (len={len(access_token)})")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "sf-probe", "version": "1.0"},
        },
    }

    async with httpx2.AsyncClient(timeout=60.0) as http:
        try:
            res = await http.post(url, json=payload, headers=headers)
            print(f"\n[MCP] Status: {res.status_code}")
            print(f"[MCP] Body (first 800 chars):\n{res.text[:800]}")
        except Exception as e:
            print(f"[MCP] EXC: {type(e).__name__}: {e}")

        inst = token_vault.get(scope_sid).get("instance_url") or os.getenv(
            "SALESFORCE_INSTANCE_URL", ""
        )
        rest_url = f"{inst.rstrip('/')}/services/data/v62.0/limits"
        try:
            res = await http.get(rest_url, headers=headers)
            print(f"\n[REST limits] Status: {res.status_code}")
            print(f"[REST limits] Body (first 400 chars):\n{res.text[:400]}")
        except Exception as e:
            print(f"[REST limits] EXC: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
