"""Temporary probe: does Salesforce OAuth password grant honor the sfap:mcp scope?

Only prints scope/instance info, never token values.
"""

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

load_dotenv(override=True)

DOMAIN = os.getenv("SALESFORCE_DOMAIN", "login")
SCOPE = os.getenv("SALESFORCE_OAUTH_SCOPE", "")


async def probe(grant_scope: str | None):
    token_url = f"https://{DOMAIN}.salesforce.com/services/oauth2/token"
    payload = {
        "grant_type": "password",
        "client_id": os.getenv("SALESFORCE_CLIENT_ID", ""),
        "client_secret": os.getenv("SALESFORCE_CLIENT_SECRET", ""),
        "username": os.getenv("SALESFORCE_USERNAME", ""),
        "password": os.getenv("SALESFORCE_PASSWORD", "") + os.getenv("SALESFORCE_SECURITY_TOKEN", ""),
    }
    if grant_scope:
        payload["scope"] = grant_scope

    label = f"scope={grant_scope!r}" if grant_scope else "no scope param"
    print(f"\n--- OAuth password grant ({label}) ---")
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            res = await http.post(token_url, data=payload)
            print(f"HTTP {res.status_code}")
            if res.status_code == 200:
                data = res.json()
                # print scope/audience only, NEVER the tokens
                print(f"  returned scope: {data.get('scope')!r}")
                print(f"  token_type: {data.get('token_type')!r}")
                print(f"  instance_url: {data.get('instance_url')!r}")
                print(f"  audience: {data.get('sfdc_audience')!r}")
                print(f"  has_access_token: {bool(data.get('access_token'))}")
                print(f"  has_refresh_token: {bool(data.get('refresh_token'))}")
            else:
                try:
                    j = res.json()
                    print(f"  error={j.get('error')!r} desc={j.get('error_description')!r}")
                except Exception:
                    print(f"  body: {res.text[:300]}")
    except Exception as e:
        print(f"  EXC: {type(e).__name__}: {e}")


async def main():
    print(f"Using scope from env: {SCOPE!r}")
    await probe(None)
    await probe(SCOPE)
    await probe("sfap:mcp:all")


if __name__ == "__main__":
    asyncio.run(main())
