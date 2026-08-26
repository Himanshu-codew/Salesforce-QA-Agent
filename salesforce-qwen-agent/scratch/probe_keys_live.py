"""Live probe: which hosts accept which Consumer Key (read-only authorize GETs)."""
import asyncio
import base64
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv("salesforce-qwen-agent/.env")

default_key = base64.b64decode(
    "M01WRzk3TDdQV2JQcTZVelRTTzAyUTBZeEdSQ1hMRWljVmtXb0dEQnZtX2trcEpGMlBoeFdSRmpEanZTQnl0NjE4TDk0NmxiQmdUZWpqa3h5Y19IbQ=="
).decode()
old_key_prefix = "3MVG97L7PwbPq6UzTSO02Q0YxGf7HtzS"
env_key = os.getenv("SALESFORCE_CLIENT_ID", "").strip()

keys = {
    "DEFAULT_APP_KEY (hardcoded)": default_key,
    "OLD dead-prefix key": old_key_prefix,
}
if env_key and env_key != default_key:
    keys[".env SALESFORCE_CLIENT_ID"] = env_key

hosts = [
    "login.salesforce.com",
    "orgfarm-d5054a6252-dev-ed.develop.my.salesforce.com",
]


def mask(k):
    return f"{k[:12]}…{k[-6:]}" if len(k) > 24 else k


async def main():
    async with httpx.AsyncClient(timeout=15.0) as h:
        for kname, k in keys.items():
            for host in hosts:
                url = (
                    f"https://{host}/services/oauth2/authorize?"
                    f"response_type=code&client_id={k}&redirect_uri=http://localhost/probe"
                )
                try:
                    r = await h.get(url, follow_redirects=False)
                    loc = r.headers.get("location", "")
                    verdict = ""
                    if "error=" in loc:
                        verdict = loc.split("?")[1]
                    elif r.status_code in (200, 302) and loc:
                        verdict = f"-> redirects to login page ({loc.split('?')[0][:80]})"
                    else:
                        verdict = r.text[:120].replace("\n", " ")
                    print(f"{mask(k):28} @ {host:60} HTTP {r.status_code}  {verdict}")
                except Exception as e:
                    print(f"{mask(k):28} @ {host:60} ERROR {type(e).__name__}: {e}")


asyncio.run(main())
