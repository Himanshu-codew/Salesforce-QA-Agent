"""
Live speed test — queries agent at localhost:8000 and measures response time.
"""
import asyncio
import sys
import time
import json
import httpx

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"

TEST_QUERIES = [
    ("Simple fetch",        "Show me all Accounts"),
    ("Filter query",        "Show me all Leads with status Open"),
    ("Count query",         "How many Contacts are there in total?"),
    ("Specific record",     "Find the Account named Edge Communications"),
    ("Multi-object",        "Show me all Opportunities that are Closed Won"),
    ("User info",           "Who am I? What is my user info?"),
    ("Schema lookup",       "What fields does the Lead object have?"),
    ("Date filter",         "Show me Leads created this month"),
]

async def run_query(client: httpx.AsyncClient, label: str, query: str, session_id: str) -> dict:
    start = time.perf_counter()
    tool_calls_seen = []
    response_text = ""

    try:
        resp = await client.post(
            f"{BASE_URL}/chat",
            json={"message": query, "session_id": session_id},
            timeout=120.0,
        )
        if resp.status_code == 200:
            res_data = resp.json()
            response_text = res_data.get("response", "")
            tool_calls = res_data.get("tool_calls", [])
            for tc in tool_calls:
                tool_calls_seen.append(tc.get("name", "?"))
        else:
            return {
                "label": label,
                "query": query,
                "elapsed": -1,
                "error": f"HTTP {resp.status_code}: {resp.text}",
                "tools": [],
                "preview": "",
            }
    except Exception as e:
        return {"label": label, "query": query, "elapsed": -1, "error": str(e), "tools": [], "preview": ""}

    elapsed = time.perf_counter() - start
    preview = response_text[:120].replace("\n", " ").strip() if response_text else "(no text)"

    return {
        "label": label,
        "query": query,
        "elapsed": elapsed,
        "tools": tool_calls_seen,
        "preview": preview,
        "error": None,
    }


async def main():
    print("=" * 65)
    print("  LOCAL MODEL SPEED TEST — qwen2.5:latest via Ollama")
    print("=" * 65)

    # Wait for server to be ready
    async with httpx.AsyncClient() as probe:
        for _ in range(10):
            try:
                r = await probe.get(f"{BASE_URL}/health", timeout=3.0)
                if r.status_code < 500:
                    print("Server: READY\n")
                    break
            except Exception:
                pass
            await asyncio.sleep(1)

    session_id = f"speedtest_{int(time.time())}"
    results = []

    async with httpx.AsyncClient() as client:
        for i, (label, query) in enumerate(TEST_QUERIES, 1):
            print(f"[{i}/{len(TEST_QUERIES)}] {label}...")
            r = await run_query(client, label, query, session_id)
            results.append(r)
            if r["error"]:
                print(f"  ERROR: {r['error']}")
            else:
                tools_str = ", ".join(r["tools"]) if r["tools"] else "NONE (no tool call!)"
                print(f"  Time:    {r['elapsed']:.1f}s")
                print(f"  Tools:   {tools_str}")
                print(f"  Preview: {r['preview'][:80]}...")
            print()

    # Summary table
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"{'#':<3} {'Query':<30} {'Time':>6}  {'Tools Called':<25}")
    print("-" * 65)
    good = 0
    for i, r in enumerate(results, 1):
        if r["error"]:
            status = "ERROR"
            t = "  -  "
        else:
            t = f"{r['elapsed']:>5.1f}s"
            status = ", ".join(r["tools"]) if r["tools"] else "NO TOOL"
            if r["tools"]:
                good += 1
        print(f"{i:<3} {r['label']:<30} {t}  {status:<25}")

    print("-" * 65)
    valid = [r for r in results if not r["error"]]
    if valid:
        avg = sum(r["elapsed"] for r in valid) / len(valid)
        fastest = min(r["elapsed"] for r in valid)
        slowest = max(r["elapsed"] for r in valid)
        print(f"\nAvg response: {avg:.1f}s  |  Fastest: {fastest:.1f}s  |  Slowest: {slowest:.1f}s")
        print(f"Tool calls:   {good}/{len(results)} queries called MCP tools correctly")
        print()
        if avg < 15:
            verdict = "FAST — production ready"
        elif avg < 30:
            verdict = "ACCEPTABLE — usable but could be faster"
        else:
            verdict = "SLOW — consider smaller model or GPU"
        print(f"VERDICT: {verdict}")

asyncio.run(main())
