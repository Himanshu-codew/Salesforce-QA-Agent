"""
Fast live speed test with instant unbuffered output.
"""
import asyncio
import time
import sys
import httpx

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"

TEST_QUERIES = [
    ("Simple fetch",        "Show me all Accounts"),
    ("Count query",         "How many Contacts are there in total?"),
    ("Schema lookup",       "What fields does the Lead object have?"),
    ("User info",           "Who am I? What is my user info?"),
]

async def run_test():
    print("=" * 60, flush=True)
    print("  LIVE MODEL SPEED & BENCHMARK TEST", flush=True)
    print("  Model: qwen2.5:latest (Ollama local)", flush=True)
    print("  Server: http://localhost:8000", flush=True)
    print("=" * 60, flush=True)

    session_id = f"speed_{int(time.time())}"
    results = []

    async with httpx.AsyncClient() as client:
        for i, (label, query) in enumerate(TEST_QUERIES, 1):
            print(f"\n[{i}/{len(TEST_QUERIES)}] Query: '{query}' ({label})", flush=True)
            start = time.perf_counter()
            try:
                resp = await client.post(
                    f"{BASE_URL}/chat",
                    json={"message": query, "session_id": session_id},
                    timeout=180.0,
                )
                elapsed = time.perf_counter() - start
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("response", "")
                    tools = [tc.get("name") for tc in data.get("tool_calls", [])]
                    print(f"  ⏱️ Time taken: {elapsed:.2f} seconds", flush=True)
                    print(f"  🛠️ Tools used: {tools if tools else 'None'}", flush=True)
                    print(f"  💬 Response:  {text[:120].strip()}...", flush=True)
                    results.append({"query": query, "time": elapsed, "tools": tools, "success": True})
                else:
                    print(f"  ❌ Error HTTP {resp.status_code}: {resp.text[:100]}", flush=True)
                    results.append({"query": query, "time": elapsed, "tools": [], "success": False})
            except Exception as e:
                elapsed = time.perf_counter() - start
                print(f"  ❌ Exception: {e}", flush=True)
                results.append({"query": query, "time": elapsed, "tools": [], "success": False})

    print("\n" + "=" * 60, flush=True)
    print("  SPEED SUMMARY & PERFORMANCE RESULTS", flush=True)
    print("=" * 60, flush=True)
    times = [r["time"] for r in results if r["success"]]
    if times:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        print(f"  Average Speed: {avg_time:.2f} seconds / query", flush=True)
        print(f"  Fastest Query: {min_time:.2f} seconds", flush=True)
        print(f"  Slowest Query: {max_time:.2f} seconds", flush=True)
        print(f"  Successful Queries: {len(times)} / {len(TEST_QUERIES)}", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    asyncio.run(run_test())
