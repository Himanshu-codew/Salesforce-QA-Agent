import httpx
import asyncio

async def check_health():
    async with httpx.AsyncClient() as http:
        try:
            r = await http.get("http://localhost:8000/health")
            print("Health response:", r.status_code, r.json())
        except Exception as e:
            print("Health error:", e)

if __name__ == "__main__":
    asyncio.run(check_health())
