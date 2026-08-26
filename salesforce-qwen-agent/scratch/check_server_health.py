import httpx
import asyncio

async def check_root():
    url = "https://salesforce-qa-agent.onrender.com/"
    async with httpx.AsyncClient(timeout=25.0) as http:
        try:
            res = await http.get(url)
            print("Render Root Status Code:", res.status_code)
            print("Render Content Length:", len(res.text))
        except Exception as e:
            print("Root check error:", e)

if __name__ == "__main__":
    asyncio.run(check_root())
