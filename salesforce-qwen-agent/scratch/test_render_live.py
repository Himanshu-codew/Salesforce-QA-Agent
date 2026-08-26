import httpx
import asyncio

async def test_render():
    url = "https://salesforce-qa-agent.onrender.com/api/auth/login?session_id=test_session&domain=login"
    async with httpx.AsyncClient(timeout=15.0) as http:
        try:
            res = await http.get(url, follow_redirects=False)
            print("Render Login Status:", res.status_code)
            print("Render Redirect Location:", res.headers.get("location"))
            if res.status_code == 400 or res.status_code == 500:
                print("Body:", res.text[:500])
        except Exception as e:
            print("Render connection error:", e)

if __name__ == "__main__":
    asyncio.run(test_render())
