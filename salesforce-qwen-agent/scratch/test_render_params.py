import httpx
import asyncio

async def check_render_oauth():
    url = "https://salesforce-qa-agent.onrender.com/api/auth/login?session_id=test_session&domain=login"
    async with httpx.AsyncClient(timeout=15.0) as http:
        try:
            res = await http.get(url, follow_redirects=False)
            print("Render Redirect Status:", res.status_code)
            location = res.headers.get("location", "")
            print("Redirect Location URL:", location)
        except Exception as e:
            print("Connection Error:", e)

if __name__ == "__main__":
    asyncio.run(check_render_oauth())
