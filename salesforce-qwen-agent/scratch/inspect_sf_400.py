import httpx
import asyncio

async def test_live_render_302():
    async with httpx.AsyncClient(timeout=15.0) as http:
        # 1. Hit Render login
        r1 = await http.get("https://salesforce-qa-agent.onrender.com/api/auth/login?session_id=render_test&domain=login", follow_redirects=False)
        sf_auth_url = r1.headers.get("location")
        print("Render generated OAuth URL:", sf_auth_url)

        # 2. Hit Salesforce OAuth endpoint
        r2 = await http.get(sf_auth_url, follow_redirects=False)
        print("Salesforce OAuth Page HTTP Status:", r2.status_code)
        print("Salesforce Redirect Target:", r2.headers.get("location"))

if __name__ == "__main__":
    asyncio.run(test_live_render_302())
