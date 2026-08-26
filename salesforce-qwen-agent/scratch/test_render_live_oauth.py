import httpx
import asyncio

async def test_live_render_oauth():
    # 1. Fetch OAuth URL from Render
    async with httpx.AsyncClient(timeout=15.0) as http:
        r1 = await http.get("https://salesforce-qa-agent.onrender.com/api/auth/login?session_id=test_live&domain=login", follow_redirects=False)
        oauth_url = r1.headers.get("location")
        print("Render generated OAuth URL:", oauth_url)

        # 2. Hit the generated OAuth URL on Salesforce
        r2 = await http.get(oauth_url, follow_redirects=False)
        print("Salesforce OAuth Page HTTP Status:", r2.status_code)
        print("Salesforce Redirect Target:", r2.headers.get("location"))

if __name__ == "__main__":
    asyncio.run(test_live_render_oauth())
