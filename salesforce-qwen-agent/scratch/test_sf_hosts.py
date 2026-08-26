import httpx
import asyncio
import urllib.parse

async def test_live_render_clean():
    async with httpx.AsyncClient(timeout=15.0) as http:
        # 1. Click login on Render
        r1 = await http.get("https://salesforce-qa-agent.onrender.com/api/auth/login?session_id=final_render&domain=login", follow_redirects=False)
        sf_url = r1.headers.get("location")
        print("Render Generated OAuth URL:", sf_url)

        # 2. Unquote for raw HTTP request
        clean_url = urllib.parse.unquote(sf_url)
        r2 = await http.get(clean_url, follow_redirects=False)
        print("Salesforce Response Status Code:", r2.status_code)
        print("Salesforce Authorization Target URL:", r2.headers.get("location"))

if __name__ == "__main__":
    asyncio.run(test_live_render_clean())
