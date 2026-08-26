import httpx
import asyncio

async def test_oauth():
    client_id = "3MVG97L7PwbPq6UzTSO02Q0YxGf7HtzS7pjx8Z0e.8DnCyjqD1SQJnH5Re3GcQ3Soun8cla0A3ODURNtz96DT"
    
    # Test 1: login.salesforce.com
    url1 = f"https://login.salesforce.com/services/oauth2/authorize?response_type=code&client_id={client_id}&redirect_uri=http://localhost:8000/api/auth/callback"
    async with httpx.AsyncClient() as http:
        r1 = await http.get(url1, follow_redirects=False)
        print("login.salesforce.com status:", r1.status_code, "headers:", r1.headers.get("location"))

    # Test 2: orgfarm instance
    url2 = f"https://orgfarm-d5054a6252-dev-ed.develop.my.salesforce.com/services/oauth2/authorize?response_type=code&client_id={client_id}&redirect_uri=http://localhost:8000/api/auth/callback"
    async with httpx.AsyncClient() as http:
        r2 = await http.get(url2, follow_redirects=False)
        print("orgfarm status:", r2.status_code, "headers:", r2.headers.get("location"))

if __name__ == "__main__":
    asyncio.run(test_oauth())
