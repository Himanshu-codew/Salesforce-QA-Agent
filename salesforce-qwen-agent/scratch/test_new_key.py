import httpx
import asyncio

async def test_new_client():
    client_id = "3MVG97L7PWbPq6UzTSO02Q0YxGRCXLEicVkWoGDBvm_kkpJF2PhzWRFjDjvSByt618L946lbBgTejjkxyc_Hm"
    url = f"https://login.salesforce.com/services/oauth2/authorize?response_type=code&client_id={client_id}&redirect_uri=http://localhost:8000/api/auth/callback"
    async with httpx.AsyncClient(timeout=10.0) as http:
        res = await http.get(url, follow_redirects=False)
        print("Status code:", res.status_code)
        print("Headers Location:", res.headers.get("location"))

if __name__ == "__main__":
    asyncio.run(test_new_client())
