import httpx
import asyncio

async def test_scratch_oauth():
    client_id = "3MVG97L7PwbPq6UzTSO02Q0YxGf7HtzS7pjx8Z0e.8DnCyjqD1SQJnH5Re3GcQ3Soun8cla0A3ODURNtz96DT"
    url = f"https://orgfarm-d5054a6252-dev-ed.develop.my.salesforce.com/services/oauth2/authorize?response_type=code&client_id={client_id}&redirect_uri=http://localhost:8000/api/auth/callback"
    async with httpx.AsyncClient(timeout=10.0) as http:
        try:
            res = await http.get(url, follow_redirects=False)
            print("Status code:", res.status_code)
            print("Location:", res.headers.get("location"))
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    asyncio.run(test_scratch_oauth())
