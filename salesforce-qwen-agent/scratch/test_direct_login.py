import httpx
import asyncio

async def test_connect():
    async with httpx.AsyncClient() as http:
        res = await http.post("http://localhost:8000/api/auth/connect_direct", json={
            "session_id": "test_session",
            "mode": "password",
            "username": "dummy_user@test.com",
            "password": "dummy_password",
            "security_token": "",
            "domain": "login"
        })
        print("Status code:", res.status_code)
        data = res.json()
        print("Response error:", data.get("error").encode("ascii", "ignore").decode("ascii"))

if __name__ == "__main__":
    asyncio.run(test_connect())
