import httpx
import asyncio

async def test_direct_login_now():
    url = "https://salesforce-qa-agent.onrender.com/api/auth/connect_direct"
    payload = {
        "username": "himanshuswami898.e86b4be632fc@agentforce.com",
        "password": "Himanshu@2026",
        "security_token": "QkxURREpDMh0dW0iVqCk3IB3y",
        "domain": "login"
    }
    async with httpx.AsyncClient(timeout=25.0) as http:
        res = await http.post(url, json=payload)
        print("Direct Login Status Code:", res.status_code)
        print("Direct Login Response:", res.json())

if __name__ == "__main__":
    asyncio.run(test_direct_login_now())
