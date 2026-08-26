import httpx
import asyncio

async def test_direct_on_render_json():
    url = "https://salesforce-qa-agent.onrender.com/api/auth/connect_direct"
    payload = {
        "username": "himanshuswami898.e86b4be632fc@agentforce.com",
        "password": "Himanshu@2026",
        "security_token": "QkxURREpDMh0dW0iVqCk3IB3y",
        "domain": "login"
    }
    async with httpx.AsyncClient(timeout=20.0) as http:
        res = await http.post(url, json=payload)
        print("Direct Login Response Status:", res.status_code)
        print("Direct Login Response Body:", res.json())

if __name__ == "__main__":
    asyncio.run(test_direct_on_render_json())
