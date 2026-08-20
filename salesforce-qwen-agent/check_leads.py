import asyncio
import json
from app import create_mcp_client
import httpx

async def check():
    client = create_mcp_client()
    await client.authenticate()
    headers = {'Authorization': f'Bearer {client._access_token}', 'Content-Type': 'application/json'}
    url = f'{client.instance_url.rstrip("/")}/services/data/v62.0/query'
    async with httpx.AsyncClient() as http:
        resp = await http.get(url, params={'q': "SELECT Id, FirstName, LastName, Name, Company, Email, CreatedDate FROM Lead WHERE CreatedDate = THIS_WEEK ORDER BY CreatedDate DESC"}, headers=headers)
        data = resp.json()
        print(json.dumps(data, indent=2))

if __name__ == '__main__':
    asyncio.run(check())
