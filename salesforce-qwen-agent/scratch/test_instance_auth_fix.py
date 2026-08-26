import httpx
import asyncio
import secrets
import hashlib
import base64

def get_pkce():
    v = secrets.token_urlsafe(32)
    d = hashlib.sha256(v.encode('utf-8')).digest()
    c = base64.urlsafe_b64encode(d).decode('utf-8').rstrip('=')
    return v, c

async def test_instance_pkce():
    client_id = "3MVG97L7PWbPq6UzTSO02Q0YxGRCXLEicVkWoGDBvm_kkpJF2PhzWRFjDjvSByt618L946lbBgTejjkxyc_Hm"
    v, c = get_pkce()
    url = f"https://orgfarm-d5054a6252-dev-ed.develop.my.salesforce.com/services/oauth2/authorize?response_type=code&client_id={client_id}&redirect_uri=http://localhost:8000/api/auth/callback&code_challenge={c}&code_challenge_method=S256"
    async with httpx.AsyncClient(timeout=10.0) as http:
        res = await http.get(url, follow_redirects=False)
        print("Instance OAuth PKCE Status:", res.status_code)
        print("Redirect Location:", res.headers.get("location"))

if __name__ == "__main__":
    asyncio.run(test_instance_pkce())
