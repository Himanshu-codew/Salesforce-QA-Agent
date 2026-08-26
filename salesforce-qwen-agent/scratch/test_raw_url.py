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

async def test_instance_raw():
    client_id = "3MVG97L7PWbPq6UzTSO02Q0YxGRCXLEicVkWoGDBvm_kkpJF2PhzWRFjDjvSByt618L946lbBgTejjkxyc_Hm"
    redirect_uri = "https://salesforce-qa-agent.onrender.com/api/auth/callback"
    state = secrets.token_urlsafe(24)
    v, c = get_pkce()

    url = f"https://orgfarm-d5054a6252-dev-ed.develop.my.salesforce.com/services/oauth2/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&state={state}&code_challenge={c}&code_challenge_method=S256"

    async with httpx.AsyncClient(timeout=15.0) as http:
        res = await http.get(url, follow_redirects=False)
        print("Instance Raw Status Code:", res.status_code)
        print("Instance Location:", res.headers.get("location"))

if __name__ == "__main__":
    asyncio.run(test_instance_raw())
