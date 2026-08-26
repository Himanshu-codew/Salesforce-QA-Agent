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

async def test_no_scope():
    client_id = "3MVG97L7PWbPq6UzTSO02Q0YxGRCXLEicVkWoGDBvm_kkpJF2PhzWRFjDjvSByt618L946lbBgTejjkxyc_Hm"
    redirect_uri = "https://salesforce-qa-agent.onrender.com/api/auth/callback"
    v, c = get_pkce()

    # 1. login.salesforce.com with NO scope param and NO prompt param
    url_login = f"https://login.salesforce.com/services/oauth2/authorize?response_type=code&client_id={client_id}&redirect_uri={redirect_uri}&code_challenge={c}&code_challenge_method=S256"
    async with httpx.AsyncClient(timeout=15.0) as http:
        res1 = await http.get(url_login, follow_redirects=False)
        print("login.salesforce.com (No Scope/Prompt) Status:", res1.status_code)
        print("login.salesforce.com Location:", res1.headers.get("location"))

if __name__ == "__main__":
    asyncio.run(test_no_scope())
