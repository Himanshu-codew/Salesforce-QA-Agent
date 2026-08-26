import httpx
import asyncio
import secrets
import hashlib
import base64
import urllib.parse

def get_pkce():
    v = secrets.token_urlsafe(32)
    d = hashlib.sha256(v.encode('utf-8')).digest()
    c = base64.urlsafe_b64encode(d).decode('utf-8').rstrip('=')
    return v, c

async def test_matrix_login():
    client_id = "3MVG97L7PWbPq6UzTSO02Q0YxGRCXLEicVkWoGDBvm_kkpJF2PhzWRFjDjvSByt618L946lbBgTejjkxyc_Hm"
    redirect_uri = "https://salesforce-qa-agent.onrender.com/api/auth/callback"
    base_host = "login.salesforce.com"
    
    v, c = get_pkce()

    test_cases = [
        ("Base Only", {"response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri}),
        ("With PKCE", {"response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri, "code_challenge": c, "code_challenge_method": "S256"}),
        ("With Scope api refresh_token id", {"response_type": "code", "client_id": client_id, "redirect_uri": redirect_uri, "scope": "api refresh_token id", "code_challenge": c, "code_challenge_method": "S256"}),
    ]

    async with httpx.AsyncClient(timeout=15.0) as http:
        for name, params in test_cases:
            url = f"https://{base_host}/services/oauth2/authorize?" + urllib.parse.urlencode(params)
            res = await http.get(url, follow_redirects=False)
            print(f"[{name}] Status: {res.status_code} | Location: {res.headers.get('location', 'None')[:80]}")

if __name__ == "__main__":
    asyncio.run(test_matrix_login())
