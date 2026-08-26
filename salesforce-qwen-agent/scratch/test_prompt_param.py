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

async def test_without_prompt():
    client_id = "3MVG97L7PWbPq6UzTSO02Q0YxGRCXLEicVkWoGDBvm_kkpJF2PhxWRFjDjvSByt618L946lbBgTejjkxyc_Hm"
    v, c = get_pkce()
    
    # 1. WITH prompt=consent
    url_with_prompt = f"https://orgfarm-d5054a6252-dev-ed.develop.my.salesforce.com/services/oauth2/authorize?response_type=code&client_id={client_id}&redirect_uri=https%3A%2F%2Fsalesforce-qa-agent.onrender.com%2Fapi%2Fauth%2Fcallback&code_challenge={c}&code_challenge_method=S256&prompt=consent"
    async with httpx.AsyncClient(timeout=15.0) as http:
        res1 = await http.get(url_with_prompt, follow_redirects=False)
        print("WITH prompt=consent Status:", res1.status_code)

    # 2. WITHOUT prompt=consent
    url_without_prompt = f"https://orgfarm-d5054a6252-dev-ed.develop.my.salesforce.com/services/oauth2/authorize?response_type=code&client_id={client_id}&redirect_uri=https%3A%2F%2Fsalesforce-qa-agent.onrender.com%2Fapi%2Fauth%2Fcallback&code_challenge={c}&code_challenge_method=S256"
    async with httpx.AsyncClient(timeout=15.0) as http:
        res2 = await http.get(url_without_prompt, follow_redirects=False)
        print("WITHOUT prompt=consent Status:", res2.status_code)
        print("WITHOUT prompt=consent Location:", res2.headers.get("location"))

if __name__ == "__main__":
    asyncio.run(test_without_prompt())
