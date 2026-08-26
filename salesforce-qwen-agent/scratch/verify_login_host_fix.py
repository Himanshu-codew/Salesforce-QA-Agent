"""Local verification of the direct-login host fix + callback forensics (no real network)."""
import importlib.util
import os
import sys

sys.path.insert(0, "salesforce-qwen-agent")
spec = importlib.util.spec_from_file_location("apptest", "salesforce-qwen-agent/app.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

# ── Stub httpx so no real Salesforce calls happen ──
recorded = {"posts": []}


class FakeRes:
    status_code = 500
    text = (
        '<soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/">'
        "<soapenv:Body><soapenv:Fault><faultcode>sf:INVALID_LOGIN</faultcode>"
        "<faultstring>bad creds</faultstring></soapenv:Fault></soapenv:Body></soapenv:Envelope>"
    )

    def json(self):
        return {}


class FakeClient:
    def __init__(self, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, **kw):
        recorded["posts"].append(url)
        return FakeRes()

    async def get(self, url, **kw):
        return FakeRes()


m.httpx.AsyncClient = FakeClient

# 1) Host mapping unit checks
assert m._resolve_login_host("login") == "login.salesforce.com"
assert m._resolve_login_host("") == "login.salesforce.com"
assert m._resolve_login_host("production") == "login.salesforce.com"
assert m._resolve_login_host("test") == "test.salesforce.com"
assert (
    m._resolve_login_host("https://myco.my.salesforce.com/") == "myco.my.salesforce.com"
), "custom My Domain must pass through"
print("1. _resolve_login_host mapping OK")

# 2) Direct password login posts SOAP + token to canonical login host
from fastapi.testclient import TestClient

c = TestClient(m.app)
r = c.post(
    "/api/auth/connect_direct",
    json={"session_id": "t", "mode": "password", "username": "u@x.com", "password": "p", "domain": "login"},
)
assert r.status_code == 401, r.status_code  # stubbed fault -> friendly 401
posts = " ".join(recorded["posts"])
assert "https://login.salesforce.com/services/Soap/u/" in posts, recorded["posts"]
assert "orgfarm" not in posts, f"instance host leaked into login flow: {recorded['posts']}"
print("2. domain=login -> SOAP+token on login.salesforce.com OK")

# 3) Custom My-Domain passes through for credential login
recorded["posts"].clear()
r = c.post(
    "/api/auth/connect_direct",
    json={
        "session_id": "t",
        "mode": "password",
        "username": "u@x.com",
        "password": "p",
        "domain": "myco.my.salesforce.com",
    },
)
assert any("https://myco.my.salesforce.com/services/Soap/u/" in u for u in recorded["posts"]), recorded["posts"]
print("3. custom My Domain passthrough OK")

# 4) Callback forensics: known state -> masked key + host in popup
verifier, _ = m._generate_pkce_pair()
state = "test-state-123"
m._oauth_pending_flows[state] = {
    "session_id": "t",
    "auth_host": "orgfarm-d5054a6252-dev-ed.develop.my.salesforce.com",
    "client_id": "3MVG97L7PWbPq6UzTSO02Q0YxGRCXLEicVkWoGDBvm_kkpJF2PhzWRFjDjvSByt618L946lbBgTejjkxyc_Hm",
    "client_secret": "s",
    "redirect_uri": "http://localhost:8000/api/auth/callback",
    "code_verifier": verifier,
    "created_at": __import__("time").time(),
}
r = c.get("/api/auth/callback", params={"error": "invalid_client_id", "error_description": "client identifier invalid", "state": state})
body = r.text
assert r.status_code == 400 and "Salesforce Authorization Failed" in body
assert "3MVG97L7PWbP…" in body and "_Hm</code>" in body, "masked key missing"
assert "orgfarm-d5054a6252-dev-ed.develop.my.salesforce.com" in body, "host missing"
print("4. callback forensics (known flow) OK")

# 5) Callback forensics: expired/unknown state still renders safely
r = c.get("/api/auth/callback", params={"error": "invalid_client_id", "state": "nope"})
assert r.status_code == 400 and "unknown (state expired)" in r.text
print("5. callback forensics (expired state) OK")

print("\nALL VERIFICATIONS PASSED")
