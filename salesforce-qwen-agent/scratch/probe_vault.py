"""Temporary probe: list token-vault sessions and which have sfap:mcp scope.

Prints ONLY non-secret metadata (session id, scope, instance url host, token presence), never token values.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sfmcp.crypto.envelope import token_vault, load_or_create_kek  # noqa: E402


def main():
    print("KEK len:", len(load_or_create_kek()), "bytes")
    print("Vault store path:", token_vault._store_path)
    sessions = token_vault.sessions()
    print("\nSession ids in vault:", sessions if sessions else "(none)")
    for sid in sessions:
        rec = token_vault.get(sid)
        print("\n--- session:", sid, "---")
        if not rec:
            print("  (no/undecryptable record)")
            continue
        scope = rec.get("oauth_scope") or ""
        tok = rec.get("access_token") or ""
        refresh = rec.get("refresh_token") or ""
        inst = rec.get("instance_url") or ""
        import urllib.parse
        host = urllib.parse.urlparse(inst).netloc if inst else ""
        print(f"  scope: {scope!r}")
        print(f"  has_access_token: {bool(tok)}  (len={len(tok)})")
        print(f"  has_refresh_token: {bool(refresh)}")
        print(f"  instance_host: {host!r}")
        print(f"  has_sfap_mcp_scope: {'sfap:mcp' in scope}")


if __name__ == "__main__":
    main()
