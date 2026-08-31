"""
Envelope encryption for OAuth token storage.

Port of the mcp-bridge approach (lib/crypto/envelope.js):
- a static key-encryption-key (KEK) protects a per-record data-encryption-key (DEK)
- each sensitive value is encrypted with its own random DEK (AES-256-GCM)
- the DEK is wrapped by the KEK and stored next to the ciphertext

Nothing sensitive is ever stored in plaintext. The in-memory token store and the
optional on-disk mirror (mcp_tokens.enc) only contain wrapped envelopes.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import threading
from functools import lru_cache
from secrets import token_bytes
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

logger = logging.getLogger("sfmcp.crypto")

_DEK_BYTES = 32
_KEK_BYTES = 32
_IV_BYTES = 12


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64d(value: str) -> bytes:
    return base64.b64decode(value.encode("ascii"))


def default_kek_path() -> str:
    """Project root .sf_kek file."""
    module_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(module_dir, ".sf_kek")


@lru_cache(maxsize=1)
def load_or_create_kek() -> bytes:
    """Load the static KEK from SF_TOKEN_KEK env or the .sf_kek file, creating it on first run."""
    raw = os.getenv("SF_TOKEN_KEK")
    if raw:
        raw = raw.strip()
        key = None
        if raw.startswith("b64:"):
            key = base64.b64decode(raw[4:])
        elif raw.startswith("hex:"):
            key = bytes.fromhex(raw[4:])
        else:
            try:
                key = base64.b64decode(raw)
            except Exception:
                key = None
        if not key or len(key) != _KEK_BYTES:
            key = hashlib.sha256(raw.encode("utf-8")).digest()
        return key

    path = os.getenv("SF_TOKEN_KEK_FILE") or default_kek_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                value = fh.read().strip()
            decoded = base64.b64decode(value) or bytes.fromhex(value)
            if len(decoded) == _KEK_BYTES:
                return decoded
        key = token_bytes(_KEK_BYTES)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(base64.b64encode(key).decode("ascii"))
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        logger.warning("Generated new token KEK at %s. Protect this file; losing it makes stored tokens unreadable.", path)
        return key
    except OSError as e:
        logger.error("Could not load/create KEK file %s: %s", path, e)
        return token_bytes(_KEK_BYTES)


def encrypt_value(kek: bytes, plaintext: str) -> dict:
    """Encrypt a string value into a JSON-safe envelope."""
    dek = token_bytes(_DEK_BYTES)
    iv = token_bytes(_IV_BYTES)
    ciphertext = AESGCM(dek).encrypt(iv, plaintext.encode("utf-8"), None)

    kiv = token_bytes(_IV_BYTES)
    wrapped_dek = AESGCM(kek).encrypt(kiv, dek, None)
    return {
        "iv": _b64(iv),
        "ct": _b64(ciphertext),
        "kiv": _b64(kiv),
        "kct": _b64(wrapped_dek),
    }


def decrypt_value(kek: bytes, record: dict) -> str:
    """Decrypt an envelope produced by encrypt_value."""
    dek = AESGCM(kek).decrypt(_b64d(record["kiv"]), _b64d(record["kct"]), None)
    plaintext = AESGCM(dek).decrypt(_b64d(record["iv"]), _b64d(record["ct"]), None)
    return plaintext.decode("utf-8")


class TokenVault:
    """
    Encrypted token store.

    Each session's OAuth credentials live in memory as encrypted envelopes and are
    mirrored to an encrypted on-disk file (mcp_tokens.enc) so they survive restarts.
    """

    def __init__(self, kek: Optional[bytes] = None, store_path: Optional[str] = None) -> None:
        self._kek = kek or load_or_create_kek()
        self._store_path = store_path or os.getenv("SF_TOKEN_STORE_PATH") or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "mcp_tokens.enc"
        )
        self._sessions: dict[str, dict] = {}
        self._lock = threading.RLock()
        self._load()

    # -- internal persistence -------------------------------------------------

    def _file_key(self) -> bytes:
        return HKDF(
            algorithm=hashes.SHA256(), length=_DEK_BYTES, salt=b"sf-token-vault-v1", info=b"tokens-file"
        ).derive(self._kek)

    def _persist_locked(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._store_path) or ".", exist_ok=True)
            payload = json.dumps(self._sessions, ensure_ascii=False).encode("utf-8")
            iv = token_bytes(_IV_BYTES)
            ct = AESGCM(self._file_key()).encrypt(iv, payload, None)
            blob = {"v": 1, "iv": _b64(iv), "data": _b64(ct)}
            tmp = f"{self._store_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(blob, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self._store_path)
            try:
                os.chmod(self._store_path, 0o600)
            except OSError:
                pass
        except OSError as e:
            logger.error("Failed to persist token vault: %s", e)

    def _load(self) -> None:
        try:
            if not os.path.exists(self._store_path):
                return
            with open(self._store_path, "r", encoding="utf-8") as fh:
                blob = json.load(fh)
            if blob.get("v") != 1:
                return
            payload = AESGCM(self._file_key()).decrypt(_b64d(blob["iv"]), _b64d(blob["data"]), None)
            self._sessions = json.loads(payload.decode("utf-8"))
        except (OSError, ValueError, InvalidTag, KeyError) as e:
            logger.error("Could not read token vault %s (KEK mismatch?): %s", self._store_path, e)
            self._sessions = {}

    # -- session API ----------------------------------------------------------

    def put(
        self,
        session_id: str,
        access_token: str,
        refresh_token: Optional[str] = None,
        instance_url: Optional[str] = None,
        expires_at: Optional[float] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        oauth_scope: Optional[str] = None,
        auth_host: Optional[str] = None,
    ) -> None:
        record: dict = {"access_token": encrypt_value(self._kek, access_token)}
        if refresh_token:
            record["refresh_token"] = encrypt_value(self._kek, refresh_token)
        if instance_url:
            record["instance_url"] = encrypt_value(self._kek, instance_url)
        if client_id:
            record["client_id"] = encrypt_value(self._kek, client_id)
        if client_secret:
            record["client_secret"] = encrypt_value(self._kek, client_secret)
        if oauth_scope:
            record["oauth_scope"] = encrypt_value(self._kek, oauth_scope)
        if auth_host:
            record["auth_host"] = encrypt_value(self._kek, auth_host)
        if expires_at is not None:
            record["expires_at"] = expires_at
        with self._lock:
            self._sessions[session_id] = record
            self._persist_locked()

    def update(self, session_id: str, **fields) -> None:
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return
            for key, value in fields.items():
                if value is None:
                    record.pop(key, None)
                elif isinstance(value, (int, float, bool)):
                    record[key] = value
                else:
                    record[key] = encrypt_value(self._kek, str(value))
            self._persist_locked()

    def get(self, session_id: str) -> Optional[dict]:
        """Return the decrypted credentials record (plaintext) for a session, or None."""
        with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return None
            out = {}
            for key, value in record.items():
                if key == "expires_at":
                    out[key] = value
                elif isinstance(value, dict):
                    try:
                        out[key] = decrypt_value(self._kek, value)
                    except (InvalidTag, KeyError) as e:
                        logger.error("Failed to decrypt %s for session %s: %s", key, session_id, e)
                        out[key] = None
            return out

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
            self._persist_locked()

    def sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())


token_vault = TokenVault()


async def encrypt_value_async(*args, **kwargs) -> dict:
    return await asyncio.to_thread(encrypt_value, *args, **kwargs)


async def decrypt_value_async(*args, **kwargs) -> str:
    return await asyncio.to_thread(decrypt_value, *args, **kwargs)