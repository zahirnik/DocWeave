# packages/core/auth.py
"""
Auth & RBAC helpers — tiny, explicit, and easy to swap for your IdP.

What this module provides
-------------------------
- authenticate_user(username, password) -> dict | None
    Tutorial-simple password check that returns a user dict with tenant/roles/scopes.
    **Replace this with your DB or OIDC check in production.**

- issue_access_token(sub, tenant_id, roles, scopes, ttl_s=None) -> (jwt, ttl_s)
    Create a signed JWT (HS256) with iss/aud/exp/iat/jti claims.

- decode_access_token(token) -> dict
    Verify signature, issuer, audience, expiration — returns claims dict.

- create_api_key(meta) -> dict    # returns {"key": "<plaintext once>", "key_id": "...", "last4": "..."}
- verify_api_key(plaintext) -> dict  # returns stored metadata (tenant_id/roles/scopes/label/...)
    API keys are stored **hashed** in a small JSON file for the tutorial.
    Swap this for your DB/Secrets Manager without changing route code.

- has_scopes(current, required) -> bool
- has_roles(current, required) -> bool
    Small all-of semantics helpers for RBAC.

Design goals
------------
- Keep the code readable like a tutorial but production-lean:
  - JWT with proper standard claims
  - API keys hashed with per-key salt (+ pepper from JWT_SECRET)
  - File-based store for dev; easy to replace with DB
  - Clear docstrings, small functions

Environment / Settings (see packages.core.config)
-------------------------------------------------
JWT_SECRET           HMAC secret (MUST set in prod)
JWT_ISSUER           default "convai"
JWT_AUDIENCE         default "convai.clients"
JWT_TTL_S            default 3600
DATA_DIR             default ./data (API key store lives under ./data/outputs/keys.json)

Dependencies
------------
- pyjwt  (a.k.a. PyJWT)
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import secrets
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jwt  # PyJWT

from packages.core.config import get_settings

# ---------------------------
# Demo user store (tutorial)
# ---------------------------

@dataclass
class DemoUser:
    id: str
    tenant_id: str
    email: str
    password: str  # plaintext for tutorial simplicity; DO NOT DO THIS IN PROD
    roles: List[str]
    scopes: List[str]


# Two demo users you can log in as (POST /auth/token)
_DEMO_USERS = {
    "demo@local": DemoUser(
        id="user_demo",
        tenant_id="t0",
        email="demo@local",
        password="password",
        roles=["user"],
        scopes=["rag:query", "rag:ingest"],
    ),
    "admin@local": DemoUser(
        id="user_admin",
        tenant_id="t0",
        email="admin@local",
        password="password",
        roles=["admin"],
        scopes=["rag:query", "rag:ingest", "admin:keys"],
    ),
}


def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """
    Tutorial-simple username/password check.

    Replace this with:
      - your DB user table + password hashing (bcrypt/argon2), OR
      - your OIDC provider's token exchange (Auth Code + PKCE).

    Returns a dict with {id, tenant_id, roles, scopes} on success, else None.
    """
    u = _DEMO_USERS.get(username.lower().strip())
    if not u or password != u.password:
        return None
    return {
        "id": u.id,
        "tenant_id": u.tenant_id,
        "email": u.email,
        "roles": u.roles,
        "scopes": u.scopes,
    }


# ---------------------------
# JWT helpers
# ---------------------------

def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def issue_access_token(
    sub: str,
    tenant_id: str,
    roles: List[str],
    scopes: List[str],
    ttl_s: Optional[int] = None,
) -> Tuple[str, int]:
    """
    Issue a signed JWT (HS256) with standard claims.

    Args:
      sub:       subject (user id or service id)
      tenant_id: tenant/workspace id
      roles:     role names (e.g., ["user"] or ["admin"])
      scopes:    fine-grained permissions (e.g., ["rag:query"])
      ttl_s:     time-to-live in seconds (default from settings.jwt_ttl_s)

    Returns:
      (token, ttl_seconds)
    """
    st = get_settings()
    ttl = int(ttl_s or st.jwt_ttl_s)
    iat = _now_utc()
    exp = iat + dt.timedelta(seconds=ttl)
    jti = uuid.uuid4().hex

    payload = {
        "sub": sub,
        "tenant_id": tenant_id,
        "roles": roles or [],
        "scopes": scopes or [],
        "iss": st.jwt_issuer,
        "aud": st.jwt_audience,
        "iat": int(iat.timestamp()),
        "exp": int(exp.timestamp()),
        "jti": jti,
    }
    token = jwt.encode(payload, st.jwt_secret, algorithm="HS256")
    return token, ttl


def decode_access_token(token: str) -> Dict:
    """
    Verify a JWT and return its claims.

    Raises:
      jwt.ExpiredSignatureError
      jwt.InvalidAudienceError
      jwt.InvalidIssuerError
      jwt.InvalidTokenError
    """
    st = get_settings()
    claims = jwt.decode(
        token,
        st.jwt_secret,
        algorithms=["HS256"],
        audience=st.jwt_audience,
        issuer=st.jwt_issuer,
        options={"require": ["exp", "iat", "iss", "aud", "sub"]},
    )
    # Normalize types
    claims["roles"] = list(claims.get("roles") or [])
    claims["scopes"] = list(claims.get("scopes") or [])
    claims["tenant_id"] = str(claims.get("tenant_id") or "t0")
    return claims


# ---------------------------
# API key store (file-based for dev)
# ---------------------------

def _keys_store_path() -> str:
    st = get_settings()
    out_dir = os.path.join(st.outputs_dir)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, "keys.json")


def _load_keys_store() -> Dict[str, Dict]:
    path = _keys_store_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def _save_keys_store(store: Dict[str, Dict]) -> None:
    path = _keys_store_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _pepper() -> bytes:
    """Extra secret mixed into API key hashing (reuse JWT_SECRET as pepper)."""
    return get_settings().jwt_secret.encode("utf-8")


def _hash_api_key(plaintext: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """
    Hash API key using HMAC-SHA256(pepper, salt || key), returning (salt_b64, hash_hex).
    We store salt (b64) and hash (hex). Plaintext is **never** persisted.
    """
    salt = salt or secrets.token_bytes(16)
    msg = salt + plaintext.encode("utf-8")
    digest = hmac.new(_pepper(), msg, hashlib.sha256).hexdigest()
    return base64.b64encode(salt).decode("ascii"), digest


def _id_from_hash(hh: str) -> str:
    """Generate a short stable id from the hash for logging/lookup."""
    return hh[:12]


def create_api_key(meta: Dict) -> Dict:
    """
    Create an API key and persist its **hash** and metadata.

    Args (meta):
      tenant_id: str
      roles:     List[str]
      scopes:    List[str]
      label:     human-friendly label
      ttl_days:  int (optional)

    Returns:
      {
        "key": "<PLAINTEXT ONCE>",
        "key_id": "<short id>",
        "last4": "abcd"
      }
    """
    # 1) Generate a 32-byte random key, url-safe base64 without padding for copy-friendliness
    raw = secrets.token_urlsafe(32)
    last4 = raw[-4:]

    # 2) Hash + salt
    salt_b64, hh = _hash_api_key(raw)
    key_id = _id_from_hash(hh)

    # 3) Persist
    store = _load_keys_store()
    now = _now_utc()
    expires_at = None
    ttl_days = int(meta.get("ttl_days") or 0)
    if ttl_days > 0:
        expires_at = (now + dt.timedelta(days=ttl_days)).isoformat()

    entry = {
        "key_id": key_id,
        "hash": hh,
        "salt_b64": salt_b64,
        "tenant_id": meta.get("tenant_id", "t0"),
        "roles": list(meta.get("roles") or []),
        "scopes": list(meta.get("scopes") or []),
        "label": str(meta.get("label") or ""),
        "created_at": now.isoformat(),
        "ttl_days": ttl_days or None,
        "expires_at": expires_at,
    }
    store[key_id] = entry
    _save_keys_store(store)

    # 4) Return plaintext once
    return {"key": raw, "key_id": key_id, "last4": last4}


def verify_api_key(plaintext: str) -> Dict:
    """
    Validate an API key plaintext and return its stored metadata.

    Raises:
      ValueError("invalid api key") if not found or mismatched.
      ValueError("api key expired") if `expires_at` has passed.
    """
    # Recreate hash with the stored salt for each entry and match.
    store = _load_keys_store()
    for entry in store.values():
        try:
            salt = base64.b64decode(entry["salt_b64"])
            _, hh = _hash_api_key(plaintext, salt=salt)
            if hmac.compare_digest(hh, entry["hash"]):
                # Check expiry
                if entry.get("expires_at"):
                    if _now_utc() > dt.datetime.fromisoformat(entry["expires_at"]):
                        raise ValueError("api key expired")
                # Return metadata (no secrets)
                return {
                    "key_id": entry["key_id"],
                    "tenant_id": entry.get("tenant_id", "t0"),
                    "roles": entry.get("roles", []) or [],
                    "scopes": entry.get("scopes", []) or [],
                    "label": entry.get("label", ""),
                }
        except Exception:
            continue
    raise ValueError("invalid api key")


# ---------------------------
# RBAC helpers
# ---------------------------

def has_scopes(current: List[str], required: List[str]) -> bool:
    """
    All-of semantics: return True if *all* required scopes are present in `current`.

    Example:
        has_scopes(["rag:query","admin:keys"], ["rag:query"])           -> True
        has_scopes(["rag:query"], ["rag:query","admin:keys"])           -> False
    """
    cur = set(current or [])
    return all(r in cur for r in (required or []))


def has_roles(current: List[str], required: List[str]) -> bool:
    """
    All-of semantics for roles (mirrors has_scopes).
    """
    cur = set(current or [])
    return all(r in cur for r in (required or []))


__all__ = [
    "authenticate_user",
    "issue_access_token",
    "decode_access_token",
    "create_api_key",
    "verify_api_key",
    "has_scopes",
    "has_roles",
]
