# apps/api/routes/auth.py
"""
Auth routes — tiny, explicit, and production-lean.

What this file provides
-----------------------
- POST /auth/token      : Exchange username/password for a short-lived JWT (Bearer).
- GET  /auth/me         : Return the current principal (from Bearer or API key).
- POST /auth/apikeys    : Create a new API key (admin-only), returned once in full.

Security model (simple, clear)
------------------------------
- Two auth paths:
  1) Bearer JWT (Authorization: Bearer <token>)
     - Issued by /auth/token with username & password.
     - Claims include: sub (user id/email), tenant_id, roles, scopes, exp, iat.
  2) API Key (X-API-KEY: <key>)
     - Meant for server-to-server ingestion jobs or headless clients.
     - Metadata (tenant_id/roles/scopes) is tied to the key at creation.

- Per-route scope checks:
  Use `require_scopes(["scope:write"])` as a dependency in other route modules.

Where the heavy lifting lives
-----------------------------
This router delegates to `packages.core.auth` for:
  - authenticate_user
  - issue_access_token / decode_access_token
  - verify_api_key / create_api_key
  - has_scopes / has_roles

Those helpers are small and documented, and we’ll implement them in the next files.

Notes
-----
- This is intentionally simple and readable like a tutorial, but it's structured
  so you can swap in your IdP (OIDC) or DB user store without touching call-sites.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Security, status
from fastapi.security import (
    OAuth2PasswordRequestForm,
    HTTPAuthorizationCredentials,
    HTTPBearer,
    APIKeyHeader,
)
from pydantic import BaseModel, Field

# Centralized auth/RBAC helpers (implemented in packages/core/auth.py)
from packages.core.auth import (
    authenticate_user,
    issue_access_token,
    decode_access_token,
    verify_api_key,
    create_api_key,
    has_scopes,
    has_roles,
)

router = APIRouter()

# Security extractors (lenient here; we validate below)
http_bearer = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)


# ---------------------------
# Pydantic response models
# ---------------------------
class TokenResponse(BaseModel):
    access_token: str = Field(..., example="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
    token_type: str = Field("bearer", example="bearer")
    expires_in: int = Field(..., example=3600)


class Principal(BaseModel):
    """Auth context that downstream routes can rely on."""
    subject: str = Field(..., description="Unique user id or key id", example="user_123")
    tenant_id: str = Field(..., example="t0")
    roles: List[str] = Field(default_factory=list, example=["user"])
    scopes: List[str] = Field(default_factory=list, example=["rag:query"])
    auth_type: str = Field(..., example="bearer")  # "bearer" | "apikey"


class CreateApiKeyRequest(BaseModel):
    label: str = Field(..., example="ingest-bot-eu-west")
    ttl_days: int = Field(90, ge=1, le=365, description="How long the key will be valid.")


class CreateApiKeyResponse(BaseModel):
    key: str = Field(..., description="The plain API key — will be shown ONCE.")
    key_id: str = Field(..., description="Stored identifier for the key.")
    last4: str = Field(..., description="Convenience mask for logs/UI.")
    tenant_id: str
    roles: List[str]
    scopes: List[str]


# ---------------------------
# Dependencies
# ---------------------------
async def get_principal(
    authorization: Optional[HTTPAuthorizationCredentials] = Security(http_bearer),
    api_key: Optional[str] = Security(api_key_header),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
) -> Principal:
    """
    Build a Principal from either Bearer JWT or API key.
    - Prefers Bearer if both are present.
    - Validates signature/expiration and returns consistent shape.
    """
    if authorization and authorization.scheme.lower() == "bearer":
        token = authorization.credentials
        try:
            claims = decode_access_token(token)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
        # Optionally allow an explicit tenant header to narrow scopes (never widen)
        tenant_id = claims.get("tenant_id")
        if x_tenant_id and x_tenant_id != tenant_id:
            # Disallow cross-tenant spoofing. You may relax this if using multi-tenant headers intentionally.
            raise HTTPException(status_code=403, detail="Tenant mismatch for bearer token.")
        return Principal(
            subject=claims.get("sub", "unknown"),
            tenant_id=tenant_id or "t0",
            roles=claims.get("roles", []) or [],
            scopes=claims.get("scopes", []) or [],
            auth_type="bearer",
        )

    if api_key:
        try:
            meta = verify_api_key(api_key)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
        tenant_id = meta.get("tenant_id", "t0")
        if x_tenant_id and x_tenant_id != tenant_id:
            raise HTTPException(status_code=403, detail="Tenant mismatch for API key.")
        return Principal(
            subject=meta.get("key_id", "key_unknown"),
            tenant_id=tenant_id,
            roles=meta.get("roles", []) or [],
            scopes=meta.get("scopes", []) or [],
            auth_type="apikey",
        )

    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials.")


def require_scopes(required: List[str]):
    """
    Simple dependency factory to enforce scopes on a route.

    Usage:
        @router.get("/admin", dependencies=[Depends(require_scopes(["admin:all"]))])
        def admin_only(principal: Principal = Depends(get_principal)):
            ...
    """
    async def _checker(principal: Principal = Depends(get_principal)):
        if not has_scopes(principal.scopes, required):
            raise HTTPException(status_code=403, detail=f"Missing scopes: {required}")
        return principal

    return _checker


# ---------------------------
# Endpoints
# ---------------------------
@router.post("/token", response_model=TokenResponse, summary="Exchange username/password for a JWT")
async def login_for_access_token(form: OAuth2PasswordRequestForm = Depends()):
    """
    Password-based login (tutorial-simple). Production guidance:
    - Replace `authenticate_user` with your DB/IdP check.
    - Consider OIDC code flow for browser clients; keep this for service accounts.

    Returns a short-lived JWT with roles/scopes bound to the user & tenant.
    """
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")

    token, ttl = issue_access_token(
        sub=user["id"],
        tenant_id=user["tenant_id"],
        roles=user.get("roles", []),
        scopes=user.get("scopes", []),
    )
    return TokenResponse(access_token=token, token_type="bearer", expires_in=ttl)


@router.get("/me", response_model=Principal, summary="Return the current principal")
async def me(principal: Principal = Depends(get_principal)):
    """
    Useful for debugging: shows what the server thinks you are (roles/scopes/tenant).
    Works with either Bearer JWT or API key auth.
    """
    return principal


@router.post(
    "/apikeys",
    response_model=CreateApiKeyResponse,
    summary="Create a new API key (admin-only; key is returned once)",
    dependencies=[Depends(require_scopes(["admin:keys"]))],
)
async def create_key(
    body: CreateApiKeyRequest,
    principal: Principal = Depends(get_principal),
):
    """
    Create an API key under your tenant. Only principals with `admin:keys` can call this.
    The plaintext `key` is returned once; store it securely client-side.
    """
    # Minimal RBAC gate (belt and suspenders in addition to scope)
    if not has_roles(principal.roles, ["admin"]):
        raise HTTPException(status_code=403, detail="Admin role required.")

    meta = {
        "tenant_id": principal.tenant_id,
        "roles": ["service"],               # Default role for keys (narrower than admin)
        "scopes": ["rag:ingest", "rag:query"],  # Practical defaults for automation
        "label": body.label,
        "ttl_days": body.ttl_days,
        "creator": principal.subject,
    }
    created = create_api_key(meta)
    return CreateApiKeyResponse(
        key=created["key"],            # <-- show ONCE
        key_id=created["key_id"],
        last4=created["last4"],
        tenant_id=meta["tenant_id"],
        roles=meta["roles"],
        scopes=meta["scopes"],
    )
