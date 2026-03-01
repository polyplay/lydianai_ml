from __future__ import annotations
import time
import jwt
from fastapi import Header, HTTPException, status, Depends
from dataclasses import dataclass

@dataclass(frozen=True)
class AuthConfig:
    jwt_secret: str
    issuer: str = "lydian-poc"
    audience: str = "lydian-poc-workers"
    token_ttl_s: int = 60 * 60 * 24 * 7  # 7 days

def create_worker_token(cfg: AuthConfig, worker_id: str) -> str:
    now = int(time.time())
    payload = {
        "sub": worker_id,
        "iat": now,
        "exp": now + cfg.token_ttl_s,
        "iss": cfg.issuer,
        "aud": cfg.audience,
        "typ": "worker",
    }
    return jwt.encode(payload, cfg.jwt_secret, algorithm="HS256")

def verify_worker_token(cfg: AuthConfig, authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(
            token,
            cfg.jwt_secret,
            algorithms=["HS256"],
            audience=cfg.audience,
            issuer=cfg.issuer,
        )
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {e}") from e
    worker_id = payload.get("sub")
    if not worker_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing subject")
    return worker_id

def worker_auth_dependency(cfg: AuthConfig):
    async def _dep(authorization: str | None = Header(default=None)) -> str:
        return verify_worker_token(cfg, authorization)
    return _dep
