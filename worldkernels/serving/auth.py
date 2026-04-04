r"""API key authentication middleware."""

from __future__ import annotations

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


def require_api_key(expected_key: str | None):
    r"""Return a FastAPI dependency that validates the Bearer token."""

    async def _verify(key: str | None = Security(_api_key_header)) -> None:
        if expected_key is None:
            return
        if key is None:
            raise HTTPException(status_code=401, detail="Missing API key.")
        token = key.removeprefix("Bearer ").strip()
        if token != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API key.")

    return _verify
