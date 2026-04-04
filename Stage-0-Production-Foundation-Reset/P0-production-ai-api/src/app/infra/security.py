from __future__ import annotations

from fastapi import Header, HTTPException, Request, status

from app.config import get_settings


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> str:
    settings = get_settings()
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
    return x_api_key


def rate_limit_request(request: Request) -> None:
    limiter = request.app.state.rate_limiter
    client_key = request.headers.get("x-api-key", "anonymous")
    if not limiter.check(client_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded.",
        )

