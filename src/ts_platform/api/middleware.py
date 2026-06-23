"""API hardening middleware for local and production-like deployments."""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from ts_platform.api.settings import APISettings


class InMemoryRateLimiter:
    """Small per-client sliding-window limiter for demo deployments."""

    def __init__(self, requests_per_minute: int | None) -> None:
        self.requests_per_minute = requests_per_minute
        self._requests: dict[str, deque[float]] = {}
        self._lock = Lock()

    def allow(self, key: str, now: float) -> bool:
        """Return whether a client can make one more request now."""

        if self.requests_per_minute is None:
            return True
        window_start = now - 60.0
        with self._lock:
            requests = self._requests.setdefault(key, deque())
            while requests and requests[0] < window_start:
                requests.popleft()
            if len(requests) >= self.requests_per_minute:
                return False
            requests.append(now)
            return True


def install_hardening_middleware(app: FastAPI, settings: APISettings) -> None:
    """Install optional auth, body-size, rate-limit, and audit middleware."""

    limiter = InMemoryRateLimiter(settings.rate_limit_requests_per_minute)

    @app.middleware("http")
    async def hardening_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        status_code = 500
        try:
            size_response = _reject_oversized_request(request, settings)
            if size_response is not None:
                status_code = size_response.status_code
                return size_response

            auth_response = _reject_unauthorized_request(request, settings)
            if auth_response is not None:
                status_code = auth_response.status_code
                return auth_response

            now = datetime.now(timezone.utc).timestamp()
            if not limiter.allow(_client_key(request), now):
                response: Response = JSONResponse(
                    status_code=429,
                    content={"detail": "rate limit exceeded"},
                )
                status_code = response.status_code
                return response

            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            _write_audit_event(request, status_code=status_code, settings=settings)


def _reject_oversized_request(
    request: Request,
    settings: APISettings,
) -> JSONResponse | None:
    limit = settings.max_request_body_bytes
    if limit is None:
        return None
    content_length = request.headers.get("content-length")
    if content_length is None:
        return None
    try:
        size = int(content_length)
    except ValueError:
        return JSONResponse(status_code=400, content={"detail": "invalid content-length"})
    if size > limit:
        return JSONResponse(
            status_code=413,
            content={"detail": f"request body exceeds maximum size: {size} > {limit} bytes"},
        )
    return None


def _reject_unauthorized_request(
    request: Request,
    settings: APISettings,
) -> JSONResponse | None:
    expected_key = settings.api_key
    if not expected_key or _is_auth_exempt(request.url.path, settings):
        return None
    provided_key = request.headers.get("x-api-key")
    authorization = request.headers.get("authorization", "")
    bearer_key = authorization.removeprefix("Bearer ").strip()
    if provided_key == expected_key or bearer_key == expected_key:
        return None
    return JSONResponse(status_code=401, content={"detail": "missing or invalid API key"})


def _is_auth_exempt(path: str, settings: APISettings) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}/") for prefix in settings.auth_exempt_paths
    )


def _client_key(request: Request) -> str:
    if request.client is None:
        return "unknown"
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",", 1)[0].strip() or request.client.host
    return request.client.host


def _write_audit_event(request: Request, *, status_code: int, settings: APISettings) -> None:
    audit_path = settings.audit_log_path
    if audit_path is None:
        return
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": request.method,
        "path": request.url.path,
        "status_code": status_code,
        "client": _client_key(request),
    }
    try:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    except OSError:
        return
