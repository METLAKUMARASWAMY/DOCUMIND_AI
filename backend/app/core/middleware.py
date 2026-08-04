from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.config import settings


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or f"req-{uuid4().hex}"
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        if request.url.path.startswith("/api/v1/auth"):
            response.headers["Cache-Control"] = "no-store"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, requests: int, window_seconds: int) -> None:
        super().__init__(app)
        self.requests = max(1, requests)
        self.window_seconds = max(1, window_seconds)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()
        self._redis = None

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in {"/", "/api/v1/health", "/docs", "/openapi.json"}:
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        if settings.trust_proxy_headers:
            client = request.headers.get("x-forwarded-for", client).split(",", 1)[0].strip()
        key = f"{client}:{request.url.path}"
        if settings.enable_redis:
            limited, retry_after = self._redis_limited(key)
            if limited:
                return self._limited_response(retry_after)
            if retry_after is not None:
                return await call_next(request)
        now = time.monotonic()
        with self._lock:
            bucket = self._hits[key]
            while bucket and now - bucket[0] >= self.window_seconds:
                bucket.popleft()
            if len(bucket) >= self.requests:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])))
                return self._limited_response(retry_after)
            bucket.append(now)
        return await call_next(request)

    def _redis_limited(self, key: str) -> tuple[bool, int | None]:
        try:
            if self._redis is None:
                from redis import Redis

                self._redis = Redis.from_url(settings.redis_url, decode_responses=True)
            window = int(time.time()) // self.window_seconds
            redis_key = f"rate-limit:{window}:{key}"
            pipeline = self._redis.pipeline()
            pipeline.incr(redis_key)
            pipeline.expire(redis_key, self.window_seconds + 1)
            count, _ = pipeline.execute()
            return int(count) > self.requests, self.window_seconds
        except Exception:
            return False, None

    def _limited_response(self, retry_after: int) -> JSONResponse:
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "case_id": None,
                "data": None,
                "errors": [{"code": "rate_limited", "message": "Request limit exceeded"}],
            },
            headers={"Retry-After": str(max(1, retry_after))},
        )


def install_security_middleware(app: FastAPI) -> None:
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        requests=settings.rate_limit_requests,
        window_seconds=settings.rate_limit_window_seconds,
    )
