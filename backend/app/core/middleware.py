from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.config import settings

ACCESS_LOG = logging.getLogger("documind.access")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id") or f"req-{uuid4().hex}"
        request.state.request_id = request_id
        started = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        # One structured line per request, carrying the correlation id that ties
        # together every downstream log for this call. Without this the JSON
        # formatter and the X-Request-ID header had nothing to correlate.
        ACCESS_LOG.info(
            "request completed",
            extra={
                "context": {
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "session_id": request.headers.get("x-session-id"),
                }
            },
        )
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
    #: Upper bound on tracked rate-limit buckets. Keys are
    #: ``client:path`` and path-parameter routes mint a distinct key per document
    #: id, so an unbounded dict grew forever. This is an LRU with eviction.
    max_tracked_keys: int = 8192

    def __init__(self, app, requests: int, window_seconds: int) -> None:
        super().__init__(app)
        self.requests = max(1, requests)
        self.window_seconds = max(1, window_seconds)
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
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
            self._evict_expired(now)
            bucket = self._hits.get(key)
            if bucket is None:
                bucket = deque()
                self._hits[key] = bucket
            self._hits.move_to_end(key)
            while bucket and now - bucket[0] >= self.window_seconds:
                bucket.popleft()
            if len(bucket) >= self.requests:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])))
                return self._limited_response(retry_after)
            bucket.append(now)
            while len(self._hits) > self.max_tracked_keys:
                self._hits.popitem(last=False)
        return await call_next(request)

    def _evict_expired(self, now: float) -> None:
        """Drop buckets whose whole window has elapsed.

        Called under the lock. Bounded work per request: it stops at the first
        key still inside its window, and insertion order approximates recency.
        """

        stale: list[str] = []
        for key, bucket in self._hits.items():
            if bucket and now - bucket[-1] < self.window_seconds:
                break
            stale.append(key)
            if len(stale) >= 64:
                break
        for key in stale:
            self._hits.pop(key, None)

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
