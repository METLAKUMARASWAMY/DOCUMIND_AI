from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, Response

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
except ImportError:  # pragma: no cover - dependency is present in production requirements
    CONTENT_TYPE_LATEST = "text/plain"
    Counter = Histogram = None
    generate_latest = None


REQUEST_COUNT = (
    Counter(
        "documind_http_requests_total",
        "HTTP requests handled by DocuMind.",
        ("method", "route", "status"),
    )
    if Counter
    else None
)
REQUEST_LATENCY = (
    Histogram(
        "documind_http_request_duration_seconds",
        "HTTP request latency without tenant or document labels.",
        ("method", "route"),
    )
    if Histogram
    else None
)


def install_observability(app: FastAPI) -> None:
    @app.middleware("http")
    async def prometheus_metrics(request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        route_path = getattr(route, "path", "unmatched")
        if REQUEST_COUNT is not None:
            REQUEST_COUNT.labels(request.method, route_path, str(response.status_code)).inc()
        if REQUEST_LATENCY is not None:
            REQUEST_LATENCY.labels(request.method, route_path).observe(time.perf_counter() - started)
        return response

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Any:
        if generate_latest is None:
            return Response(content="prometheus-client unavailable\n", media_type="text/plain", status_code=503)
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
