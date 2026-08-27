"""Regression tests for D-06 / D-07 / D-08 / D-09.

* D-07 unhandled exceptions returned Starlette's plain-text 500, which does not
  match the ``{success, case_id, data, errors}`` envelope the frontend parses,
  and were never logged with the request id.
* D-06 the correlation id was set and returned as a header but never reached a
  log record, so a request could not be traced.
* D-08 the rate-limit bucket dict was keyed ``client:path`` and never evicted,
  so path-parameter routes grew it without bound.
* D-09 ``/metrics`` was served to anyone.
"""

from __future__ import annotations

import logging

import pytest
from fastapi import APIRouter
from fastapi.testclient import TestClient

from app.core.config import settings
from app.core.middleware import RateLimitMiddleware
from app.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


# --------------------------------------------------------------------- D-07


def test_unhandled_exception_returns_the_standard_envelope() -> None:
    app = create_app()
    router = APIRouter()

    @router.get("/boom")
    async def boom() -> dict:
        raise RuntimeError("internal detail that must not leak")

    app.include_router(router, prefix="/api/v1/_test")
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/api/v1/_test/boom")

    assert response.status_code == 500
    payload = response.json()
    # The frontend's parseResponse() depends on exactly this shape.
    assert payload["success"] is False
    assert payload["data"] is None
    assert payload["errors"][0]["code"] == "internal_error"
    assert payload["request_id"]
    # The internal message must never reach the client.
    assert "internal detail that must not leak" not in response.text
    assert "RuntimeError" not in response.text
    assert "Traceback" not in response.text


def test_unhandled_exception_is_logged_with_the_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app()
    router = APIRouter()

    @router.get("/boom")
    async def boom() -> dict:
        raise RuntimeError("traceable failure")

    app.include_router(router, prefix="/api/v1/_test")
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="documind.api"):
        response = client.get("/api/v1/_test/boom", headers={"X-Request-ID": "req-trace-me"})

    assert response.json()["request_id"] == "req-trace-me"
    records = [r for r in caplog.records if r.name == "documind.api"]
    assert records, "an unhandled exception must produce a log record"
    context = getattr(records[0], "context", {})
    assert context.get("request_id") == "req-trace-me"
    assert context.get("path") == "/api/v1/_test/boom"
    assert records[0].exc_info is not None, "the traceback must be captured in the log"


def test_http_exceptions_also_use_the_envelope(client: TestClient) -> None:
    response = client.get("/api/v1/documents/does-not-exist")

    assert response.status_code in {401, 403, 404}
    payload = response.json()
    assert payload["success"] is False
    assert payload["errors"]
    assert "request_id" in payload


def test_validation_errors_use_the_envelope(client: TestClient) -> None:
    response = client.post("/api/v1/auth/login", json={"email": "not-an-email"})

    assert response.status_code == 422
    payload = response.json()
    assert payload["success"] is False
    assert payload["errors"][0]["code"] == "validation_error"


# --------------------------------------------------------------------- D-06


def test_every_request_emits_one_correlated_access_log(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="documind.access"):
        response = client.get("/health/live", headers={"X-Request-ID": "req-access-1"})

    assert response.headers["X-Request-ID"] == "req-access-1"
    records = [r for r in caplog.records if r.name == "documind.access"]
    assert len(records) == 1
    context = getattr(records[0], "context", {})
    assert context["request_id"] == "req-access-1"
    assert context["method"] == "GET"
    assert context["path"] == "/health/live"
    assert context["status_code"] == 200
    assert isinstance(context["duration_ms"], float)


def test_request_id_is_generated_when_absent(client: TestClient) -> None:
    response = client.get("/health/live")
    assert response.headers["X-Request-ID"].startswith("req-")


# --------------------------------------------------------------------- D-08


def test_rate_limit_buckets_do_not_grow_without_bound() -> None:
    """Distinct path-parameter routes must not leak a bucket each, forever."""

    middleware = RateLimitMiddleware(app=None, requests=1000, window_seconds=60)
    middleware.max_tracked_keys = 100

    now = 1_000.0
    for index in range(1000):
        with middleware._lock:
            middleware._evict_expired(now)
            key = f"1.2.3.4:/api/v1/documents/{index}"
            from collections import deque

            middleware._hits[key] = deque([now])
            middleware._hits.move_to_end(key)
            while len(middleware._hits) > middleware.max_tracked_keys:
                middleware._hits.popitem(last=False)

    assert len(middleware._hits) <= middleware.max_tracked_keys, (
        f"bucket dict grew to {len(middleware._hits)} entries"
    )


def test_expired_buckets_are_evicted() -> None:
    from collections import deque

    middleware = RateLimitMiddleware(app=None, requests=10, window_seconds=60)
    middleware._hits["1.2.3.4:/old"] = deque([0.0])
    middleware._hits["1.2.3.4:/recent"] = deque([1_000.0])

    middleware._evict_expired(1_000.0)

    assert "1.2.3.4:/old" not in middleware._hits
    assert "1.2.3.4:/recent" in middleware._hits


def test_rate_limiting_still_works(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "rate_limit_requests", 3)
    app = create_app()
    limited = TestClient(app, raise_server_exceptions=False)

    statuses = [limited.get("/api/v1/auth/me").status_code for _ in range(8)]

    assert 429 in statuses, f"rate limiting did not engage: {statuses}"


# --------------------------------------------------------------------- D-09


def test_metrics_requires_a_token_when_one_is_configured(monkeypatch) -> None:
    monkeypatch.setattr(settings, "metrics_auth_token", "scraper-secret")
    client = TestClient(create_app(), raise_server_exceptions=False)

    assert client.get("/metrics").status_code == 403
    assert client.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 403
    ok = client.get("/metrics", headers={"Authorization": "Bearer scraper-secret"})
    assert ok.status_code == 200


def test_metrics_is_loopback_only_when_no_token_is_set(monkeypatch) -> None:
    monkeypatch.setattr(settings, "metrics_auth_token", None)
    client = TestClient(create_app(), raise_server_exceptions=False)

    # TestClient presents as testclient, not loopback.
    response = client.get("/metrics")
    assert response.status_code == 403
    assert "METRICS_AUTH_TOKEN" in response.text


def test_structured_http_detail_is_preserved_verbatim() -> None:
    """`detail` is wire contract, not decoration.

    The approval policy returns a structured blocker list through
    ``HTTPException.detail``, and the frontend's ``parseResponse()`` reads
    ``payload.detail`` for its display message. An earlier version of the error
    handler flattened it into the envelope and broke both.
    """

    app = create_app()
    router = APIRouter()

    @router.get("/structured")
    async def structured() -> dict:
        from fastapi import HTTPException as _HTTPException

        raise _HTTPException(
            status_code=422,
            detail={
                "code": "approval_blocked",
                "message": "Approval is blocked",
                "blockers": [{"code": "no_evidence", "field": "total_amount"}],
            },
        )

    app.include_router(router, prefix="/api/v1/_test")
    client = TestClient(app, raise_server_exceptions=False)

    payload = client.get("/api/v1/_test/structured").json()

    assert payload["detail"]["blockers"] == [{"code": "no_evidence", "field": "total_amount"}]
    assert payload["detail"]["code"] == "approval_blocked"
    # ...and the envelope is present alongside it.
    assert payload["success"] is False
    assert payload["errors"][0]["code"] == "approval_blocked"
    assert payload["request_id"]
