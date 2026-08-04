#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import statistics
import time
import urllib.error
import urllib.request
from uuid import uuid4

API_BASE = os.environ.get("DOCUMIND_BENCHMARK_API", "http://127.0.0.1:8000/api/v1")
EMAIL = os.environ.get("DOCUMIND_BENCHMARK_EMAIL")
PASSWORD = os.environ.get("DOCUMIND_BENCHMARK_PASSWORD")


def main() -> None:
    if not EMAIL or not PASSWORD:
        raise SystemExit(
            "DOCUMIND_BENCHMARK_EMAIL and DOCUMIND_BENCHMARK_PASSWORD are required."
        )
    login, _ = request(
        "POST",
        "/auth/login",
        {"email": EMAIL, "password": PASSWORD},
    )
    token = login["data"]["access_token"]
    suffix = uuid4().hex[:10]
    document, analysis_ms = request(
        "POST",
        "/vision/documents/analyze",
        {
            "filename": f"benchmark-invoice-{suffix}.txt",
            "text": (
                f"Invoice INV-BENCH-{suffix}\n"
                "Vendor: Benchmark Services\n"
                "Invoice Date: 2026-08-04\n"
                "Subtotal: USD 100.00\n"
                "Tax: USD 18.00\n"
                "Total Amount: USD 118.00\n"
                "Currency: USD\n"
            ),
            "metadata": {"document_type_hint": "invoice"},
        },
        token,
    )
    document_id = document["data"]["document_id"]
    reads = [request("GET", f"/documents/{document_id}", token=token)[1] for _ in range(10)]
    _, ask_ms = request(
        "POST",
        "/ask",
        {"question": "What is the total amount?", "document_id": document_id},
        token,
    )
    _, analytics_ms = request("GET", "/analytics/overview", token=token)
    print(
        json.dumps(
            {
                "environment": "local_loopback_single_process",
                "sample_sizes": {
                    "document_analysis": 1,
                    "document_reads": len(reads),
                    "grounded_question": 1,
                    "analytics_overview": 1,
                },
                "document_analysis_ms": round(analysis_ms, 3),
                "document_read_ms": {
                    "median": round(statistics.median(reads), 3),
                    "p95_nearest_rank": round(sorted(reads)[-1], 3),
                    "minimum": round(min(reads), 3),
                    "maximum": round(max(reads), 3),
                },
                "grounded_question_ms": round(ask_ms, 3),
                "analytics_overview_ms": round(analytics_ms, 3),
                "note": "Observed timings are not an SLA or load-test result.",
            },
            indent=2,
            sort_keys=True,
        )
    )


def request(
    method: str,
    path: str,
    payload: dict | None = None,
    token: str | None = None,
) -> tuple[dict, float]:
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    prepared = urllib.request.Request(
        f"{API_BASE}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(prepared, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8")
        raise SystemExit(f"{method} {path} failed: {exc.code} {detail}") from exc
    return result, (time.perf_counter() - started) * 1000


if __name__ == "__main__":
    main()
