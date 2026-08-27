"""Regression tests for the Qdrant honesty and resilience defects (D-01/D-02/D-03).

Each test here fails against the pre-hardening implementation:

* D-01 ``QdrantHybridBackend.status()`` reported ``active`` from configuration and
  package presence alone, without ever contacting the server.
* D-03 ``LocalHybridRetrievalIndex._external_error`` latched permanently: one
  transient failure disabled the Qdrant mirror for the lifetime of the process.
* D-02 the vector backend actually serving a search was never reported, so a
  local-only answer was indistinguishable from a Qdrant-backed one.
"""

from __future__ import annotations

import time

import pytest

from app.core.config import settings
from app.models.vision import ReviewerStatus, VisionDocumentType, VisionEvidenceChunk
from app.services.vision.retrieval_adapters import (
    LocalHybridRetrievalIndex,
    QdrantHybridBackend,
)

DEAD_ENDPOINT = "http://127.0.0.1:6399"


@pytest.fixture
def qdrant_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "enable_qdrant", True)
    return settings


def _chunk(chunk_id: str, text: str, *, tenant: str = "tenant-probe") -> VisionEvidenceChunk:
    return VisionEvidenceChunk(
        chunk_id=chunk_id,
        document_id="doc-probe",
        tenant_id=tenant,
        page_number=1,
        text=text,
        document_type=VisionDocumentType.invoice,
        confidence=0.9,
        model_name="test",
        model_version="v1",
        review_status=ReviewerStatus.unverified,
    )


# --------------------------------------------------------------------------- D-01


def test_status_is_not_active_when_the_server_is_unreachable(
    qdrant_enabled, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Configuration must never be reported as a live backend."""

    monkeypatch.setattr(settings, "qdrant_url", DEAD_ENDPOINT)
    backend = QdrantHybridBackend(vector_namespace="probe", expected_vector_size=768)

    status = backend.status()

    assert status.status == "unavailable", (
        "A configured-but-unreachable Qdrant must not report 'active'; "
        f"got {status.status!r} with reason {status.reason!r}"
    )
    assert status.metadata["server_reachable"] is False
    assert "liveness probe" in status.reason


def test_status_reports_disabled_without_probing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_qdrant", False)
    backend = QdrantHybridBackend(vector_namespace="probe", expected_vector_size=768)

    status = backend.status()

    assert status.status == "disabled"
    assert status.metadata["server_reachable"] is None


def test_liveness_probe_result_is_cached(qdrant_enabled, monkeypatch: pytest.MonkeyPatch) -> None:
    """status() runs on every upsert and search; it must not probe every call."""

    monkeypatch.setattr(settings, "qdrant_url", DEAD_ENDPOINT)
    backend = QdrantHybridBackend(vector_namespace="probe", expected_vector_size=768)
    calls = {"count": 0}

    def counting_urlopen(*args, **kwargs):
        calls["count"] += 1
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", counting_urlopen)

    for _ in range(5):
        backend.status()

    assert calls["count"] == 1, f"expected one cached probe, made {calls['count']}"

    backend.invalidate_liveness_cache()
    backend.status()
    assert calls["count"] == 2


# --------------------------------------------------------------------------- D-03


def test_transient_failure_does_not_disable_qdrant_permanently(
    qdrant_enabled, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One failed call must not take the mirror out for the process lifetime."""

    index = LocalHybridRetrievalIndex(force_local=True)
    index.force_local = False
    index.external_retry_backoff_seconds = 0.0

    attempts: list[str] = []

    class FlakyQdrant:
        def __init__(self) -> None:
            self.fail_next = True

        def status(self):
            from app.services.vision.adapters import AdapterStatus

            return AdapterStatus(
                adapter_id="qdrant-hybrid-retrieval",
                stage="vector_store",
                status="active",
                reason="stub",
                enabled=True,
                installed=True,
            )

        def invalidate_liveness_cache(self) -> None:
            return None

        def replace_document(self, document_id, *, tenant_id, chunks, vectors):
            return None

        def delete_document(self, document_id, *, tenant_id):
            return None

        def candidate_scores(self, dense, *, tenant_id, document_ids, limit):
            attempts.append("call")
            if self.fail_next:
                self.fail_next = False
                raise ConnectionError("qdrant went away")
            return {}

    index.qdrant = FlakyQdrant()
    index.upsert_document("doc-probe", [_chunk("chunk-1", "invoice total payable")])

    index.search("invoice total", tenant_id="tenant-probe")
    assert index._external_error is not None, "the failure should be recorded"
    assert len(attempts) == 1

    # Pre-fix, this second search skipped Qdrant forever because the latch was set.
    index.search("invoice total", tenant_id="tenant-probe")
    assert len(attempts) == 2, "Qdrant must be retried once the backoff has elapsed"
    assert index._external_error is None, "a successful retry must clear the degraded state"
    assert index.status()["degraded"] is False


def test_failure_is_held_off_for_the_backoff_window(
    qdrant_enabled, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = LocalHybridRetrievalIndex(force_local=True)
    index.force_local = False
    index.external_retry_backoff_seconds = 60.0
    attempts: list[str] = []

    class DeadQdrant:
        def status(self):
            from app.services.vision.adapters import AdapterStatus

            return AdapterStatus(
                adapter_id="qdrant-hybrid-retrieval",
                stage="vector_store",
                status="active",
                reason="stub",
                enabled=True,
                installed=True,
            )

        def invalidate_liveness_cache(self) -> None:
            return None

        def replace_document(self, document_id, *, tenant_id, chunks, vectors):
            return None

        def delete_document(self, document_id, *, tenant_id):
            return None

        def candidate_scores(self, dense, *, tenant_id, document_ids, limit):
            attempts.append("call")
            raise ConnectionError("still down")

    index.qdrant = DeadQdrant()
    index.upsert_document("doc-probe", [_chunk("chunk-1", "invoice total payable")])

    index.search("invoice total", tenant_id="tenant-probe")
    index.search("invoice total", tenant_id="tenant-probe")

    assert len(attempts) == 1, "the backoff window must suppress immediate retries"
    report = index.status()
    assert report["degraded"] is True
    assert report["external_retry_after_seconds"] > 0


# --------------------------------------------------------------------------- D-02


def test_status_names_the_backend_that_actually_served_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local-only answer must be distinguishable from a Qdrant-backed one."""

    monkeypatch.setattr(settings, "enable_qdrant", False)
    index = LocalHybridRetrievalIndex(force_local=True)
    index.upsert_document("doc-probe", [_chunk("chunk-1", "invoice total payable")])

    report = index.status()

    assert report["vector_backend"] == "local_hybrid_only"
    assert report["degraded"] is False


def test_search_still_answers_when_the_mirror_is_down(
    qdrant_enabled, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Degradation must be disclosed, not silent, and must not lose the answer."""

    monkeypatch.setattr(settings, "qdrant_url", DEAD_ENDPOINT)
    index = LocalHybridRetrievalIndex()
    index.upsert_document(
        "doc-probe",
        [
            _chunk("chunk-1", "Total Amount: INR 472,000.00 payable to the vendor"),
            _chunk("chunk-2", "Governing law and confidentiality clauses"),
        ],
    )

    hits = index.search("what is the total amount", tenant_id="tenant-probe", top_k=1)

    assert hits, "local retrieval must still answer when Qdrant is unreachable"
    assert "Total Amount" in hits[0].chunk.text
    assert index.status()["vector_backend"] == "local_hybrid_only"


def test_probe_is_bounded_in_time(qdrant_enabled, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead endpoint must not stall a request thread."""

    monkeypatch.setattr(settings, "qdrant_url", DEAD_ENDPOINT)
    backend = QdrantHybridBackend(vector_namespace="probe", expected_vector_size=768)

    started = time.perf_counter()
    backend.status()
    elapsed = time.perf_counter() - started

    assert elapsed < backend.liveness_timeout_seconds + 1.0, f"liveness probe took {elapsed:.2f}s"
