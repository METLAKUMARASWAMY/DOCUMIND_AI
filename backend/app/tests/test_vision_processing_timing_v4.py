from __future__ import annotations

from app.models.vision import (
    VisionAuditEvent,
    VisionDocumentRecord,
    VisionProcessingStatus,
)
from app.services.vision.analytics import VisionAnalyticsService


def _document(*, pipeline: str = "documind-vision-v4") -> VisionDocumentRecord:
    return VisionDocumentRecord(
        document_id="vision-doc-timing",
        case_id="vision-case-timing",
        filename="timing.txt",
        file_hash="a" * 64,
        status=VisionProcessingStatus.review_required,
        created_at="2026-08-06T10:00:00+00:00",
        updated_at="2026-08-07T10:00:00+00:00",
        metadata={"pipeline_version": pipeline},
    )


def _event(document: VisionDocumentRecord, event_type: str, created_at: str) -> None:
    document.audit_events.append(
        VisionAuditEvent(
            document_id=document.document_id,
            event_type=event_type,
            message=event_type,
            created_at=created_at,
        )
    )


def test_processing_timing_stops_at_pipeline_terminal_event_not_later_review_events():
    document = _document()
    _event(document, "document_queued", "2026-08-06T10:00:00+00:00")
    _event(document, "preprocessing", "2026-08-06T10:00:01+00:00")
    _event(document, "indexed", "2026-08-06T10:00:03+00:00")
    _event(document, "review_required", "2026-08-06T10:00:04+00:00")
    _event(document, "review_decision_recorded", "2026-08-07T10:00:00+00:00")

    overview = VisionAnalyticsService().overview([document])

    assert overview["average_processing_time_seconds"] == 3.0
    assert overview["median_processing_time_seconds"] == 3.0
    assert overview["p95_processing_time_seconds"] == 3.0
    assert overview["processing_timing"]["mean_ms"] == 3000.0
    assert overview["processing_timing"]["queued"]["mean_ms"] == 1000.0


def test_processing_timing_excludes_legacy_and_implausible_records():
    legacy = _document(pipeline="legacy")
    _event(legacy, "preprocessing", "2026-08-06T10:00:00+00:00")
    _event(legacy, "review_required", "2026-08-07T10:00:00+00:00")
    invalid = _document()
    invalid.document_id = "vision-doc-invalid-timing"
    invalid.metadata["processing_timing"] = {
        "processing_duration_ms": 43_683_100,
        "processing_started_at": "2026-08-06T10:00:00+00:00",
        "processing_completed_at": "2026-08-07T00:00:00+00:00",
    }

    overview = VisionAnalyticsService().overview([legacy, invalid])

    assert overview["average_processing_time_seconds"] is None
    assert overview["sample_sizes"]["processing_duration_samples"] == 0


def test_processing_timing_reports_real_millisecond_stage_summaries():
    document = _document()
    document.metadata["processing_timing"] = {
        "processing_started_at": "2026-08-06T10:00:00+00:00",
        "processing_completed_at": "2026-08-06T10:00:02+00:00",
        "processing_duration_ms": 2000,
        "queued_duration_ms": 250,
        "stage_durations_ms": {
            "ingestion": 100,
            "preprocessing": 200,
            "ocr": 900,
            "layout": 100,
            "extraction": 300,
            "validation": 200,
            "indexing": 100,
            "graph_projection": 100,
        },
    }

    overview = VisionAnalyticsService().overview([document])
    timing = overview["processing_timing"]

    assert timing["mean_ms"] == 2000.0
    assert timing["sample_size"] == 1
    assert timing["per_stage"]["ocr"] == {
        "mean_ms": 900.0,
        "median_ms": 900.0,
        "p95_ms": 900.0,
        "sample_size": 1,
    }


def test_processing_timing_excludes_explicitly_unreliable_failed_run():
    document = _document()
    document.metadata["processing_timing"] = {
        "processing_started_at": "2026-08-06T10:00:00+00:00",
        "processing_completed_at": "2026-08-06T10:00:02+00:00",
        "processing_duration_ms": 2000,
        "reliable": False,
        "exclusion_reason": "processing_failed",
    }

    overview = VisionAnalyticsService().overview([document])

    assert overview["average_processing_time_seconds"] is None
    assert overview["sample_sizes"]["processing_duration_samples"] == 0
