from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from statistics import mean, median
from typing import Any

from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VisionConfidenceSummary,
    VisionDocumentRecord,
    VisionProcessingStatus,
    VisionReviewTask,
)
from app.services.vision.legacy_audit import CURRENT_PIPELINE_VERSION
from app.services.vision.schema_registry import vision_schema_registry

MAX_PROCESSING_DURATION_MS = 60 * 60 * 1000


def _valid(values: Iterable[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and 0 <= value <= 1]


def _average(values: Iterable[float | None]) -> float | None:
    samples = _valid(values)
    return round(mean(samples), 4) if samples else None


def _field_confidence(field: ExtractedVisionField) -> float | None:
    for value in (
        field.calibrated_confidence,
        field.combined_confidence,
        field.confidence,
    ):
        if value is not None:
            return value
    return None


def _first_defined(*values: float | None) -> float | None:
    return next((value for value in values if value is not None), None)


def _elapsed_seconds(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        elapsed = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except (TypeError, ValueError):
        return None
    return max(0.0, elapsed)


def _elapsed_milliseconds(start: str | None, end: str | None) -> float | None:
    seconds = _elapsed_seconds(start, end)
    return seconds * 1000 if seconds is not None else None


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    samples = sorted(float(value) for value in values)
    if not samples:
        return None
    if len(samples) == 1:
        return samples[0]
    position = (len(samples) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(samples) - 1)
    fraction = position - lower
    return samples[lower] + (samples[upper] - samples[lower]) * fraction


def _duration_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values]
    return {
        "mean_ms": round(mean(samples), 3) if samples else None,
        "median_ms": round(median(samples), 3) if samples else None,
        "p95_ms": (round(value, 3) if (value := _percentile(samples, 0.95)) is not None else None),
        "sample_size": len(samples),
    }


class VisionAnalyticsService:
    calculation_version = "vision-analytics-v3"

    def refresh_document(self, document: VisionDocumentRecord) -> None:
        approved = [
            field
            for field in document.extracted_fields
            if field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
        ]
        document.confidence_summary = VisionConfidenceSummary(
            mean_ocr_confidence=_average(word.confidence for word in document.ocr_words),
            mean_extraction_confidence=_average(
                _first_defined(field.extraction_confidence, field.confidence)
                for field in document.extracted_fields
            ),
            mean_field_confidence=_average(
                _field_confidence(field) for field in document.extracted_fields
            ),
            mean_calibrated_confidence=_average(
                field.calibrated_confidence for field in document.extracted_fields
            ),
            approved_field_confidence=_average(_field_confidence(field) for field in approved),
            ocr_word_count=len(document.ocr_words),
            field_count=len(document.extracted_fields),
            approved_field_count=len(approved),
            confidence_available=bool(document.extracted_fields),
        )
        document.field_completeness = vision_schema_registry.summarize(
            document.document_type,
            document.extracted_fields,
            extraction_diagnostics=document.metadata.get("extraction_diagnostics"),
        )
        if document.review_task:
            document.review_task.filename = document.filename
            document.review_task.document_type = document.document_type
            document.review_task.page_count = document.page_count
            document.review_task.confidence_value = (
                document.confidence_summary.mean_field_confidence
            )
            document.review_task.confidence_status = (
                "available" if document.confidence_summary.confidence_available else "not_available"
            )
            document.review_task.expected_required_fields = (
                document.field_completeness.expected_required_count
            )
            document.review_task.unresolved_required_fields = (
                document.field_completeness.unresolved_required_count
            )

    def overview(
        self,
        documents: Iterable[VisionDocumentRecord],
        reviews: Iterable[VisionReviewTask] = (),
        *,
        question_records: Iterable[dict[str, Any]] = (),
        narrative_count: int = 0,
        report_count: int = 0,
    ) -> dict[str, Any]:
        document_list = list(documents)
        review_list = list(reviews)
        for document in document_list:
            self.refresh_document(document)
        fields = [field for document in document_list for field in document.extracted_fields]
        approved_fields = [
            field
            for field in fields
            if field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
        ]
        ocr_words = [word for document in document_list for word in document.ocr_words]
        model_runs = [run for document in document_list for run in document.model_runs]
        risks = [risk for document in document_list for risk in document.risk_indicators]
        warnings = [
            warning for document in document_list for warning in document.processing_warnings
        ]
        questions = list(question_records)
        dates = sorted(document.created_at for document in document_list if document.created_at)
        updated = sorted(document.updated_at for document in document_list if document.updated_at)
        with_fields = sum(bool(document.extracted_fields) for document in document_list)
        extraction_coverage = round(with_fields / len(document_list), 4) if document_list else None
        exact_schema_count = sum(
            (document.metadata.get("schema_selection") or {}).get("schema_source") == "exact"
            for document in document_list
        )
        generic_schema_count = sum(
            (document.metadata.get("schema_selection") or {}).get("schema_source")
            in {"generic_key_value", "generic_metadata"}
            for document in document_list
        )
        fallback_schema_count = sum(
            bool((document.metadata.get("schema_selection") or {}).get("fallback_used"))
            for document in document_list
        )
        schema_unavailable_count = sum(
            not document.field_completeness.schema_available for document in document_list
        )
        expected_required = sum(
            document.field_completeness.expected_required_count or 0 for document in document_list
        )
        unresolved_required = sum(
            document.field_completeness.unresolved_required_count or 0 for document in document_list
        )
        ocr_answers = [
            item
            for item in questions
            if str((item.get("answer") or {}).get("retrieval_path", "")).startswith("ocr_")
            and (item.get("answer") or {}).get("answer_status") == "supported"
        ]
        structured_questions = [
            item
            for item in questions
            if (item.get("answer") or {}).get("query_type")
            in {"structured_field_lookup", "structured_lookup"}
        ]
        processing_samples = [
            sample
            for document in document_list
            if (sample := self._document_processing_sample(document)) is not None
        ]
        processing_durations_ms = [
            float(sample["processing_duration_ms"]) for sample in processing_samples
        ]
        queue_durations_ms = [
            float(sample["queued_duration_ms"])
            for sample in processing_samples
            if sample.get("queued_duration_ms") is not None
        ]
        per_stage_samples: dict[str, list[float]] = {}
        for sample in processing_samples:
            for stage, duration in dict(sample.get("stage_durations_ms") or {}).items():
                if self._valid_duration_ms(duration, allow_zero=True):
                    per_stage_samples.setdefault(stage, []).append(float(duration))
        processing_summary = _duration_summary(processing_durations_ms)
        review_durations = [
            duration
            for review in review_list
            if (duration := self._review_duration(review)) is not None
        ]
        model_failure_count = sum("fail" in item.status.lower() for item in model_runs)
        processing_failure_count = sum(
            any(event.event_type == "processing_failed" for event in item.audit_events)
            or item.status == VisionProcessingStatus.confirmed_error
            for item in document_list
        )
        return {
            "calculation_version": self.calculation_version,
            "last_updated": updated[-1] if updated else None,
            "date_range": {
                "from": dates[0] if dates else None,
                "to": dates[-1] if dates else None,
            },
            "documents": len(document_list),
            "document_count": len(document_list),
            "documents_with_fields": with_fields,
            "documents_without_fields": len(document_list) - with_fields,
            "structured_extraction_coverage": extraction_coverage,
            "extraction_coverage": extraction_coverage,
            "average_fields_per_supported_document": (
                round(len(fields) / with_fields, 3) if with_fields else None
            ),
            "required_field_completion_rate": (
                round((expected_required - unresolved_required) / expected_required, 4)
                if expected_required
                else None
            ),
            "documents_using_exact_schema": exact_schema_count,
            "documents_using_fallback_schema": fallback_schema_count,
            "documents_using_generic_schema": generic_schema_count,
            "documents_schema_unavailable": schema_unavailable_count,
            "ocr_only_answer_success_count": len(ocr_answers),
            "ocr_only_answer_success_rate": (
                round(len(ocr_answers) / len(structured_questions), 4)
                if structured_questions
                else None
            ),
            "field_count": len(fields),
            "approved_field_count": len(approved_fields),
            "ocr_word_count": len(ocr_words),
            "mean_ocr_confidence": _average(word.confidence for word in ocr_words),
            "mean_extraction_confidence": _average(
                _first_defined(field.extraction_confidence, field.confidence) for field in fields
            ),
            "mean_field_confidence": _average(_field_confidence(field) for field in fields),
            "mean_calibrated_field_confidence": _average(
                field.calibrated_confidence for field in fields
            ),
            "mean_approved_field_confidence": _average(
                _field_confidence(field) for field in approved_fields
            ),
            "average_field_confidence": _average(_field_confidence(field) for field in fields),
            "confidence_exclusions": {
                "documents_without_fields": len(document_list) - with_fields,
                "fields_without_confidence": sum(
                    _field_confidence(field) is None for field in fields
                ),
            },
            "status_counts": dict(Counter(item.status.value for item in document_list)),
            "document_type_counts": dict(
                Counter(item.document_type.value for item in document_list)
            ),
            "review_tasks": len(review_list),
            "review_priority_counts": dict(Counter(item.priority.value for item in review_list)),
            "review_outcome_counts": dict(Counter(item.status.value for item in review_list)),
            "risk_counts": dict(Counter(item.severity.value for item in risks)),
            "risk_category_counts": dict(Counter(item.category for item in risks)),
            "processing_warning_count": len(warnings),
            "processing_warning_counts": dict(Counter(item.title for item in warnings)),
            "model_usage": dict(Counter(item.model_name for item in model_runs)),
            "model_failure_count": model_failure_count,
            "model_run_count": len(model_runs),
            "model_failure_rate": (
                round(model_failure_count / len(model_runs), 4) if model_runs else None
            ),
            "processing_success_count": len(document_list) - processing_failure_count,
            "processing_failure_count": processing_failure_count,
            "average_processing_time_seconds": (
                round(float(processing_summary["mean_ms"]) / 1000, 3)
                if processing_summary["mean_ms"] is not None
                else None
            ),
            "median_processing_time_seconds": (
                round(float(processing_summary["median_ms"]) / 1000, 3)
                if processing_summary["median_ms"] is not None
                else None
            ),
            "p95_processing_time_seconds": (
                round(float(processing_summary["p95_ms"]) / 1000, 3)
                if processing_summary["p95_ms"] is not None
                else None
            ),
            "processing_timing": {
                "duration_unit": "milliseconds",
                **processing_summary,
                "queued": _duration_summary(queue_durations_ms),
                "per_stage": {
                    stage: _duration_summary(values)
                    for stage, values in sorted(per_stage_samples.items())
                },
                "pipeline_versions": dict(
                    Counter(str(sample["pipeline_version"]) for sample in processing_samples)
                ),
                "exclusion_policy": {
                    "incomplete_jobs": True,
                    "missing_timestamps": True,
                    "negative_durations": True,
                    "legacy_or_incompatible_pipeline_versions": True,
                    "maximum_processing_duration_ms": MAX_PROCESSING_DURATION_MS,
                },
            },
            "average_review_time_seconds": (
                round(mean(review_durations), 3) if review_durations else None
            ),
            "rag_answer_status": dict(
                Counter(
                    str((item.get("answer") or {}).get("answer_status", "unknown"))
                    for item in questions
                )
            ),
            "citation_coverage": self._citation_coverage(questions),
            "narrative_count": narrative_count,
            "report_count": report_count,
            "sample_sizes": {
                "documents": len(document_list),
                "documents_with_fields": with_fields,
                "ocr_words": len(ocr_words),
                "fields": len(fields),
                "approved_fields": len(approved_fields),
                "review_tasks": len(review_list),
                "risk_indicators": len(risks),
                "processing_warnings": len(warnings),
                "model_runs": len(model_runs),
                "processing_duration_samples": len(processing_samples),
                "review_duration_samples": len(review_durations),
                "rag_questions": len(questions),
            },
            "fake_metrics_reported": False,
        }

    def _document_processing_sample(
        self,
        document: VisionDocumentRecord,
    ) -> dict[str, Any] | None:
        pipeline_version = str(document.metadata.get("pipeline_version") or "legacy")
        if pipeline_version != CURRENT_PIPELINE_VERSION:
            return None

        timing = dict(document.metadata.get("processing_timing") or {})
        if timing.get("reliable") is False:
            return None
        processing_started_at = timing.get("processing_started_at")
        processing_completed_at = timing.get("processing_completed_at")
        duration_ms = timing.get("processing_duration_ms")
        if duration_ms is None:
            duration_ms = _elapsed_milliseconds(
                str(processing_started_at) if processing_started_at else None,
                str(processing_completed_at) if processing_completed_at else None,
            )

        queued_duration_ms = timing.get("queued_duration_ms")
        stage_durations_ms = dict(timing.get("stage_durations_ms") or {})
        if not self._valid_duration_ms(duration_ms):
            event_timing = self._event_processing_timing(document)
            if event_timing is None:
                return None
            duration_ms = event_timing["processing_duration_ms"]
            queued_duration_ms = event_timing.get("queued_duration_ms")

        if not self._valid_duration_ms(queued_duration_ms, allow_none=True):
            queued_duration_ms = None
        return {
            "document_id": document.document_id,
            "pipeline_version": pipeline_version,
            "processing_duration_ms": float(duration_ms),
            "queued_duration_ms": (
                float(queued_duration_ms) if queued_duration_ms is not None else None
            ),
            "stage_durations_ms": stage_durations_ms,
        }

    def _event_processing_timing(
        self,
        document: VisionDocumentRecord,
    ) -> dict[str, float] | None:
        events = document.audit_events
        start_indexes = [
            index for index, event in enumerate(events) if event.event_type == "preprocessing"
        ]
        if not start_indexes:
            return None
        start_index = start_indexes[-1]
        start = events[start_index].created_at
        terminal = next(
            (
                event.created_at
                for event in events[start_index + 1 :]
                if event.event_type in {"review_required", "completed", "processing_failed"}
            ),
            None,
        )
        duration_ms = _elapsed_milliseconds(start, terminal)
        if not self._valid_duration_ms(duration_ms):
            return None
        queued_events = [
            event
            for event in events[: start_index + 1]
            if event.event_type in {"document_queued", "document_reprocess_queued"}
        ]
        queued_duration_ms = (
            _elapsed_milliseconds(queued_events[-1].created_at, start) if queued_events else None
        )
        result = {"processing_duration_ms": float(duration_ms)}
        if self._valid_duration_ms(queued_duration_ms, allow_zero=True):
            result["queued_duration_ms"] = float(queued_duration_ms)
        return result

    def _valid_duration_ms(
        self,
        value: Any,
        *,
        allow_none: bool = False,
        allow_zero: bool = False,
    ) -> bool:
        if value is None:
            return allow_none
        try:
            duration = float(value)
        except (TypeError, ValueError):
            return False
        lower_bound = 0 if allow_zero else 0.001
        return lower_bound <= duration <= MAX_PROCESSING_DURATION_MS

    def _review_duration(self, review: VisionReviewTask) -> float | None:
        if review.status in {
            VisionProcessingStatus.review_required,
            VisionProcessingStatus.under_review,
        }:
            return None
        return _elapsed_seconds(review.created_at, review.updated_at)

    def _citation_coverage(self, questions: list[dict[str, Any]]) -> dict[str, Any]:
        supported = [
            item
            for item in questions
            if (item.get("answer") or {}).get("answer_status")
            in {"supported", "partially_supported"}
        ]
        cited = [item for item in supported if (item.get("answer") or {}).get("citations")]
        return {
            "supported_answer_count": len(supported),
            "supported_answers_with_citations": len(cited),
            "coverage": round(len(cited) / len(supported), 4) if supported else None,
        }


vision_analytics_service = VisionAnalyticsService()
