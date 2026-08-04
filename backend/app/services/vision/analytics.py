from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from datetime import datetime
from statistics import mean
from typing import Any

from app.models.vision import (
    ExtractedVisionField,
    ReviewerStatus,
    VisionConfidenceSummary,
    VisionDocumentRecord,
    VisionProcessingStatus,
    VisionReviewTask,
)
from app.services.vision.schema_registry import vision_schema_registry


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
    except ValueError:
        return None
    return max(0.0, elapsed)


class VisionAnalyticsService:
    calculation_version = "vision-analytics-v2"

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
            approved_field_confidence=_average(
                _field_confidence(field) for field in approved
            ),
            ocr_word_count=len(document.ocr_words),
            field_count=len(document.extracted_fields),
            approved_field_count=len(approved),
            confidence_available=bool(document.extracted_fields),
        )
        document.field_completeness = vision_schema_registry.summarize(
            document.document_type,
            document.extracted_fields,
        )
        if document.review_task:
            document.review_task.filename = document.filename
            document.review_task.document_type = document.document_type
            document.review_task.page_count = document.page_count
            document.review_task.confidence_value = (
                document.confidence_summary.mean_field_confidence
            )
            document.review_task.confidence_status = (
                "available"
                if document.confidence_summary.confidence_available
                else "not_available"
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
        extraction_coverage = (
            round(with_fields / len(document_list), 4) if document_list else None
        )
        exact_schema_count = sum(
            (document.metadata.get("schema_selection") or {}).get("schema_source")
            == "exact"
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
            document.field_completeness.expected_required_count or 0
            for document in document_list
        )
        unresolved_required = sum(
            document.field_completeness.unresolved_required_count or 0
            for document in document_list
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
            if (item.get("answer") or {}).get("query_type") == "structured_lookup"
        ]
        processing_durations = [
            duration
            for document in document_list
            if (
                duration := self._document_processing_duration(document)
            )
            is not None
        ]
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
                _first_defined(field.extraction_confidence, field.confidence)
                for field in fields
            ),
            "mean_field_confidence": _average(_field_confidence(field) for field in fields),
            "mean_calibrated_field_confidence": _average(
                field.calibrated_confidence for field in fields
            ),
            "mean_approved_field_confidence": _average(
                _field_confidence(field) for field in approved_fields
            ),
            "average_field_confidence": _average(
                _field_confidence(field) for field in fields
            ),
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
            "review_priority_counts": dict(
                Counter(item.priority.value for item in review_list)
            ),
            "review_outcome_counts": dict(
                Counter(item.status.value for item in review_list)
            ),
            "risk_counts": dict(Counter(item.severity.value for item in risks)),
            "risk_category_counts": dict(Counter(item.category for item in risks)),
            "processing_warning_count": len(warnings),
            "processing_warning_counts": dict(Counter(item.title for item in warnings)),
            "model_usage": dict(Counter(item.model_name for item in model_runs)),
            "model_failure_count": model_failure_count,
            "model_run_count": len(model_runs),
            "model_failure_rate": (
                round(model_failure_count / len(model_runs), 4)
                if model_runs
                else None
            ),
            "processing_success_count": len(document_list) - processing_failure_count,
            "processing_failure_count": processing_failure_count,
            "average_processing_time_seconds": (
                round(mean(processing_durations), 3)
                if processing_durations
                else None
            ),
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
                "processing_duration_samples": len(processing_durations),
                "review_duration_samples": len(review_durations),
                "rag_questions": len(questions),
            },
            "fake_metrics_reported": False,
        }

    def _document_processing_duration(
        self,
        document: VisionDocumentRecord,
    ) -> float | None:
        measured = [
            run.processing_duration_ms
            for run in document.model_runs
            if run.processing_duration_ms is not None
        ]
        if measured:
            return sum(measured) / 1000
        timestamps = [event.created_at for event in document.audit_events]
        return _elapsed_seconds(min(timestamps), max(timestamps)) if timestamps else None

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
