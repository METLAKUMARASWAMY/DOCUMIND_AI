"""Analytics aggregates and audit history, computed from persisted records.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.core.security import utc_now_iso
from app.models.vision import (
    VisionDocumentRecord,
    VisionProcessingStatus,
)
from app.services.vision.analytics import vision_analytics_service


class AnalyticsMixin:
    def analytics_overview(
        self,
        tenant_id: str | None = None,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        documents = self._analytics_documents(tenant_id, date_from, date_to)
        reviews = [item.review_task for item in documents if item.review_task]
        resolved_tenant = tenant_id or "tenant-local"
        return vision_analytics_service.overview(
            documents,
            reviews,
            question_records=[
                item
                for item in self.questions.values()
                if item.get("tenant_id", "tenant-local") == resolved_tenant
                and self._date_in_range(item.get("created_at"), date_from, date_to)
            ],
            narrative_count=sum(
                self._date_in_range(item.created_at, date_from, date_to)
                for item in self.list_finance_narratives(resolved_tenant)
            ),
            report_count=sum(
                self._date_in_range(item.created_at, date_from, date_to)
                for item in self.list_reports(resolved_tenant)
            ),
        )

    def analytics_vendors(
        self,
        tenant_id: str | None = None,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        vendors: dict[str, dict[str, Any]] = {}
        for document in self._analytics_documents(tenant_id, date_from, date_to):
            field_map = self._field_map(document)
            vendor = field_map.get("vendor_name") or field_map.get("party_name") or "unknown"
            key = self._normalize(vendor) or "unknown"
            amount = self._money(field_map.get("total_amount")) or 0.0
            entry = vendors.setdefault(
                key,
                {
                    "vendor_name": vendor,
                    "document_count": 0,
                    "total_amount": 0.0,
                    "review_required": 0,
                    "risk_indicators": 0,
                    "document_ids": [],
                },
            )
            entry["document_count"] += 1
            entry["total_amount"] = round(entry["total_amount"] + amount, 2)
            entry["review_required"] += 1 if document.review_task else 0
            entry["risk_indicators"] += len(document.risk_indicators)
            entry["document_ids"].append(document.document_id)
        return {
            "total": len(vendors),
            "last_updated": utc_now_iso(),
            "date_range": {"from": date_from, "to": date_to},
            "vendors": sorted(vendors.values(), key=lambda item: item["vendor_name"]),
        }

    def analytics_documents(
        self,
        tenant_id: str | None = None,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        documents = self._analytics_documents(tenant_id, date_from, date_to)
        fields = [field for document in documents for field in document.extracted_fields]
        available_confidences = [
            field.confidence for field in fields if field.confidence is not None
        ]
        return {
            "total": len(documents),
            "by_status": dict(Counter(item.status.value for item in documents)),
            "by_document_type": dict(Counter(item.document_type.value for item in documents)),
            "review_required": sum(item.review_task is not None for item in documents),
            "documents_without_fields": sum(not item.extracted_fields for item in documents),
            "extracted_field_count": len(fields),
            "field_confidence_sample_size": len(available_confidences),
            "mean_field_confidence": (
                round(sum(available_confidences) / len(available_confidences), 4)
                if available_confidences
                else None
            ),
            "sample_size": len(documents),
            "last_updated": max((item.updated_at for item in documents), default=None),
            "date_range": {"from": date_from, "to": date_to},
        }

    def analytics_risk(
        self,
        tenant_id: str | None = None,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        documents = self._analytics_documents(tenant_id, date_from, date_to)
        indicators = [risk for document in documents for risk in document.risk_indicators]
        warnings = [warning for document in documents for warning in document.processing_warnings]
        by_severity = Counter(risk.severity.value for risk in indicators)
        by_category = Counter(risk.category for risk in indicators)
        return {
            "total_indicators": len(indicators),
            "by_severity": dict(by_severity),
            "by_category": dict(by_category),
            "processing_warning_count": len(warnings),
            "processing_warnings_by_title": dict(Counter(warning.title for warning in warnings)),
            "confirmed_fraud_count": sum(
                1
                for document in documents
                if document.status == VisionProcessingStatus.confirmed_fraud
            ),
            "automatic_fraud_claims": 0,
            "sample_size": len(documents),
            "last_updated": max((item.updated_at for item in documents), default=None),
            "date_range": {"from": date_from, "to": date_to},
        }

    def analytics_review(
        self,
        tenant_id: str | None = None,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        document_ids = {
            item.document_id for item in self._analytics_documents(tenant_id, date_from, date_to)
        }
        reviews = [
            item for item in self.list_reviews(tenant_id) if item.document_id in document_ids
        ]
        duration_metrics = vision_analytics_service.overview([], reviews)
        return {
            "total": len(reviews),
            "by_status": dict(Counter(review.status.value for review in reviews)),
            "by_priority": dict(Counter(review.priority.value for review in reviews)),
            "pending": sum(
                1
                for review in reviews
                if review.status
                in {
                    VisionProcessingStatus.review_required,
                    VisionProcessingStatus.under_review,
                }
            ),
            "schema_unresolved_total": sum(
                item.unresolved_required_fields or 0 for item in reviews
            ),
            "average_review_time_seconds": duration_metrics["average_review_time_seconds"],
            "review_duration_sample_size": duration_metrics["sample_sizes"][
                "review_duration_samples"
            ],
            "sample_size": len(reviews),
            "last_updated": max((item.updated_at for item in reviews), default=None),
            "date_range": {"from": date_from, "to": date_to},
        }

    def analytics_models(
        self,
        tenant_id: str | None = None,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        documents = self._analytics_documents(tenant_id, date_from, date_to)
        runs = [run for document in documents for run in document.model_runs]
        failed = sum("fail" in run.status.lower() for run in runs)
        return {
            "registry": self.model_registry.registry(),
            "run_count": len(runs),
            "runs_by_stage": dict(Counter(run.stage for run in runs)),
            "runs_by_status": dict(Counter(run.status for run in runs)),
            "failure_count": failed,
            "failure_rate": round(failed / len(runs), 4) if runs else None,
            "sample_size": len(runs),
            "last_updated": max((item.updated_at for item in documents), default=None),
            "date_range": {"from": date_from, "to": date_to},
            "fake_metrics_reported": False,
        }

    def analytics_rag(
        self,
        tenant_id: str | None = None,
        *,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> dict[str, Any]:
        resolved_tenant = tenant_id or "tenant-local"
        questions = [
            item
            for item in self.questions.values()
            if item.get("tenant_id", "tenant-local") == resolved_tenant
            and self._date_in_range(item.get("created_at"), date_from, date_to)
        ]
        supported = [
            item
            for item in questions
            if (item.get("answer") or {}).get("answer_status")
            in {"supported", "partially_supported"}
        ]
        cited = [item for item in supported if (item.get("answer") or {}).get("citations")]
        return {
            "total_questions": len(questions),
            "by_answer_status": dict(
                Counter(
                    str((item.get("answer") or {}).get("answer_status", "unknown"))
                    for item in questions
                )
            ),
            "by_query_type": dict(
                Counter(
                    str((item.get("answer") or {}).get("query_type", "unknown"))
                    for item in questions
                )
            ),
            "by_retriever": dict(
                Counter(
                    str((item.get("answer") or {}).get("retriever", "unknown"))
                    for item in questions
                )
            ),
            "supported_answer_count": len(supported),
            "supported_answers_with_citations": len(cited),
            "citation_coverage": (round(len(cited) / len(supported), 4) if supported else None),
            "citation_coverage_denominator": len(supported),
            "feedback_count": sum(len(item.get("feedback") or []) for item in questions),
            "sample_size": len(questions),
            "last_updated": max(
                (str(item.get("created_at")) for item in questions),
                default=None,
            ),
            "date_range": {"from": date_from, "to": date_to},
        }

    def _analytics_documents(
        self,
        tenant_id: str | None,
        date_from: str | None,
        date_to: str | None,
    ) -> list[VisionDocumentRecord]:
        return [
            document
            for document in self.list_documents(tenant_id)
            if self._date_in_range(document.created_at, date_from, date_to)
        ]

    def _date_in_range(
        self,
        timestamp: str | None,
        date_from: str | None,
        date_to: str | None,
    ) -> bool:
        if not timestamp:
            return False
        day = timestamp[:10]
        return (not date_from or day >= date_from) and (not date_to or day <= date_to)

    def audit_history(self, tenant_id: str | None = None) -> dict[str, Any]:
        events = [
            event.model_dump(mode="json")
            for document in self.list_documents(tenant_id)
            for event in document.audit_events
        ]
        events.extend(
            event
            for event in self.system_audit_events
            if tenant_id is None or event.get("tenant_id") == tenant_id
        )
        events.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return {"total": len(events), "events": events}
