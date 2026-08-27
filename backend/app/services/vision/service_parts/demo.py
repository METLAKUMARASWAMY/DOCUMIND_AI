"""Seeding of the connected demonstration corpus.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

from typing import Any

from app.models.schemas import TextAnalysisRequest
from app.models.vision import (
    ReviewerStatus,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionFinanceNarrative,
    VisionFinanceNarrativeRequest,
    VisionFinanceNarrativeType,
    VisionProcessingStatus,
    VisionReportRecord,
    VisionReportRequest,
    VisionReportType,
)


class DemoSeedMixin:
    def seed_connected_demo(self, *, tenant_id: str, actor: str) -> dict[str, Any]:
        from app.services.vision.demo_data import CONNECTED_DEMO_DOCUMENTS

        documents: list[VisionDocumentRecord] = []
        for item in CONNECTED_DEMO_DOCUMENTS:
            document = self.process_text(
                TextAnalysisRequest(
                    filename=str(item["filename"]),
                    text=str(item["text"]),
                    metadata=dict(item["metadata"]),
                ),
                actor=actor,
                idempotency_key=f"connected-vision-demo:{item['key']}",
                tenant_id=tenant_id,
            )
            self._approve_demo_document(document, actor=actor)
            documents.append(document)

        primary = next(
            (
                document
                for document in documents
                if document.document_type == VisionDocumentType.invoice
            ),
            documents[0],
        )
        narrative = self._demo_narrative(primary, tenant_id=tenant_id, actor=actor)
        report = self._demo_report(primary, tenant_id=tenant_id, actor=actor)
        ocr_status = self.local_ocr.status()
        return {
            "message": "Connected demo case is ready.",
            "document_ids": [document.document_id for document in documents],
            "primary_document_id": primary.document_id,
            "documents": [
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "status": document.status.value,
                    "document_type": document.document_type.value,
                    "fields": len(document.extracted_fields),
                    "chunks": len(document.evidence_chunks),
                    "graph_nodes": len(document.graph.nodes),
                    "graph_edges": len(document.graph.edges),
                }
                for document in documents
            ],
            "narrative_id": narrative.narrative_id if narrative else None,
            "report_id": report.report_id if report else None,
            "ask_examples": [
                "What is the total amount?",
                "Which bank account is linked to Balu Enterprises?",
                "Compare invoice totals by vendor",
            ],
            "local_ocr_status": {
                "status": ocr_status.status,
                "reason": ocr_status.reason,
            },
        }

    def _approve_demo_document(self, document: VisionDocumentRecord, *, actor: str) -> None:
        if document.status == VisionProcessingStatus.approved and document.review_task:
            document.evidence_chunks = self._chunks(document)
            document.graph = self._graph(document)
            self._index_document(document)
            self._persist_document(document)
            self._persist_review(document.review_task)
            return

        for field in document.extracted_fields:
            field.reviewer_status = ReviewerStatus.accepted
            field.evidence.reviewer_status = ReviewerStatus.accepted
        for detection in document.visual_detections:
            detection.reviewer_status = ReviewerStatus.accepted
        for risk in document.risk_indicators:
            risk.reviewer_status = ReviewerStatus.accepted
            for evidence in risk.evidence:
                evidence.reviewer_status = ReviewerStatus.accepted

        if document.review_task is None:
            document.review_task = self._create_review_task(document)
            self.review_tasks[document.review_task.review_id] = document.review_task

        event = self._audit(
            document,
            "demo_document_approved",
            "Connected local demo document approved with reviewer-verified fields.",
            actor=actor,
            metadata={"demo_seed": "connected-vision-demo"},
        )
        document.status = VisionProcessingStatus.approved
        document.updated_at = event.created_at
        document.review_task.status = VisionProcessingStatus.approved
        document.review_task.reviewer_comments = (
            "Auto-approved for the connected local demo dataset."
        )
        document.review_task.updated_at = event.created_at
        document.review_task.history.append(event)
        self._refresh_verified_facts(document, reviewer=actor)
        document.evidence_chunks = self._chunks(document)
        document.graph = self._graph(document)
        self._index_document(document)
        self.persistence.upsert_graph(document.tenant_id, document.document_id, document.graph)
        self._persist_document(document)
        self._persist_review(document.review_task)

    def _demo_narrative(
        self,
        primary: VisionDocumentRecord,
        *,
        tenant_id: str,
        actor: str,
    ) -> VisionFinanceNarrative | None:
        for narrative in self.list_finance_narratives(tenant_id):
            if (
                narrative.narrative_type == VisionFinanceNarrativeType.monthly_invoice_summary
                and narrative.document_ids == [primary.document_id]
            ):
                return narrative
        try:
            return self.generate_finance_narrative(
                VisionFinanceNarrativeRequest(
                    narrative_type=VisionFinanceNarrativeType.monthly_invoice_summary,
                    vendor_name="Balu Enterprises",
                    document_ids=[primary.document_id],
                ),
                tenant_id=tenant_id,
                actor=actor,
            )
        except ValueError:
            return None

    def _demo_report(
        self,
        primary: VisionDocumentRecord,
        *,
        tenant_id: str,
        actor: str,
    ) -> VisionReportRecord | None:
        for report in self.list_reports(tenant_id):
            if (
                report.report_type == VisionReportType.document_intelligence
                and report.document_ids == [primary.document_id]
            ):
                return report
        try:
            return self.generate_report(
                VisionReportRequest(
                    report_type=VisionReportType.document_intelligence,
                    document_id=primary.document_id,
                ),
                tenant_id=tenant_id,
                actor=actor,
            )
        except ValueError:
            return None
