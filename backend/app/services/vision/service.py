from __future__ import annotations

import json
import re
import threading
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import settings
from app.core.security import hash_document, utc_now_iso
from app.models.schemas import TextAnalysisRequest, UserRole
from app.models.vision import (
    DuplicateCandidate,
    ExtractedVisionField,
    ModelRunRecord,
    OCRWord,
    ReviewerStatus,
    VisionAskCitation,
    VisionAskRequest,
    VisionAskResponse,
    VisionAuditEvent,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionEvidence,
    VisionEvidenceChunk,
    VisionFinanceNarrative,
    VisionFinanceNarrativeRequest,
    VisionFinanceNarrativeType,
    VisionGraphProjection,
    VisionProcessingStatus,
    VisionReportRecord,
    VisionReportRequest,
    VisionReportType,
    VisionReviewDecisionRequest,
    VisionReviewTask,
    VisionRiskIndicator,
    VisionRiskSeverity,
    VisionUploadResponse,
)
from app.services.classification import DocumentClassificationService
from app.services.ingestion import DocumentIngestionService
from app.services.model_registry import ModelRegistryService
from app.services.vision.advanced_adapters import PaddleOCRVLAdapter
from app.services.vision.analytics import vision_analytics_service
from app.services.vision.anomaly_adapters import IsolationForestRiskAdapter, XGBoostRiskAdapter
from app.services.vision.approval_policy import (
    ApprovalPolicyEvaluation,
    ApprovalPolicyViolation,
    ReviewDecisionValidationError,
    vision_approval_policy_service,
)
from app.services.vision.contract_adapters import DebertaContractClauseAdapter
from app.services.vision.deterministic_extraction import deterministic_schema_extractor
from app.services.vision.duplicate_detection import (
    DinoV2DuplicateAdapter,
    cosine_similarity,
    document_dhash,
    perceptual_similarity,
)
from app.services.vision.extraction_adapters import (
    DonutExtractionAdapter,
    LayoutLMv3ClassificationAdapter,
    LayoutLMv3ExtractionAdapter,
)
from app.services.vision.finance_narratives import verified_finance_narrative_service
from app.services.vision.generator_adapters import GroundedGeneratorRouter
from app.services.vision.graph_builder import evidence_graph_builder
from app.services.vision.language_analysis import vision_language_analysis_service
from app.services.vision.layout_adapters import GeometryLayoutAdapter, PaddleStructureV3Adapter
from app.services.vision.model_smoke import ModelSmokeService
from app.services.vision.ocr_adapters import PaddleOCRV5Adapter, TesseractOCRAdapter
from app.services.vision.persistence import vision_persistence
from app.services.vision.preprocessing import DocumentPreprocessor
from app.services.vision.query_router import VisionQueryRouter
from app.services.vision.reporting import vision_report_service
from app.services.vision.retrieval_adapters import LocalHybridRetrievalIndex
from app.services.vision.schema_registry import vision_schema_registry
from app.services.vision.tamper_adapters import LocalForensicsAdapter, SegFormerTamperAdapter
from app.services.vision.visual_adapters import (
    OpenCVVisualDetectionAdapter,
    Yolo26VisualDetectionAdapter,
)

CONTRACT_CLAUSE_FIELDS = {
    "parties",
    "agreement_date",
    "effective_date",
    "expiry_date",
    "governing_law",
    "renewal_term",
    "termination_for_convenience",
    "notice_period",
    "anti_assignment",
    "exclusivity",
    "non_compete",
    "audit_rights",
    "insurance",
    "cap_on_liability",
    "uncapped_liability",
    "perpetual_licence",
    "revenue_sharing",
    "change_of_control",
    "confidentiality",
    "indemnification",
    "payment_terms",
}


class VisionAIService:
    def __init__(self) -> None:
        self.ingestion = DocumentIngestionService()
        self.classifier = DocumentClassificationService()
        self.model_registry = ModelRegistryService()
        self.model_smoke = ModelSmokeService(registry=self.model_registry)
        self.preprocessor = DocumentPreprocessor()
        self.local_ocr = TesseractOCRAdapter()
        self.paddle_ocr = PaddleOCRV5Adapter()
        self.local_layout = GeometryLayoutAdapter()
        self.paddle_structure = PaddleStructureV3Adapter()
        self.local_visual_detector = OpenCVVisualDetectionAdapter()
        self.yolo_detector = Yolo26VisualDetectionAdapter()
        self.layoutlmv3_classifier = LayoutLMv3ClassificationAdapter()
        self.layoutlmv3_extractor = LayoutLMv3ExtractionAdapter()
        self.donut_extractor = DonutExtractionAdapter()
        self.dinov2_duplicate = DinoV2DuplicateAdapter()
        self.paddleocr_vl = PaddleOCRVLAdapter()
        self.local_forensics = LocalForensicsAdapter()
        self.segformer_forensics = SegFormerTamperAdapter()
        self.isolation_forest = IsolationForestRiskAdapter()
        self.xgboost_risk = XGBoostRiskAdapter()
        self.contract_extractor = DebertaContractClauseAdapter()
        self.retrieval_index = LocalHybridRetrievalIndex()
        self.query_router = VisionQueryRouter()
        self.generator_router = GroundedGeneratorRouter()
        self.persistence = vision_persistence
        self.documents: dict[str, VisionDocumentRecord] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.review_tasks: dict[str, VisionReviewTask] = {}
        self._pending_content: dict[str, tuple[str, bytes, dict[str, Any]]] = {}
        self._idempotency_index: dict[str, str] = {}
        self._hash_index: dict[str, str] = {}
        self._semantic_index: dict[str, str] = {}
        self.questions: dict[str, dict[str, Any]] = {}
        self.question_feedback: dict[str, list[dict[str, Any]]] = {}
        self.finance_narratives: dict[str, VisionFinanceNarrative] = {}
        self.reports: dict[str, VisionReportRecord] = {}
        self.model_overrides: dict[str, dict[str, Any]] = {}
        self.rule_overrides: dict[str, dict[str, Any]] = {}
        self.system_audit_events: list[dict[str, Any]] = []
        self._artifact_lock = threading.RLock()
        self._state_root = Path(settings.upload_dir).parent / "vision_state"
        self.confidence_policy = self._load_confidence_policy()
        self._ensure_state_dirs()
        self._load_admin_overrides()
        self._load_indexes()

    def queue_documents(
        self,
        files: list[tuple[str, bytes, dict[str, Any]]],
        *,
        actor: str = "system",
        process_inline: bool = False,
        idempotency_key: str | None = None,
        tenant_id: str = "tenant-local",
    ) -> VisionUploadResponse:
        if not files:
            raise ValueError("At least one document is required")
        self._load_indexes()
        scoped_idempotency_key = f"{tenant_id}:{idempotency_key}" if idempotency_key else None
        if scoped_idempotency_key:
            existing_job_id = self._idempotency_index.get(scoped_idempotency_key)
            if existing_job_id:
                self.get_job(existing_job_id)
            if existing_job_id and existing_job_id in self.jobs:
                return self._upload_response_from_job(existing_job_id)
        job_id = f"vision-job-{uuid4().hex[:10]}"
        case_id = f"vision-case-{uuid4().hex[:10]}"
        document_ids: list[str] = []
        for filename, content, metadata in files:
            self.ingestion.validate_file(
                filename,
                content,
                declared_mime=metadata.get("declared_mime"),
            )
            now = utc_now_iso()
            document = VisionDocumentRecord(
                case_id=case_id,
                filename=Path(filename).name or "uploaded-document",
                tenant_id=tenant_id,
                status=VisionProcessingStatus.queued,
                file_hash=hash_document(content),
                created_at=now,
                updated_at=now,
                metadata=metadata,
            )
            self._audit(
                document,
                "document_queued",
                "Document queued for Vision AI processing.",
                actor=actor,
            )
            self.documents[document.document_id] = document
            self._pending_content[document.document_id] = (filename, content, metadata)
            source_uri = self.persistence.store_original(
                document,
                filename,
                content,
                metadata.get("declared_mime"),
            )
            if source_uri:
                document.metadata["source_object_uri"] = source_uri
            self._persist_document(document)
            self._persist_pending_content(document.document_id, filename, content, metadata)
            document_ids.append(document.document_id)
        self.jobs[job_id] = {
            "job_id": job_id,
            "case_id": case_id,
            "document_ids": document_ids,
            "status": VisionProcessingStatus.queued.value,
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "errors": [],
            "idempotency_key": idempotency_key,
            "tenant_id": tenant_id,
        }
        if scoped_idempotency_key:
            self._idempotency_index[scoped_idempotency_key] = job_id
            self._persist_indexes()
        self._persist_job(self.jobs[job_id])
        if process_inline:
            self.process_job(job_id)
        return self._upload_response_from_job(job_id)

    def process_text(
        self,
        request: TextAnalysisRequest,
        *,
        actor: str = "system",
        idempotency_key: str | None = None,
        tenant_id: str = "tenant-local",
    ) -> VisionDocumentRecord:
        response = self.queue_documents(
            [(request.filename, request.text.encode("utf-8"), request.metadata)],
            actor=actor,
            process_inline=True,
            idempotency_key=idempotency_key,
            tenant_id=tenant_id,
        )
        return self.get_document(response.document_ids[0])

    def queue_reprocess(
        self,
        document_id: str,
        content: bytes,
        *,
        actor: str,
        tenant_id: str,
    ) -> VisionUploadResponse:
        document = self.get_document(document_id, tenant_id)
        metadata = {
            **document.metadata,
            "reprocessed_from": document.document_id,
            "reprocess_count": int(document.metadata.get("reprocess_count", 0)) + 1,
        }
        if document.review_task:
            metadata["_reprocess_review_id"] = document.review_task.review_id
        self._reset_for_reprocess(document, metadata, actor)

        job_id = f"vision-job-{uuid4().hex[:10]}"
        now = utc_now_iso()
        self.jobs[job_id] = {
            "job_id": job_id,
            "case_id": document.case_id,
            "document_ids": [document.document_id],
            "status": VisionProcessingStatus.queued.value,
            "created_at": now,
            "updated_at": now,
            "errors": [],
            "idempotency_key": None,
            "tenant_id": tenant_id,
            "reprocess": True,
        }
        self._pending_content[document.document_id] = (
            document.filename,
            content,
            metadata,
        )
        self._persist_pending_content(
            document.document_id,
            document.filename,
            content,
            metadata,
        )
        self._persist_document(document)
        self._persist_job(self.jobs[job_id])
        return self._upload_response_from_job(job_id)

    def process_job(self, job_id: str, tenant_id: str | None = None) -> None:
        job = self.get_job(job_id, tenant_id)
        if not job:
            return
        for document_id in job["document_ids"]:
            if document_id not in self._pending_content:
                self._load_pending_content(document_id)
            if document_id not in self.documents:
                self._load_document(document_id)
        if not any(document_id in self._pending_content for document_id in job["document_ids"]):
            return
        job["status"] = VisionProcessingStatus.preprocessing.value
        job["updated_at"] = utc_now_iso()
        self._persist_job(job)
        for document_id in job["document_ids"]:
            pending = self._pending_content.get(document_id)
            document = self.documents.get(document_id)
            if not pending or not document:
                continue
            filename, content, metadata = pending
            success = False
            try:
                self._process_document(document, filename, content, metadata)
                success = True
            except Exception as exc:  # pragma: no cover - defensive safety net
                document.status = VisionProcessingStatus.review_required
                document.updated_at = utc_now_iso()
                warning = self._risk(
                    document,
                    title="Processing Failure",
                    severity=VisionRiskSeverity.high,
                    category="processing",
                    description="The processing pipeline failed before a complete result was available.",
                    recommended_action="Retry processing or assign a reviewer.",
                    source_text=str(exc),
                )
                document.processing_warnings.append(warning)
                self._audit(
                    document,
                    "processing_failed",
                    "Vision AI processing failed and was routed to review.",
                    metadata={"error": str(exc)},
                )
                if document.review_task is None:
                    document.review_task = self._create_review_task(document)
                    self.review_tasks[document.review_task.review_id] = document.review_task
                job["errors"].append({"document_id": document_id, "error": str(exc)})
            finally:
                if success:
                    self._pending_content.pop(document_id, None)
                    self._remove_pending_content(document_id)
                if document:
                    self._persist_document(document)
                    if document.review_task:
                        self._persist_review(document.review_task)
        job["status"] = VisionProcessingStatus.completed.value if not job["errors"] else VisionProcessingStatus.review_required.value
        job["updated_at"] = utc_now_iso()
        self._persist_job(job)
        self._persist_indexes()

    def _upload_response_from_job(self, job_id: str) -> VisionUploadResponse:
        job = self.get_job(job_id) or self.jobs[job_id]
        return VisionUploadResponse(
            case_id=job["case_id"],
            job_id=job_id,
            document_ids=list(job["document_ids"]),
            status=VisionProcessingStatus(job["status"]),
            status_url=f"/api/v1/vision/jobs/{job_id}",
        )

    def get_job(self, job_id: str, tenant_id: str | None = None) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        if not job:
            job = self._load_job(job_id)
        if not job and tenant_id:
            job = self.persistence.get_job(job_id, tenant_id)
            if job:
                self.jobs[job_id] = job
        if job and tenant_id and job.get("tenant_id", "tenant-local") != tenant_id:
            return None
        return job

    def retry_job(self, job_id: str, tenant_id: str | None = None) -> dict[str, Any]:
        job = self.get_job(job_id, tenant_id)
        if not job:
            raise ValueError("job_id not found")
        job["errors"] = []
        job["status"] = VisionProcessingStatus.queued.value
        job["updated_at"] = utc_now_iso()
        self._persist_job(job)
        return job

    def record_job_dispatch(
        self,
        job_id: str,
        dispatch: dict[str, Any],
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        job = self.get_job(job_id, tenant_id)
        if not job:
            return None
        job["dispatch"] = dispatch
        job["updated_at"] = utc_now_iso()
        self._persist_job(job)
        return job

    def delete_document(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> VisionDocumentRecord:
        document = self.get_document(document_id, tenant_id)
        self.documents.pop(document_id, None)
        self._remove_document(document_id)
        self._pending_content.pop(document_id, None)
        self._remove_pending_content(document_id)
        if document.review_task:
            self.review_tasks.pop(document.review_task.review_id, None)
            self._remove_review(document.review_task.review_id)
        self._hash_index = {
            key: value for key, value in self._hash_index.items() if value != document_id
        }
        self._semantic_index = {
            key: value for key, value in self._semantic_index.items() if value != document_id
        }
        self._persist_indexes()
        self.persistence.delete_document(document_id, document.tenant_id)
        return document

    def get_document(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> VisionDocumentRecord:
        document = self.documents.get(document_id) or self._load_document(document_id)
        if not document and tenant_id:
            document = self.persistence.get_document(document_id, tenant_id)
            if document:
                self.documents[document.document_id] = document
                self._index_document(document)
        if not document or (tenant_id and document.tenant_id != tenant_id):
            raise ValueError("document_id not found")
        vision_analytics_service.refresh_document(document)
        self._ensure_review_task(document)
        vision_analytics_service.refresh_document(document)
        return document

    def list_documents(self, tenant_id: str | None = None) -> list[VisionDocumentRecord]:
        self._load_all_documents()
        if tenant_id:
            for document in self.persistence.list_documents(tenant_id):
                self.documents[document.document_id] = document
                self._index_document(document)
        documents = self.documents.values()
        if tenant_id:
            documents = [item for item in documents if item.tenant_id == tenant_id]
        for document in list(documents):
            vision_analytics_service.refresh_document(document)
            self._ensure_review_task(document)
            vision_analytics_service.refresh_document(document)
        return sorted(documents, key=lambda item: item.created_at, reverse=True)

    def list_reviews(self, tenant_id: str | None = None) -> list[VisionReviewTask]:
        self._load_all_reviews()
        if tenant_id:
            for review in self.persistence.list_reviews(tenant_id):
                self.review_tasks[review.review_id] = review
        reviews = self.review_tasks.values()
        if tenant_id:
            reviews = [
                item
                for item in reviews
                if self._review_belongs_to_tenant(item, tenant_id)
            ]
        return sorted(reviews, key=lambda item: item.created_at, reverse=True)

    def review_queue(
        self,
        tenant_id: str,
        *,
        search: str = "",
        document_type: str = "all",
        priority: str = "all",
        reason: str = "all",
        assignee: str = "all",
        status: str = "open",
        sort_by: str = "created_at",
        sort_direction: str = "desc",
        page: int = 1,
        page_size: int = 25,
    ) -> dict[str, Any]:
        all_reviews = self.list_reviews(tenant_id)
        needle = self._normalize(search)
        rows = []
        for review in all_reviews:
            document = self.get_document(review.document_id, tenant_id)
            searchable = self._normalize(
                " ".join(
                    [
                        review.review_id,
                        review.document_id,
                        review.filename or document.filename,
                        document.document_type.value,
                        *review.reasons,
                        *review.reason_labels,
                    ]
                )
            )
            if needle and needle not in searchable:
                continue
            if document_type != "all" and document.document_type.value != document_type:
                continue
            if priority != "all" and review.priority.value != priority:
                continue
            if reason != "all" and reason not in review.reasons:
                continue
            if assignee == "unassigned" and review.assigned_reviewer:
                continue
            if assignee not in {"all", "unassigned"} and review.assigned_reviewer != assignee:
                continue
            if status == "open" and review.status not in {
                VisionProcessingStatus.review_required,
                VisionProcessingStatus.under_review,
            }:
                continue
            if status not in {"all", "open"} and review.status.value != status:
                continue
            rows.append(review)

        priority_order = {
            VisionRiskSeverity.low: 1,
            VisionRiskSeverity.medium: 2,
            VisionRiskSeverity.high: 3,
            VisionRiskSeverity.critical: 4,
        }
        sorters = {
            "created_at": lambda item: item.created_at,
            "priority": lambda item: priority_order[item.priority],
            "filename": lambda item: (item.filename or "").lower(),
            "status": lambda item: item.status.value,
            "confidence": lambda item: item.confidence_value or -1.0,
        }
        rows.sort(
            key=sorters.get(sort_by, sorters["created_at"]),
            reverse=sort_direction != "asc",
        )
        total = len(rows)
        safe_page_size = min(max(page_size, 1), 100)
        safe_page = max(page, 1)
        start = (safe_page - 1) * safe_page_size
        paged = rows[start : start + safe_page_size]
        return {
            "total": total,
            "unfiltered_total": len(all_reviews),
            "page": safe_page,
            "page_size": safe_page_size,
            "total_pages": (total + safe_page_size - 1) // safe_page_size,
            "reviews": [item.model_dump(mode="json") for item in paged],
            "facets": {
                "document_types": sorted(
                    {
                        self.get_document(item.document_id, tenant_id).document_type.value
                        for item in all_reviews
                    }
                ),
                "priorities": sorted({item.priority.value for item in all_reviews}),
                "reasons": [
                    {"value": machine, "label": label}
                    for machine, label in sorted(
                        {
                            (machine, label)
                            for item in all_reviews
                            for machine, label in zip(
                                item.reasons,
                                item.reason_labels,
                                strict=False,
                            )
                        }
                    )
                ],
                "assignees": sorted(
                    {item.assigned_reviewer for item in all_reviews if item.assigned_reviewer}
                ),
                "statuses": sorted({item.status.value for item in all_reviews}),
            },
        }

    def get_review(self, review_id: str, tenant_id: str | None = None) -> VisionReviewTask:
        review = self.review_tasks.get(review_id) or self._load_review(review_id)
        if not review and tenant_id:
            review = self.persistence.get_review(review_id, tenant_id)
            if review:
                self.review_tasks[review.review_id] = review
        if not review or (tenant_id and not self._review_belongs_to_tenant(review, tenant_id)):
            raise ValueError("review_id not found")
        return review

    def patch_field(
        self,
        review_id: str,
        field_id: str,
        *,
        reviewed_value: str | None,
        reviewer_status: ReviewerStatus,
        comment: str,
        actor: str,
        tenant_id: str | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> ExtractedVisionField:
        review = self.get_review(review_id, tenant_id)
        document = self.get_document(review.document_id, tenant_id)
        field = next((item for item in document.extracted_fields if item.field_id == field_id), None)
        if not field:
            raise ValueError("field_id not found")
        old_value = field.reviewed_value or field.value
        field.reviewed_value = reviewed_value if reviewed_value is not None else field.value
        field.reviewer_status = reviewer_status
        field.evidence.reviewer_status = reviewer_status
        field.verification_status = reviewer_status.value
        event = self._audit(
            document,
            "field_reviewed",
            "Reviewer updated an extracted field.",
            actor=actor,
            metadata={
                "field_id": field_id,
                "field_name": field.name,
                "old_value": self._mask_sensitive(field.name, old_value),
                "new_value": self._mask_sensitive(field.name, field.reviewed_value or ""),
                "comment": comment,
                **(request_metadata or {}),
            },
        )
        review.history.append(event)
        review.updated_at = event.created_at
        document.data_version += 1
        document.analytics_cache_version += 1
        document.evidence_chunks = self._chunks(document)
        document.graph = self._graph(document)
        vision_analytics_service.refresh_document(document)
        self._index_document(document)
        self._persist_document(document)
        self._persist_review(review)
        return field

    def record_review_decision(
        self,
        review_id: str,
        *,
        decision: VisionProcessingStatus,
        request: VisionReviewDecisionRequest | None,
        actor: str,
        actor_role: UserRole,
        tenant_id: str | None = None,
        request_metadata: dict[str, Any] | None = None,
        reauthentication_payload: dict[str, Any] | None = None,
    ) -> VisionReviewTask:
        allowed = {
            VisionProcessingStatus.approved,
            VisionProcessingStatus.rejected,
            VisionProcessingStatus.confirmed_error,
            VisionProcessingStatus.confirmed_fraud,
            VisionProcessingStatus.under_review,
            VisionProcessingStatus.review_required,
        }
        if decision not in allowed:
            raise ValueError("Unsupported review decision")
        decision_request = request or VisionReviewDecisionRequest(decision=decision)
        if decision_request.decision != decision:
            raise ValueError("Request decision does not match the action endpoint")
        review = self.get_review(review_id, tenant_id)
        document = self.get_document(review.document_id, tenant_id)
        policy_evaluation: ApprovalPolicyEvaluation | None = None
        if decision == VisionProcessingStatus.approved:
            policy_evaluation = vision_approval_policy_service.evaluate(
                document,
                request=decision_request,
                actor_role=actor_role,
                confidence_threshold=self._mandatory_review_threshold(document),
            )
            if not policy_evaluation.allowed:
                raise ApprovalPolicyViolation(policy_evaluation)
            if policy_evaluation.mode == "manual_override":
                self._apply_manual_confirmations(
                    document,
                    decision_request.confirmed_fields,
                    actor=actor,
                )
        if decision == VisionProcessingStatus.confirmed_fraud:
            self._validate_fraud_confirmation(
                document,
                decision_request,
                actor=actor,
                actor_role=actor_role,
                reauthentication_payload=reauthentication_payload,
            )
        comment = decision_request.comment.strip()
        review.status = decision
        review.reviewer_comments = comment
        review.updated_at = utc_now_iso()
        document.status = decision
        document.updated_at = review.updated_at
        if decision == VisionProcessingStatus.approved:
            for field in document.extracted_fields:
                if field.reviewer_status == ReviewerStatus.unverified:
                    field.reviewer_status = ReviewerStatus.accepted
                    field.evidence.reviewer_status = ReviewerStatus.accepted
                    field.verification_status = ReviewerStatus.accepted.value
            for detection in document.visual_detections:
                if detection.reviewer_status == ReviewerStatus.unverified:
                    detection.reviewer_status = ReviewerStatus.accepted
        document.evidence_chunks = self._chunks(document)
        document.graph = self._graph(document)
        document.data_version += 1
        document.analytics_cache_version += 1
        vision_analytics_service.refresh_document(document)
        self._index_document(document)
        event_type = "review_decision_recorded"
        if decision == VisionProcessingStatus.confirmed_fraud:
            event_type = "confirmed_fraud_by_reviewer"
        decision_metadata = {
            "comment": comment,
            "approval_reason": decision_request.approval_reason.strip(),
            "approval_mode": policy_evaluation.mode if policy_evaluation else None,
            "confirmed_field_names": sorted(decision_request.confirmed_fields),
            "evidence_ids": sorted(set(decision_request.evidence_ids)),
            "decision_rationale": decision_request.decision_rationale.strip(),
            "automated_indicators_acknowledged": (
                decision_request.automated_indicators_acknowledged
            ),
            "reauthenticated_at": (
                reauthentication_payload.get("auth_time")
                if reauthentication_payload
                else None
            ),
            "overridden_policy_conditions": (
                [item.code for item in policy_evaluation.observed_conditions]
                if policy_evaluation and policy_evaluation.mode == "manual_override"
                else []
            ),
            **(request_metadata or {}),
        }
        event = self._audit(
            document,
            event_type,
            f"Reviewer set document status to {decision.value}.",
            actor=actor,
            metadata=decision_metadata,
        )
        review.history.append(event)
        if decision == VisionProcessingStatus.approved:
            self.persistence.upsert_graph(
                document.tenant_id,
                document.document_id,
                document.graph,
            )
        self._persist_document(document)
        self._persist_review(review)
        return review

    def approval_policy_for_review(
        self,
        review_id: str,
        *,
        actor_role: UserRole,
        tenant_id: str,
    ) -> ApprovalPolicyEvaluation:
        review = self.get_review(review_id, tenant_id)
        document = self.get_document(review.document_id, tenant_id)
        return vision_approval_policy_service.evaluate(
            document,
            request=None,
            actor_role=actor_role,
            confidence_threshold=self._mandatory_review_threshold(document),
        )

    def _apply_manual_confirmations(
        self,
        document: VisionDocumentRecord,
        confirmed_fields: dict[str, str],
        *,
        actor: str,
    ) -> None:
        schema = vision_schema_registry.get(document.document_type)
        schema_names = {item.name for item in schema.fields} if schema else set()
        by_name = {field.name: field for field in document.extracted_fields}
        confirmed_names: list[str] = []
        for raw_name, raw_value in confirmed_fields.items():
            name = raw_name.strip().lower()
            value = raw_value.strip()
            if not value or (schema_names and name not in schema_names):
                continue
            field = by_name.get(name)
            if field:
                field.reviewed_value = value
                field.reviewer_status = ReviewerStatus.accepted
                field.evidence.reviewer_status = ReviewerStatus.accepted
                field.verification_status = ReviewerStatus.accepted.value
            else:
                spec = schema.field(name) if schema else None
                identity = hash_document(
                    f"{document.document_id}|manual-review|{name}".encode()
                )
                evidence = VisionEvidence(
                    evidence_id=f"vis-ev-manual-{identity[:10]}",
                    document_id=document.document_id,
                    page_number=1,
                    source_text=f"Reviewer-entered field: {name.replace('_', ' ')}",
                    model_confidence=0.0,
                    model_name="human-manual-verification",
                    model_version="v1",
                    processing_timestamp=utc_now_iso(),
                    reviewer_status=ReviewerStatus.accepted,
                    metadata={
                        "confidence_available": False,
                        "source": "manual_approval_confirmation",
                        "actor": actor,
                    },
                )
                field = ExtractedVisionField(
                    field_id=f"field-manual-{identity[:10]}",
                    name=name,
                    schema_field=name,
                    display_name=(
                        spec.display_name if spec else name.replace("_", " ").title()
                    ),
                    value=value,
                    normalized_value=self._normalize(value),
                    confidence=None,
                    evidence=evidence,
                    model_name="human-manual-verification",
                    model_version="v1",
                    extractor="manual-review",
                    verification_status=ReviewerStatus.accepted.value,
                    required=bool(spec and spec.required),
                    reviewer_status=ReviewerStatus.accepted,
                    reviewed_value=value,
                )
                document.extracted_fields.append(field)
                by_name[name] = field
            confirmed_names.append(name)
        vision_analytics_service.refresh_document(document)
        self._audit(
            document,
            "manual_approval_fields_confirmed",
            "Reviewer entered or confirmed required fields for manual approval.",
            actor=actor,
            metadata={"field_names": sorted(set(confirmed_names))},
        )

    def _validate_fraud_confirmation(
        self,
        document: VisionDocumentRecord,
        request: VisionReviewDecisionRequest,
        *,
        actor: str,
        actor_role: UserRole,
        reauthentication_payload: dict[str, Any] | None,
    ) -> None:
        if actor_role != UserRole.admin:
            raise PermissionError("Admin role is required to confirm fraud")
        if (
            not reauthentication_payload
            or reauthentication_payload.get("email") != actor
            or reauthentication_payload.get("purpose") != "sensitive_review_decision"
        ):
            raise PermissionError("Recent reauthentication is required to confirm fraud")
        if len(request.comment.strip()) < 10:
            raise ReviewDecisionValidationError(
                "A reviewer comment of at least 10 characters is required"
            )
        if len(request.decision_rationale.strip()) < 20:
            raise ReviewDecisionValidationError(
                "A written fraud decision rationale of at least 20 characters is required"
            )
        if not request.automated_indicators_acknowledged:
            raise ReviewDecisionValidationError(
                "Acknowledge that automated indicators alone do not prove fraud"
            )
        if not request.confirmation_acknowledged:
            raise ReviewDecisionValidationError(
                "The fraud confirmation modal acknowledgement is required"
            )
        available_evidence = self._document_evidence_ids(document)
        selected_evidence = set(request.evidence_ids)
        if not selected_evidence:
            raise ReviewDecisionValidationError(
                "At least one supporting evidence item must be selected"
            )
        unavailable = sorted(selected_evidence - available_evidence)
        if unavailable:
            raise ReviewDecisionValidationError(
                f"Unknown evidence IDs: {', '.join(unavailable)}"
            )
        if settings.fraud_second_reviewer_required:
            second_reviewer = (request.second_reviewer_email or "").strip().lower()
            if (
                not request.second_reviewer_approved
                or not second_reviewer
                or second_reviewer == actor.lower()
            ):
                raise ReviewDecisionValidationError(
                    "A distinct second reviewer approval is required"
                )

    def _document_evidence_ids(self, document: VisionDocumentRecord) -> set[str]:
        return {
            evidence.evidence_id
            for evidence in [
                *(field.evidence for field in document.extracted_fields),
                *(
                    evidence
                    for risk in [
                        *document.risk_indicators,
                        *document.processing_warnings,
                    ]
                    for evidence in risk.evidence
                ),
                *(
                    evidence
                    for duplicate in document.duplicate_candidates
                    for evidence in duplicate.evidence
                ),
            ]
        }

    def accept_high_confidence_fields(
        self,
        review_id: str,
        *,
        actor: str,
        tenant_id: str,
        request_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        review = self.get_review(review_id, tenant_id)
        document = self.get_document(review.document_id, tenant_id)
        threshold = self._review_threshold(
            "auto_approval_min_confidence",
            document,
        )
        accepted = []
        for field in document.extracted_fields:
            if (
                field.reviewer_status != ReviewerStatus.unverified
                or field.confidence is None
                or field.confidence < threshold
            ):
                continue
            field.reviewer_status = ReviewerStatus.accepted
            field.evidence.reviewer_status = ReviewerStatus.accepted
            field.verification_status = ReviewerStatus.accepted.value
            accepted.append(field.field_id)
        event = self._audit(
            document,
            "high_confidence_fields_accepted",
            f"Reviewer accepted {len(accepted)} field(s) at or above the configured threshold.",
            actor=actor,
            metadata={
                "field_ids": accepted,
                "threshold": threshold,
                **(request_metadata or {}),
            },
        )
        review.history.append(event)
        review.updated_at = document.updated_at
        document.data_version += 1
        document.analytics_cache_version += 1
        vision_analytics_service.refresh_document(document)
        self._persist_document(document)
        self._persist_review(review)
        return {"accepted_field_ids": accepted, "threshold": threshold}

    def active_learning_export(self, tenant_id: str | None = None) -> dict[str, Any]:
        corrections = []
        for document in self.list_documents(tenant_id):
            for field in document.extracted_fields:
                if field.reviewer_status == ReviewerStatus.edited and field.reviewed_value is not None:
                    corrections.append(
                        {
                            "document_id": document.document_id,
                            "field_id": field.field_id,
                            "field_name": field.name,
                            "model_value": self._mask_sensitive(field.name, field.value),
                            "reviewed_value": self._mask_sensitive(field.name, field.reviewed_value),
                            "evidence": field.evidence.model_dump(mode="json"),
                        }
                    )
        return {
            "total_corrections": len(corrections),
            "format": "active_learning_jsonl_ready",
            "corrections": corrections,
        }

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
            (document for document in documents if document.document_type == VisionDocumentType.invoice),
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
        document.review_task.reviewer_comments = "Auto-approved for the connected local demo dataset."
        document.review_task.updated_at = event.created_at
        document.review_task.history.append(event)
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

    def ask(
        self,
        request: VisionAskRequest,
        tenant_id: str | None = None,
    ) -> VisionAskResponse:
        question = request.question.strip()
        query_type = self.query_router.route(question)
        casual_documents = (
            [self.get_document(request.document_id, tenant_id)]
            if request.document_id
            else []
        )
        casual_responses = {
            "greeting": VisionAskResponse(
                answer=(
                    "Hello. Ask me about extracted fields, tables, clauses, validation "
                    "issues, risk indicators or relationships in the selected documents."
                ),
                answer_status="supported",
                query_type="greeting",
                reasoning_summary="Conversational greeting; no document retrieval was run.",
                requires_review=False,
                retriever="not_used",
            ),
            "help": VisionAskResponse(
                answer=(
                    "Ask about extracted fields, tables, clauses, calculations, visual "
                    "elements, validation issues, risk indicators or document relationships."
                ),
                answer_status="supported",
                query_type="help",
                reasoning_summary="Help response; no document retrieval was run.",
                requires_review=False,
                retriever="not_used",
            ),
            "ambiguous_question": VisionAskResponse(
                answer=(
                    "Do you want me to check document quality, financial consistency, "
                    "suspicious indicators, signature presence or approval status?"
                ),
                answer_status="clarification_needed",
                query_type="ambiguous_question",
                reasoning_summary="The requested meaning is too broad for an evidence-bound answer.",
                requires_review=False,
                retriever="not_used",
            ),
            "unsupported_question": VisionAskResponse(
                answer=(
                    "That request is outside the available document evidence. Ask about "
                    "fields, tables, clauses, calculations, visual elements, risks or relationships."
                ),
                answer_status="not_found",
                query_type="unsupported_question",
                reasoning_summary="No document retrieval was run for an unsupported request.",
                requires_review=False,
                retriever="not_used",
            ),
        }
        if query_type in casual_responses:
            return self._record_question(
                request,
                casual_responses[query_type],
                casual_documents,
                tenant_id=tenant_id,
            )
        documents = (
            casual_documents
            if request.document_id
            else self.list_documents(tenant_id)
        )
        if not documents:
            response = self._not_found_answer(requires_review=False)
            response.query_type = query_type
            return self._record_question(request, response, [], tenant_id=tenant_id)

        resolved_tenant = tenant_id or documents[0].tenant_id
        for document in documents:
            self.retrieval_index.upsert_document(document.document_id, document.evidence_chunks)
        handlers = {
            "structured_lookup": self._answer_structured,
            "semantic_question": self._answer_semantic,
            "relationship_question": self._answer_relationship,
            "arithmetic_question": self._answer_arithmetic,
            "visual_question": self._answer_visual,
            "comparison_question": self._answer_comparison,
            "risk_question": self._answer_risk,
        }
        response = handlers[query_type](
            question,
            documents,
            request.top_k,
            resolved_tenant,
        )
        response.query_type = query_type
        return self._record_question(
            request,
            response,
            documents,
            tenant_id=tenant_id,
        )

    def _answer_structured(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del tenant_id
        requested_names = self._requested_field_names(question)
        matches = self._field_matches(question.lower(), self._documents_for_question(question, documents))
        if requested_names:
            matches = [field for field in matches if field.name in requested_names]
        matches.sort(
            key=lambda field: (
                field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited},
                field.confidence or 0.0,
            ),
            reverse=True,
        )
        matches = matches[:top_k]
        if not matches:
            return self._not_found_answer(requires_review=False)
        return VisionAskResponse(
            answer=", ".join(
                f"{field.name}: {self._mask_sensitive(field.name, field.reviewed_value or field.value)}"
                for field in matches
            ),
            answer_status="supported",
            citations=[self._field_citation(field) for field in matches],
            confidence=self._mean_available_field_confidence(matches) or 0.0,
            reasoning_summary="Returned extracted structured fields and their source evidence.",
            requires_review=any(
                field.reviewer_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
                or not field.calibrated
                for field in matches
            ),
            retriever="persisted-structured-field-store",
        )

    def _answer_semantic(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        hits = self.retrieval_index.search(
            question,
            tenant_id=tenant_id,
            top_k=min(max(top_k, 5), 8),
            document_ids={document.document_id for document in documents},
        )
        hits = self._relevant_semantic_hits(question, hits)
        if not hits:
            return self._not_found_answer(requires_review=False)
        citations = [
            VisionAskCitation(
                document_id=hit.chunk.document_id,
                filename=self.get_document(hit.chunk.document_id, tenant_id).filename,
                page_number=hit.chunk.page_number,
                source_text=hit.chunk.text,
                bounding_boxes=hit.chunk.bounding_boxes,
                chunk_id=hit.chunk.chunk_id,
                retrieval_score=round(max(0.0, min(1.0, hit.score)), 4),
                retrieval_backend=hit.backend,
                model_confidence=hit.chunk.confidence,
                model_name=hit.chunk.model_name,
                model_version=hit.chunk.model_version,
                processing_timestamp=self.get_document(hit.chunk.document_id, tenant_id).updated_at,
                reviewer_status=hit.chunk.review_status,
            )
            for hit in hits
        ]
        response = VisionAskResponse(
            answer=" ".join(hit.chunk.text for hit in hits)[:1200],
            answer_status="supported",
            citations=citations,
            confidence=round(sum(hit.chunk.confidence for hit in hits) / len(hits), 4),
            reasoning_summary="Returned the highest-ranked structure-aware evidence blocks.",
            requires_review=any(
                hit.chunk.review_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
                for hit in hits
            ),
            retriever=hits[0].backend,
        )
        return self._apply_grounded_generator(question, hits, response)

    def _apply_grounded_generator(
        self,
        question: str,
        hits: list[Any],
        fallback: VisionAskResponse,
    ) -> VisionAskResponse:
        generator = self.generator_router.active()
        if generator is None:
            return fallback
        evidence = [
            {
                "id": hit.chunk.chunk_id,
                "text": hit.chunk.text,
                "document_id": hit.chunk.document_id,
                "page_number": hit.chunk.page_number,
                "review_status": hit.chunk.review_status.value,
            }
            for hit in hits
        ]
        try:
            generated = generator.generate(question, evidence)
        except Exception:
            return fallback
        citation_by_id = {
            citation.chunk_id: citation
            for citation in fallback.citations
            if citation.chunk_id is not None
        }
        selected = [
            citation_by_id[item]
            for item in generated["citation_ids"]
            if item in citation_by_id
        ]
        if generated["answer_status"] != "not_found" and not selected:
            return fallback
        return VisionAskResponse(
            answer=generated["answer"]
            or "The available documents do not contain sufficient evidence to answer this question.",
            answer_status=generated["answer_status"],
            query_type=fallback.query_type,
            citations=selected,
            confidence=fallback.confidence if selected else 0.0,
            reasoning_summary=generated["reasoning_summary"],
            requires_review=fallback.requires_review or generated["requires_review"],
            retriever=fallback.retriever,
            generator_name=generator.adapter_id,
            generator_version=generator.model_version,
        )

    def _answer_arithmetic(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del tenant_id
        requested = self._requested_field_names(question)
        numeric_names = requested & {"subtotal", "tax", "discount", "total_amount"}
        if not numeric_names:
            numeric_names = {"total_amount"}
        selected_documents = self._documents_for_question(question, documents)
        fields = [
            field
            for document in selected_documents
            for field in document.extracted_fields
            if field.name in numeric_names
        ][: max(top_k, 8)]
        parsed = [
            (field, value)
            for field in fields
            if (value := self._decimal_value(field.reviewed_value or field.value)) is not None
        ]
        if not parsed:
            return self._not_found_answer(requires_review=False)
        currencies = {
            self._field_map(document).get("currency")
            for document in selected_documents
            if self._field_map(document).get("currency")
        }
        if len(currencies) > 1:
            return VisionAskResponse(
                answer="The calculation spans multiple currencies and cannot be combined without verified conversion data.",
                answer_status="partially_supported",
                citations=[self._field_citation(field) for field, _ in parsed],
                confidence=0.0,
                reasoning_summary="Deterministic calculation stopped at the currency-consistency guardrail.",
                requires_review=True,
            )
        total = sum((value for _, value in parsed), Decimal(0))
        currency = next(iter(currencies), "")
        return VisionAskResponse(
            answer=f"{currency + ' ' if currency else ''}{total.quantize(Decimal('0.01'))}",
            answer_status=(
                "supported"
                if all(
                    field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
                    for field, _ in parsed
                )
                else "partially_supported"
            ),
            citations=[self._field_citation(field) for field, _ in parsed],
            confidence=(
                self._mean_available_field_confidence(field for field, _ in parsed)
                or 0.0
            ),
            reasoning_summary=f"Summed {len(parsed)} extracted value(s) using Decimal arithmetic.",
            requires_review=any(
                field.reviewer_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
                for field, _ in parsed
            ),
            retriever="deterministic-decimal-aggregation",
        )

    def _answer_visual(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del tenant_id
        lower = question.lower()
        class_terms = {
            "signature": ("signature", "signed"),
            "stamp": ("stamp",),
            "seal": ("seal",),
            "checkbox_checked": ("checkbox", "checked"),
            "checkbox_unchecked": ("checkbox", "unchecked"),
            "company_logo": ("logo",),
            "handwritten_note": ("handwritten",),
        }
        requested_classes = {
            name for name, terms in class_terms.items() if any(term in lower for term in terms)
        }
        detections = [
            (document, detection)
            for document in documents
            for detection in document.visual_detections
            if not requested_classes or detection.detection_class in requested_classes
        ][:top_k]
        if not detections:
            return self._not_found_answer(requires_review=False)
        citations = [
            VisionAskCitation(
                document_id=document.document_id,
                filename=document.filename,
                page_number=detection.page_number,
                source_text=f"Visual candidate: {detection.detection_class}",
                bounding_boxes=[detection.bounding_box],
                model_confidence=detection.confidence,
                model_name=detection.model_name,
                model_version=detection.model_version,
                processing_timestamp=document.updated_at,
                reviewer_status=detection.reviewer_status,
            )
            for document, detection in detections
        ]
        return VisionAskResponse(
            answer=(
                f"Detected {len(detections)} visual candidate(s): "
                + ", ".join(detection.detection_class for _, detection in detections)
                + ". Detection does not establish authenticity."
            ),
            answer_status="supported",
            citations=citations,
            confidence=round(sum(item.confidence for _, item in detections) / len(detections), 4),
            reasoning_summary="Returned visual detector candidates and their page regions.",
            requires_review=True,
            retriever="stored-visual-detections",
        )

    def _answer_relationship(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del tenant_id
        tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        matches: list[tuple[VisionDocumentRecord, dict[str, Any]]] = []
        for document in documents:
            for edge in document.graph.edges:
                haystack = " ".join(
                    str(edge.get(key, ""))
                    for key in ("type", "source", "target", "evidence_text")
                ).lower()
                if tokens & set(re.findall(r"[a-z0-9]+", haystack)):
                    matches.append((document, edge))
        matches = matches[:top_k]
        if not matches:
            return self._not_found_answer(requires_review=False)
        citations = [
            VisionAskCitation(
                document_id=document.document_id,
                filename=document.filename,
                page_number=int(edge.get("source_page", 1)),
                source_text=str(edge.get("evidence_text", "")),
                bounding_boxes=[VisionBoundingBox.model_validate(box) for box in edge.get("bounding_boxes", [])],
                evidence_id=edge.get("evidence_id"),
                model_confidence=float(edge.get("confidence", 0.0)),
                model_name=str(edge.get("model_name", "deterministic-graph-builder")),
                model_version=str(edge.get("model_version", "v1")),
                processing_timestamp=document.updated_at,
                reviewer_status=ReviewerStatus(str(edge.get("review_status", "unverified"))),
            )
            for document, edge in matches
        ]
        return VisionAskResponse(
            answer="; ".join(
                f"{edge.get('source')} {edge.get('type')} {edge.get('target')}"
                for _, edge in matches
            ),
            answer_status=(
                "supported"
                if all(edge.get("status") == "verified" for _, edge in matches)
                else "partially_supported"
            ),
            citations=citations,
            confidence=round(
                sum(float(edge.get("confidence", 0.0)) for _, edge in matches) / len(matches),
                4,
            ),
            reasoning_summary="Returned matching graph edges with their originating evidence.",
            requires_review=any(edge.get("status") != "verified" for _, edge in matches),
            retriever="verified-local-graph-fallback",
        )

    def _answer_risk(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del question, tenant_id
        matches = [
            (document, risk)
            for document in documents
            for risk in document.risk_indicators
        ][:top_k]
        if not matches:
            response = self._not_found_answer(requires_review=False)
            response.reasoning_summary = "No stored business or visual risk indicator matched the scope."
            response.retriever = "stored-risk-registry"
            return response
        citations = [
            VisionAskCitation(
                document_id=document.document_id,
                filename=document.filename,
                page_number=evidence.page_number,
                source_text=evidence.source_text,
                bounding_boxes=[evidence.bounding_box] if evidence.bounding_box else [],
                evidence_id=evidence.evidence_id,
                model_confidence=evidence.model_confidence,
                model_name=evidence.model_name,
                model_version=evidence.model_version,
                processing_timestamp=evidence.processing_timestamp,
                reviewer_status=evidence.reviewer_status,
            )
            for document, risk in matches
            for evidence in risk.evidence[:2]
        ]
        if not citations:
            response = self._not_found_answer(requires_review=False)
            response.reasoning_summary = "Stored risk records had no source-linked evidence."
            response.retriever = "stored-risk-registry"
            return response
        confidence_values = [
            risk.confidence
            for _, risk in matches
            if risk.confidence is not None
        ]
        return VisionAskResponse(
            answer="; ".join(
                f"{risk.severity.value}: {risk.title}. {risk.recommended_action}"
                for _, risk in matches
            ),
            answer_status=(
                "supported"
                if all(risk.reviewer_status == ReviewerStatus.accepted for _, risk in matches)
                else "partially_supported"
            ),
            citations=citations,
            confidence=(
                round(sum(confidence_values) / len(confidence_values), 4)
                if confidence_values
                else 0.0
            ),
            reasoning_summary="Returned stored risk indicators and their source-linked evidence.",
            requires_review=any(
                risk.reviewer_status != ReviewerStatus.accepted for _, risk in matches
            ),
            retriever="stored-risk-registry",
        )

    def _answer_comparison(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del tenant_id
        if len(documents) < 2:
            return self._not_found_answer(requires_review=False)
        requested = self._requested_field_names(question)
        by_name: dict[str, list[ExtractedVisionField]] = {}
        for document in documents:
            for field in document.extracted_fields:
                if requested and field.name not in requested:
                    continue
                by_name.setdefault(field.name, []).append(field)
        comparable = [
            (name, fields)
            for name, fields in by_name.items()
            if len({field.evidence.document_id for field in fields}) >= 2
        ][:top_k]
        if not comparable:
            return self._not_found_answer(requires_review=False)
        fields = [field for _, values in comparable for field in values]
        return VisionAskResponse(
            answer="; ".join(
                f"{name}: "
                + ", ".join(
                    f"{field.evidence.document_id}={self._mask_sensitive(name, field.reviewed_value or field.value)}"
                    for field in values
                )
                for name, values in comparable
            ),
            answer_status="partially_supported",
            citations=[self._field_citation(field) for field in fields],
            confidence=self._mean_available_field_confidence(fields) or 0.0,
            reasoning_summary="Compared like-named structured fields across documents.",
            requires_review=any(
                field.reviewer_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
                for field in fields
            ),
            retriever="cross-document-field-comparison",
        )

    def _record_question(
        self,
        request: VisionAskRequest,
        response: VisionAskResponse,
        documents: list[VisionDocumentRecord],
        tenant_id: str | None = None,
    ) -> VisionAskResponse:
        question_id = f"vision-question-{uuid4().hex[:10]}"
        response.question_id = question_id
        self.questions[question_id] = {
            "question_id": question_id,
            "question": request.question,
            "document_id": request.document_id,
            "document_ids": [document.document_id for document in documents],
            "top_k": request.top_k,
            "answer": response.model_dump(mode="json"),
            "feedback": [],
            "tenant_id": tenant_id or (documents[0].tenant_id if documents else "tenant-local"),
            "created_at": utc_now_iso(),
        }
        self._persist_question(question_id, self.questions[question_id])
        return response

    def get_question(self, question_id: str, tenant_id: str | None = None) -> dict[str, Any]:
        question = self.questions.get(question_id) or self._load_question(question_id)
        if not question and tenant_id:
            question = self.persistence.get_question(question_id, tenant_id)
            if question:
                self.questions[question_id] = question
        if not question or (
            tenant_id and question.get("tenant_id", "tenant-local") != tenant_id
        ):
            raise ValueError("question_id not found")
        return question

    def record_question_feedback(
        self,
        question_id: str,
        *,
        rating: str,
        comment: str,
        actor: str = "system",
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        question = self.get_question(question_id, tenant_id)
        feedback = {
            "feedback_id": f"vision-feedback-{uuid4().hex[:10]}",
            "question_id": question_id,
            "rating": rating,
            "comment": comment,
            "actor": actor,
            "created_at": utc_now_iso(),
        }
        self.question_feedback.setdefault(question_id, []).append(feedback)
        question["feedback"] = list(self.question_feedback[question_id])
        self._persist_question(question_id, question)
        return feedback

    def verify_graph_relationship(
        self,
        relationship_id: str,
        *,
        tenant_id: str,
        actor: str,
        comment: str,
        evidence_acknowledged: bool,
    ) -> dict[str, Any]:
        if not evidence_acknowledged:
            raise ValueError("Relationship evidence must be acknowledged before verification")
        if len(comment.strip()) < 10:
            raise ValueError("A reviewer comment of at least 10 characters is required")

        document: VisionDocumentRecord | None = None
        relationship: dict[str, Any] | None = None
        for candidate in self.list_documents(tenant_id):
            relationship = next(
                (
                    edge
                    for edge in candidate.graph.edges
                    if edge.get("id") == relationship_id
                ),
                None,
            )
            if relationship:
                document = candidate
                break
        if document is None or relationship is None:
            raise ValueError("graph relationship not found")
        if not relationship.get("source_linked") or not relationship.get("evidence_text"):
            raise ValueError("A relationship without source evidence cannot be verified")
        if not relationship.get("verification_eligible"):
            raise ValueError("Relationship is not eligible for manual verification")

        now = utc_now_iso()
        relationship.update(
            {
                "status": "verified",
                "review_status": ReviewerStatus.accepted.value,
                "verification_eligible": False,
                "verified_by": actor,
                "verified_at": now,
                "verification_comment": comment.strip(),
            }
        )
        document.updated_at = now
        self._audit(
            document,
            "graph_relationship_verified",
            "A reviewer verified a source-linked graph relationship.",
            actor=actor,
            metadata={
                "relationship_id": relationship_id,
                "relationship_type": relationship.get("type"),
                "evidence_id": relationship.get("evidence_id"),
                "comment": comment.strip(),
            },
        )
        self.persistence.upsert_graph(
            document.tenant_id,
            document.document_id,
            document.graph,
        )
        self._persist_document(document)
        return relationship

    def model_comparison(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        document = self.get_document(document_id, tenant_id)
        registry = self.model_registry.registry()
        schema = vision_schema_registry.get(document.document_type)
        selected = {field.name: field for field in document.extracted_fields}
        deterministic = {field.name: field for field in self._extract_fields(document)}
        donut_payload = dict(document.model_routing.get("donut_comparison") or {})
        donut_values = self._flatten_model_output(donut_payload.get("output") or {})
        field_names = list(
            dict.fromkeys(
                [
                    *([field.name for field in schema.fields] if schema else []),
                    *selected,
                    *deterministic,
                    *(key.rsplit(".", 1)[-1] for key in donut_values),
                ]
            )
        )
        extraction_runs = [
            run for run in document.model_runs if run.stage == "extraction"
        ]
        primary_run = next(
            (run for run in reversed(extraction_runs) if run.status == "completed"),
            extraction_runs[-1] if extraction_runs else None,
        )
        availability = [
            item
            for item in registry["backends"]
            if item["stage"]
            in {
                "classification_extraction",
                "ocr_free_baseline",
                "advanced_parsing",
            }
        ]

        rows = []
        for field_name in field_names:
            selected_field = selected.get(field_name)
            fallback_field = deterministic.get(field_name)
            layout_field = (
                selected_field
                if selected_field
                and "layoutlm" in selected_field.model_name.lower()
                else None
            )
            donut_value = next(
                (
                    value
                    for key, value in donut_values.items()
                    if key.rsplit(".", 1)[-1] == field_name.lower()
                ),
                None,
            )
            observed_values = [
                value
                for value in (
                    selected_field.value if selected_field else None,
                    fallback_field.value if fallback_field else None,
                    str(donut_value) if donut_value is not None else None,
                )
                if value is not None
            ]
            disagreement = len({self._normalize(value) for value in observed_values}) > 1
            required = bool(schema and schema.field(field_name) and schema.field(field_name).required)
            if selected_field:
                validation_result = (
                    "accepted"
                    if selected_field.reviewer_status
                    in {ReviewerStatus.accepted, ReviewerStatus.edited}
                    else "requires_review"
                    if selected_field.confidence is None
                    or selected_field.confidence
                    < self._mandatory_review_threshold(document)
                    else "unverified"
                )
            else:
                validation_result = "missing_required" if required else "not_extracted"
            rows.append(
                {
                    "field_id": selected_field.field_id if selected_field else None,
                    "field_name": field_name,
                    "required": required,
                    "primary_result": self._mask_sensitive(field_name, selected_field.value)
                    if selected_field
                    else None,
                    "primary_confidence": selected_field.confidence if selected_field else None,
                    "primary_model": selected_field.model_name if selected_field else None,
                    "fallback_result": self._mask_sensitive(field_name, fallback_field.value)
                    if fallback_field
                    else None,
                    "fallback_confidence": fallback_field.confidence if fallback_field else None,
                    "fallback_model": fallback_field.model_name if fallback_field else None,
                    "layoutlmv3_result": self._mask_sensitive(field_name, layout_field.value)
                    if layout_field
                    else None,
                    "layoutlmv3_confidence": layout_field.confidence if layout_field else None,
                    "donut_result": self._mask_sensitive(field_name, str(donut_value))
                    if donut_value is not None
                    else None,
                    "donut_confidence": None,
                    "validation_result": validation_result,
                    "selected_result": self._mask_sensitive(field_name, selected_field.reviewed_value or selected_field.value)
                    if selected_field
                    else None,
                    "review_status": selected_field.reviewer_status.value
                    if selected_field
                    else "unresolved" if required else "not_available",
                    "disagreement": disagreement,
                    "evidence": selected_field.evidence.model_dump(mode="json")
                    if selected_field
                    else None,
                }
            )
        return {
            "document_id": document_id,
            "comparison_mode": "observed_runtime_outputs",
            "primary_adapter": {
                "model_id": primary_run.model_name if primary_run else None,
                "status": primary_run.status if primary_run else "not_recorded",
                "runtime_ms": primary_run.processing_duration_ms if primary_run else None,
                "device": primary_run.device if primary_run else None,
                "checkpoint": primary_run.checkpoint if primary_run else None,
            },
            "fallback_adapter": {
                "model_id": deterministic_schema_extractor.adapter_id,
                "status": "observed" if deterministic else "no_fields_observed",
                "field_count": len(deterministic),
            },
            "availability": availability,
            "field_output": rows,
            "fallback_routing": document.model_routing,
            "routing_reason": document.model_routing.get("reasons", []),
            "model_runs": [run.model_dump(mode="json") for run in document.model_runs],
            "note": (
                "Only observed outputs are shown. Unavailable model cells stay blank until a "
                "checkpoint-backed adapter completes inference for this document."
            ),
        }

    def _flatten_model_output(self, payload: Any) -> dict[str, Any]:
        flattened: dict[str, Any] = {}

        def visit(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(f"{prefix}.{key}" if prefix else str(key), child)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(f"{prefix}.{index}", child)
            elif value is not None:
                flattened[prefix.lower().replace(" ", "_")] = value

        visit("", payload)
        return flattened

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
            "by_document_type": dict(
                Counter(item.document_type.value for item in documents)
            ),
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
        warnings = [
            warning for document in documents for warning in document.processing_warnings
        ]
        by_severity = Counter(risk.severity.value for risk in indicators)
        by_category = Counter(risk.category for risk in indicators)
        return {
            "total_indicators": len(indicators),
            "by_severity": dict(by_severity),
            "by_category": dict(by_category),
            "processing_warning_count": len(warnings),
            "processing_warnings_by_title": dict(
                Counter(warning.title for warning in warnings)
            ),
            "confirmed_fraud_count": sum(
                1 for document in documents if document.status == VisionProcessingStatus.confirmed_fraud
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
            item.document_id
            for item in self._analytics_documents(tenant_id, date_from, date_to)
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
            "average_review_time_seconds": duration_metrics[
                "average_review_time_seconds"
            ],
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
        cited = [
            item for item in supported if (item.get("answer") or {}).get("citations")
        ]
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
            "citation_coverage": (
                round(len(cited) / len(supported), 4) if supported else None
            ),
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

    def admin_models(self) -> dict[str, Any]:
        registry = self.model_registry.registry()
        for backend in registry["backends"]:
            override = self.model_overrides.get(str(backend["model_id"]))
            backend["requested"] = (
                bool(override["requested_enabled"])
                if override and override.get("requested_enabled") is not None
                else bool(backend.get("enabled"))
            )
        return {
            "registry": registry,
            "overrides": self.model_overrides,
            "smoke_results": self.model_smoke.store.all(),
            "policy": (
                "Requested state is deployment intent only. Active status requires the current "
                "package, checkpoint, adapter initialization, inference, and output-schema smoke test."
            ),
        }

    def admin_model_health(self, model_id: str) -> dict[str, Any]:
        models = self.admin_models()["registry"]["backends"]
        match = next(
            (
                item
                for item in models
                if str(item["model_id"]) == model_id
                or str(item.get("adapter_id")) == model_id
            ),
            None,
        )
        if match is None:
            raise ValueError(f"Model is not registered: {model_id}")
        return match

    def patch_admin_model(
        self,
        model_id: str,
        payload: dict[str, Any],
        *,
        actor: str = "system",
        tenant_id: str = "tenant-local",
    ) -> dict[str, Any]:
        registered = {
            str(item["model_id"])
            for item in self.model_registry.backends(include_runtime_health=False)
        }
        if model_id not in registered:
            raise ValueError(f"Model is not registered: {model_id}")
        override = {
            "model_id": model_id,
            "requested_enabled": payload.get("requested", payload.get("enabled")),
            "profile": payload.get("profile"),
            "notes": payload.get("notes", ""),
            "actor": actor,
            "updated_at": utc_now_iso(),
            "status": "recorded_for_deployment_review",
        }
        self.model_overrides[model_id] = override
        self._write_json(self._state_root / "model_overrides.json", self.model_overrides)
        self._system_audit(
            "model_state_change",
            "Recorded a requested model deployment state without changing runtime availability.",
            actor=actor,
            tenant_id=tenant_id,
            metadata={
                "model_id": model_id,
                "requested_enabled": override["requested_enabled"],
                "status": override["status"],
            },
        )
        return override

    def smoke_test_admin_model(
        self,
        model_id: str,
        *,
        actor: str = "system",
        tenant_id: str = "tenant-local",
    ) -> dict[str, Any]:
        result = self.model_smoke.run(model_id)
        self._system_audit(
            "model_smoke_test",
            "Executed adapter initialization, inference, and output-schema verification.",
            actor=actor,
            tenant_id=tenant_id,
            metadata={
                "model_id": result["model_id"],
                "status": result["status"],
                "health_status": result["health_status"],
                "configuration_fingerprint": result["configuration_fingerprint"],
            },
        )
        return result

    def admin_rules(self) -> dict[str, Any]:
        return {
            "confidence_policy": self.confidence_policy,
            "validation_rules": [
                "amount_arithmetic",
                "due_date_after_invoice_date",
                "approved_bank_match",
                "required_signature",
                "vendor_tax_id_format",
                "payment_terms_conflict",
                "duplicate_detection",
                "advanced_fallback_review",
            ],
            "overrides": self.rule_overrides,
        }

    def patch_admin_rule(
        self,
        rule_id: str,
        payload: dict[str, Any],
        *,
        actor: str = "system",
        tenant_id: str = "tenant-local",
    ) -> dict[str, Any]:
        override = {
            "rule_id": rule_id,
            "enabled": payload.get("enabled"),
            "threshold": payload.get("threshold"),
            "notes": payload.get("notes", ""),
            "actor": actor,
            "updated_at": utc_now_iso(),
            "status": "recorded_for_local_runtime",
        }
        self.rule_overrides[rule_id] = override
        self._write_json(self._state_root / "rule_overrides.json", self.rule_overrides)
        self._system_audit(
            "rule_state_change",
            "Recorded a validation-rule override.",
            actor=actor,
            tenant_id=tenant_id,
            metadata={
                "rule_id": rule_id,
                "enabled": override["enabled"],
                "threshold": override["threshold"],
            },
        )
        return override

    def generate_finance_narrative(
        self,
        request: VisionFinanceNarrativeRequest,
        *,
        tenant_id: str,
        actor: str,
    ) -> VisionFinanceNarrative:
        with self._artifact_lock:
            narrative = verified_finance_narrative_service.generate(
                request,
                self.list_documents(tenant_id),
                tenant_id=tenant_id,
                actor=actor,
            )
            selected = [
                self.get_document(document_id, tenant_id)
                for document_id in narrative.document_ids
            ]
            source_data_version = self._artifact_source_data_version(selected)
            request_scope = request.model_dump(
                mode="json",
                exclude={"generate_new_version"},
            )
            request_scope["document_ids"] = sorted(narrative.document_ids)
            identity_hash = self._artifact_identity_hash(
                {
                    "tenant_id": tenant_id,
                    "narrative_type": request.narrative_type.value,
                    "date_range": [request.date_from, request.date_to],
                    "vendor_filter": self._normalize(request.vendor_name or ""),
                    "document_ids": sorted(narrative.document_ids),
                    "verified_data_version": source_data_version,
                }
            )
            matches = [
                item
                for item in self.list_finance_narratives(tenant_id)
                if item.identity_hash == identity_hash
            ]
            latest = max(matches, key=lambda item: item.version, default=None)
            if latest and not request.generate_new_version:
                return latest
            narrative.identity_hash = identity_hash
            narrative.source_data_version = source_data_version
            narrative.request_scope = request_scope
            if latest:
                latest.status = "superseded"
                self._persist_finance_narrative(latest)
                narrative.version = latest.version + 1
                narrative.supersedes = latest.narrative_id
            self._persist_finance_narrative(narrative)
            return narrative

    def get_finance_narrative(
        self,
        narrative_id: str,
        tenant_id: str,
    ) -> VisionFinanceNarrative:
        narrative = self.finance_narratives.get(narrative_id)
        if not narrative:
            payload = self._read_json(self._state_file("narratives", narrative_id))
            if payload:
                try:
                    narrative = VisionFinanceNarrative.model_validate(payload)
                except ValueError:
                    narrative = None
        if not narrative:
            narrative = self.persistence.get_narrative(narrative_id, tenant_id)
        if not narrative or narrative.tenant_id != tenant_id:
            raise ValueError("narrative_id not found")
        self.finance_narratives[narrative.narrative_id] = narrative
        return narrative

    def list_finance_narratives(self, tenant_id: str) -> list[VisionFinanceNarrative]:
        for path in (self._state_root / "narratives").glob("*.json"):
            payload = self._read_json(path)
            if not payload:
                continue
            try:
                narrative = VisionFinanceNarrative.model_validate(payload)
            except ValueError:
                continue
            self.finance_narratives[narrative.narrative_id] = narrative
        for narrative in self.persistence.list_narratives(tenant_id):
            self.finance_narratives[narrative.narrative_id] = narrative
        return sorted(
            [item for item in self.finance_narratives.values() if item.tenant_id == tenant_id],
            key=lambda item: item.created_at,
            reverse=True,
        )

    def generate_report(
        self,
        request: VisionReportRequest,
        *,
        tenant_id: str,
        actor: str,
    ) -> VisionReportRecord:
        with self._artifact_lock:
            report = vision_report_service.generate(
                request,
                self.list_documents(tenant_id),
                self.list_finance_narratives(tenant_id),
                tenant_id=tenant_id,
                actor=actor,
            )
            selected = [
                self.get_document(document_id, tenant_id)
                for document_id in report.document_ids
            ]
            source_data_version = self._artifact_source_data_version(selected)
            request_scope = request.model_dump(
                mode="json",
                exclude={"generate_new_version"},
            )
            request_scope["document_ids"] = sorted(report.document_ids)
            identity_hash = self._artifact_identity_hash(
                {
                    "tenant_id": tenant_id,
                    "report_type": request.report_type.value,
                    "document_ids": sorted(report.document_ids),
                    "document_data_versions": source_data_version,
                    "review_state": sorted(
                        (item.document_id, item.status.value) for item in selected
                    ),
                    "template_version": report.template_version,
                    "narrative_id": request.narrative_id,
                    "vendor_filter": self._normalize(request.vendor_name or ""),
                    "evaluation": request.evaluation,
                }
            )
            matches = [
                item
                for item in self.list_reports(tenant_id)
                if item.identity_hash == identity_hash
            ]
            latest = max(matches, key=lambda item: item.version, default=None)
            if latest and not request.generate_new_version:
                return latest
            report.identity_hash = identity_hash
            report.source_data_version = source_data_version
            report.request_scope = request_scope
            if latest:
                latest.status = "superseded"
                self._persist_report_record(latest)
                report.version = latest.version + 1
                report.supersedes = latest.report_id
            pdf = vision_report_service.pdf(report)
            report.object_uri = self.persistence.save_report(
                report.model_dump(mode="json"),
                pdf,
            )
            self._persist_report_record(report)
            for document in selected:
                self._audit(
                    document,
                    "report_generated",
                    f"Generated {report.report_type.value} version {report.version}.",
                    actor=actor,
                    metadata={
                        "report_id": report.report_id,
                        "identity_hash": report.identity_hash,
                        "version": report.version,
                    },
                )
                self._persist_document(document)
            return report

    def archive_report(
        self,
        report_id: str,
        *,
        tenant_id: str,
        actor: str,
    ) -> VisionReportRecord:
        report = self.get_report(report_id, tenant_id)
        report.status = "archived"
        self._persist_report_record(report)
        for document_id in report.document_ids:
            document = self.get_document(document_id, tenant_id)
            self._audit(
                document,
                "report_archived",
                "Archived a generated report without deleting its audit record.",
                actor=actor,
                metadata={"report_id": report_id, "version": report.version},
            )
            self._persist_document(document)
        return report

    def audit_report_download(
        self,
        resource_id: str,
        *,
        tenant_id: str,
        actor: str,
    ) -> None:
        try:
            report = self.get_report(resource_id, tenant_id)
            document_ids = report.document_ids
            report_id = report.report_id
        except ValueError:
            document = self.get_document(resource_id, tenant_id)
            document_ids = [document.document_id]
            report_id = None
        for document_id in document_ids:
            document = self.get_document(document_id, tenant_id)
            self._audit(
                document,
                "report_downloaded",
                "Downloaded an evidence report as PDF.",
                actor=actor,
                metadata={"resource_id": resource_id, "report_id": report_id},
            )
            self._persist_document(document)
        self._system_audit(
            "report_downloaded",
            "Downloaded an evidence report as PDF.",
            actor=actor,
            tenant_id=tenant_id,
            metadata={"resource_id": resource_id, "report_id": report_id},
        )

    def get_report(self, report_id: str, tenant_id: str) -> VisionReportRecord:
        report = self.reports.get(report_id)
        if not report:
            payload = self._read_json(self._state_file("reports", report_id))
            if payload:
                try:
                    report = VisionReportRecord.model_validate(payload)
                except ValueError:
                    report = None
        if not report:
            report = self.persistence.get_report(report_id, tenant_id)
        if not report or report.tenant_id != tenant_id:
            raise ValueError("report_id not found")
        self.reports[report.report_id] = report
        return report

    def list_reports(self, tenant_id: str) -> list[VisionReportRecord]:
        for path in (self._state_root / "reports").glob("*.json"):
            payload = self._read_json(path)
            if not payload:
                continue
            try:
                report = VisionReportRecord.model_validate(payload)
            except ValueError:
                continue
            self.reports[report.report_id] = report
        for report in self.persistence.list_reports(tenant_id):
            self.reports[report.report_id] = report
        return sorted(
            [item for item in self.reports.values() if item.tenant_id == tenant_id],
            key=lambda item: item.created_at,
            reverse=True,
        )

    def render_stored_report_markdown(self, report_id: str, tenant_id: str) -> str:
        return vision_report_service.markdown(self.get_report(report_id, tenant_id))

    def render_stored_report_pdf(self, report_id: str, tenant_id: str) -> bytes:
        return vision_report_service.pdf(self.get_report(report_id, tenant_id))

    def render_report_payload(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        document = self.get_document(document_id, tenant_id)
        return {
            "report_type": "document_intelligence_report",
            "document": document.model_dump(mode="json"),
            "summary": self._document_summary(document),
            "risk_policy": "Suspicious indicators require human review; this report does not confirm fraud.",
            "active_learning_available": bool(
                any(field.reviewer_status == ReviewerStatus.edited for field in document.extracted_fields)
            ),
        }

    def render_report_markdown(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> str:
        payload = self.render_report_payload(document_id, tenant_id)
        document = self.get_document(document_id, tenant_id)
        fields = "\n".join(
            f"| {field.name} | {self._mask_sensitive(field.name, field.reviewed_value or field.value)} | "
            f"{field.confidence:.0%} | {field.evidence.evidence_id} | {field.reviewer_status.value} |"
            for field in document.extracted_fields
            if field.confidence is not None
        ) or "| None | - | - | - | - |"
        risks = "\n".join(
            f"| {risk.title} | {risk.severity.value} | {risk.category} | {risk.recommended_action} |"
            for risk in document.risk_indicators
        ) or "| None | - | - | - |"
        tables = "\n\n".join(table.markdown for table in document.tables) or "No structural tables detected."
        return f"""# DOCUMIND VISION AI Report

## Summary
{payload["summary"]}

## Document
| Field | Value |
| --- | --- |
| Document ID | {document.document_id} |
| Filename | {document.filename} |
| Type | {document.document_type.value} |
| Status | {document.status.value} |
| Pages | {document.page_count} |

## Extracted Fields
| Field | Value | Confidence | Evidence | Review |
| --- | --- | ---: | --- | --- |
{fields}

## Risk Indicators
| Indicator | Severity | Category | Recommendation |
| --- | --- | --- | --- |
{risks}

## Tables
{tables}

## Policy
Suspicious indicators require human review. Only an authorised reviewer may set CONFIRMED_FRAUD.
"""

    def render_pdf_bytes(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> bytes:
        text = self.render_report_markdown(document_id, tenant_id)
        lines = [line.replace("|", " ").strip() for line in text.splitlines()]
        stream_lines = []
        y = 780
        for index, line in enumerate(lines[:58]):
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            size = 16 if index == 0 else 9
            stream_lines.append(f"BT /F1 {size} Tf 40 {y - index * 12} Td ({escaped}) Tj ET")
        stream = "\n".join(stream_lines)
        objects = [
            "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
            "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
            "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
            "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
            f"5 0 obj << /Length {len(stream)} >> stream\n{stream}\nendstream endobj",
        ]
        body = "%PDF-1.4\n" + "\n".join(objects) + "\n"
        offsets = []
        cursor = len("%PDF-1.4\n")
        for item in objects:
            offsets.append(cursor)
            cursor += len(item) + 1
        xref_start = len(body)
        xref = "xref\n0 6\n0000000000 65535 f \n" + "".join(
            f"{offset:010d} 00000 n \n" for offset in offsets
        )
        trailer = f"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF"
        return (body + xref + trailer).encode("utf-8")

    def _process_document(
        self,
        document: VisionDocumentRecord,
        filename: str,
        content: bytes,
        metadata: dict[str, Any],
    ) -> None:
        self._transition(document, VisionProcessingStatus.preprocessing, "Started quality analysis and preprocessing.")
        text, pages, parse_evidence, stored_path = self.ingestion.parse(filename, content)
        parse_confidence = parse_evidence[0].confidence if parse_evidence else 0.5
        document.parsed_text = text
        document.original_storage_path = stored_path
        preprocessing = self.preprocessor.process(
            document.document_id,
            filename,
            content,
            parsed_text=text,
        )
        document.page_artifacts = preprocessing.artifacts
        document.quality = preprocessing.quality
        document.page_count = len(preprocessing.artifacts) or pages
        if not document.parsed_text.strip():
            document.parsed_text = "\n\n".join(
                preprocessing.page_text.get(page_number, "")
                for page_number in sorted(preprocessing.page_text)
            ).strip()
        quality_confidence = (
            sum(item.image_quality_score for item in document.quality) / len(document.quality)
            if document.quality
            else 0.0
        )
        document.model_runs.append(
            self._model_run(
                "preprocessing",
                "OpenCV-Pillow-PyMuPDF",
                "v1",
                "completed",
                round(quality_confidence, 3),
            )
        )

        ocr_adapter = self.paddle_ocr if self.paddle_ocr.status().status == "active" else self.local_ocr
        try:
            document.ocr_words = ocr_adapter.extract_words(document, preprocessing.native_words)
        except Exception as exc:
            if ocr_adapter is self.local_ocr:
                raise
            document.model_runs.append(
                self._model_run(
                    "ocr",
                    ocr_adapter.adapter_id,
                    ocr_adapter.model_version,
                    "failed_fallback_used",
                    None,
                    notes=[str(exc)],
                )
            )
            document.ocr_words = self.local_ocr.extract_words(document, preprocessing.native_words)
            ocr_adapter = self.local_ocr
        if not document.parsed_text.strip() and document.ocr_words:
            document.parsed_text = self._text_from_ocr(document.ocr_words)
        text = document.parsed_text
        parse_confidence = self._mean_word_confidence(document) or parse_confidence
        document.model_runs.append(
            self._model_run(
                "ocr",
                ocr_adapter.adapter_id,
                ocr_adapter.model_version,
                "completed",
                self._mean_word_confidence(document),
                notes=[
                    "Native PDF/source text geometry was preferred where available."
                    if preprocessing.native_words
                    else "OCR executed on processed page images."
                ],
            )
        )
        self._transition(document, VisionProcessingStatus.ocr_completed, "Completed OCR with page evidence.")

        layout_adapter = (
            self.paddle_structure
            if self.paddle_structure.status().status == "active"
            else self.local_layout
        )
        try:
            document.layout_regions, document.tables = layout_adapter.analyze_layout(document)
        except Exception as exc:
            if layout_adapter is self.local_layout:
                raise
            document.model_runs.append(
                self._model_run(
                    "layout",
                    layout_adapter.adapter_id,
                    layout_adapter.model_version,
                    "failed_fallback_used",
                    None,
                    notes=[str(exc)],
                )
            )
            layout_adapter = self.local_layout
            document.layout_regions, document.tables = layout_adapter.analyze_layout(document)
        document.model_runs.append(
            self._model_run(
                "layout",
                layout_adapter.adapter_id,
                layout_adapter.model_version,
                "completed",
                self._layout_confidence(document),
            )
        )

        document_type_hint = self._document_type_hint(metadata)
        if document_type_hint is not None:
            document.document_type = document_type_hint
            document.model_runs.append(
                self._model_run(
                    "document_classification",
                    "reviewer-selected-document-type",
                    "v1",
                    "completed",
                    None,
                    notes=["The explicit upload selection was used instead of model classification."],
                )
            )
            self._audit(
                document,
                "document_type_selected",
                "Applied the document type selected during upload.",
                metadata={"document_type": document_type_hint.value},
            )
        elif self.layoutlmv3_classifier.status().status == "active":
            try:
                document.document_type, classification_confidence = (
                    self.layoutlmv3_classifier.classify(document)
                )
                document.model_runs.append(
                    self._model_run(
                        "document_classification",
                        self.layoutlmv3_classifier.adapter_id,
                        self.layoutlmv3_classifier.model_version,
                        "completed",
                        classification_confidence,
                    )
                )
            except Exception as exc:
                document.document_type = self._classify_document(text, filename)
                document.model_runs.append(
                    self._model_run(
                        "document_classification",
                        self.layoutlmv3_classifier.adapter_id,
                        self.layoutlmv3_classifier.model_version,
                        "failed_fallback_used",
                        None,
                        notes=[str(exc)],
                    )
                )
        else:
            document.document_type = self._classify_document(text, filename)
            document.model_runs.append(
                self._model_run(
                    "document_classification",
                    "deterministic-document-classifier",
                    "v1",
                    "completed",
                    None,
                    notes=["Rule classification is not a learned probability."],
                )
            )

        extraction_adapter_name = deterministic_schema_extractor.adapter_id
        extraction_adapter_version = deterministic_schema_extractor.model_version
        if self.layoutlmv3_extractor.status().status == "active":
            try:
                learned_fields = self.layoutlmv3_extractor.extract_fields(document)
                learned_names = {field.name for field in learned_fields}
                supplemental_fields = [
                    field for field in self._extract_fields(document) if field.name not in learned_names
                ]
                document.extracted_fields = [*learned_fields, *supplemental_fields]
                extraction_adapter_name = self.layoutlmv3_extractor.adapter_id
                extraction_adapter_version = self.layoutlmv3_extractor.model_version
            except Exception as exc:
                document.extracted_fields = self._extract_fields(document)
                document.model_runs.append(
                    self._model_run(
                        "extraction",
                        self.layoutlmv3_extractor.adapter_id,
                        self.layoutlmv3_extractor.model_version,
                        "failed_fallback_used",
                        None,
                        notes=[str(exc)],
                    )
                )
        else:
            document.extracted_fields = self._extract_fields(document)
        if document.document_type == VisionDocumentType.contract:
            self._run_contract_extraction(document)
        visual_adapter = (
            self.yolo_detector
            if self.yolo_detector.status().status == "active"
            else self.local_visual_detector
        )
        try:
            document.visual_detections = visual_adapter.detect(document)
        except Exception as exc:
            if visual_adapter is self.local_visual_detector:
                raise
            document.model_runs.append(
                self._model_run(
                    "visual_detection",
                    visual_adapter.adapter_id,
                    visual_adapter.model_version,
                    "failed_fallback_used",
                    None,
                    notes=[str(exc)],
                )
            )
            visual_adapter = self.local_visual_detector
            document.visual_detections = visual_adapter.detect(document)
        document.model_runs.append(
            self._model_run(
                "extraction",
                extraction_adapter_name,
                extraction_adapter_version,
                "completed",
                self._mean_field_confidence(document),
                notes=[
                    "Confidence is uncalibrated and therefore cannot imply model accuracy."
                    if not bool(self.confidence_policy.get("calibration", {}).get("calibrated"))
                    else "Confidence calibration artefact is active."
                ],
            )
        )
        document.model_runs.append(
            self._model_run(
                "visual_detection",
                visual_adapter.adapter_id,
                visual_adapter.model_version,
                "completed",
                round(
                    sum(item.confidence for item in document.visual_detections)
                    / len(document.visual_detections),
                    3,
                )
                if document.visual_detections
                else None,
                notes=["Candidate detection does not establish authenticity."],
            )
        )
        document.language_analysis = vision_language_analysis_service.analyze(
            document.ocr_words
        )
        document.model_routing = self.model_registry.route_document(
            filename=filename,
            text=text,
            parse_confidence=parse_confidence,
            layout_result={
                "engine": "local-layout-adapter",
                "regions": sorted({region.region_type for region in document.layout_regions}),
                "evidence": {"confidence": self._layout_confidence(document) or 0.0},
            },
            mixed_languages=document.language_analysis.mixed_language_routing,
            language_analysis=document.language_analysis.model_dump(mode="json"),
        )
        if document.model_routing.get("fallback_required"):
            self._run_selective_fallbacks(document)
        self._transition(document, VisionProcessingStatus.extraction_completed, "Completed structured extraction.")

        forensics_adapter = (
            self.segformer_forensics
            if self.segformer_forensics.status().status == "active"
            else self.local_forensics
        )
        try:
            document.tamper_analysis = forensics_adapter.analyze(document)
        except Exception as exc:
            if forensics_adapter is self.local_forensics:
                raise
            document.model_runs.append(
                self._model_run(
                    "tamper_screening",
                    forensics_adapter.adapter_id,
                    forensics_adapter.model_version,
                    "failed_fallback_used",
                    None,
                    notes=[str(exc)],
                )
            )
            forensics_adapter = self.local_forensics
            document.tamper_analysis = forensics_adapter.analyze(document)
        document.model_runs.append(
            self._model_run(
                "tamper_screening",
                forensics_adapter.adapter_id,
                forensics_adapter.model_version,
                "completed",
                max((item.screening_score for item in document.tamper_analysis), default=0.0),
                notes=[
                    "Local screening scores are measured signals, not calibrated tampering probabilities."
                    if forensics_adapter is self.local_forensics
                    else "Segmentation softmax is uncalibrated and requires human verification."
                ],
            )
        )
        validation_risks = self._validation_risks(document, metadata)
        validation_risks.extend(self._contract_risks(document))
        self._enrich_field_confidence(document, validation_risks)
        vision_analytics_service.refresh_document(document)
        tamper_risks = self._tamper_risks(document)
        document.perceptual_hash = document_dhash(document)
        dino_status = self.dinov2_duplicate.status()
        if dino_status.status == "active":
            try:
                document.visual_embedding = self.dinov2_duplicate.embed(document)
                document.visual_embedding_model = self.dinov2_duplicate.model_version
                document.model_runs.append(
                    self._model_run(
                        self.dinov2_duplicate.stage,
                        self.dinov2_duplicate.adapter_id,
                        self.dinov2_duplicate.model_version,
                        "completed",
                        None,
                        notes=[f"Embedding dimensions: {len(document.visual_embedding)}."],
                    )
                )
            except Exception as exc:
                document.model_runs.append(
                    self._model_run(
                        self.dinov2_duplicate.stage,
                        self.dinov2_duplicate.adapter_id,
                        self.dinov2_duplicate.model_version,
                        "failed_fallback_used",
                        None,
                        notes=[str(exc), "Perceptual-hash duplicate screening remained active."],
                    )
                )
        document.duplicate_candidates = self._duplicate_candidates(document)
        duplicate_risks = self._duplicate_risks(document)
        document.processing_warnings = self._processing_warnings(document)
        document.risk_indicators = [
            *validation_risks,
            *tamper_risks,
            *duplicate_risks,
        ]
        self._run_anomaly_models(document)
        self._transition(document, VisionProcessingStatus.validation_completed, "Completed validation and risk screening.")

        document.evidence_chunks = self._chunks(document)
        self._enrich_field_confidence(document, document.risk_indicators)
        vision_analytics_service.refresh_document(document)
        document.graph = self._graph(document)
        self._index_document(document)
        self._transition(document, VisionProcessingStatus.indexed, "Indexed structure-aware chunks and graph projection.")

        previous_review_id = document.metadata.pop("_reprocess_review_id", None)
        if self._requires_review(document):
            document.review_task = self._create_review_task(
                document,
                review_id=str(previous_review_id) if previous_review_id else None,
            )
            self.review_tasks[document.review_task.review_id] = document.review_task
            self._transition(document, VisionProcessingStatus.review_required, "Document routed to human review.")
        else:
            self._transition(document, VisionProcessingStatus.completed, "Document completed without mandatory review.")
        vision_analytics_service.refresh_document(document)
        self._persist_document(document)

    def _reset_for_reprocess(
        self,
        document: VisionDocumentRecord,
        metadata: dict[str, Any],
        actor: str,
    ) -> None:
        previous_review_id = document.review_task.review_id if document.review_task else None
        if previous_review_id:
            self.review_tasks.pop(previous_review_id, None)
        self.retrieval_index.remove_document(document.document_id)
        document.status = VisionProcessingStatus.queued
        document.parsed_text = ""
        document.page_count = 1
        document.page_artifacts = []
        document.quality = []
        document.ocr_words = []
        document.layout_regions = []
        document.tables = []
        document.visual_detections = []
        document.tamper_analysis = []
        document.extracted_fields = []
        document.confidence_summary = type(document.confidence_summary)()
        document.field_completeness = type(document.field_completeness)()
        document.language_analysis = type(document.language_analysis)()
        document.processing_warnings = []
        document.risk_indicators = []
        document.duplicate_candidates = []
        document.anomaly_analysis = []
        document.evidence_chunks = []
        document.graph = VisionGraphProjection()
        document.model_runs = []
        document.model_routing = {}
        document.review_task = None
        document.perceptual_hash = None
        document.visual_embedding = None
        document.visual_embedding_model = None
        document.metadata = metadata
        document.data_version += 1
        document.analytics_cache_version += 1
        document.updated_at = utc_now_iso()
        self._audit(
            document,
            "document_reprocess_queued",
            "Document queued for in-place reprocessing.",
            actor=actor,
            metadata={
                "document_id_reused": True,
                "review_id_reused": previous_review_id,
                "reprocess_count": metadata["reprocess_count"],
            },
        )

    def _run_selective_fallbacks(self, document: VisionDocumentRecord) -> None:
        requested = {
            str(item.get("model_id"))
            for item in document.model_routing.get("fallback_models", [])
            if isinstance(item, dict)
        }
        reasons = ", ".join(document.model_routing.get("reasons", [])) or "selective routing"

        if "PaddleOCR-VL-1.6" in requested:
            status = self.paddleocr_vl.status()
            if status.status == "active":
                try:
                    regions, tables, blocks = self.paddleocr_vl.parse(document)
                    existing_regions = {
                        (
                            region.page_number,
                            region.region_type,
                            region.text,
                            region.bounding_box.x1,
                            region.bounding_box.y1,
                            region.bounding_box.x2,
                            region.bounding_box.y2,
                        )
                        for region in document.layout_regions
                    }
                    document.layout_regions.extend(
                        region
                        for region in regions
                        if (
                            region.page_number,
                            region.region_type,
                            region.text,
                            region.bounding_box.x1,
                            region.bounding_box.y1,
                            region.bounding_box.x2,
                            region.bounding_box.y2,
                        )
                        not in existing_regions
                    )
                    document.tables.extend(tables)
                    document.model_routing["advanced_parsing"] = {
                        "model": self.paddleocr_vl.model_version,
                        "blocks": blocks,
                        "confidence_available": False,
                    }
                    document.model_runs.append(
                        self._model_run(
                            self.paddleocr_vl.stage,
                            self.paddleocr_vl.adapter_id,
                            self.paddleocr_vl.model_version,
                            "completed",
                            None,
                            routing_reason=reasons,
                            notes=["PaddleOCR-VL does not expose calibrated block confidence."],
                        )
                    )
                except Exception as exc:
                    document.model_runs.append(
                        self._model_run(
                            self.paddleocr_vl.stage,
                            self.paddleocr_vl.adapter_id,
                            self.paddleocr_vl.model_version,
                            "failed",
                            None,
                            routing_reason=reasons,
                            notes=[str(exc)],
                        )
                    )
            else:
                document.model_runs.append(
                    self._model_run(
                        self.paddleocr_vl.stage,
                        self.paddleocr_vl.adapter_id,
                        self.paddleocr_vl.model_version,
                        f"skipped_{status.status}",
                        None,
                        routing_reason=reasons,
                        notes=[status.reason],
                    )
                )

        if "Donut" in requested:
            status = self.donut_extractor.status()
            if status.status == "active":
                try:
                    output = self.donut_extractor.extract_json(document)
                    disagreements = self._donut_disagreements(document, output)
                    document.model_routing["donut_comparison"] = {
                        "model": self.donut_extractor.model_version,
                        "output": output,
                        "disagreements": disagreements,
                        "primary_fields_overwritten": False,
                    }
                    document.model_runs.append(
                        self._model_run(
                            self.donut_extractor.stage,
                            self.donut_extractor.adapter_id,
                            self.donut_extractor.model_version,
                            "completed",
                            None,
                            routing_reason=reasons,
                            notes=[f"Observed {len(disagreements)} field disagreement(s)."],
                        )
                    )
                except Exception as exc:
                    document.model_runs.append(
                        self._model_run(
                            self.donut_extractor.stage,
                            self.donut_extractor.adapter_id,
                            self.donut_extractor.model_version,
                            "failed",
                            None,
                            routing_reason=reasons,
                            notes=[str(exc)],
                        )
                    )
            else:
                document.model_runs.append(
                    self._model_run(
                        self.donut_extractor.stage,
                        self.donut_extractor.adapter_id,
                        self.donut_extractor.model_version,
                        f"skipped_{status.status}",
                        None,
                        routing_reason=reasons,
                        notes=[status.reason],
                    )
                )

    def _donut_disagreements(
        self,
        document: VisionDocumentRecord,
        output: dict[str, Any],
    ) -> list[dict[str, str]]:
        flattened: dict[str, str] = {}

        def visit(prefix: str, value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(f"{prefix}.{key}" if prefix else str(key), child)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(f"{prefix}.{index}", child)
            elif value is not None:
                flattened[prefix.lower().replace(" ", "_")] = str(value)

        visit("", output)
        disagreements: list[dict[str, str]] = []
        for field in document.extracted_fields:
            match = next(
                (
                    value
                    for key, value in flattened.items()
                    if key.rsplit(".", 1)[-1] == field.name.lower()
                ),
                None,
            )
            if match is not None and self._normalize(match) != self._normalize(field.value):
                disagreements.append(
                    {
                        "field_name": field.name,
                        "primary_value": field.value,
                        "donut_value": match,
                    }
                )
        return disagreements

    def _run_contract_extraction(self, document: VisionDocumentRecord) -> None:
        status = self.contract_extractor.status()
        if status.status != "active":
            document.model_runs.append(
                self._model_run(
                    self.contract_extractor.stage,
                    self.contract_extractor.adapter_id,
                    self.contract_extractor.model_version,
                    f"skipped_{status.status}",
                    None,
                    notes=[
                        status.reason,
                        "Deterministic evidence-localized clause detection remains active.",
                    ],
                )
            )
            return
        try:
            learned = self.contract_extractor.extract_fields(document)
        except Exception as exc:
            document.model_runs.append(
                self._model_run(
                    self.contract_extractor.stage,
                    self.contract_extractor.adapter_id,
                    self.contract_extractor.model_version,
                    "failed_fallback_used",
                    None,
                    notes=[str(exc)],
                )
            )
            return
        learned_names = {field.name for field in learned}
        retained = [
            field
            for field in document.extracted_fields
            if field.name not in CONTRACT_CLAUSE_FIELDS or field.name not in learned_names
        ]
        document.extracted_fields = self._dedupe_fields([*retained, *learned])
        document.model_runs.append(
            self._model_run(
                self.contract_extractor.stage,
                self.contract_extractor.adapter_id,
                self.contract_extractor.model_version,
                "completed",
                self._mean_available_field_confidence(learned),
                notes=["Clause scores are uncalibrated and are not legal judgements."],
            )
        )

    def _document_type_hint(
        self,
        metadata: dict[str, Any],
    ) -> VisionDocumentType | None:
        raw_value = metadata.get("document_type_hint")
        if raw_value in {None, "", "auto", "unknown"}:
            return None
        normalized = str(raw_value).strip().lower().replace(" ", "_")
        aliases = {
            "purchaseorder": "purchase_order",
            "po": "purchase_order",
            "bank": "bank_document",
            "bank_statement": "bank_document",
        }
        try:
            return VisionDocumentType(aliases.get(normalized, normalized))
        except ValueError:
            return None

    def _classify_document(self, text: str, filename: str) -> VisionDocumentType:
        haystack = f"{filename}\n{text}".lower()
        if "receipt" in haystack:
            return VisionDocumentType.receipt
        if "insurance" in haystack or "policy number" in haystack:
            return VisionDocumentType.insurance_document
        form_signal = re.search(
            r"\b(?:application|contact|intake|registration|request)\s+form\b|"
            r"\bform\s+(?:name|number|title)\b",
            haystack,
        )
        if form_signal or "checkbox" in haystack or "[x]" in haystack or "[ ]" in haystack:
            return VisionDocumentType.form
        if "report" in haystack and "invoice" not in haystack:
            return VisionDocumentType.report
        if "bank statement" in haystack or "account number" in haystack:
            return VisionDocumentType.bank_document
        doc_type, _ = self.classifier.classify(text, filename)
        mapping = {
            "invoice": VisionDocumentType.invoice,
            "purchase_order": VisionDocumentType.purchase_order,
            "contract": VisionDocumentType.contract,
            "bank_change_request": VisionDocumentType.bank_document,
            "bank_statement": VisionDocumentType.bank_document,
        }
        return mapping.get(doc_type.value, VisionDocumentType.unknown)

    def _extract_fields(self, document: VisionDocumentRecord) -> list[ExtractedVisionField]:
        fields = deterministic_schema_extractor.extract(document)
        if document.document_type == VisionDocumentType.contract:
            fields.extend(self._clause_fields(document))
        return self._dedupe_fields(fields)

    def _clause_fields(self, document: VisionDocumentRecord) -> list[ExtractedVisionField]:
        clauses = {
            "parties": ("between the parties", "by and between", "the parties agree"),
            "agreement_date": ("agreement date", "dated as of", "made on"),
            "effective_date": ("effective date", "effective as of"),
            "expiry_date": ("expiry date", "expiration date", "expires on"),
            "governing_law": ("governing law", "governed by the laws"),
            "renewal_term": ("renewal term", "automatic renewal", "auto-renew"),
            "termination_for_convenience": (
                "termination for convenience",
                "terminate for convenience",
            ),
            "notice_period": ("notice period", "days' notice", "days notice"),
            "anti_assignment": ("anti-assignment", "may not assign", "assignment consent"),
            "exclusivity": ("exclusivity", "exclusive supplier", "exclusive right"),
            "non_compete": ("non-compete", "non compete", "covenant not to compete"),
            "audit_rights": ("audit rights", "right to audit", "inspect the records"),
            "insurance": ("insurance coverage", "maintain insurance", "insurance policy"),
            "cap_on_liability": ("cap on liability", "liability shall not exceed"),
            "uncapped_liability": ("uncapped liability", "unlimited liability"),
            "perpetual_licence": ("perpetual licence", "perpetual license"),
            "revenue_sharing": ("revenue sharing", "share of revenue"),
            "change_of_control": ("change of control", "change in control"),
            "confidentiality": ("confidentiality", "non-disclosure", "confidential information"),
            "indemnification": ("indemnification", "indemnify", "hold harmless"),
            "payment_terms": ("payment terms", "payment shall be due", "net 30", "net 60"),
        }
        lower = document.parsed_text.lower()
        fields: list[ExtractedVisionField] = []
        for name, keywords in clauses.items():
            for keyword in keywords:
                index = lower.find(keyword)
                if index >= 0:
                    snippet = document.parsed_text[max(0, index - 90) : index + 180].strip()
                    fields.append(self._field(document, name, snippet or keyword, snippet or keyword))
                    break
        return fields

    def _validation_risks(
        self,
        document: VisionDocumentRecord,
        metadata: dict[str, Any],
    ) -> list[VisionRiskIndicator]:
        risks: list[VisionRiskIndicator] = []
        fields = self._field_map(document)
        subtotal = self._money(fields.get("subtotal"))
        tax = self._money(fields.get("tax"))
        discount = self._money(fields.get("discount")) or 0.0
        total = self._money(fields.get("total_amount"))
        if subtotal is not None and tax is not None and total is not None:
            expected = subtotal + tax - discount
            tolerance = max(1.0, abs(total) * 0.01)
            if abs(expected - total) > tolerance:
                risks.append(
                    self._risk(
                        document,
                        title="Amount Arithmetic Mismatch",
                        severity=VisionRiskSeverity.high,
                        category="validation",
                        description="The document contains suspicious indicators and requires human review.",
                        recommended_action="Review subtotal, tax, discount, and total before approval.",
                        source_text=f"subtotal + tax - discount = {expected:.2f}; total = {total:.2f}",
                        field_names=["subtotal", "tax", "discount", "total_amount"],
                    )
                )
        invoice_date = self._date(fields.get("invoice_date"))
        due_date = self._date(fields.get("due_date"))
        if invoice_date and due_date and due_date < invoice_date:
            risks.append(
                self._risk(
                    document,
                    title="Due Date Before Invoice Date",
                    severity=VisionRiskSeverity.medium,
                    category="validation",
                    description="The document contains suspicious indicators and requires human review.",
                    recommended_action="Verify invoice and due dates.",
                    source_text=f"invoice_date={fields.get('invoice_date')}; due_date={fields.get('due_date')}",
                    field_names=["invoice_date", "due_date"],
                )
            )
        approved_bank = metadata.get("approved_bank_identifier") or metadata.get("approved_bank_account")
        bank_identifier = fields.get("bank_account_number") or fields.get("IFSC_or_bank_code")
        if approved_bank and bank_identifier and self._normalize(approved_bank) != self._normalize(bank_identifier):
            risks.append(
                self._risk(
                    document,
                    title="Bank Detail Mismatch",
                    severity=VisionRiskSeverity.critical,
                    category="cross_document",
                    description="The document contains suspicious indicators and requires human review.",
                    recommended_action="Pause payment until vendor bank details are independently verified.",
                    source_text=f"document={self._mask_account(bank_identifier)}; approved={self._mask_account(str(approved_bank))}",
                    field_names=["bank_account_number", "IFSC_or_bank_code"],
                )
            )
        required_signature = metadata.get("required_signature") is True
        if required_signature and not any(item.detection_class == "signature" for item in document.visual_detections):
            risks.append(
                self._risk(
                    document,
                    title="Required Signature Missing",
                    severity=VisionRiskSeverity.high,
                    category="visual_validation",
                    description="The document contains suspicious indicators and requires human review.",
                    recommended_action="Request a signed copy or reviewer confirmation.",
                    source_text="No signature visual cue detected by the local adapter.",
                )
            )
        tax_id = fields.get("vendor_tax_id")
        if tax_id and not re.fullmatch(r"[A-Z0-9-]{5,30}", tax_id.strip(), re.IGNORECASE):
            risks.append(
                self._risk(
                    document,
                    title="Vendor Tax ID Format Review",
                    severity=VisionRiskSeverity.medium,
                    category="format_validation",
                    description="The document contains suspicious indicators and requires human review.",
                    recommended_action="Verify vendor tax identifier format.",
                    source_text=tax_id,
                    field_names=["vendor_tax_id"],
                )
            )
        contract_terms = metadata.get("contract_payment_terms")
        payment_terms = fields.get("payment_terms")
        if contract_terms and payment_terms and self._normalize(contract_terms) != self._normalize(payment_terms):
            risks.append(
                self._risk(
                    document,
                    title="Payment Terms Conflict",
                    severity=VisionRiskSeverity.medium,
                    category="cross_document",
                    description="The document contains suspicious indicators and requires human review.",
                    recommended_action="Compare invoice terms against the governing contract.",
                    source_text=f"document={payment_terms}; contract={contract_terms}",
                    field_names=["payment_terms"],
                )
            )
        return risks

    def _tamper_risks(self, document: VisionDocumentRecord) -> list[VisionRiskIndicator]:
        risks: list[VisionRiskIndicator] = []
        threshold = settings.tamper_screening_review_threshold
        for analysis in document.tamper_analysis:
            score = analysis.tampering_probability or analysis.screening_score
            regions = [
                region for region in analysis.suspicious_regions if region.screening_score >= threshold
            ]
            if score < threshold or not regions:
                continue
            evidence = [
                self._evidence(
                    document,
                    f"Possible visual inconsistency: {region.manipulation_class}",
                    analysis.model_name,
                    region.screening_score,
                    bounding_box=region.bounding_box,
                    page_number=analysis.page_number,
                    metadata={
                        "page_number": analysis.page_number,
                        "heatmap_path": analysis.heatmap_path,
                        "calibrated": analysis.calibrated,
                        "screening_only": True,
                    },
                )
                for region in regions[:6]
            ]
            risks.append(
                self._risk(
                    document,
                    title="Possible Visual Manipulation",
                    severity=VisionRiskSeverity.high,
                    category="tamper_screening",
                    description=(
                        "Measured visual inconsistencies crossed the configured screening threshold; "
                        "this is not a finding of forgery."
                    ),
                    recommended_action=(
                        "Inspect the heatmap and source regions, then obtain human forensic verification."
                    ),
                    source_text=analysis.explanation,
                    evidence=evidence,
                )
            )
        return risks

    def _contract_risks(self, document: VisionDocumentRecord) -> list[VisionRiskIndicator]:
        if document.document_type != VisionDocumentType.contract:
            return []
        fields = {field.name: field for field in document.extracted_fields}
        policies = {
            "uncapped_liability": (
                "Uncapped Liability Clause Review",
                VisionRiskSeverity.high,
                "Review liability scope, exceptions, and organisational policy with qualified counsel.",
            ),
            "perpetual_licence": (
                "Perpetual Licence Clause Review",
                VisionRiskSeverity.high,
                "Review licence duration, termination effects, and approved contracting policy.",
            ),
            "renewal_term": (
                "Renewal Clause Review",
                VisionRiskSeverity.medium,
                "Review renewal dates, notice windows, and ownership before the deadline.",
            ),
            "non_compete": (
                "Non-Compete Clause Review",
                VisionRiskSeverity.medium,
                "Review scope, duration, and applicable policy with qualified counsel.",
            ),
            "exclusivity": (
                "Exclusivity Clause Review",
                VisionRiskSeverity.medium,
                "Review exclusivity scope and commercial approval requirements.",
            ),
        }
        risks: list[VisionRiskIndicator] = []
        for field_name, (title, severity, action) in policies.items():
            field = fields.get(field_name)
            if not field:
                continue
            risks.append(
                self._risk(
                    document,
                    title=title,
                    severity=severity,
                    category="contract_clause_review",
                    description=(
                        "A contract clause matched a configured review category. "
                        "This is an evidence-location aid, not a legal judgement."
                    ),
                    recommended_action=action,
                    source_text=field.evidence.source_text,
                    field_names=[field_name],
                )
            )
        return risks

    def _duplicate_candidates(self, document: VisionDocumentRecord) -> list[DuplicateCandidate]:
        self._load_all_documents()
        candidates: list[DuplicateCandidate] = []
        exact_key = f"{document.tenant_id}:{document.file_hash}"
        existing_id = self._hash_index.get(exact_key)
        if existing_id and existing_id != document.document_id:
            existing = self.documents.get(existing_id) or self._load_document(existing_id)
            if existing and existing.tenant_id == document.tenant_id:
                candidates.append(
                    DuplicateCandidate(
                        status="exact_duplicate",
                        matched_document_id=existing_id,
                        matched_filename=existing.filename,
                        similarity=1.0,
                        evidence=[
                            self._evidence(
                                document,
                                "Exact SHA-256 content hash match.",
                                "sha256-duplicate-detector",
                                1.0,
                                metadata={"matched_document_id": existing_id, "algorithm": "SHA-256"},
                            )
                        ],
                        reason="The source bytes have the same SHA-256 hash.",
                    )
                )

        for existing in self.documents.values():
            if (
                existing.document_id == document.document_id
                or existing.tenant_id != document.tenant_id
                or existing.document_id == existing_id
            ):
                continue
            dhash_similarity = perceptual_similarity(
                document.perceptual_hash,
                existing.perceptual_hash,
            )
            dino_similarity = cosine_similarity(
                document.visual_embedding,
                existing.visual_embedding,
            )
            dhash_match = (
                dhash_similarity is not None
                and dhash_similarity >= settings.visual_duplicate_threshold
            )
            dino_match = (
                dino_similarity is not None
                and dino_similarity >= settings.dinov2_duplicate_threshold
            )
            if not dhash_match and not dino_match:
                continue
            measured = [
                value for value in (dhash_similarity, dino_similarity) if value is not None
            ]
            similarity = max(measured)
            evidence: list[VisionEvidence] = []
            if dhash_similarity is not None:
                evidence.append(
                    self._evidence(
                        document,
                        f"Document-image dHash similarity: {dhash_similarity:.4f}",
                        "document-dhash-detector",
                        dhash_similarity,
                        metadata={
                            "matched_document_id": existing.document_id,
                            "algorithm": "multi-page-dHash",
                            "threshold": settings.visual_duplicate_threshold,
                            "crossed_threshold": dhash_match,
                        },
                    )
                )
            if dino_similarity is not None:
                evidence.append(
                    self._evidence(
                        document,
                        f"DINOv2 document-image cosine similarity: {dino_similarity:.4f}",
                        self.dinov2_duplicate.adapter_id,
                        max(0.0, dino_similarity),
                        metadata={
                            "matched_document_id": existing.document_id,
                            "algorithm": "DINOv2-cosine",
                            "threshold": settings.dinov2_duplicate_threshold,
                            "crossed_threshold": dino_match,
                            "embedding_model": document.visual_embedding_model,
                        },
                    )
                )
            candidates.append(
                DuplicateCandidate(
                    status="probable_duplicate",
                    matched_document_id=existing.document_id,
                    matched_filename=existing.filename,
                    similarity=round(similarity, 4),
                    evidence=evidence,
                    reason=(
                        "The rendered page images crossed a configured visual-similarity threshold "
                        "using perceptual hash, DINOv2, or both."
                    ),
                )
            )

        semantic_key = self._semantic_key(document)
        if semantic_key:
            matched_id = self._semantic_index.get(f"{document.tenant_id}:{semantic_key}")
            if matched_id and matched_id != document.document_id:
                existing = self.documents.get(matched_id) or self._load_document(matched_id)
                similarity, matches, compared = self._structured_duplicate_similarity(document, existing)
                if (
                    existing
                    and existing.tenant_id == document.tenant_id
                    and compared >= 2
                    and similarity >= 0.8
                    and not any(item.matched_document_id == matched_id for item in candidates)
                ):
                    candidates.append(
                        DuplicateCandidate(
                            status="probable_duplicate",
                            matched_document_id=matched_id,
                            matched_filename=existing.filename,
                            similarity=round(similarity, 4),
                            evidence=[
                                self._evidence(
                                    document,
                                    f"Matching structured fields: {', '.join(matches)}",
                                    "structured-duplicate-detector",
                                    similarity,
                                    metadata={
                                        "matched_document_id": matched_id,
                                        "matched_fields": matches,
                                        "compared_fields": compared,
                                    },
                                )
                            ],
                            reason="Identity and financial fields agree across the two documents.",
                        )
                    )

        related_match = self._related_document(document)
        if related_match:
            related, similarity = related_match
            if not any(candidate.matched_document_id == related.document_id for candidate in candidates):
                candidates.append(
                    DuplicateCandidate(
                        status="related_document",
                        matched_document_id=related.document_id,
                        matched_filename=related.filename,
                        similarity=round(similarity, 4),
                        evidence=[
                            self._evidence(
                                document,
                                f"Normalized token Jaccard similarity: {similarity:.4f}",
                                "token-jaccard-related-document-detector",
                                similarity,
                                metadata={
                                    "matched_document_id": related.document_id,
                                    "threshold": settings.related_document_threshold,
                                },
                            )
                        ],
                        reason="The document text crossed the configured related-document threshold.",
                    )
                )
        if not candidates:
            candidates.append(
                DuplicateCandidate(
                    status="no_duplicate",
                    similarity=0.0,
                    reason="No measured duplicate signal crossed a configured threshold.",
                )
            )
        return candidates

    def _duplicate_risks(self, document: VisionDocumentRecord) -> list[VisionRiskIndicator]:
        risks = []
        for candidate in document.duplicate_candidates:
            if candidate.status in {"exact_duplicate", "probable_duplicate"}:
                risks.append(
                    self._risk(
                        document,
                        title="Duplicate Document Review",
                        severity=VisionRiskSeverity.high if candidate.status == "exact_duplicate" else VisionRiskSeverity.medium,
                        category="duplicate_detection",
                        description="The document contains suspicious indicators and requires human review.",
                        recommended_action="Compare duplicate match before approval.",
                        source_text=candidate.reason,
                        extra_evidence=candidate.evidence,
                    )
                )
        return risks

    def _run_anomaly_models(self, document: VisionDocumentRecord) -> None:
        self._load_all_documents()
        corpus = [
            item for item in self.documents.values() if item.tenant_id == document.tenant_id
        ]
        unsupervised = self.isolation_forest.analyze(document, corpus)
        document.anomaly_analysis.append(unsupervised)
        document.model_runs.append(
            self._model_run(
                self.isolation_forest.stage,
                self.isolation_forest.adapter_id,
                self.isolation_forest.model_version,
                unsupervised.status,
                unsupervised.anomaly_score,
                notes=[
                    f"Corpus samples: {unsupervised.sample_count}; minimum: {settings.anomaly_min_samples}.",
                    "Anomaly percentile is an empirical rank, not a fraud probability.",
                ],
            )
        )
        if (
            unsupervised.status == "completed"
            and unsupervised.anomaly_score is not None
            and unsupervised.anomaly_score >= settings.anomaly_review_percentile
        ):
            evidence = self._evidence(
                document,
                "Unsupervised financial and document feature anomaly.",
                self.isolation_forest.adapter_id,
                unsupervised.anomaly_score,
                metadata={
                    "features": unsupervised.features,
                    "explanations": unsupervised.explanations,
                    "sample_count": unsupervised.sample_count,
                    "score_semantics": "empirical_anomaly_percentile_not_probability",
                },
            )
            document.risk_indicators.append(
                self._risk(
                    document,
                    title="Unusual Document Pattern",
                    severity=VisionRiskSeverity.medium,
                    category="unsupervised_anomaly",
                    description=(
                        "Measured document and financial features are unusual relative to this "
                        "organisation's processed corpus."
                    ),
                    recommended_action="Review the contributing features and supporting documents.",
                    source_text="Isolation Forest empirical anomaly percentile crossed the review threshold.",
                    evidence=[evidence],
                )
            )

        supervised_status = self.xgboost_risk.status()
        if supervised_status.status != "active":
            document.model_runs.append(
                self._model_run(
                    self.xgboost_risk.stage,
                    self.xgboost_risk.adapter_id,
                    self.xgboost_risk.model_version,
                    f"skipped_{supervised_status.status}",
                    None,
                    notes=[supervised_status.reason],
                )
            )
            return
        try:
            supervised = self.xgboost_risk.analyze(unsupervised.features)
        except Exception as exc:
            document.model_runs.append(
                self._model_run(
                    self.xgboost_risk.stage,
                    self.xgboost_risk.adapter_id,
                    self.xgboost_risk.model_version,
                    "failed",
                    None,
                    notes=[str(exc)],
                )
            )
            return
        document.anomaly_analysis.append(supervised)
        document.model_runs.append(
            self._model_run(
                self.xgboost_risk.stage,
                self.xgboost_risk.adapter_id,
                self.xgboost_risk.model_version,
                supervised.status,
                supervised.supervised_probability,
                notes=["XGBoost prediction is uncalibrated until validation calibration is supplied."],
            )
        )
        if (
            supervised.supervised_probability is not None
            and supervised.supervised_probability >= settings.xgboost_review_threshold
        ):
            evidence = self._evidence(
                document,
                "Supervised risk-model review signal.",
                self.xgboost_risk.adapter_id,
                supervised.supervised_probability,
                metadata={
                    "features": supervised.features,
                    "shap_explanations": supervised.explanations,
                    "calibrated": supervised.calibrated,
                },
            )
            document.risk_indicators.append(
                self._risk(
                    document,
                    title="Supervised Risk Model Review",
                    severity=VisionRiskSeverity.high,
                    category="supervised_anomaly",
                    description="The trained risk model crossed the configured human-review threshold.",
                    recommended_action="Review SHAP feature contributions and source evidence.",
                    source_text="XGBoost review score crossed the configured threshold.",
                    evidence=[evidence],
                )
            )

    def _processing_warnings(
        self,
        document: VisionDocumentRecord,
    ) -> list[VisionRiskIndicator]:
        warnings = []
        if not document.ocr_words:
            warnings.append(
                self._risk(
                    document,
                    title="No OCR Text Available",
                    severity=VisionRiskSeverity.high,
                    category="processing_warning",
                    description="No reliable OCR text was available for structured processing.",
                    recommended_action="Run a production OCR adapter or request a clearer scan.",
                    source_text="No OCR words were extracted by the local adapter.",
                )
            )
        low_words = [word for word in document.ocr_words if word.review_required]
        if low_words:
            warnings.append(
                self._risk(
                    document,
                    title="Low OCR Confidence Words",
                    severity=VisionRiskSeverity.medium,
                    category="processing_warning",
                    description="Some OCR tokens fell below the configured reliability threshold.",
                    recommended_action="Review low-confidence words and consider advanced parsing.",
                    source_text=" ".join(word.text for word in low_words[:16]),
                    evidence=[
                        self._word_evidence(document, word)
                        for word in low_words[:8]
                    ],
                )
            )
        if document.model_routing.get("fallback_required"):
            warnings.append(
                self._risk(
                    document,
                    title="Advanced Parser Fallback Requested",
                    severity=VisionRiskSeverity.medium,
                    category="processing_warning",
                    description="One or more measured processing conditions requested an advanced parser.",
                    recommended_action="Enable or run the configured fallback parser for this document.",
                    source_text=", ".join(document.model_routing.get("reasons", [])),
                )
            )
        if document.language_analysis.mixed_language_routing:
            warnings.append(
                self._risk(
                    document,
                    title="Multiple-language uncertainty",
                    severity=VisionRiskSeverity.medium,
                    category="processing_warning",
                    description=(
                        "Cleaned OCR text contains a second script above the configured "
                        "character-ratio and confidence thresholds."
                    ),
                    recommended_action=(
                        "Review the detected-language evidence and run multilingual OCR "
                        "only if the source document confirms both language groups."
                    ),
                    source_text=document.language_analysis.sample_text,
                    rule_id="processing_warning.mixed_languages_detected",
                )
            )
        if not document.field_completeness.schema_available:
            warnings.append(
                self._risk(
                    document,
                    title="Expected Schema Unavailable",
                    severity=VisionRiskSeverity.low,
                    category="processing_warning",
                    description="No expected-field schema is registered for this document type.",
                    recommended_action="Select a supported document type or review the document manually.",
                    source_text=document.document_type.value,
                )
            )
        return warnings

    def _chunks(self, document: VisionDocumentRecord) -> list[VisionEvidenceChunk]:
        chunks: list[VisionEvidenceChunk] = []
        for region in document.layout_regions:
            if not region.text.strip():
                continue
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(
                        document,
                        region.page_number,
                        "layout",
                        region.text,
                    ),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=region.page_number,
                    section=region.region_type,
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=[region.bounding_box],
                    source_type=(
                        "form_section" if region.region_type == "form region" else "paragraph"
                    ),
                    model_name=region.model_name,
                    model_version=region.model_version,
                    text=region.text,
                    confidence=region.confidence,
                )
            )
        for field in document.extracted_fields:
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(
                        document,
                        field.evidence.page_number,
                        f"field:{field.name}",
                        field.reviewed_value or field.value,
                    ),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=field.evidence.page_number,
                    section=field.name,
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=[field.evidence.bounding_box] if field.evidence.bounding_box else [],
                    source_type="field",
                    review_status=field.reviewer_status,
                    model_name=field.model_name,
                    model_version=field.model_version,
                    text=f"{field.name}: {field.reviewed_value or field.value}",
                    confidence=field.confidence or 0.0,
                )
            )
        for table in document.tables:
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(
                        document,
                        table.page_number,
                        "table",
                        table.markdown,
                    ),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=table.page_number,
                    section="table",
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=[table.bounding_box],
                    source_type="table",
                    model_name=table.model_name,
                    model_version=table.model_version,
                    text=table.markdown,
                    confidence=table.confidence,
                )
            )
        retained: dict[tuple[int, str, str], VisionEvidenceChunk] = {}
        for chunk in chunks:
            key = (chunk.page_number, chunk.source_type, self._normalize(chunk.text))
            current = retained.get(key)
            if current is None or chunk.confidence > current.confidence:
                retained[key] = chunk
        return list(retained.values())

    def _chunk_id(
        self,
        document: VisionDocumentRecord,
        page_number: int,
        source_type: str,
        text: str,
    ) -> str:
        identity = hash_document(
            (
                f"{document.document_id}|{page_number}|{source_type}|"
                f"{self._normalize(text)}"
            ).encode()
        )
        return f"vis-chk-{identity[:16]}"

    def _graph(self, document: VisionDocumentRecord) -> VisionGraphProjection:
        return evidence_graph_builder.build(document)

    def _legacy_graph(self, document: VisionDocumentRecord) -> VisionGraphProjection:
        nodes: dict[str, dict[str, Any]] = {
            document.document_id: {
                "id": document.document_id,
                "label": document.filename,
                "type": "document",
                "tenant_id": document.tenant_id,
                "status": document.status.value,
            }
        }
        edges: list[dict[str, Any]] = []
        fields = {field.name: field for field in document.extracted_fields}

        def add_field_edge(
            field_name: str,
            node_type: str,
            relationship: str,
            *,
            reverse: bool = False,
            masked: bool = False,
        ) -> str | None:
            field = fields.get(field_name)
            if not field:
                return None
            value = field.reviewed_value or field.value
            node_value = self._mask_sensitive(field_name, value) if masked else value
            node_id = (
                f"{node_type}:{hash_document(value.encode('utf-8'))[:16]}"
                if masked
                else f"{node_type}:{self._normalize(value)}"
            )
            nodes[node_id] = {
                "id": node_id,
                "label": node_value,
                "type": node_type,
                "review_status": field.reviewer_status.value,
                "tenant_id": document.tenant_id,
            }
            source, target = (
                (node_id, document.document_id)
                if reverse
                else (document.document_id, node_id)
            )
            verified = field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
            edges.append(
                {
                    "id": f"edge:{document.document_id}:{relationship}:{node_id}",
                    "source": source,
                    "target": target,
                    "type": relationship,
                    "status": "verified" if verified else "proposed",
                    "source_document": document.document_id,
                    "source_page": field.evidence.page_number,
                    "evidence_id": field.evidence.evidence_id,
                    "evidence_text": field.evidence.source_text,
                    "bounding_boxes": [
                        field.evidence.bounding_box.model_dump(mode="json")
                    ]
                    if field.evidence.bounding_box
                    else [],
                    "confidence": field.confidence,
                    "review_status": field.reviewer_status.value,
                    "model_name": field.model_name,
                    "model_version": field.model_version,
                    "deterministic": True,
                }
            )
            return node_id

        vendor_id = add_field_edge("vendor_name", "vendor", "SUBMITTED", reverse=True)
        add_field_edge("invoice_number", "invoice", "REFERENCES")
        add_field_edge("purchase_order_number", "purchase_order", "REFERENCES")
        add_field_edge("contract_number", "contract", "GOVERNED_BY")
        bank_id = add_field_edge("bank_account_number", "bank_account", "PAID_TO", masked=True)
        if bank_id is None:
            bank_id = add_field_edge("IFSC_or_bank_code", "bank_account", "PAID_TO", masked=True)
        tax_id = add_field_edge("vendor_tax_id", "tax_identifier", "REGISTERED_WITH", masked=True)
        add_field_edge("signature_name", "approver", "APPROVED_BY")
        add_field_edge("party_name", "party", "REFERENCES")
        if vendor_id and bank_id:
            evidence_field = fields.get("bank_account_number") or fields.get("IFSC_or_bank_code")
            if evidence_field:
                edges.append(
                    {
                        "id": f"edge:{vendor_id}:PAID_TO:{bank_id}",
                        "source": vendor_id,
                        "target": bank_id,
                        "type": "PAID_TO",
                        "status": "verified"
                        if evidence_field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
                        else "proposed",
                        "source_document": document.document_id,
                        "source_page": evidence_field.evidence.page_number,
                        "evidence_id": evidence_field.evidence.evidence_id,
                        "evidence_text": evidence_field.evidence.source_text,
                        "bounding_boxes": [
                            evidence_field.evidence.bounding_box.model_dump(mode="json")
                        ]
                        if evidence_field.evidence.bounding_box
                        else [],
                        "confidence": evidence_field.confidence,
                        "review_status": evidence_field.reviewer_status.value,
                        "model_name": evidence_field.model_name,
                        "model_version": evidence_field.model_version,
                        "deterministic": True,
                    }
                )
        if vendor_id and tax_id:
            evidence_field = fields["vendor_tax_id"]
            edges.append(
                {
                    "id": f"edge:{vendor_id}:REGISTERED_WITH:{tax_id}",
                    "source": vendor_id,
                    "target": tax_id,
                    "type": "REGISTERED_WITH",
                    "status": "verified"
                    if evidence_field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
                    else "proposed",
                    "source_document": document.document_id,
                    "source_page": evidence_field.evidence.page_number,
                    "evidence_id": evidence_field.evidence.evidence_id,
                    "evidence_text": evidence_field.evidence.source_text,
                    "bounding_boxes": [
                        evidence_field.evidence.bounding_box.model_dump(mode="json")
                    ]
                    if evidence_field.evidence.bounding_box
                    else [],
                    "confidence": evidence_field.confidence,
                    "review_status": evidence_field.reviewer_status.value,
                    "model_name": evidence_field.model_name,
                    "model_version": evidence_field.model_version,
                    "deterministic": True,
                }
            )

        for duplicate in document.duplicate_candidates:
            if not duplicate.matched_document_id:
                continue
            nodes.setdefault(
                duplicate.matched_document_id,
                {
                    "id": duplicate.matched_document_id,
                    "label": duplicate.matched_filename or duplicate.matched_document_id,
                    "type": "document",
                    "tenant_id": document.tenant_id,
                },
            )
            evidence = duplicate.evidence[0] if duplicate.evidence else None
            relationship = "DUPLICATE_OF" if duplicate.status != "related_document" else "MATCHES"
            edges.append(
                {
                    "id": f"edge:{document.document_id}:{relationship}:{duplicate.matched_document_id}",
                    "source": document.document_id,
                    "target": duplicate.matched_document_id,
                    "type": relationship,
                    "status": "proposed",
                    "source_document": document.document_id,
                    "source_page": evidence.page_number if evidence else 1,
                    "evidence_id": evidence.evidence_id if evidence else None,
                    "evidence_text": evidence.source_text if evidence else duplicate.reason,
                    "bounding_boxes": [evidence.bounding_box.model_dump(mode="json")]
                    if evidence and evidence.bounding_box
                    else [],
                    "confidence": duplicate.similarity,
                    "review_status": ReviewerStatus.unverified.value,
                    "model_name": evidence.model_name if evidence else "duplicate-detector",
                    "model_version": evidence.model_version if evidence else "v1",
                    "deterministic": True,
                }
            )

        for risk in document.risk_indicators:
            if not risk.rule_id.startswith("cross_document."):
                continue
            target_id = f"trusted_reference:{self._normalize(risk.title)}"
            nodes[target_id] = {
                "id": target_id,
                "label": risk.title,
                "type": "trusted_reference",
                "tenant_id": document.tenant_id,
            }
            evidence = risk.evidence[0] if risk.evidence else None
            edges.append(
                {
                    "id": f"edge:{document.document_id}:CONFLICTS_WITH:{target_id}",
                    "source": document.document_id,
                    "target": target_id,
                    "type": "CONFLICTS_WITH",
                    "status": "proposed",
                    "source_document": document.document_id,
                    "source_page": evidence.page_number if evidence else 1,
                    "evidence_id": evidence.evidence_id if evidence else None,
                    "evidence_text": evidence.source_text if evidence else risk.title,
                    "bounding_boxes": [evidence.bounding_box.model_dump(mode="json")]
                    if evidence and evidence.bounding_box
                    else [],
                    "confidence": evidence.model_confidence if evidence else 1.0,
                    "review_status": risk.reviewer_status.value,
                    "model_name": risk.model_name,
                    "model_version": risk.model_version,
                    "deterministic": True,
                }
            )
        return VisionGraphProjection(nodes=list(nodes.values()), edges=edges)

    def _create_review_task(
        self,
        document: VisionDocumentRecord,
        *,
        review_id: str | None = None,
    ) -> VisionReviewTask:
        reasons = []
        if document.model_routing.get("fallback_required"):
            reasons.extend(document.model_routing.get("reasons", []))
        reasons.extend(risk.title for risk in document.risk_indicators if risk.severity in {VisionRiskSeverity.high, VisionRiskSeverity.critical})
        reasons.extend(
            warning.title
            for warning in document.processing_warnings
            if warning.severity in {VisionRiskSeverity.high, VisionRiskSeverity.critical}
        )
        reasons.extend(
            f"missing_required_field:{name}"
            for name in document.field_completeness.missing_required_fields
        )
        if not reasons:
            reasons.append("field_confidence_requires_review")
        severity = self._review_priority(document)
        now = utc_now_iso()
        task_data: dict[str, Any] = {
            "document_id": document.document_id,
            "priority": severity,
            "reasons": sorted(set(reasons)),
            "filename": document.filename,
            "document_type": document.document_type,
            "page_count": document.page_count,
            "confidence_status": (
                "available"
                if document.confidence_summary.confidence_available
                else "not_available"
            ),
            "confidence_value": document.confidence_summary.mean_field_confidence,
            "expected_required_fields": document.field_completeness.expected_required_count,
            "unresolved_required_fields": document.field_completeness.unresolved_required_count,
            "created_at": now,
            "updated_at": now,
        }
        if review_id:
            task_data["review_id"] = review_id
        task = VisionReviewTask(**task_data)
        event = self._audit(
            document,
            "review_task_created",
            "Human review task created from evidence-backed routing rules.",
            metadata={"review_id": task.review_id, "priority": severity.value, "reasons": task.reasons},
        )
        task.history.append(event)
        return task

    def _ensure_review_task(self, document: VisionDocumentRecord) -> None:
        if document.review_task or document.status not in {
            VisionProcessingStatus.review_required,
            VisionProcessingStatus.under_review,
        }:
            return
        document.review_task = self._create_review_task(document)
        self.review_tasks[document.review_task.review_id] = document.review_task
        self._persist_document(document)
        self._persist_review(document.review_task)

    def _requires_review(self, document: VisionDocumentRecord) -> bool:
        if not bool(self.confidence_policy.get("calibration", {}).get("calibrated")):
            return True
        if document.model_routing.get("fallback_required"):
            return True
        if not document.ocr_words:
            return True
        if (
            document.field_completeness.schema_available
            and document.field_completeness.unresolved_required_count
        ):
            return True
        if any(risk.severity in {VisionRiskSeverity.high, VisionRiskSeverity.critical} for risk in document.risk_indicators):
            return True
        if any(
            field.confidence is None
            or field.confidence < self._mandatory_review_threshold(document)
            for field in document.extracted_fields
        ):
            return True
        return False

    def _risk(
        self,
        document: VisionDocumentRecord,
        *,
        title: str,
        severity: VisionRiskSeverity,
        category: str,
        description: str,
        recommended_action: str,
        source_text: str,
        field_names: list[str] | None = None,
        evidence: list[VisionEvidence] | None = None,
        extra_evidence: list[VisionEvidence] | None = None,
        rule_id: str | None = None,
    ) -> VisionRiskIndicator:
        evidence_items = evidence or []
        if not evidence_items:
            fields = [
                field for field in document.extracted_fields if not field_names or field.name in field_names
            ]
            evidence_items.extend(field.evidence for field in fields[:6])
        if extra_evidence:
            evidence_items.extend(extra_evidence)
        if not evidence_items:
            evidence_items.append(
                self._evidence(
                    document,
                    source_text,
                    "local-risk-rule-engine",
                    1.0,
                    metadata={
                        "category": category,
                        "confidence_semantics": "deterministic_rule_condition_observed",
                    },
                )
            )
        return VisionRiskIndicator(
            title=title,
            severity=severity,
            category=self._indicator_category(category),
            description=description,
            recommended_action=recommended_action,
            evidence=evidence_items,
            rule_id=rule_id or self._rule_id(category, title),
            affected_fields=field_names or [],
        )

    def _field(self, document: VisionDocumentRecord, name: str, value: str, source_text: str) -> ExtractedVisionField:
        box = self._box_for_text(value, document.ocr_words) or self._box_for_text(source_text, document.ocr_words)
        confidence, components = self._field_confidence(value, document.ocr_words, box)
        evidence = self._evidence(
            document,
            source_text,
            "local-schema-extraction-adapter",
            confidence,
            bounding_box=box,
            metadata={
                "confidence_basis": "weighted_observed_components_not_calibrated_model_accuracy",
                "confidence_components": components,
                "confidence_policy": self.confidence_policy.get("calibration", {}),
            },
        )
        return ExtractedVisionField(
            name=name,
            schema_field=name,
            display_name=(
                schema_field.display_name
                if (schema_field := (
                    vision_schema_registry.get(document.document_type).field(name)
                    if vision_schema_registry.get(document.document_type)
                    else None
                ))
                else name.replace("_", " ").title()
            ),
            value=value,
            normalized_value=self._normalize(value),
            confidence=confidence,
            combined_confidence=confidence,
            ocr_confidence=components.get("ocr_confidence"),
            extraction_confidence=components.get("extraction_confidence"),
            validation_confidence=components.get("field_format_validity"),
            evidence=evidence,
            confidence_components=components,
            calibrated=bool(self.confidence_policy.get("calibration", {}).get("calibrated")),
            extractor="deterministic-fallback",
            required=bool(schema_field and schema_field.required),
        )

    def _evidence(
        self,
        document: VisionDocumentRecord,
        source_text: str,
        model_name: str,
        confidence: float,
        *,
        bounding_box: VisionBoundingBox | None = None,
        page_number: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VisionEvidence:
        matched_words = self._words_for_text(source_text, document.ocr_words)
        resolved_page = matched_words[0].page if matched_words else (page_number or 1)
        return VisionEvidence(
            document_id=document.document_id,
            page_number=resolved_page,
            source_text=(source_text or document.parsed_text[:400] or "No source text available.")[:700],
            bounding_box=bounding_box or self._box_for_text(source_text, document.ocr_words),
            visual_region=f"page_{resolved_page}",
            model_confidence=max(0.0, min(1.0, confidence)),
            model_name=model_name,
            model_version="v1",
            processing_timestamp=utc_now_iso(),
            metadata=metadata or {},
        )

    def _word_evidence(self, document: VisionDocumentRecord, word: OCRWord) -> VisionEvidence:
        return VisionEvidence(
            document_id=document.document_id,
            page_number=word.page,
            source_text=word.text,
            bounding_box=word.bounding_box,
            visual_region=word.line_id,
            model_confidence=word.confidence,
            model_name=word.model_name,
            model_version=word.model_version,
            processing_timestamp=utc_now_iso(),
            metadata={"reading_order": word.reading_order, "low_confidence": word.review_required},
        )

    def _audit(
        self,
        document: VisionDocumentRecord,
        event_type: str,
        message: str,
        *,
        actor: str = "system",
        metadata: dict[str, Any] | None = None,
    ) -> VisionAuditEvent:
        event = VisionAuditEvent(
            document_id=document.document_id,
            actor=actor,
            event_type=event_type,
            message=message,
            created_at=utc_now_iso(),
            metadata=metadata or {},
        )
        document.audit_events.append(event)
        document.updated_at = event.created_at
        return event

    def _transition(self, document: VisionDocumentRecord, status: VisionProcessingStatus, message: str) -> None:
        document.status = status
        self._audit(document, status.value.lower(), message)
        self._persist_document(document)

    def _model_run(
        self,
        stage: str,
        model_name: str,
        model_version: str,
        status: str,
        confidence: float | None,
        *,
        routing_reason: str | None = None,
        notes: list[str] | None = None,
    ) -> ModelRunRecord:
        now = utc_now_iso()
        return ModelRunRecord(
            stage=stage,
            model_name=model_name,
            model_version=model_version,
            status=status,
            started_at=now,
            completed_at=now,
            confidence=confidence,
            routing_reason=routing_reason,
            notes=notes or [],
        )

    def _field_matches(self, question: str, documents: list[VisionDocumentRecord]) -> list[ExtractedVisionField]:
        question_tokens = set(re.findall(r"[a-z0-9_]+", question))
        matches = []
        for document in documents:
            for field in document.extracted_fields:
                name_tokens = set(field.name.lower().split("_"))
                if name_tokens & question_tokens or field.name.lower() in question:
                    matches.append(field)
        return sorted(matches, key=lambda item: item.confidence or 0.0, reverse=True)

    def _field_citation(self, field: ExtractedVisionField) -> VisionAskCitation:
        source_document = self.documents.get(field.evidence.document_id)
        return VisionAskCitation(
            document_id=field.evidence.document_id,
            filename=source_document.filename if source_document else None,
            page_number=field.evidence.page_number,
            source_text=field.evidence.source_text,
            bounding_boxes=[field.evidence.bounding_box] if field.evidence.bounding_box else [],
            evidence_id=field.evidence.evidence_id,
            model_confidence=field.evidence.model_confidence,
            model_name=field.evidence.model_name,
            model_version=field.evidence.model_version,
            processing_timestamp=field.evidence.processing_timestamp,
            reviewer_status=field.reviewer_status,
        )

    def _requested_field_names(self, question: str) -> set[str]:
        lower = question.lower()
        aliases = {
            "organisation_name": ("organisation", "organization", "company name"),
            "address": ("address",),
            "contact_name": ("contact name", "contact person", "contact"),
            "phone_number": ("phone", "telephone"),
            "fax_number": ("fax", "fax number"),
            "email": ("email", "e-mail"),
            "form_title": ("form title", "form name"),
            "vendor_name": ("vendor", "supplier", "seller"),
            "invoice_number": ("invoice number", "invoice id"),
            "invoice_date": ("invoice date",),
            "due_date": ("due date",),
            "purchase_order_number": ("purchase order", "po number", "po reference"),
            "contract_number": ("contract number",),
            "bank_account_number": ("bank account", "account number"),
            "bank_code": ("ifsc", "swift", "routing number", "bank code"),
            "subtotal": ("subtotal",),
            "tax": ("tax",),
            "discount": ("discount",),
            "total_amount": ("total amount", "grand total", "amount due", " total"),
            "currency": ("currency",),
            "payment_terms": ("payment terms",),
            "effective_date": ("effective date",),
            "expiry_date": ("expiry date", "expiration date"),
            "governing_law": ("governing law",),
            "termination_clause": ("termination", "termination clause"),
            "notice_period": ("notice period",),
            "liability_clause": ("liability clause",),
            "confidentiality_clause": ("confidentiality", "non-disclosure"),
            "indemnification_clause": ("indemnification", "indemnity"),
        }
        return {
            name for name, terms in aliases.items() if any(term in lower for term in terms)
        }

    def _documents_for_question(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
    ) -> list[VisionDocumentRecord]:
        normalized_question = self._normalize(question)
        identity_fields = {
            "invoice_number",
            "purchase_order_number",
            "contract_number",
            "vendor_name",
        }
        matches = [
            document
            for document in documents
            if any(
                field.name in identity_fields
                and len(self._normalize(field.reviewed_value or field.value)) >= 3
                and self._normalize(field.reviewed_value or field.value) in normalized_question
                for field in document.extracted_fields
            )
        ]
        return matches or documents

    def _decimal_value(self, value: str) -> Decimal | None:
        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
        if not match:
            return None
        try:
            return Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            return None

    def _relevant_semantic_hits(self, question: str, hits: list[Any]) -> list[Any]:
        stopwords = {
            "about",
            "and",
            "are",
            "does",
            "document",
            "for",
            "from",
            "have",
            "how",
            "is",
            "of",
            "the",
            "this",
            "to",
            "what",
            "which",
            "who",
        }
        query_tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", question.lower())
            if len(token) >= 3 and token not in stopwords
        }
        advanced_embedding = (
            self.retrieval_index.embedding.adapter_id != "local-hashing-embedding"
        )
        retained = []
        for hit in hits:
            chunk_tokens = set(re.findall(r"[a-z0-9]+", hit.chunk.text.lower()))
            lexical_match = bool(query_tokens & chunk_tokens)
            if hit.score < 0.12:
                continue
            if lexical_match or (advanced_embedding and hit.score >= 0.25):
                retained.append(hit)
        return retained

    def _not_found_answer(self, *, requires_review: bool) -> VisionAskResponse:
        return VisionAskResponse(
            answer="The available documents do not contain sufficient evidence to answer this question.",
            answer_status="not_found",
            citations=[],
            confidence=0.0,
            reasoning_summary="No extracted field, chunk, or graph evidence matched the question.",
            requires_review=requires_review,
        )

    def _index_document(self, document: VisionDocumentRecord) -> None:
        self._hash_index.setdefault(
            f"{document.tenant_id}:{document.file_hash}",
            document.document_id,
        )
        semantic_key = self._semantic_key(document)
        if semantic_key:
            self._semantic_index.setdefault(
                f"{document.tenant_id}:{semantic_key}",
                document.document_id,
            )
        self.retrieval_index.upsert_document(document.document_id, document.evidence_chunks)

    def _semantic_key(self, document: VisionDocumentRecord) -> str:
        fields = self._field_map(document)
        parts = [
            fields.get("vendor_name", ""),
            fields.get("invoice_number", ""),
            fields.get("purchase_order_number", ""),
            fields.get("invoice_date", ""),
            fields.get("total_amount", ""),
        ]
        normalized = [self._normalize(part) for part in parts if part]
        return "|".join(normalized)

    def _structured_duplicate_similarity(
        self,
        document: VisionDocumentRecord,
        candidate: VisionDocumentRecord | None,
    ) -> tuple[float, list[str], int]:
        if candidate is None:
            return 0.0, [], 0
        left = self._field_map(document)
        right = self._field_map(candidate)
        weights = {
            "vendor_name": 1.5,
            "invoice_number": 2.0,
            "total_amount": 1.5,
            "invoice_date": 1.0,
            "purchase_order_number": 1.5,
        }
        compared = [name for name in weights if left.get(name) and right.get(name)]
        if not compared:
            return 0.0, [], 0
        matches = [
            name
            for name in compared
            if self._normalize(left[name]) == self._normalize(right[name])
        ]
        denominator = sum(weights[name] for name in compared)
        score = sum(weights[name] for name in matches) / denominator
        return score, matches, len(compared)

    def _related_document(
        self,
        document: VisionDocumentRecord,
    ) -> tuple[VisionDocumentRecord, float] | None:
        self._load_all_documents()
        tokens = set(re.findall(r"[a-z0-9]+", document.parsed_text.lower()))
        if not tokens:
            return None
        best: tuple[VisionDocumentRecord, float] | None = None
        for candidate in self.documents.values():
            if (
                candidate.document_id == document.document_id
                or candidate.tenant_id != document.tenant_id
                or not candidate.parsed_text
            ):
                continue
            other_tokens = set(re.findall(r"[a-z0-9]+", candidate.parsed_text.lower()))
            if not other_tokens:
                continue
            similarity = len(tokens & other_tokens) / max(1, len(tokens | other_tokens))
            if similarity >= settings.related_document_threshold and (
                best is None or similarity > best[1]
            ):
                best = (candidate, similarity)
        return best

    def _document_summary(self, document: VisionDocumentRecord) -> str:
        return (
            f"Processed {document.filename} as {document.document_type.value} with "
            f"{len(document.ocr_words)} OCR words, {len(document.extracted_fields)} fields, "
            f"{len(document.tables)} table(s), {len(document.visual_detections)} visual cue(s), "
            f"and {len(document.risk_indicators)} review indicator(s)."
        )

    def _backend_status(self, registry: dict[str, Any], model_id: str) -> str:
        for backend in registry.get("backends", []):
            if backend["model_id"] == model_id or model_id.lower() in backend["model_id"].lower():
                return backend["status"]
        return "unregistered"

    def _load_confidence_policy(self) -> dict[str, Any]:
        fallback = {
            "field_confidence_weights": {
                "ocr_confidence": 0.18,
                "layoutlmv3_probability": 0.18,
                "layout_agreement": 0.12,
                "donut_agreement": 0.10,
                "visual_parser_agreement": 0.08,
                "field_format_validity": 0.10,
                "arithmetic_consistency": 0.10,
                "cross_document_consistency": 0.08,
                "retrieval_evidence_quality": 0.06,
            },
            "review_thresholds": {
                "auto_approval_min_confidence": 0.90,
                "medium_review_min_confidence": 0.70,
                "mandatory_review_below_confidence": 0.70,
            },
            "calibration": {"calibrated": False},
        }
        path = Path(__file__).resolve().parents[3] / "config" / "vision_confidence.yml"
        if not path.exists():
            return fallback
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return {**fallback, **loaded}
        except Exception:
            return fallback

    def _ensure_state_dirs(self) -> None:
        for child in (
            "documents",
            "jobs",
            "reviews",
            "pending",
            "questions",
            "narratives",
            "reports",
        ):
            (self._state_root / child).mkdir(parents=True, exist_ok=True)

    def _load_admin_overrides(self) -> None:
        payload = self._read_json(self._state_root / "model_overrides.json") or {}
        self.model_overrides.update(
            {
                str(model_id): dict(value)
                for model_id, value in payload.items()
                if isinstance(value, dict)
            }
        )
        rule_payload = self._read_json(self._state_root / "rule_overrides.json") or {}
        self.rule_overrides.update(
            {
                str(rule_id): dict(value)
                for rule_id, value in rule_payload.items()
                if isinstance(value, dict)
            }
        )
        audit_payload = self._read_json(self._state_root / "system_audit.json") or {}
        self.system_audit_events.extend(
            dict(value)
            for value in audit_payload.get("events", [])
            if isinstance(value, dict)
        )

    def _system_audit(
        self,
        event_type: str,
        message: str,
        *,
        actor: str,
        tenant_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": f"sys-aud-{uuid4().hex[:12]}",
            "document_id": "system",
            "tenant_id": tenant_id,
            "actor": actor,
            "event_type": event_type,
            "message": message,
            "created_at": utc_now_iso(),
            "metadata": metadata or {},
        }
        self.system_audit_events.append(event)
        self._write_json(
            self._state_root / "system_audit.json",
            {"events": self.system_audit_events},
        )
        return event

    def _state_file(self, collection: str, item_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", item_id)
        return self._state_root / collection / f"{safe_id}.json"

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)

    def _read_json(self, path: Path) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _persist_document(self, document: VisionDocumentRecord) -> None:
        self.persistence.save_document(document)
        self._write_json(
            self._state_file("documents", document.document_id),
            document.model_dump(mode="json"),
        )

    def _load_document(self, document_id: str) -> VisionDocumentRecord | None:
        payload = self._read_json(self._state_file("documents", document_id))
        if not payload:
            return None
        try:
            document = VisionDocumentRecord.model_validate(payload)
        except ValueError:
            return None
        self.documents[document.document_id] = document
        if document.review_task:
            self.review_tasks[document.review_task.review_id] = document.review_task
        self._index_document(document)
        return document

    def _load_all_documents(self) -> None:
        for path in (self._state_root / "documents").glob("*.json"):
            document_id = path.stem
            if document_id not in self.documents:
                self._load_document(document_id)

    def _remove_document(self, document_id: str) -> None:
        self.retrieval_index.remove_document(document_id)
        self._state_file("documents", document_id).unlink(missing_ok=True)

    def _persist_job(self, job: dict[str, Any]) -> None:
        self.jobs[job["job_id"]] = job
        self._write_json(self._state_file("jobs", job["job_id"]), job)
        self.persistence.save_job(job)

    def _load_job(self, job_id: str) -> dict[str, Any] | None:
        payload = self._read_json(self._state_file("jobs", job_id))
        if not payload:
            return None
        self.jobs[job_id] = payload
        return payload

    def _persist_review(self, review: VisionReviewTask) -> None:
        self.review_tasks[review.review_id] = review
        self._write_json(self._state_file("reviews", review.review_id), review.model_dump(mode="json"))
        document = self.documents.get(review.document_id)
        if document:
            self.persistence.save_review(review, document.tenant_id)

    def _load_review(self, review_id: str) -> VisionReviewTask | None:
        payload = self._read_json(self._state_file("reviews", review_id))
        if not payload:
            return None
        try:
            review = VisionReviewTask.model_validate(payload)
        except ValueError:
            return None
        self.review_tasks[review.review_id] = review
        return review

    def _load_all_reviews(self) -> None:
        for path in (self._state_root / "reviews").glob("*.json"):
            review_id = path.stem
            if review_id not in self.review_tasks:
                self._load_review(review_id)

    def _remove_review(self, review_id: str) -> None:
        self._state_file("reviews", review_id).unlink(missing_ok=True)

    def _persist_pending_content(
        self,
        document_id: str,
        filename: str,
        content: bytes,
        metadata: dict[str, Any],
    ) -> None:
        pending_dir = self._state_root / "pending" / document_id
        pending_dir.mkdir(parents=True, exist_ok=True)
        (pending_dir / "content.bin").write_bytes(content)
        self._write_json(pending_dir / "metadata.json", {"filename": filename, "metadata": metadata})

    def _load_pending_content(self, document_id: str) -> tuple[str, bytes, dict[str, Any]] | None:
        pending_dir = self._state_root / "pending" / document_id
        metadata = self._read_json(pending_dir / "metadata.json")
        content_path = pending_dir / "content.bin"
        if not metadata or not content_path.exists():
            return None
        try:
            pending = (
                str(metadata.get("filename") or "uploaded-document"),
                content_path.read_bytes(),
                dict(metadata.get("metadata") or {}),
            )
        except OSError:
            return None
        self._pending_content[document_id] = pending
        return pending

    def _remove_pending_content(self, document_id: str) -> None:
        pending_dir = self._state_root / "pending" / document_id
        for filename in ("content.bin", "metadata.json"):
            (pending_dir / filename).unlink(missing_ok=True)
        try:
            pending_dir.rmdir()
        except OSError:
            pass

    def _persist_question(self, question_id: str, question: dict[str, Any]) -> None:
        self.questions[question_id] = question
        self._write_json(self._state_file("questions", question_id), question)
        self.persistence.save_question(question)

    def _persist_finance_narrative(
        self,
        narrative: VisionFinanceNarrative,
    ) -> None:
        self.finance_narratives[narrative.narrative_id] = narrative
        self._write_json(
            self._state_file("narratives", narrative.narrative_id),
            narrative.model_dump(mode="json"),
        )
        self.persistence.save_narrative(narrative)

    def _persist_report_record(self, report: VisionReportRecord) -> None:
        self.reports[report.report_id] = report
        self._write_json(
            self._state_file("reports", report.report_id),
            report.model_dump(mode="json"),
        )
        self.persistence.save_report(report.model_dump(mode="json"))

    def _artifact_source_data_version(
        self,
        documents: list[VisionDocumentRecord],
    ) -> str:
        snapshot = [
            {
                "document_id": document.document_id,
                "data_version": document.data_version,
                "status": document.status.value,
                "review_status": (
                    document.review_task.status.value
                    if document.review_task
                    else document.status.value
                ),
                "verified_fields": sorted(
                    (
                        field.name,
                        field.reviewed_value or field.value,
                        field.reviewer_status.value,
                    )
                    for field in document.extracted_fields
                    if field.reviewer_status
                    in {ReviewerStatus.accepted, ReviewerStatus.edited}
                ),
            }
            for document in sorted(documents, key=lambda item: item.document_id)
        ]
        return self._artifact_identity_hash(snapshot)

    def _artifact_identity_hash(self, payload: Any) -> str:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return hash_document(encoded)

    def _load_question(self, question_id: str) -> dict[str, Any] | None:
        payload = self._read_json(self._state_file("questions", question_id))
        if payload:
            self.questions[question_id] = payload
            self.question_feedback[question_id] = list(payload.get("feedback") or [])
        return payload

    def _persist_indexes(self) -> None:
        self._write_json(
            self._state_root / "indexes.json",
            {
                "idempotency": self._idempotency_index,
                "hash": self._hash_index,
                "semantic": self._semantic_index,
            },
        )

    def _load_indexes(self) -> None:
        payload = self._read_json(self._state_root / "indexes.json") or {}
        self._idempotency_index.update(dict(payload.get("idempotency") or {}))
        self._hash_index.update(dict(payload.get("hash") or {}))
        self._semantic_index.update(dict(payload.get("semantic") or {}))

    def _field_confidence(
        self,
        value: str,
        words: list[OCRWord],
        box: VisionBoundingBox | None,
    ) -> tuple[float, dict[str, float]]:
        weights = self.confidence_policy["field_confidence_weights"]
        matched_words = self._words_for_text(value, words)
        components = {
            "ocr_confidence": round(
                sum(word.confidence for word in matched_words) / len(matched_words),
                4,
            )
            if matched_words
            else 0.0,
            "layout_agreement": 1.0 if box and matched_words else 0.0,
            "field_format_validity": 1.0 if self._format_valid(value) else 0.0,
        }
        return self._score_confidence_components(components, weights), components

    def _enrich_field_confidence(
        self,
        document: VisionDocumentRecord,
        risks: list[VisionRiskIndicator],
    ) -> None:
        weights = self.confidence_policy["field_confidence_weights"]
        risk_evidence_ids = {
            evidence.evidence_id
            for risk in risks
            for evidence in risk.evidence
        }
        donut = document.model_routing.get("donut_comparison") or {}
        donut_output = json.dumps(donut.get("output") or {}, sort_keys=True).lower()
        donut_disagreements = {
            str(item.get("field_name"))
            for item in donut.get("disagreements", [])
            if isinstance(item, dict)
        }
        advanced = document.model_routing.get("advanced_parsing") or {}
        advanced_text = json.dumps(advanced.get("blocks") or {}, sort_keys=True).lower()
        field_map = self._field_map(document)
        arithmetic_available = all(
            field_map.get(name) for name in ("subtotal", "tax", "total_amount")
        )
        arithmetic_failed = any(risk.title == "Amount Arithmetic Mismatch" for risk in risks)
        cross_metadata_present = any(
            key in document.metadata
            for key in (
                "approved_bank_identifier",
                "approved_bank_account",
                "po_vendor",
                "purchase_order_vendor",
                "po_amount",
                "purchase_order_amount",
                "contract_payment_terms",
                "contract_effective_date",
                "contract_expiry_date",
            )
        )
        for field in document.extracted_fields:
            components = dict(field.confidence_components)
            if field.model_name.lower().startswith("layoutlm"):
                if field.confidence is not None:
                    components.setdefault("layoutlmv3_probability", field.confidence)
            normalized_value = self._normalize(field.reviewed_value or field.value)
            if donut_output and field.name.lower() in donut_output:
                components["donut_agreement"] = (
                    0.0 if field.name in donut_disagreements else 1.0
                )
            if advanced_text and normalized_value:
                components["visual_parser_agreement"] = float(
                    normalized_value in self._normalize(advanced_text)
                )
            if arithmetic_available and field.name in {
                "subtotal",
                "tax",
                "discount",
                "total_amount",
            }:
                components["arithmetic_consistency"] = 0.0 if arithmetic_failed else 1.0
            if cross_metadata_present and field.name in {
                "vendor_name",
                "purchase_order_number",
                "bank_account_number",
                "IFSC_or_bank_code",
                "total_amount",
                "currency",
                "payment_terms",
                "invoice_date",
            }:
                components["cross_document_consistency"] = float(
                    field.evidence.evidence_id not in risk_evidence_ids
                )
            if document.evidence_chunks:
                components["retrieval_evidence_quality"] = float(
                    any(
                        normalized_value
                        and normalized_value in self._normalize(chunk.text)
                        for chunk in document.evidence_chunks
                    )
                )
            field.confidence_components = components
            field.confidence = self._score_confidence_components(components, weights)
            field.combined_confidence = field.confidence
            field.ocr_confidence = components.get("ocr_confidence")
            field.extraction_confidence = components.get(
                "layoutlmv3_probability",
                components.get("extraction_confidence", field.extraction_confidence),
            )
            field.validation_confidence = components.get("field_format_validity")
            agreement_values = [
                components.get(name)
                for name in (
                    "layout_agreement",
                    "donut_agreement",
                    "visual_parser_agreement",
                    "cross_document_consistency",
                )
                if components.get(name) is not None
            ]
            field.agreement_confidence = (
                round(sum(agreement_values) / len(agreement_values), 4)
                if agreement_values
                else None
            )
            field.calibrated_confidence = (
                field.confidence if field.calibrated else None
            )
            field.evidence.model_confidence = field.confidence
            field.evidence.metadata["confidence_components"] = components

    def _score_confidence_components(
        self,
        components: dict[str, float],
        weights: dict[str, Any],
    ) -> float:
        active_weights = {
            key: float(weights[key])
            for key in components
            if key in weights and components[key] is not None
        }
        denominator = sum(active_weights.values())
        score = (
            sum(components[key] * weight for key, weight in active_weights.items())
            / denominator
            if denominator
            else 0.0
        )
        return round(max(0.0, min(1.0, score)), 4)

    def _mean_word_confidence(self, document: VisionDocumentRecord) -> float | None:
        if not document.ocr_words:
            return None
        return round(sum(word.confidence for word in document.ocr_words) / len(document.ocr_words), 3)

    def _text_from_ocr(self, words: list[OCRWord]) -> str:
        grouped: dict[tuple[int, str], list[OCRWord]] = {}
        for word in sorted(words, key=lambda item: item.reading_order):
            grouped.setdefault((word.page, word.line_id), []).append(word)
        pages: dict[int, list[str]] = {}
        for (page_number, _), line_words in grouped.items():
            line = " ".join(
                word.text for word in sorted(line_words, key=lambda item: item.bounding_box.x1)
            )
            pages.setdefault(page_number, []).append(line)
        return "\n\n".join(
            "\n".join(pages[page_number]) for page_number in sorted(pages)
        )

    def _mean_field_confidence(self, document: VisionDocumentRecord) -> float | None:
        if not document.extracted_fields:
            return None
        values = [
            field.confidence
            for field in document.extracted_fields
            if field.confidence is not None
        ]
        return round(sum(values) / len(values), 3) if values else None

    def _mean_available_field_confidence(
        self,
        fields: Any,
    ) -> float | None:
        values = [field.confidence for field in fields if field.confidence is not None]
        return round(sum(values) / len(values), 4) if values else None

    def _layout_confidence(self, document: VisionDocumentRecord) -> float | None:
        if not document.layout_regions:
            return None
        return round(sum(region.confidence for region in document.layout_regions) / len(document.layout_regions), 3)

    def _review_threshold(
        self,
        name: str,
        document: VisionDocumentRecord | None = None,
    ) -> float:
        value = self.confidence_policy["review_thresholds"][name]
        if document:
            document_override = self.confidence_policy.get("document_type_overrides", {}).get(
                document.document_type.value,
                {},
            )
            organisation_override = self.confidence_policy.get("organisation_overrides", {}).get(
                document.tenant_id,
                {},
            )
            value = document_override.get(name, value)
            value = organisation_override.get(name, value)
        return float(value)

    def _mandatory_review_threshold(
        self,
        document: VisionDocumentRecord | None = None,
    ) -> float:
        return self._review_threshold("mandatory_review_below_confidence", document)

    def _review_priority(self, document: VisionDocumentRecord) -> VisionRiskSeverity:
        severities = [
            item.severity
            for item in [*document.risk_indicators, *document.processing_warnings]
        ]
        if VisionRiskSeverity.critical in severities:
            return VisionRiskSeverity.critical
        if VisionRiskSeverity.high in severities:
            return VisionRiskSeverity.high
        if VisionRiskSeverity.medium in severities:
            return VisionRiskSeverity.medium
        return VisionRiskSeverity.low

    def _field_map(self, document: VisionDocumentRecord) -> dict[str, str]:
        result = {}
        for field in document.extracted_fields:
            result.setdefault(field.name, field.reviewed_value or field.value)
        return result

    def _review_belongs_to_tenant(self, review: VisionReviewTask, tenant_id: str) -> bool:
        try:
            return self.get_document(review.document_id).tenant_id == tenant_id
        except ValueError:
            return False

    def _field_confidence_by_name(self, document: VisionDocumentRecord, name: str) -> float:
        field = next((item for item in document.extracted_fields if item.name == name), None)
        return field.confidence if field and field.confidence is not None else 0.0

    def _indicator_category(self, category: str) -> str:
        if category == "processing_warning" or category in {
            "processing",
            "ocr_quality",
            "model_routing",
        }:
            return "processing_warning"
        if category in {"validation", "format_validation", "visual_validation"}:
            return "data_quality"
        if category in {"tamper_screening", "visual_risk"}:
            return "visual_risk"
        return "business_risk"

    def _rule_id(self, category: str, title: str) -> str:
        prefix = "cross_document" if category == "cross_document" else self._indicator_category(category)
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        return f"{prefix}.{slug}"

    def _box_for_text(self, text: str, words: list[OCRWord]) -> VisionBoundingBox | None:
        matched = self._words_for_text(text, words)
        return self._union_box([word.bounding_box for word in matched])

    def _words_for_text(self, text: str, words: list[OCRWord]) -> list[OCRWord]:
        targets = [
            normalized
            for token in re.findall(r"\S+", text or "")
            if (normalized := self._normalize_token(token))
        ]
        if not targets:
            return []
        by_page: dict[int, list[OCRWord]] = {}
        for word in sorted(words, key=lambda item: item.reading_order):
            by_page.setdefault(word.page, []).append(word)

        best: list[OCRWord] = []
        for page_words in by_page.values():
            normalized_words = [self._normalize_token(word.text) for word in page_words]
            for target_start in range(len(targets)):
                for word_start, normalized_word in enumerate(normalized_words):
                    if normalized_word != targets[target_start]:
                        continue
                    matched: list[OCRWord] = []
                    target_index = target_start
                    word_index = word_start
                    while (
                        target_index < len(targets)
                        and word_index < len(page_words)
                        and len(matched) < 24
                        and normalized_words[word_index] == targets[target_index]
                    ):
                        matched.append(page_words[word_index])
                        target_index += 1
                        word_index += 1
                    if len(matched) > len(best):
                        best = matched
                    if target_start == 0 and target_index == len(targets):
                        return matched
        minimum = 1 if len(targets) == 1 else min(2, len(targets))
        return best if len(best) >= minimum else []

    def _union_box(self, boxes: list[VisionBoundingBox]) -> VisionBoundingBox | None:
        if not boxes:
            return None
        return VisionBoundingBox(
            x1=min(box.x1 for box in boxes),
            y1=min(box.y1 for box in boxes),
            x2=max(box.x2 for box in boxes),
            y2=max(box.y2 for box in boxes),
        )

    def _dedupe_fields(self, fields: list[ExtractedVisionField]) -> list[ExtractedVisionField]:
        seen = set()
        result = []
        for field in fields:
            key = (field.name, field.normalized_value)
            if key in seen:
                continue
            seen.add(key)
            result.append(field)
        return result

    def _format_valid(self, value: str) -> bool:
        value = value.strip()
        if not value:
            return False
        if re.search(r"\d", value):
            return len(value) >= 2
        return any(character.isalpha() for character in value)

    def _money(self, value: str | None) -> float | None:
        if not value:
            return None
        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
        if not match:
            return None
        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            return None

    def _date(self, value: str | None) -> datetime | None:
        if not value:
            return None
        cleaned = value.strip().rstrip(".")
        formats = ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y", "%b %d, %Y", "%B %d, %Y")
        for fmt in formats:
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
        return None

    def _clean_value(self, value: str) -> str:
        value = re.sub(r"\s+", " ", value).strip(" .,:;-")
        value = re.split(
            r"\b(?:invoice date|due date|payment terms|subtotal|tax|discount|total|amount due|signature)\b",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .,:;-")
        return value

    def _normalize(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value).strip().lower()).strip(" .,:;-")

    def _normalize_token(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _mask_sensitive(self, field_name: str, value: str) -> str:
        if "bank" in field_name or "account" in field_name or "IFSC" in field_name:
            return self._mask_account(value)
        return value

    def _mask_account(self, value: str) -> str:
        raw = re.sub(r"\s+", "", str(value))
        if len(raw) <= 4:
            return "****"
        return f"{'*' * max(4, len(raw) - 4)}{raw[-4:]}"


vision_ai_service = VisionAIService()
