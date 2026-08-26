from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
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
    TypedVisionEntity,
    VerifiedVisionFact,
    VisionAskCalculationInput,
    VisionAskCalculationOperation,
    VisionAskCalculationTrace,
    VisionAskCitation,
    VisionAskRequest,
    VisionAskResponse,
    VisionAuditEvent,
    VisionBoundingBox,
    VisionClarificationOption,
    VisionConversationState,
    VisionDocumentRecord,
    VisionDocumentType,
    VisionEvidence,
    VisionEvidenceChunk,
    VisionExtractionOutcome,
    VisionFinanceNarrative,
    VisionFinanceNarrativeRequest,
    VisionFinanceNarrativeType,
    VisionGraphProjection,
    VisionProcessingStatus,
    VisionReportRecord,
    VisionReportRequest,
    VisionReportType,
    VisionRetrievalTrace,
    VisionReviewDecisionRequest,
    VisionReviewTask,
    VisionRiskIndicator,
    VisionRiskSeverity,
    VisionSemanticType,
    VisionUploadResponse,
)
from app.models.vision import (
    GenericDiscoveredField as VisionGenericDiscoveredField,
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
from app.services.vision.document_classification import hybrid_document_classifier
from app.services.vision.duplicate_detection import (
    DinoV2DuplicateAdapter,
    cosine_similarity,
    document_dhash,
    perceptual_similarity,
)
from app.services.vision.entity_resolution import conservative_entity_resolver
from app.services.vision.extraction_adapters import (
    DonutExtractionAdapter,
    LayoutLMv3ClassificationAdapter,
    LayoutLMv3ExtractionAdapter,
)
from app.services.vision.finance_narratives import verified_finance_narrative_service
from app.services.vision.generator_adapters import GroundedGeneratorRouter
from app.services.vision.generic_field_discovery import generic_field_discovery_engine
from app.services.vision.graph_builder import evidence_graph_builder
from app.services.vision.language_analysis import vision_language_analysis_service
from app.services.vision.layout_adapters import GeometryLayoutAdapter, PaddleStructureV3Adapter
from app.services.vision.legacy_audit import CURRENT_PIPELINE_VERSION
from app.services.vision.model_smoke import ModelSmokeService
from app.services.vision.ocr_adapters import PaddleOCRV5Adapter, TesseractOCRAdapter
from app.services.vision.ocr_lines import ocr_line_reconstructor
from app.services.vision.persistence import vision_persistence
from app.services.vision.preprocessing import DocumentPreprocessor
from app.services.vision.query_router import VisionQueryRouter
from app.services.vision.reporting import vision_report_service
from app.services.vision.retrieval_adapters import LocalHybridRetrievalIndex
from app.services.vision.reviewer_feedback import ReviewerFeedbackDataset
from app.services.vision.schema_registry import vision_schema_registry
from app.services.vision.semantic_facts import (
    normalize_semantic_value,
    semantic_type_for_name,
)
from app.services.vision.semantic_risk import (
    SemanticRiskFinding,
    generic_semantic_risk_engine,
)
from app.services.vision.semantic_values import (
    SemanticContext,
    SemanticValueType,
    semantic_value_registry,
)
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

CLARIFICATION_OPTIONS = (
    VisionClarificationOption(
        option_id="document_quality",
        label="Check document quality",
        resolved_question="What document quality and unresolved data-quality issues require human review?",
        query_type="quality_question",
    ),
    VisionClarificationOption(
        option_id="suspicious_indicators",
        label="Check suspicious indicators",
        resolved_question="What suspicious business-rule or visual-manipulation indicators are recorded?",
        query_type="risk_question",
    ),
    VisionClarificationOption(
        option_id="financial_consistency",
        label="Check financial consistency",
        resolved_question="Are the extracted financial values consistent across the selected documents?",
        query_type="comparison_question",
    ),
    VisionClarificationOption(
        option_id="signature_or_stamp",
        label="Check signature or stamp presence",
        resolved_question="Is there a signature or stamp present?",
        query_type="visual_question",
    ),
    VisionClarificationOption(
        option_id="approval_status",
        label="Check approval status",
        resolved_question="What is the current approval status?",
        query_type="approval_question",
    ),
)

RETRIEVAL_SOURCE_SEQUENCE = (
    "verified_structured_fields",
    "reviewed_structured_fields",
    "unverified_extracted_fields",
    "field_aliases",
    "ocr_key_value_pairs",
    "reconstructed_ocr_lines",
    "lexical_bm25_chunks",
    "dense_vector_retrieval",
    "reranked_evidence",
    "graph_facts_and_paths",
    "visual_detections",
    "deterministic_calculations",
)


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
        self._document_tenant_index: dict[str, str] = {}
        self._semantic_label_embedding_cache: dict[str, dict[str, list[float]]] = {}
        self.questions: dict[str, dict[str, Any]] = {}
        self.question_feedback: dict[str, list[dict[str, Any]]] = {}
        self.conversations: dict[str, VisionConversationState] = {}
        self.finance_narratives: dict[str, VisionFinanceNarrative] = {}
        self.reports: dict[str, VisionReportRecord] = {}
        self.model_overrides: dict[str, dict[str, Any]] = {}
        self.rule_overrides: dict[str, dict[str, Any]] = {}
        self.system_audit_events: list[dict[str, Any]] = []
        self._artifact_lock = threading.RLock()
        self._state_root = Path(settings.upload_dir).parent / "vision_state"
        self.feedback_dataset = ReviewerFeedbackDataset(self._state_root / "active_learning")
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
        retained_metadata = {
            key: value
            for key, value in document.metadata.items()
            if key
            not in {
                "classification",
                "schema_selection",
                "extraction",
                "extraction_diagnostics",
                "pipeline_status",
                "decision_status",
                "status_summary",
                "legacy_flags",
            }
        }
        reprocessing_history = list(retained_metadata.get("reprocessing_history") or [])
        reprocessing_history.append(
            {
                "status": document.status.value,
                "updated_at": document.updated_at,
                "review": (
                    {
                        "review_id": document.review_task.review_id,
                        "status": document.review_task.status.value,
                        "reviewer_comments": document.review_task.reviewer_comments,
                        "history_event_ids": [
                            event.event_id for event in document.review_task.history
                        ],
                    }
                    if document.review_task
                    else None
                ),
                "data_version": document.data_version,
                "pipeline_version": document.metadata.get("pipeline_version", "legacy"),
            }
        )
        metadata = {
            **retained_metadata,
            "reprocessed_from": document.document_id,
            "reprocess_count": int(document.metadata.get("reprocess_count", 0)) + 1,
            "reprocessing_history": reprocessing_history,
        }
        if document.review_task:
            metadata["_reprocess_review_id"] = document.review_task.review_id
        stale_artifacts = self._mark_derived_artifacts_stale(document, actor=actor)
        metadata["stale_derived_artifacts"] = stale_artifacts
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
                timing = dict(document.metadata.get("processing_timing") or {})
                processing_started_at = timing.get("processing_started_at")
                processing_completed_at = utc_now_iso()
                processing_duration_ms: float | None = None
                if processing_started_at:
                    try:
                        processing_duration_ms = max(
                            0.0,
                            (
                                datetime.fromisoformat(processing_completed_at)
                                - datetime.fromisoformat(str(processing_started_at))
                            ).total_seconds()
                            * 1000,
                        )
                    except (TypeError, ValueError):
                        processing_duration_ms = None
                document.metadata["processing_timing"] = {
                    **timing,
                    "processing_completed_at": processing_completed_at,
                    "processing_duration_ms": (
                        round(processing_duration_ms, 3)
                        if processing_duration_ms is not None
                        else None
                    ),
                    "reliable": False,
                    "exclusion_reason": "processing_failed",
                }
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
        job["status"] = (
            VisionProcessingStatus.completed.value
            if not job["errors"]
            else VisionProcessingStatus.review_required.value
        )
        processed_documents = [
            self.documents[document_id]
            for document_id in job["document_ids"]
            if document_id in self.documents
        ]
        decision_statuses = sorted({document.status.value for document in processed_documents})
        job["pipeline_status"] = (
            VisionProcessingStatus.completed.value
            if not job["errors"]
            else VisionProcessingStatus.review_required.value
        )
        job["decision_status"] = (
            VisionProcessingStatus.review_required.value
            if any(
                document.status == VisionProcessingStatus.review_required
                for document in processed_documents
            )
            else decision_statuses[0]
            if len(decision_statuses) == 1
            else "MIXED"
        )
        job["status_summary"] = (
            "Processing completed - awaiting human review"
            if job["decision_status"] == VisionProcessingStatus.review_required.value
            and not job["errors"]
            else "Processing completed"
            if not job["errors"]
            else "Processing requires technical review"
        )
        job["extraction_outcomes"] = [
            {
                "document_id": document.document_id,
                **dict(document.metadata.get("extraction") or {}),
            }
            for document in processed_documents
        ]
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
        # API and Celery worker are separate processes.  Prefer the durable job
        # snapshot so a queued in-memory object cannot mask worker progress.
        job = self._load_job(job_id) or self.jobs.get(job_id)
        if not job and tenant_id:
            job = self.persistence.get_job(job_id, tenant_id)
            if job:
                self.jobs[job_id] = job
        if job and tenant_id and job.get("tenant_id", "tenant-local") != tenant_id:
            return None
        return job

    def retry_job(
        self,
        job_id: str,
        tenant_id: str | None = None,
        *,
        actor: str = "system",
    ) -> dict[str, Any]:
        job = self.get_job(job_id, tenant_id)
        if not job:
            raise ValueError("job_id not found")
        job["errors"] = []
        job["status"] = VisionProcessingStatus.queued.value
        job["updated_at"] = utc_now_iso()
        self._persist_job(job)
        self._system_audit(
            "job_retry_queued",
            "Queued a failed or incomplete processing job for retry.",
            actor=actor,
            tenant_id=str(job.get("tenant_id") or tenant_id or "tenant-local"),
            metadata={"job_id": job_id, "document_ids": list(job.get("document_ids") or [])},
        )
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
        *,
        actor: str = "system",
    ) -> VisionDocumentRecord:
        document = self.get_document(document_id, tenant_id)
        self.documents.pop(document_id, None)
        self._document_tenant_index.pop(document_id, None)
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
        self._system_audit(
            "document_deleted",
            "Deleted a document and its local derived state.",
            actor=actor,
            tenant_id=document.tenant_id,
            metadata={
                "document_id": document.document_id,
                "case_id": document.case_id,
                "filename": document.filename,
            },
        )
        return document

    def get_document(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> VisionDocumentRecord:
        # Reload the shared durable snapshot first.  Without this, the API keeps
        # returning its pre-queue object after a Celery worker has completed the
        # same document in another process.
        document = self._load_document(document_id, tenant_id=tenant_id) or self.documents.get(
            document_id
        )
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
        self._load_all_documents(tenant_id)
        if tenant_id:
            for document in self.persistence.list_documents(tenant_id):
                current = self.documents.get(document.document_id)
                if current is None or (
                    current.updated_at,
                    current.data_version,
                ) != (document.updated_at, document.data_version):
                    self.documents[document.document_id] = document
                    self._document_tenant_index[document.document_id] = document.tenant_id
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
            reviews = [item for item in reviews if self._review_belongs_to_tenant(item, tenant_id)]
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
        reviewed_field_name: str | None = None,
        reviewer_status: ReviewerStatus,
        comment: str,
        actor: str,
        tenant_id: str | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> ExtractedVisionField:
        review = self.get_review(review_id, tenant_id)
        document = self.get_document(review.document_id, tenant_id)
        field = next(
            (item for item in document.extracted_fields if item.field_id == field_id), None
        )
        if not field:
            raise ValueError("field_id not found")
        old_name = field.name
        if reviewed_field_name and reviewed_field_name != field.name:
            if any(
                item.field_id != field.field_id and item.name == reviewed_field_name
                for item in document.extracted_fields
            ):
                raise ValueError("reviewed_field_name already exists on this document")
            field.name = reviewed_field_name
            field.schema_field = reviewed_field_name
            field.display_name = reviewed_field_name.replace("_", " ").title()
        old_value = field.reviewed_value or field.value
        field.reviewed_value = reviewed_value if reviewed_value is not None else field.value
        semantic_type = self._semantic_type_for_reviewed_field(document, field)
        field.normalized_value = normalize_semantic_value(
            field.reviewed_value,
            semantic_type,
            label=field.name,
            document_type=document.document_type.value,
        )
        field.reviewer_status = reviewer_status
        field.evidence.reviewer_status = reviewer_status
        field.verification_status = reviewer_status.value
        self._apply_field_review_to_generic_layers(
            document,
            field,
            semantic_type=semantic_type,
        )
        event = self._audit(
            document,
            "field_reviewed",
            "Reviewer updated an extracted field.",
            actor=actor,
            metadata={
                "field_id": field_id,
                "field_name": field.name,
                "old_field_name": old_name,
                "old_value": self._mask_sensitive(field.name, old_value),
                "new_value": self._mask_sensitive(field.name, field.reviewed_value or ""),
                "comment": comment,
                **(request_metadata or {}),
            },
        )
        review.history.append(event)
        review.updated_at = event.created_at
        if reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}:
            self._record_active_learning_correction(
                document,
                field,
                original_field_name=old_name,
                original_value=old_value,
                comment=comment,
                reviewer=actor,
                reviewed_at=event.created_at,
            )
        document.data_version += 1
        document.analytics_cache_version += 1
        self._refresh_verified_facts(document, reviewer=actor)
        self._resolve_semantic_entities(document)
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
            self._refresh_verified_facts(document, reviewer=actor)
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
                reauthentication_payload.get("auth_time") if reauthentication_payload else None
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
        evaluation = vision_approval_policy_service.evaluate(
            document,
            request=None,
            actor_role=actor_role,
            confidence_threshold=self._mandatory_review_threshold(document),
        )
        evaluation.fraud_second_reviewer_required = settings.fraud_second_reviewer_required
        return evaluation

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
                identity = hash_document(f"{document.document_id}|manual-review|{name}".encode())
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
                    display_name=(spec.display_name if spec else name.replace("_", " ").title()),
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
        if actor_role not in {
            UserRole.admin,
            UserRole.senior_reviewer,
            UserRole.compliance_officer,
        }:
            raise PermissionError(
                "Compliance Officer, Senior Reviewer, or Admin role is required to confirm fraud"
            )
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
            raise ReviewDecisionValidationError(f"Unknown evidence IDs: {', '.join(unavailable)}")
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
        self._refresh_verified_facts(document, reviewer=actor)
        document.evidence_chunks = self._chunks(document)
        document.graph = self._graph(document)
        vision_analytics_service.refresh_document(document)
        self._index_document(document)
        self.persistence.upsert_graph(document.tenant_id, document.document_id, document.graph)
        self._persist_document(document)
        self._persist_review(review)
        return {"accepted_field_ids": accepted, "threshold": threshold}

    def active_learning_export(self, tenant_id: str | None = None) -> dict[str, Any]:
        resolved_tenant = str(tenant_id or "").strip()
        if not resolved_tenant:
            raise ValueError("tenant_id is required for active-learning export")
        corrections: list[dict[str, Any]] = []
        persisted_keys: set[tuple[str, str]] = set()
        for record in self.feedback_dataset.records(resolved_tenant):
            field_name = str(
                record.get("corrected_field_name") or record.get("original_field_name") or "unknown"
            )
            semantic_type = str(record.get("semantic_type") or "")
            sensitive = semantic_type == VisionSemanticType.bank_account.value or any(
                token in field_name.casefold()
                for token in ("bank", "account", "ifsc", "routing", "iban")
            )
            document_id = str(record.get("document_id") or "")
            field_id = str(record.get("field_id") or "")
            persisted_keys.add((document_id, field_id))
            corrections.append(
                {
                    "correction_id": record.get("correction_id"),
                    "schema_version": record.get("schema_version"),
                    "tenant_id": resolved_tenant,
                    "document_id": document_id,
                    "document_type": record.get("document_type"),
                    "field_id": field_id,
                    "field_name": field_name,
                    "original_field_name": record.get("original_field_name"),
                    "raw_label": record.get("raw_label"),
                    "semantic_type": record.get("semantic_type"),
                    "model_value": self._mask_sensitive(
                        field_name,
                        str(record.get("original_value") or ""),
                    ),
                    "reviewed_value": self._mask_sensitive(
                        field_name,
                        str(record.get("corrected_value") or ""),
                    ),
                    "evidence": {
                        "evidence_id": record.get("evidence_id"),
                        "page": record.get("page"),
                        "bounding_box": record.get("bounding_box"),
                        "source_text": (
                            "Sensitive source text withheld from export."
                            if sensitive
                            else record.get("source_text")
                        ),
                    },
                    "extraction_method": record.get("extraction_method"),
                    "layout_relation": record.get("layout_relation"),
                    "schema_mode": record.get("schema_mode"),
                    "schema_status": record.get("schema_status"),
                    "processing_version": record.get("processing_version"),
                    "status": record.get("status"),
                    "reviewed_at": record.get("reviewed_at"),
                    "training_policy": record.get("training_policy"),
                }
            )

        # Keep edited fields created before the durable v2 dataset exportable.
        legacy_count = 0
        for document in self.list_documents(resolved_tenant):
            for field in document.extracted_fields:
                if (
                    field.reviewer_status == ReviewerStatus.edited
                    and field.reviewed_value is not None
                    and (document.document_id, field.field_id) not in persisted_keys
                ):
                    legacy_count += 1
                    corrections.append(
                        {
                            "document_id": document.document_id,
                            "field_id": field.field_id,
                            "field_name": field.name,
                            "tenant_id": document.tenant_id,
                            "document_type": document.document_type.value,
                            "model_value": self._mask_sensitive(field.name, field.value),
                            "reviewed_value": self._mask_sensitive(
                                field.name, field.reviewed_value
                            ),
                            "evidence": field.evidence.model_dump(mode="json"),
                        }
                    )
        return {
            "total_corrections": len(corrections),
            "persisted_corrections": len(corrections) - legacy_count,
            "legacy_live_corrections": legacy_count,
            "format": "active_learning_jsonl_v2_export",
            "summary": self.feedback_dataset.summary(resolved_tenant),
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

    def ask(
        self,
        request: VisionAskRequest,
        tenant_id: str | None = None,
        *,
        conversation_id: str | None = None,
        user_id: str | None = None,
    ) -> VisionAskResponse:
        started_at = time.perf_counter()
        resolved_tenant = tenant_id or "tenant-local"
        resolved_user = user_id or "anonymous"
        resolved_conversation_id = (
            request.conversation_id or conversation_id or f"vision-conversation-{uuid4().hex[:12]}"
        )
        conversation = self._conversation_state(
            resolved_conversation_id,
            resolved_tenant,
            resolved_user,
        )
        question = request.question.strip()
        query_type = self.query_router.route(question)

        pending = dict(conversation.pending_clarification)
        if query_type == "clarification_response":
            request = self._request_with_conversation_scope(request, pending)
            selected_option = self._selected_clarification_option(
                request.clarification_option,
                question,
                pending,
            )
            if not pending:
                documents, effective_scope = self._ask_scope_documents(request, tenant_id)
                response = VisionAskResponse(
                    answer=(
                        "I do not have a pending clarification for this conversation. "
                        "Please ask the document question again."
                    ),
                    answer_status="clarification_needed",
                    query_type="clarification_response",
                    reasoning_summary="An acknowledgement cannot be resolved without a pending option set.",
                    requires_review=False,
                    retriever="not_used",
                )
                return self._finish_ask(
                    request,
                    response,
                    documents,
                    effective_scope,
                    conversation,
                    started_at,
                    question,
                    tenant_id,
                )
            if question.lower().rstrip(".! ") in {"no", "n"}:
                conversation.pending_clarification = {}
                documents, effective_scope = self._ask_scope_documents(request, tenant_id)
                response = VisionAskResponse(
                    answer="Okay. Ask a more specific question whenever you are ready.",
                    answer_status="supported",
                    query_type="clarification_response",
                    reasoning_summary="The pending clarification was cancelled without document retrieval.",
                    requires_review=False,
                    retriever="not_used",
                )
                return self._finish_ask(
                    request,
                    response,
                    documents,
                    effective_scope,
                    conversation,
                    started_at,
                    question,
                    tenant_id,
                )
            if selected_option is None:
                documents, effective_scope = self._ask_scope_documents(request, tenant_id)
                response = VisionAskResponse(
                    answer="Please select which type of safety check you want me to perform.",
                    answer_status="clarification_needed",
                    query_type="clarification_response",
                    clarification_options=list(CLARIFICATION_OPTIONS),
                    reasoning_summary=(
                        "The acknowledgement preserved the pending clarification; it did not "
                        "become an independent semantic query."
                    ),
                    requires_review=False,
                    retriever="not_used",
                )
                return self._finish_ask(
                    request,
                    response,
                    documents,
                    effective_scope,
                    conversation,
                    started_at,
                    question,
                    tenant_id,
                )
            question = selected_option.resolved_question
            query_type = selected_option.query_type
            conversation.pending_clarification = {}

        if query_type == "conversation_follow_up":
            request = self._request_with_selected_conversation_scope(request, conversation)
            documents, effective_scope = self._ask_scope_documents(request, tenant_id)
            response = self._conversation_follow_up_response(question, conversation)
            return self._finish_ask(
                request,
                response,
                documents,
                effective_scope,
                conversation,
                started_at,
                conversation.last_resolved_question or question,
                tenant_id,
            )

        documents, effective_scope = self._ask_scope_documents(request, tenant_id)
        conversation.selected_scope = self._conversation_scope(request, effective_scope)
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
                clarification_options=list(CLARIFICATION_OPTIONS),
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
            if query_type == "ambiguous_question":
                conversation.pending_clarification = {
                    "clarification_id": f"clarification-{uuid4().hex[:12]}",
                    "original_question": question,
                    "options": [item.model_dump(mode="json") for item in CLARIFICATION_OPTIONS],
                    "selected_scope": self._conversation_scope(request, effective_scope),
                    "created_at": utc_now_iso(),
                }
            return self._finish_ask(
                request,
                casual_responses[query_type],
                documents,
                effective_scope,
                conversation,
                started_at,
                question,
                tenant_id,
            )
        if not documents:
            response = self._not_found_answer(requires_review=False)
            response.query_type = query_type
            return self._finish_ask(
                request,
                response,
                [],
                effective_scope,
                conversation,
                started_at,
                question,
                tenant_id,
            )

        resolved_tenant = tenant_id or documents[0].tenant_id
        for document in documents:
            self.retrieval_index.upsert_document(document.document_id, document.evidence_chunks)
        handlers = {
            "structured_field_lookup": self._answer_structured,
            "structured_lookup": self._answer_structured,
            "ocr_lookup": self._answer_structured,
            "semantic_question": self._answer_semantic,
            "relationship_question": self._answer_relationship,
            "arithmetic_question": self._answer_arithmetic,
            "visual_question": self._answer_visual,
            "comparison_question": self._answer_comparison,
            "risk_question": self._answer_risk,
            "quality_question": self._answer_quality,
            "approval_question": self._answer_approval,
            "safety_question": self._answer_safety,
        }
        response = handlers[query_type](
            question,
            documents,
            request.top_k,
            resolved_tenant,
        )
        if response.answer_status == "not_found" and query_type not in {
            "structured_field_lookup",
            "structured_lookup",
            "ocr_lookup",
            "arithmetic_question",
        }:
            cascade_response = self._answer_structured(
                question,
                documents,
                request.top_k,
                resolved_tenant,
            )
            if cascade_response.answer_status != "not_found":
                response = cascade_response
                response.reasoning_summary = (
                    f"{response.reasoning_summary} The primary {query_type.replace('_', ' ')} "
                    "source had no match, so the remaining evidence cascade was checked."
                )
        response.query_type = query_type
        response.resolved_question = question if question != request.question.strip() else None
        if response.answer_status == "not_found" and effective_scope in {
            "current_document",
            "selected_documents",
        }:
            broader = [
                document
                for document in self.list_documents(tenant_id)
                if document.document_id not in {item.document_id for item in documents}
            ]
            if self._question_has_document_evidence(question, broader):
                requested = self._requested_field_names(question)
                label = (
                    next(iter(sorted(requested))).replace("_", " ")
                    if requested
                    else "matching evidence"
                )
                response.answer = (
                    f"The selected document does not contain a {label} field. "
                    "Search all accessible documents instead?"
                )
                response.answer_status = "clarification_needed"
                response.reasoning_summary = (
                    "No evidence matched the selected scope; matching evidence exists outside it."
                )
                response.broader_scope_available = True
        return self._finish_ask(
            request,
            response,
            documents,
            effective_scope,
            conversation,
            started_at,
            question,
            tenant_id,
        )

    def _finish_ask(
        self,
        request: VisionAskRequest,
        response: VisionAskResponse,
        documents: list[VisionDocumentRecord],
        effective_scope: str,
        conversation: VisionConversationState,
        started_at: float,
        resolved_question: str,
        tenant_id: str | None,
    ) -> VisionAskResponse:
        response.scope = self._ask_scope_payload(effective_scope, documents)
        response.conversation_id = conversation.conversation_id
        if resolved_question != request.question.strip():
            response.resolved_question = resolved_question
        self._attach_retrieval_trace(
            response,
            resolved_question,
            documents,
            tenant_id or conversation.organisation_id,
            started_at,
        )
        conversation.last_query_type = response.query_type
        if response.answer_status != "clarification_needed":
            conversation.last_resolved_question = resolved_question
            conversation.last_answer = response.model_dump(mode="json")
        return self._record_question(
            request,
            response,
            documents,
            tenant_id=tenant_id,
            conversation=conversation,
        )

    def _conversation_state(
        self,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
    ) -> VisionConversationState:
        cache_key = self._conversation_cache_key(conversation_id, tenant_id, user_id)
        state = self.conversations.get(cache_key)
        if state is None:
            payload = self._read_json(self._state_file("conversations", cache_key))
            if payload is None:
                payload = self.persistence.get_conversation(
                    conversation_id,
                    tenant_id,
                    user_id,
                )
            try:
                state = VisionConversationState.model_validate(payload) if payload else None
            except ValueError:
                state = None
        if state is None or self._conversation_expired(state):
            now = datetime.now(UTC)
            state = VisionConversationState(
                conversation_id=conversation_id,
                user_id=user_id,
                organisation_id=tenant_id,
                expires_at=(now + timedelta(minutes=30)).isoformat(),
                updated_at=now.isoformat(),
            )
        self.conversations[cache_key] = state
        return state

    def _persist_conversation(self, state: VisionConversationState) -> None:
        now = datetime.now(UTC)
        state.updated_at = now.isoformat()
        state.expires_at = (now + timedelta(minutes=30)).isoformat()
        state.turns = state.turns[-100:]
        cache_key = self._conversation_cache_key(
            state.conversation_id,
            state.organisation_id,
            state.user_id,
        )
        self.conversations[cache_key] = state
        payload = state.model_dump(mode="json")
        self._write_json(self._state_file("conversations", cache_key), payload)
        self.persistence.save_conversation(payload)

    def _conversation_cache_key(
        self,
        conversation_id: str,
        tenant_id: str,
        user_id: str,
    ) -> str:
        identity = f"{tenant_id}|{user_id}|{conversation_id}"
        return hash_document(identity.encode())

    def _conversation_expired(self, state: VisionConversationState) -> bool:
        try:
            expires_at = datetime.fromisoformat(state.expires_at.replace("Z", "+00:00"))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            return expires_at <= datetime.now(UTC)
        except ValueError:
            return True

    def _conversation_scope(
        self,
        request: VisionAskRequest,
        effective_scope: str,
    ) -> dict[str, Any]:
        return {
            "scope": effective_scope,
            "document_id": request.document_id,
            "document_ids": list(request.document_ids),
            "top_k": request.top_k,
        }

    def _request_with_conversation_scope(
        self,
        request: VisionAskRequest,
        pending: dict[str, Any],
    ) -> VisionAskRequest:
        selected = dict(pending.get("selected_scope") or {})
        if not selected:
            return request
        return request.model_copy(
            update={
                "scope": selected.get("scope", request.scope),
                "document_id": selected.get("document_id"),
                "document_ids": list(selected.get("document_ids") or []),
                "top_k": int(selected.get("top_k") or request.top_k),
            }
        )

    def _request_with_selected_conversation_scope(
        self,
        request: VisionAskRequest,
        conversation: VisionConversationState,
    ) -> VisionAskRequest:
        selected = dict(conversation.selected_scope or {})
        if not selected:
            return request
        return request.model_copy(
            update={
                "scope": selected.get("scope", request.scope),
                "document_id": selected.get("document_id"),
                "document_ids": list(selected.get("document_ids") or []),
                "top_k": int(selected.get("top_k") or request.top_k),
            }
        )

    def _conversation_follow_up_response(
        self,
        question: str,
        conversation: VisionConversationState,
    ) -> VisionAskResponse:
        previous = dict(conversation.last_answer or {})
        citations = [
            VisionAskCitation.model_validate(item)
            for item in previous.get("citations", [])
            if isinstance(item, dict)
        ]
        if not previous or not citations:
            response = self._not_found_answer(requires_review=False)
            response.query_type = "conversation_follow_up"
            response.reasoning_summary = (
                "No prior source-backed answer exists in this conversation."
            )
            return response
        normalized = " ".join(question.lower().split())
        if "page" in normalized or "where" in normalized:
            pages = ", ".join(
                f"{citation.filename or citation.document_id}, page {citation.page_number}"
                for citation in citations
            )
            answer = f"The cited evidence is on {pages}."
        elif "evidence" in normalized or normalized.rstrip("?.! ") == "source":
            answer = "Evidence: " + "; ".join(
                f"{citation.filename or citation.document_id}, page {citation.page_number}: "
                f"{citation.source_text}"
                for citation in citations
            )
        else:
            answer = (
                str(
                    previous.get("reasoning_summary")
                    or "The answer is supported by the cited evidence."
                )
                + " Evidence: "
                + "; ".join(citation.source_text for citation in citations)
            )
        return VisionAskResponse(
            answer=answer,
            answer_status=str(previous.get("answer_status") or "supported"),
            query_type="conversation_follow_up",
            citations=citations,
            confidence=float(previous.get("confidence") or 0.0),
            reasoning_summary="Resolved the follow-up against the preceding source-backed answer.",
            requires_review=bool(previous.get("requires_review")),
            generator_name="conversation-evidence-resolver",
            generator_version="v1",
            retriever="conversation_citation_state",
            retrieval_path="prior_turn_citations",
            resolved_question=conversation.last_resolved_question or None,
        )

    def _selected_clarification_option(
        self,
        requested_option: str | None,
        question: str,
        pending: dict[str, Any],
    ) -> VisionClarificationOption | None:
        allowed_ids = {
            str(item.get("option_id"))
            for item in pending.get("options", [])
            if isinstance(item, dict) and item.get("option_id")
        }
        candidate = " ".join((requested_option or question).lower().split()).rstrip(".! ")
        for option in CLARIFICATION_OPTIONS:
            if allowed_ids and option.option_id not in allowed_ids:
                continue
            if candidate in {
                option.option_id.lower(),
                option.label.lower(),
            }:
                return option
        return None

    def _attach_retrieval_trace(
        self,
        response: VisionAskResponse,
        question: str,
        documents: list[VisionDocumentRecord],
        tenant_id: str,
        started_at: float,
    ) -> None:
        if response.retriever == "not_used":
            sources_checked: list[str] = []
            candidate_counts: dict[str, int] = {}
            versions: dict[str, str] = {}
            reranker = "not_used"
        else:
            candidate_counts, versions, reranker = self._retrieval_inventory(
                question,
                documents,
                tenant_id,
            )
            sources_checked = list(RETRIEVAL_SOURCE_SEQUENCE)
        selected = list(
            dict.fromkeys(
                str(
                    citation.evidence_id
                    or citation.chunk_id
                    or f"{citation.document_id}:page:{citation.page_number}"
                )
                for citation in response.citations
            )
        )
        trace = VisionRetrievalTrace(
            sources_checked=sources_checked,
            candidate_counts=candidate_counts,
            selected_evidence=selected,
            retrieval_duration_ms=round(
                max(0.0, (time.perf_counter() - started_at) * 1000),
                3,
            ),
            retriever_versions=versions,
        )
        response.retrieval_trace = trace
        response.retrieval_trace_id = trace.trace_id
        response.reranker = reranker

    def _retrieval_inventory(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        tenant_id: str,
    ) -> tuple[dict[str, int], dict[str, str], str]:
        requested_names = self._requested_field_names(question)
        matched_fields = self._field_matches(question.lower(), documents)
        if not matched_fields and any(
            term in question.lower() for term in ("quality", "safe", "approval")
        ):
            matched_fields = [
                field for document in documents for field in document.extracted_fields
            ]
        verified = [
            field for field in matched_fields if field.reviewer_status == ReviewerStatus.accepted
        ]
        reviewed = [
            field for field in matched_fields if field.reviewer_status == ReviewerStatus.edited
        ]
        unverified = [
            field
            for field in matched_fields
            if field.reviewer_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
        ]
        ocr_matches = self._ocr_lookup(question, documents, 50)
        relevant_chunks = [
            chunk
            for document in documents
            for chunk in document.evidence_chunks
            if self._text_relevant_to_question(question, chunk.text)
        ]
        vector_hits: list[Any] = []
        if documents:
            try:
                vector_hits = self.retrieval_index.search(
                    question,
                    tenant_id=tenant_id,
                    top_k=12,
                    document_ids={document.document_id for document in documents},
                )
            except Exception:
                vector_hits = []
        graph_edges = self._matching_graph_edges(question, documents)
        if requested_names:
            graph_edges = [
                item
                for item in graph_edges
                if str(item[1].get("source_field") or "") in requested_names
            ]
        visual = self._matching_visual_detections(question, documents)
        numeric = [
            field
            for document in documents
            for field in document.extracted_fields
            if self._decimal_value(field.reviewed_value or field.value) is not None
        ]
        counts = {
            "verified_structured_fields": len(verified),
            "reviewed_structured_fields": len(reviewed),
            "unverified_extracted_fields": len(unverified),
            "field_aliases": len(requested_names),
            "ocr_key_value_pairs": sum(
                1 for chunk in relevant_chunks if chunk.source_type == "key_value"
            ),
            "reconstructed_ocr_lines": len(ocr_matches)
            or sum(
                1
                for document in documents
                for line in document.ocr_lines
                if self._text_relevant_to_question(question, line.text)
            ),
            "lexical_bm25_chunks": sum(1 for hit in vector_hits if hit.sparse_score > 0),
            "dense_vector_retrieval": sum(1 for hit in vector_hits if hit.dense_score > 0),
            "reranked_evidence": sum(1 for hit in vector_hits if hit.rerank_score is not None),
            "graph_facts_and_paths": len(graph_edges),
            "visual_detections": len(visual),
            "deterministic_calculations": len(numeric),
        }
        reranker_status = self.retrieval_index.reranker.status()
        reranker = (
            f"{self.retrieval_index.reranker.adapter_id}:{self.retrieval_index.reranker.model_version}"
            if reranker_status.status == "active"
            else "not_used"
        )
        versions = {
            "verified_structured_fields": "persisted-structured-field-store-v1",
            "reviewed_structured_fields": "persisted-structured-field-store-v1",
            "unverified_extracted_fields": "persisted-structured-field-store-v1",
            "field_aliases": "vision-field-aliases-v3",
            "ocr_key_value_pairs": "structure-aware-chunker-v1",
            "reconstructed_ocr_lines": "ocr-line-reconstructor-v1",
            "lexical_bm25_chunks": "local-hybrid-sparse-v1",
            "dense_vector_retrieval": self.retrieval_index.embedding.model_version,
            "reranked_evidence": (
                self.retrieval_index.reranker.model_version
                if reranker_status.status == "active"
                else "not_enabled"
            ),
            "graph_facts_and_paths": "documind-evidence-graph-v4",
            "visual_detections": "stored-visual-detections-v1",
            "deterministic_calculations": "deterministic-decimal-v1",
        }
        return counts, versions, reranker

    def _text_relevant_to_question(self, question: str, text: str) -> bool:
        stopwords = {
            "about",
            "and",
            "are",
            "document",
            "does",
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
        text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
        return bool(query_tokens & text_tokens)

    def _ask_scope_documents(
        self,
        request: VisionAskRequest,
        tenant_id: str | None,
    ) -> tuple[list[VisionDocumentRecord], str]:
        if request.document_id:
            return [self.get_document(request.document_id, tenant_id)], "current_document"
        if request.document_ids:
            seen: set[str] = set()
            selected = []
            for document_id in request.document_ids:
                if document_id in seen:
                    continue
                selected.append(self.get_document(document_id, tenant_id))
                seen.add(document_id)
            return selected, "selected_documents"
        documents = self.list_documents(tenant_id)
        if request.scope == "approved_documents":
            return (
                [
                    document
                    for document in documents
                    if document.status == VisionProcessingStatus.approved
                ],
                "approved_documents",
            )
        return documents, request.scope

    def _ask_scope_payload(
        self,
        scope: str,
        documents: list[VisionDocumentRecord],
    ) -> dict[str, Any]:
        return {
            "scope": scope,
            "document_count": len(documents),
            "documents": [
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "document_type": document.document_type.value,
                    "thumbnail_url": f"/documents/{document.document_id}/pages/1/image?variant=original",
                    "page_count": document.page_count,
                    "field_count": len(document.extracted_fields),
                    "processing_version": document.metadata.get("pipeline_version", "legacy"),
                }
                for document in documents
            ],
        }

    def _question_has_document_evidence(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
    ) -> bool:
        if self._field_matches(question.lower(), documents):
            return True
        return bool(self._ocr_lookup(question, documents, 1))

    def _answer_structured(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        requested_names = self._requested_field_names(question)
        matches = self._field_matches(
            question.lower(), self._documents_for_question(question, documents)
        )
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
        if matches:
            logical_name = self._logical_requested_field(question)
            if logical_name and len(matches) == 1:
                shown = self._mask_sensitive(
                    matches[0].name,
                    matches[0].reviewed_value or matches[0].value,
                )
                answer = f"The {logical_name.replace('_', ' ')} shown is {shown}."
            else:
                answer = ", ".join(
                    f"{field.name}: {self._mask_sensitive(field.name, field.reviewed_value or field.value)}"
                    for field in matches
                )
            return VisionAskResponse(
                answer=answer,
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
                retrieval_path="structured_field_lookup",
            )

        ocr_matches = self._ocr_lookup(question, documents, top_k)
        if ocr_matches:
            logical_name = self._logical_requested_field(question) or "value"
            citations = [
                VisionAskCitation(
                    document_id=document.document_id,
                    filename=document.filename,
                    page_number=line.page,
                    source_text=line.text,
                    bounding_boxes=[line.bounding_box],
                    retrieval_score=line.mean_confidence,
                    retrieval_backend="ocr-line-retriever-v1",
                    model_confidence=line.mean_confidence or 0.0,
                    model_name="ocr-line-retriever",
                    model_version="v1",
                    processing_timestamp=document.updated_at,
                    reviewer_status=ReviewerStatus.unverified,
                )
                for document, line, _ in ocr_matches
            ]
            values = list(dict.fromkeys(value for _, _, value in ocr_matches))
            return VisionAskResponse(
                answer=(
                    f"The {logical_name.replace('_', ' ')} shown is {values[0]}."
                    if len(values) == 1
                    else f"The {logical_name.replace('_', ' ')} values shown are {', '.join(values)}."
                ),
                answer_status="supported",
                citations=citations,
                confidence=round(
                    sum(line.mean_confidence or 0.0 for _, line, _ in ocr_matches)
                    / len(ocr_matches),
                    4,
                ),
                reasoning_summary="Matched a field alias to source-backed OCR label-value evidence.",
                requires_review=any(
                    (line.mean_confidence or 0.0) < settings.ocr_low_confidence_threshold
                    for _, line, _ in ocr_matches
                ),
                retriever="ocr-line-retriever-v1",
                retrieval_path="ocr_label_value_fallback",
            )

        graph = self._answer_relationship(
            question,
            documents,
            top_k,
            tenant_id,
            source_fields=requested_names or None,
        )
        if graph.answer_status != "not_found":
            graph.retrieval_path = "graph_lookup"
            return graph
        # Dense semantic similarity cannot establish an exact field value. It is
        # still probed by the retrieval trace, but a related prose chunk must not
        # be substituted for a missing requested field.
        if requested_names:
            response = self._not_found_answer(requires_review=False)
            response.retrieval_path = "cascade_exhausted"
            return response
        semantic = self._answer_semantic(question, documents, top_k, tenant_id)
        if semantic.answer_status != "not_found":
            semantic.retrieval_path = "local_chunk_retrieval"
            return semantic
        response = self._not_found_answer(requires_review=False)
        response.retrieval_path = "cascade_exhausted"
        return response

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
            citation_by_id[item] for item in generated["citation_ids"] if item in citation_by_id
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
        del top_k, tenant_id
        lower = question.lower()
        tokens = set(re.findall(r"[a-z0-9]+", lower))
        selected_documents = self._documents_for_question(question, documents)
        if any(term in lower for term in ("date interval", "days between")):
            return self._answer_date_interval(question, selected_documents)
        if (
            any(term in lower for term in ("count", "how many", "number of"))
            and "invoice" in lower
        ):
            return self._answer_verified_invoice_count(selected_documents)
        invoice_po_question = bool(tokens & {"invoice", "invoices"}) and (
            "po" in tokens or "purchase order" in lower
        )
        if invoice_po_question and (
            "percentage" in tokens
            or "percent" in tokens
            or "%" in question
        ):
            return self._answer_invoice_po_variance(selected_documents)
        if (
            invoice_po_question
            and bool(
                tokens
                & {"difference", "delta", "differ", "different", "minus", "variance", "versus"}
            )
        ):
            return self._answer_invoice_po_difference(selected_documents)
        return self._answer_decimal_sum(question, selected_documents)

    def _answer_decimal_sum(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
    ) -> VisionAskResponse:
        requested = self._requested_field_names(question)
        numeric_names = requested & {"subtotal", "tax", "discount", "total_amount"}
        if not numeric_names:
            numeric_names = {"total_amount"}
        if "invoice" in question.lower():
            documents = [
                document
                for document in documents
                if document.document_type == VisionDocumentType.invoice
            ]
        parsed: list[
            tuple[VisionDocumentRecord, ExtractedVisionField, Decimal, str | None]
        ] = []
        for document in documents:
            for field_name in sorted(numeric_names):
                field = self._verified_calculation_field(document, field_name)
                if field is None:
                    continue
                value = self._decimal_value(self._calculation_field_value(field))
                if value is not None:
                    parsed.append(
                        (document, field, value, self._calculation_currency(document, field))
                    )
        if not parsed:
            return self._not_found_answer(requires_review=False)
        inputs = [
            self._calculation_input(
                document,
                field,
                role="addend",
                normalized_value=format(value, "f"),
                value_type="decimal",
                currency=currency,
            )
            for document, field, value, currency in parsed
        ]
        citations = [self._field_citation(field) for _, field, _, _ in parsed]
        currency, currency_error = self._consistent_calculation_currency(
            [item[3] for item in parsed]
        )
        if currency_error:
            return self._calculation_guardrail_response(
                operation=VisionAskCalculationOperation.sum,
                inputs=inputs,
                citations=citations,
                formula="sum(Decimal(verified input))",
                message=currency_error,
            )
        total = sum((value for _, _, value, _ in parsed), Decimal(0)).quantize(
            Decimal("0.01")
        )
        return VisionAskResponse(
            answer=f"{currency + ' ' if currency else ''}{total}",
            answer_status="supported",
            citations=citations,
            confidence=(
                self._mean_available_field_confidence(field for _, field, _, _ in parsed) or 0.0
            ),
            reasoning_summary=(
                f"Summed {len(parsed)} reviewed value(s) using deterministic Decimal arithmetic."
            ),
            requires_review=False,
            retriever="deterministic-decimal-aggregation",
            retrieval_path="deterministic_calculation",
            calculation_trace=VisionAskCalculationTrace(
                operation=VisionAskCalculationOperation.sum,
                inputs=inputs,
                result=format(total, ".2f"),
                result_type="decimal",
                currency=currency,
                formula="sum(Decimal(verified input))",
            ),
        )

    def _answer_verified_invoice_count(
        self,
        documents: list[VisionDocumentRecord],
    ) -> VisionAskResponse:
        counted: list[tuple[VisionDocumentRecord, ExtractedVisionField]] = []
        verified_document_states = {
            VisionProcessingStatus.approved,
            VisionProcessingStatus.completed,
        }
        for document in documents:
            if (
                document.document_type != VisionDocumentType.invoice
                or document.status not in verified_document_states
            ):
                continue
            identity = self._verified_calculation_field(document, "invoice_number")
            if identity is not None:
                counted.append((document, identity))
        inputs = [
            self._calculation_input(
                document,
                field,
                role="counted_invoice",
                normalized_value=self._calculation_field_value(field),
                value_type="document",
            )
            for document, field in counted
        ]
        count = len({document.document_id for document, _ in counted})
        return VisionAskResponse(
            answer=f"{count} verified invoice(s).",
            answer_status="supported",
            citations=[self._field_citation(field) for _, field in counted],
            confidence=(
                self._mean_available_field_confidence(field for _, field in counted) or 1.0
            ),
            reasoning_summary=(
                "Counted distinct approved or completed invoice documents with a reviewed "
                "invoice-number field."
            ),
            requires_review=False,
            retriever="deterministic-document-count",
            retrieval_path="deterministic_calculation",
            calculation_trace=VisionAskCalculationTrace(
                operation=VisionAskCalculationOperation.count_verified_invoices,
                inputs=inputs,
                result=str(count),
                result_type="integer",
                unit="invoices",
                formula=(
                    "count(distinct approved/completed invoice document_id with reviewed "
                    "invoice_number evidence)"
                ),
            ),
        )

    def _answer_invoice_po_difference(
        self,
        documents: list[VisionDocumentRecord],
    ) -> VisionAskResponse:
        return self._answer_invoice_po_calculation(
            documents,
            VisionAskCalculationOperation.invoice_po_difference,
        )

    def _answer_invoice_po_variance(
        self,
        documents: list[VisionDocumentRecord],
    ) -> VisionAskResponse:
        return self._answer_invoice_po_calculation(
            documents,
            VisionAskCalculationOperation.percentage_variance,
        )

    def _answer_invoice_po_calculation(
        self,
        documents: list[VisionDocumentRecord],
        operation: VisionAskCalculationOperation,
    ) -> VisionAskResponse:
        pair = self._linked_invoice_po_pair(documents)
        if pair is None:
            return VisionAskResponse(
                answer=(
                    "Select one invoice and its linked purchase order so the calculation has "
                    "an unambiguous evidence pair."
                ),
                answer_status="clarification_needed",
                confidence=0.0,
                reasoning_summary="No unique reviewed invoice-to-purchase-order pair was found.",
                requires_review=False,
                retriever="deterministic-calculation-engine-v1",
                retrieval_path="calculation_pair_resolution",
            )
        invoice, purchase_order = pair
        invoice_total = self._verified_calculation_field(invoice, "total_amount")
        po_total = self._verified_calculation_field(purchase_order, "total_amount")
        if invoice_total is None or po_total is None:
            return self._not_found_answer(requires_review=False)
        invoice_value = self._decimal_value(self._calculation_field_value(invoice_total))
        po_value = self._decimal_value(self._calculation_field_value(po_total))
        if invoice_value is None or po_value is None:
            return self._not_found_answer(requires_review=False)
        invoice_currency = self._calculation_currency(invoice, invoice_total)
        po_currency = self._calculation_currency(purchase_order, po_total)
        inputs = [
            self._calculation_input(
                invoice,
                invoice_total,
                role="invoice_amount",
                normalized_value=format(invoice_value, "f"),
                value_type="decimal",
                currency=invoice_currency,
            ),
            self._calculation_input(
                purchase_order,
                po_total,
                role="purchase_order_amount",
                normalized_value=format(po_value, "f"),
                value_type="decimal",
                currency=po_currency,
            ),
        ]
        citations = [
            self._field_citation(invoice_total),
            self._field_citation(po_total),
        ]
        currency, currency_error = self._consistent_calculation_currency(
            [invoice_currency, po_currency]
        )
        formula = (
            "((Decimal(invoice_total) - Decimal(purchase_order_total)) / "
            "Decimal(purchase_order_total)) * 100"
            if operation == VisionAskCalculationOperation.percentage_variance
            else "Decimal(invoice_total) - Decimal(purchase_order_total)"
        )
        if currency_error:
            return self._calculation_guardrail_response(
                operation=operation,
                inputs=inputs,
                citations=citations,
                formula=formula,
                message=currency_error,
            )
        difference = invoice_value - po_value
        if operation == VisionAskCalculationOperation.percentage_variance:
            if po_value == 0:
                return self._calculation_guardrail_response(
                    operation=operation,
                    inputs=inputs,
                    citations=citations,
                    formula=formula,
                    message=(
                        "Percentage variance cannot be calculated because the reviewed "
                        "purchase-order total is zero."
                    ),
                )
            result = ((difference / po_value) * Decimal(100)).quantize(Decimal("0.01"))
            answer = f"The invoice-to-purchase-order percentage variance is {result}%."
            result_type = "percentage"
            unit = "percent"
            result_currency = None
        else:
            result = difference.quantize(Decimal("0.01"))
            answer = (
                "Invoice total minus purchase-order total is "
                f"{currency + ' ' if currency else ''}{result}."
            )
            result_type = "decimal"
            unit = None
            result_currency = currency
        return VisionAskResponse(
            answer=answer,
            answer_status="supported",
            citations=citations,
            confidence=self._mean_available_field_confidence(
                [invoice_total, po_total]
            )
            or 0.0,
            reasoning_summary=(
                "Used one uniquely linked invoice/PO pair and reviewed totals with deterministic "
                "Decimal arithmetic."
            ),
            requires_review=False,
            retriever="deterministic-calculation-engine-v1",
            retrieval_path="deterministic_calculation",
            calculation_trace=VisionAskCalculationTrace(
                operation=operation,
                inputs=inputs,
                result=format(result, ".2f"),
                result_type=result_type,
                currency=result_currency,
                unit=unit,
                formula=formula,
            ),
        )

    def _answer_date_interval(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
    ) -> VisionAskResponse:
        lower = question.lower()
        if "order date" in lower or "delivery date" in lower:
            start_name, end_name = "order_date", "delivery_date"
        elif "effective date" in lower or "expiry date" in lower:
            start_name, end_name = "effective_date", "expiry_date"
        else:
            start_name, end_name = "invoice_date", "due_date"
        candidates: list[
            tuple[
                VisionDocumentRecord,
                ExtractedVisionField,
                ExtractedVisionField,
                datetime,
                datetime,
            ]
        ] = []
        for document in documents:
            start_field = self._verified_calculation_field(document, start_name)
            end_field = self._verified_calculation_field(document, end_name)
            if start_field is None or end_field is None:
                continue
            start_date = self._date(self._calculation_field_value(start_field))
            end_date = self._date(self._calculation_field_value(end_field))
            if start_date is not None and end_date is not None:
                candidates.append((document, start_field, end_field, start_date, end_date))
        if len(candidates) > 1:
            return VisionAskResponse(
                answer="Select one document so the requested date interval is unambiguous.",
                answer_status="clarification_needed",
                confidence=0.0,
                reasoning_summary="More than one document contains the reviewed date pair.",
                requires_review=False,
                retriever="deterministic-calculation-engine-v1",
                retrieval_path="calculation_pair_resolution",
            )
        if not candidates:
            return self._not_found_answer(requires_review=False)
        document, start_field, end_field, start_date, end_date = candidates[0]
        interval = (end_date.date() - start_date.date()).days
        inputs = [
            self._calculation_input(
                document,
                start_field,
                role="start_date",
                normalized_value=start_date.date().isoformat(),
                value_type="date",
            ),
            self._calculation_input(
                document,
                end_field,
                role="end_date",
                normalized_value=end_date.date().isoformat(),
                value_type="date",
            ),
        ]
        return VisionAskResponse(
            answer=(
                f"The interval from {start_name.replace('_', ' ')} to "
                f"{end_name.replace('_', ' ')} is {interval} day(s)."
            ),
            answer_status="supported",
            citations=[self._field_citation(start_field), self._field_citation(end_field)],
            confidence=self._mean_available_field_confidence([start_field, end_field]) or 0.0,
            reasoning_summary="Parsed reviewed dates and subtracted calendar dates deterministically.",
            requires_review=False,
            retriever="deterministic-date-calculation",
            retrieval_path="deterministic_calculation",
            calculation_trace=VisionAskCalculationTrace(
                operation=VisionAskCalculationOperation.date_interval,
                inputs=inputs,
                result=str(interval),
                result_type="integer",
                unit="days",
                formula="date(end_date) - date(start_date)",
            ),
        )

    def _linked_invoice_po_pair(
        self,
        documents: list[VisionDocumentRecord],
    ) -> tuple[VisionDocumentRecord, VisionDocumentRecord] | None:
        invoices = [
            item for item in documents if item.document_type == VisionDocumentType.invoice
        ]
        purchase_orders = [
            item
            for item in documents
            if item.document_type == VisionDocumentType.purchase_order
        ]
        linked = []
        for invoice in invoices:
            invoice_reference = self._verified_calculation_field(
                invoice, "purchase_order_number"
            )
            if invoice_reference is None:
                continue
            normalized_reference = self._normalize(
                self._calculation_field_value(invoice_reference)
            )
            for purchase_order in purchase_orders:
                po_reference = self._verified_calculation_field(
                    purchase_order, "purchase_order_number"
                )
                if (
                    po_reference is not None
                    and normalized_reference
                    and normalized_reference
                    == self._normalize(self._calculation_field_value(po_reference))
                ):
                    linked.append((invoice, purchase_order))
        if len(linked) == 1:
            return linked[0]
        if not linked and len(invoices) == 1 and len(purchase_orders) == 1:
            return invoices[0], purchase_orders[0]
        return None

    def _verified_calculation_field(
        self,
        document: VisionDocumentRecord,
        field_name: str,
    ) -> ExtractedVisionField | None:
        return next(
            (
                field
                for field in document.extracted_fields
                if field.name == field_name
                and field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
            ),
            None,
        )

    def _calculation_field_value(self, field: ExtractedVisionField) -> str:
        if field.reviewed_value is not None:
            return field.reviewed_value
        return field.normalized_value or field.value

    def _calculation_input(
        self,
        document: VisionDocumentRecord,
        field: ExtractedVisionField,
        *,
        role: str,
        normalized_value: str,
        value_type: str,
        currency: str | None = None,
    ) -> VisionAskCalculationInput:
        return VisionAskCalculationInput(
            document_id=document.document_id,
            document_type=document.document_type,
            role=role,
            field_name=field.name,
            raw_value=self._calculation_field_value(field),
            normalized_value=normalized_value,
            value_type=value_type,
            currency=currency,
            evidence_id=field.evidence.evidence_id,
            page_number=field.evidence.page_number,
            reviewer_status=field.reviewer_status,
        )

    def _calculation_currency(
        self,
        document: VisionDocumentRecord,
        amount_field: ExtractedVisionField,
    ) -> str | None:
        currency_field = self._verified_calculation_field(document, "currency")
        candidates = []
        if currency_field is not None:
            candidates.append(self._calculation_field_value(currency_field))
        candidates.append(self._calculation_field_value(amount_field))
        known_codes = {
            "AED",
            "AUD",
            "CAD",
            "CHF",
            "CNY",
            "EUR",
            "GBP",
            "INR",
            "JPY",
            "SGD",
            "USD",
        }
        symbol_codes = {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "¥": "JPY"}
        for candidate in candidates:
            upper = candidate.upper()
            code = next(
                (item for item in re.findall(r"\b[A-Z]{3}\b", upper) if item in known_codes),
                None,
            )
            if code:
                return code
            for symbol, symbol_code in symbol_codes.items():
                if symbol in candidate:
                    return symbol_code
        return None

    def _consistent_calculation_currency(
        self,
        currencies: list[str | None],
    ) -> tuple[str | None, str | None]:
        known = {currency for currency in currencies if currency}
        if len(known) > 1:
            return None, (
                "The calculation spans multiple currencies and cannot be combined without "
                "verified conversion data."
            )
        if known and any(currency is None for currency in currencies):
            return None, (
                "The calculation mixes currency-labelled and unlabelled values, so it was "
                "stopped until every input currency is verified."
            )
        return next(iter(known), None), None

    def _calculation_guardrail_response(
        self,
        *,
        operation: VisionAskCalculationOperation,
        inputs: list[VisionAskCalculationInput],
        citations: list[VisionAskCitation],
        formula: str,
        message: str,
    ) -> VisionAskResponse:
        return VisionAskResponse(
            answer=message,
            answer_status="partially_supported",
            citations=citations,
            confidence=0.0,
            reasoning_summary="Deterministic calculation stopped at a calculation guardrail.",
            requires_review=True,
            retriever="deterministic-calculation-engine-v1",
            retrieval_path="calculation_guardrail",
            calculation_trace=VisionAskCalculationTrace(
                operation=operation,
                inputs=inputs,
                result=None,
                result_type="not_calculated",
                formula=formula,
            ),
        )

    def _answer_visual(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del tenant_id
        detections = self._matching_visual_detections(question, documents)[:top_k]
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
                extraction_status=detection.reviewer_status.value,
                evidence_type="visual_detection",
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
            retrieval_path="visual_detection_lookup",
        )

    def _matching_visual_detections(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
    ) -> list[tuple[VisionDocumentRecord, Any]]:
        lower = question.lower()
        class_terms = {
            "signature": ("signature", "signed", "safety", "safe"),
            "stamp": ("stamp", "safety", "safe"),
            "seal": ("seal", "safety", "safe"),
            "checkbox_checked": ("checkbox", "checked"),
            "checkbox_unchecked": ("checkbox", "unchecked"),
            "company_logo": ("logo",),
            "handwritten_note": ("handwritten",),
        }
        requested_classes = {
            name for name, terms in class_terms.items() if any(term in lower for term in terms)
        }
        return [
            (document, detection)
            for document in documents
            for detection in document.visual_detections
            if not requested_classes or detection.detection_class in requested_classes
        ]

    def _answer_relationship(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
        *,
        source_fields: set[str] | None = None,
    ) -> VisionAskResponse:
        neo4j_paths = self.persistence.query_graph_paths(
            tenant_id,
            question,
            [document.document_id for document in documents],
            limit=top_k,
        )
        matches = self._neo4j_graph_matches(neo4j_paths, documents)
        graph_backend = "neo4j-evidence-path-v1" if matches else "local-semantic-graph-path-v1"
        if not matches:
            matches = self._matching_graph_edges(question, documents)
        if source_fields:
            matches = [
                item for item in matches if str(item[1].get("source_field") or "") in source_fields
            ]
        matches = matches[:top_k]
        if not matches:
            return self._not_found_answer(requires_review=False)
        citations = [
            VisionAskCitation(
                document_id=document.document_id,
                filename=document.filename,
                page_number=int(edge.get("source_page", 1)),
                source_text=str(edge.get("evidence_text", "")),
                bounding_boxes=[
                    VisionBoundingBox.model_validate(box) for box in edge.get("bounding_boxes", [])
                ],
                evidence_id=edge.get("evidence_id"),
                retrieval_score=float(edge.get("confidence", 0.0) or 0.0),
                retrieval_backend=graph_backend,
                model_confidence=float(edge.get("confidence", 0.0) or 0.0),
                model_name=str(edge.get("model_name", "deterministic-graph-builder")),
                model_version=str(edge.get("model_version", "v1")),
                processing_timestamp=document.updated_at,
                reviewer_status=self._graph_reviewer_status(edge),
                extraction_status=str(edge.get("status", "proposed")),
                evidence_type="graph_path",
            )
            for document, edge, _, _ in matches
        ]
        paths = [
            {
                "path_id": f"graph-path-{edge.get('id')}",
                "document_id": document.document_id,
                "node_ids": [str(edge.get("source")), str(edge.get("target"))],
                "relationships": [str(edge.get("type"))],
                "display_path": (
                    f"{source_node.get('label', edge.get('source'))} → "
                    f"{str(edge.get('type', '')).replace('_', ' ')} → "
                    f"{target_node.get('label', edge.get('target'))}"
                ),
                "explanation": self._graph_path_explanation(
                    str(edge.get("type", "")),
                    str(source_node.get("label", edge.get("source", ""))),
                    str(target_node.get("label", edge.get("target", ""))),
                ),
                "verification_status": str(edge.get("status", "proposed")),
                "evidence_id": edge.get("evidence_id"),
                "source_page": int(edge.get("source_page", 1)),
                "source_text": str(edge.get("evidence_text", "")),
                "source_field": edge.get("source_field"),
            }
            for document, edge, source_node, target_node in matches
        ]
        answer = "; ".join(path["explanation"] for path in paths)
        return VisionAskResponse(
            answer=answer,
            answer_status=(
                "supported"
                if all(edge.get("status") == "verified" for _, edge, _, _ in matches)
                else "partially_supported"
            ),
            citations=citations,
            confidence=round(
                sum(float(edge.get("confidence", 0.0) or 0.0) for _, edge, _, _ in matches)
                / len(matches),
                4,
            ),
            reasoning_summary=(
                "Traversed semantic graph relationships and returned each path with its "
                "originating field or OCR evidence."
            ),
            requires_review=any(edge.get("status") != "verified" for _, edge, _, _ in matches),
            retriever=graph_backend,
            retrieval_path="graph_path_query",
            graph_paths=paths,
        )

    def _neo4j_graph_matches(
        self,
        paths: list[dict[str, Any]],
        documents: list[VisionDocumentRecord],
    ) -> list[
        tuple[
            VisionDocumentRecord,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ]
    ]:
        by_document = {document.document_id: document for document in documents}
        matches = []
        for path in paths:
            document = by_document.get(str(path.get("document_id") or ""))
            if document is None:
                continue
            properties = dict(path.get("relationship_properties") or {})
            bounding_boxes = self._neo4j_json_value(
                properties.get("bounding_boxes"),
                [],
            )
            bounding_box = self._neo4j_json_value(
                properties.get("bounding_box"),
                None,
            )
            if not bounding_boxes and isinstance(bounding_box, dict):
                bounding_boxes = [bounding_box]
            relationship = str(path.get("relationship_type") or "")
            edge = {
                **properties,
                "id": properties.get("edge_id") or f"neo4j:{document.document_id}:{relationship}",
                "source": document.document_id,
                "target": str(path.get("target_id") or ""),
                "type": relationship,
                "source_page": int(properties.get("source_page") or 1),
                "evidence_text": str(properties.get("evidence_text") or ""),
                "status": str(properties.get("status") or "proposed"),
                "confidence": float(properties.get("confidence") or 0.0),
                "model_name": "neo4j-evidence-path",
                "model_version": "v1",
                "bounding_boxes": (bounding_boxes if isinstance(bounding_boxes, list) else []),
            }
            source_node = {
                "id": document.document_id,
                "label": str(path.get("source_label") or document.filename),
            }
            target_properties = dict(path.get("target_properties") or {})
            target_node = {
                **target_properties,
                "id": str(path.get("target_id") or ""),
                "label": str(
                    path.get("target_label")
                    or target_properties.get("display_label")
                    or path.get("target_id")
                    or "related entity"
                ),
            }
            matches.append((document, edge, source_node, target_node))
        return matches

    def _neo4j_json_value(self, value: object, default: object) -> object:
        if not isinstance(value, str):
            return value if value is not None else default
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default

    def _matching_graph_edges(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
    ) -> list[
        tuple[
            VisionDocumentRecord,
            dict[str, Any],
            dict[str, Any],
            dict[str, Any],
        ]
    ]:
        lower = question.lower()
        intents: set[str] = set()
        intent_terms = {
            "REQUESTED_BY": ("who requested", "original request", "requested by", "requester"),
            "SPECIFICATIONS_WRITTEN_BY": (
                "who wrote",
                "sample specifications",
                "specifications written by",
            ),
            "HAS_SAMPLE_NUMBER": ("sample number", "sample no", "sample id"),
            "HAS_PRODUCT_TYPE": ("product type", "type of product", "type of cigarette"),
            "HAS_BATCH_SIZE": ("batch size", "lot size", "quantity"),
            "HAS_DATE": ("what date", "document date", "form date"),
            "HAS_REQUEST_DATE": ("request date", "when was the request", "request made"),
            "HAS_PURPOSE": ("purpose", "what was the purpose"),
            "HAS_FAX": ("fax number", "fax no", "facsimile", "telefax"),
            "ISSUED_BY": ("issued by", "vendor", "supplier"),
            "ISSUED_TO": ("issued to", "buyer", "customer"),
            "SIGNED_BY": ("signed by", "who signed"),
            "REFERENCES": ("references", "linked", "related"),
            "HAS_BANK_ACCOUNT": ("bank account", "paid to"),
            "HAS_RISK": ("risk", "suspicious", "warning"),
        }
        for relationship, terms in intent_terms.items():
            if any(term in lower for term in terms):
                intents.add(relationship)
        stopwords = {
            "are",
            "does",
            "document",
            "for",
            "from",
            "how",
            "is",
            "of",
            "the",
            "this",
            "to",
            "what",
            "which",
            "who",
            "with",
        }
        tokens = {
            token
            for token in re.findall(r"[a-z0-9]+", lower)
            if len(token) >= 3 and token not in stopwords
        }
        ranked: list[
            tuple[
                int,
                VisionDocumentRecord,
                dict[str, Any],
                dict[str, Any],
                dict[str, Any],
            ]
        ] = []
        for document in documents:
            nodes = {str(node.get("id")): node for node in document.graph.nodes}
            for edge in document.graph.edges:
                relationship = str(edge.get("type", ""))
                if relationship in {"HAS_FIELD", "MATCHES", "CONTAINS_PAGE", "CONTAINS"}:
                    continue
                source_node = nodes.get(str(edge.get("source")), {})
                target_node = nodes.get(str(edge.get("target")), {})
                haystack = " ".join(
                    (
                        relationship,
                        str(source_node.get("label", "")),
                        str(target_node.get("label", "")),
                        str(edge.get("evidence_text", "")),
                        str(edge.get("source_field", "")),
                    )
                ).lower()
                edge_tokens = set(re.findall(r"[a-z0-9]+", haystack))
                if intents and relationship not in intents:
                    continue
                overlap = len(tokens & edge_tokens)
                if not intents and not overlap:
                    continue
                score = (100 if relationship in intents else 0) + overlap * 5
                score += 3 if edge.get("status") == "verified" else 0
                ranked.append((score, document, edge, source_node, target_node))
        ranked.sort(
            key=lambda item: (item[0], float(item[2].get("confidence", 0.0) or 0.0)),
            reverse=True,
        )
        return [(document, edge, source, target) for _, document, edge, source, target in ranked]

    def _graph_reviewer_status(self, edge: dict[str, Any]) -> ReviewerStatus:
        try:
            return ReviewerStatus(str(edge.get("review_status", "unverified")))
        except ValueError:
            return (
                ReviewerStatus.accepted
                if edge.get("status") == "verified"
                else ReviewerStatus.unverified
            )

    def _graph_path_explanation(
        self,
        relationship: str,
        source_label: str,
        target_label: str,
    ) -> str:
        templates = {
            "REQUESTED_BY": f"{target_label} is recorded as the person who requested the sample",
            "SPECIFICATIONS_WRITTEN_BY": (
                f"{target_label} is recorded as the person who wrote the sample specifications"
            ),
            "HAS_SAMPLE_NUMBER": f"{source_label} has sample number {target_label}",
            "HAS_PRODUCT_TYPE": f"{source_label} identifies the product as {target_label}",
            "HAS_BATCH_SIZE": f"{source_label} records a batch size of {target_label}",
            "HAS_DATE": f"{source_label} records the date as {target_label}",
            "HAS_REQUEST_DATE": f"{source_label} records the request date as {target_label}",
            "HAS_PURPOSE": f"{source_label} records the purpose as {target_label}",
            "ISSUED_BY": f"{source_label} was issued by {target_label}",
            "ISSUED_TO": f"{source_label} was issued to {target_label}",
            "SIGNED_BY": f"{source_label} is linked to signer {target_label}",
            "HAS_BANK_ACCOUNT": f"{source_label} is associated with bank account {target_label}",
            "HAS_RISK": f"{source_label} has the source-linked risk indicator {target_label}",
        }
        return templates.get(
            relationship,
            f"{source_label} is connected to {target_label} by {relationship.replace('_', ' ').lower()}",
        )

    def _answer_quality(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del question, tenant_id
        unverified = [
            field
            for document in documents
            for field in document.extracted_fields
            if field.reviewer_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
        ]
        missing = sum(
            len(document.field_completeness.missing_required_fields) for document in documents
        )
        disputed = sum(len(document.field_completeness.disputed_fields) for document in documents)
        quality_scores = [
            report.image_quality_score for document in documents for report in document.quality
        ]
        warnings = [
            (document, warning)
            for document in documents
            for warning in document.processing_warnings
        ]
        citations = [self._field_citation(field) for field in unverified[:top_k]]
        citations.extend(
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
                extraction_status=evidence.reviewer_status.value,
                evidence_type="processing_warning",
            )
            for document, warning in warnings
            for evidence in warning.evidence[:1]
        )
        citations.extend(
            self._document_state_citation(
                document,
                (
                    f"Persisted data-quality summary: {len(document.extracted_fields)} extracted "
                    f"field(s), {len(document.field_completeness.missing_required_fields)} "
                    "missing required field(s)."
                ),
                "data_quality_summary",
            )
            for document in documents
        )
        citations = self._deduplicate_citations(citations)[: max(top_k, 8)]
        if not citations:
            return self._not_found_answer(requires_review=False)
        quality_label = (
            f"mean page-quality score {sum(quality_scores) / len(quality_scores):.0%}"
            if quality_scores
            else "page-quality score not available"
        )
        requires_review = bool(unverified or missing or disputed or warnings)
        return VisionAskResponse(
            answer=(
                f"Processing quality: {quality_label}; {len(warnings)} source-linked warning(s). "
                f"Data quality: {len(unverified)} unverified field(s), {missing} unresolved "
                f"required field(s), and {disputed} disputed field(s). "
                + (
                    "Human review is required before relying on the document."
                    if requires_review
                    else "No unresolved data-quality item is recorded in the selected scope."
                )
            ),
            answer_status="partially_supported" if requires_review else "supported",
            citations=citations,
            confidence=self._mean_available_field_confidence(unverified) or 1.0,
            reasoning_summary=(
                "Combined persisted page-quality results, processing warnings, field verification "
                "states and schema completeness without making a fraud or legal determination."
            ),
            requires_review=requires_review,
            retriever="deterministic-document-quality-summary-v1",
            retrieval_path="multi_source_quality_summary",
        )

    def _answer_approval(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del question, tenant_id
        selected = documents[:top_k]
        if not selected:
            return self._not_found_answer(requires_review=False)
        citations = [
            self._document_state_citation(
                document,
                (
                    f"Persisted workflow status for {document.filename}: {document.status.value}; "
                    f"unresolved required fields: "
                    f"{len(document.field_completeness.missing_required_fields)}."
                ),
                "workflow_state",
            )
            for document in selected
        ]
        return VisionAskResponse(
            answer="; ".join(
                (
                    f"{document.filename}: {document.status.value.replace('_', ' ').title()}"
                    f" ({len(document.field_completeness.missing_required_fields)} unresolved "
                    "required field(s))"
                )
                for document in selected
            ),
            answer_status="supported",
            citations=citations,
            confidence=1.0,
            reasoning_summary="Returned persisted workflow state and deterministic completeness counts.",
            requires_review=any(
                document.status != VisionProcessingStatus.approved for document in selected
            ),
            retriever="persisted-approval-workflow-v1",
            retrieval_path="approval_status_lookup",
        )

    def _answer_safety(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del question, tenant_id
        if not documents:
            return self._not_found_answer(requires_review=False)
        unverified = [
            field
            for document in documents
            for field in document.extracted_fields
            if field.reviewer_status not in {ReviewerStatus.accepted, ReviewerStatus.edited}
        ]
        missing = sum(
            len(document.field_completeness.missing_required_fields) for document in documents
        )
        linked_risks = [
            (document, risk)
            for document in documents
            for risk in document.risk_indicators
            if risk.evidence
        ]
        manipulation_regions = [
            (document, analysis, region)
            for document in documents
            for analysis in document.tamper_analysis
            for region in analysis.suspicious_regions
        ]
        confirmed_fraud = any(
            document.status == VisionProcessingStatus.confirmed_fraud for document in documents
        )
        statuses = ", ".join(
            f"{document.filename}: {document.status.value.replace('_', ' ').title()}"
            for document in documents
        )
        citations = [self._field_citation(field) for field in unverified[:top_k]]
        citations.extend(
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
                extraction_status=evidence.reviewer_status.value,
                evidence_type="risk_indicator",
            )
            for document, risk in linked_risks
            for evidence in risk.evidence[:1]
        )
        citations.extend(
            VisionAskCitation(
                document_id=document.document_id,
                filename=document.filename,
                page_number=analysis.page_number,
                source_text=(
                    f"Visual-manipulation screening candidate: {region.manipulation_class}. "
                    "This is not a fraud determination."
                ),
                bounding_boxes=[region.bounding_box],
                evidence_id=region.region_id,
                model_confidence=region.screening_score,
                model_name=analysis.model_name,
                model_version=analysis.model_version,
                processing_timestamp=document.updated_at,
                reviewer_status=analysis.reviewer_status,
                extraction_status=analysis.reviewer_status.value,
                evidence_type="visual_manipulation_screening",
            )
            for document, analysis, region in manipulation_regions
        )
        citations.extend(
            self._document_state_citation(
                document, f"Persisted workflow status: {document.status.value}.", "workflow_state"
            )
            for document in documents
        )
        citations = self._deduplicate_citations(citations)[: max(top_k, 8)]
        requires_review = any(
            document.status != VisionProcessingStatus.approved for document in documents
        ) or bool(unverified or missing or linked_risks or manipulation_regions)
        determination = (
            "A reviewer-confirmed fraud workflow status is recorded."
            if confirmed_fraud
            else "No confirmed fraud determination is available."
        )
        return VisionAskResponse(
            answer=(
                f"{determination} Processing quality: {sum(len(d.quality) for d in documents)} "
                f"page-quality record(s). Data-quality issues: {len(unverified)} unverified "
                f"field(s) and {missing} unresolved required field(s). Business-rule indicators: "
                f"{len(linked_risks)} source-linked indicator(s). Visual-manipulation indicators: "
                f"{len(manipulation_regions)} screening candidate(s). Approval status: {statuses}. "
                "This evidence summary is not a legal, authenticity or fraud conclusion."
            ),
            answer_status="partially_supported" if requires_review else "supported",
            citations=citations,
            confidence=self._mean_available_field_confidence(unverified) or 1.0,
            reasoning_summary=(
                "Separated processing quality, data quality, business-rule indicators, visual "
                "screening and workflow status using only persisted evidence and deterministic counts."
            ),
            requires_review=requires_review,
            retriever="multi-source-safety-summary-v1",
            retrieval_path="multi_source_safety_summary",
        )

    def _document_state_citation(
        self,
        document: VisionDocumentRecord,
        source_text: str,
        evidence_type: str,
    ) -> VisionAskCitation:
        return VisionAskCitation(
            document_id=document.document_id,
            filename=document.filename,
            page_number=1,
            source_text=source_text,
            bounding_boxes=[],
            evidence_id=f"state:{document.document_id}:{evidence_type}",
            retrieval_score=1.0,
            retrieval_backend="persisted-document-state-v1",
            model_confidence=1.0,
            model_name="persisted-document-state",
            model_version="v1",
            processing_timestamp=document.updated_at,
            reviewer_status=(
                ReviewerStatus.accepted
                if document.status == VisionProcessingStatus.approved
                else ReviewerStatus.unverified
            ),
            extraction_status="persisted",
            evidence_type=evidence_type,
        )

    def _deduplicate_citations(
        self,
        citations: list[VisionAskCitation],
    ) -> list[VisionAskCitation]:
        unique: dict[str, VisionAskCitation] = {}
        for citation in citations:
            key = str(
                citation.evidence_id
                or citation.chunk_id
                or f"{citation.document_id}:{citation.page_number}:{citation.source_text}"
            )
            unique.setdefault(key, citation)
        return list(unique.values())

    def _answer_risk(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
        tenant_id: str,
    ) -> VisionAskResponse:
        del question, tenant_id
        matches = [(document, risk) for document in documents for risk in document.risk_indicators][
            :top_k
        ]
        if not matches:
            response = self._not_found_answer(requires_review=False)
            response.reasoning_summary = (
                "No stored business or visual risk indicator matched the scope."
            )
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
        confidence_values = [risk.confidence for _, risk in matches if risk.confidence is not None]
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
        conversation: VisionConversationState | None = None,
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
        if conversation is not None:
            conversation.turns.extend(
                [
                    {
                        "role": "user",
                        "question_id": question_id,
                        "content": request.question,
                        "clarification_option": request.clarification_option,
                        "created_at": self.questions[question_id]["created_at"],
                    },
                    {
                        "role": "assistant",
                        "question_id": question_id,
                        "answer_status": response.answer_status,
                        "query_type": response.query_type,
                        "content": response.answer,
                        "retrieval_trace_id": response.retrieval_trace_id,
                        "created_at": self.questions[question_id]["created_at"],
                    },
                ]
            )
            self._persist_conversation(conversation)
        return response

    def get_question(self, question_id: str, tenant_id: str | None = None) -> dict[str, Any]:
        question = self.questions.get(question_id) or self._load_question(question_id)
        if not question and tenant_id:
            question = self.persistence.get_question(question_id, tenant_id)
            if question:
                self.questions[question_id] = question
        if not question or (tenant_id and question.get("tenant_id", "tenant-local") != tenant_id):
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
                (edge for edge in candidate.graph.edges if edge.get("id") == relationship_id),
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
        extraction_runs = [run for run in document.model_runs if run.stage == "extraction"]
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
                if selected_field and "layoutlm" in selected_field.model_name.lower()
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
            required = bool(
                schema and schema.field(field_name) and schema.field(field_name).required
            )
            if selected_field:
                validation_result = (
                    "accepted"
                    if selected_field.reviewer_status
                    in {ReviewerStatus.accepted, ReviewerStatus.edited}
                    else "requires_review"
                    if selected_field.confidence is None
                    or selected_field.confidence < self._mandatory_review_threshold(document)
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
                    "selected_result": self._mask_sensitive(
                        field_name, selected_field.reviewed_value or selected_field.value
                    )
                    if selected_field
                    else None,
                    "review_status": selected_field.reviewer_status.value
                    if selected_field
                    else "unresolved"
                    if required
                    else "not_available",
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
                if str(item["model_id"]) == model_id or str(item.get("adapter_id")) == model_id
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
            contributing = [
                self.get_document(document_id, tenant_id) for document_id in narrative.document_ids
            ]
            source_ids = narrative.source_document_ids or narrative.document_ids
            selected = [self.get_document(document_id, tenant_id) for document_id in source_ids]
            source_data_version = self._artifact_source_data_version(selected)
            request_scope = request.model_dump(
                mode="json",
                exclude={"generate_new_version"},
            )
            request_scope["document_ids"] = sorted(request.document_ids or source_ids)
            request_scope["requested_document_ids"] = sorted(request.document_ids)
            request_scope["source_document_ids"] = sorted(source_ids)
            request_scope["excluded_document_ids"] = sorted(narrative.excluded_document_ids)
            duplicate_exclusion_signature = sorted(
                {
                    trace.calculation_function
                    for trace in narrative.calculations
                    if trace.label == "Duplicate invoice documents excluded"
                }
            )
            request_scope["duplicate_invoice_exclusions"] = duplicate_exclusion_signature
            identity_hash = self._artifact_identity_hash(
                {
                    "organisation_id": tenant_id,
                    "narrative_type": request.narrative_type.value,
                    "vendor_filter": self._normalize(request.vendor_name or ""),
                    "from_date": request.date_from,
                    "to_date": request.date_to,
                    "currency": sorted(
                        {
                            self._field_map(document).get("currency", "UNSPECIFIED")
                            for document in selected
                        }
                    ),
                    "document_ids": sorted(set(narrative.document_ids)),
                    "source_document_ids": sorted(set(source_ids)),
                    "duplicate_invoice_exclusions": duplicate_exclusion_signature,
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
            for document in selected:
                self._audit(
                    document,
                    "finance_narrative_generated",
                    f"Generated {narrative.narrative_type.value} version {narrative.version}.",
                    actor=actor,
                    metadata={
                        "narrative_id": narrative.narrative_id,
                        "identity_hash": narrative.identity_hash,
                        "version": narrative.version,
                        "contributing": document.document_id
                        in {item.document_id for item in contributing},
                        "excluded_duplicate": document.document_id
                        in set(narrative.excluded_document_ids),
                    },
                )
                self._persist_document(document)
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
        scoped = [item for item in self.finance_narratives.values() if item.tenant_id == tenant_id]
        canonical: dict[str, VisionFinanceNarrative] = {}
        for item in scoped:
            identity = item.identity_hash or self._artifact_identity_hash(
                {
                    "organisation_id": item.tenant_id,
                    "narrative_type": item.narrative_type.value,
                    "request_scope": item.request_scope,
                    "document_ids": sorted(set(item.document_ids)),
                    "source_document_ids": sorted(
                        set(item.source_document_ids or item.document_ids)
                    ),
                    "verified_data_version": item.source_data_version,
                }
            )
            current = canonical.get(identity)
            if current is None or (item.version, item.created_at) > (
                current.version,
                current.created_at,
            ):
                canonical[identity] = item
        active = [
            item
            for item in canonical.values()
            if item.status not in {"archived", "superseded", "stale"}
        ]
        return sorted(active, key=lambda item: item.created_at, reverse=True)

    def generate_report(
        self,
        request: VisionReportRequest,
        *,
        tenant_id: str,
        actor: str,
    ) -> VisionReportRecord:
        with self._artifact_lock:
            narratives = self.list_finance_narratives(tenant_id)
            if request.narrative_id and not any(
                item.narrative_id == request.narrative_id for item in narratives
            ):
                try:
                    narratives.append(self.get_finance_narrative(request.narrative_id, tenant_id))
                except ValueError:
                    pass
            report = vision_report_service.generate(
                request,
                self.list_documents(tenant_id),
                narratives,
                tenant_id=tenant_id,
                actor=actor,
            )
            selected = [
                self.get_document(document_id, tenant_id) for document_id in report.document_ids
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
                item for item in self.list_reports(tenant_id) if item.identity_hash == identity_hash
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
                any(
                    field.reviewer_status == ReviewerStatus.edited
                    for field in document.extracted_fields
                )
            ),
        }

    def render_report_markdown(
        self,
        document_id: str,
        tenant_id: str | None = None,
    ) -> str:
        payload = self.render_report_payload(document_id, tenant_id)
        document = self.get_document(document_id, tenant_id)
        fields = (
            "\n".join(
                f"| {field.name} | {self._mask_sensitive(field.name, field.reviewed_value or field.value)} | "
                f"{field.confidence:.0%} | {field.evidence.evidence_id} | {field.reviewer_status.value} |"
                for field in document.extracted_fields
                if field.confidence is not None
            )
            or "| None | - | - | - | - |"
        )
        risks = (
            "\n".join(
                f"| {risk.title} | {risk.severity.value} | {risk.category} | {risk.recommended_action} |"
                for risk in document.risk_indicators
            )
            or "| None | - | - | - |"
        )
        tables = (
            "\n\n".join(table.markdown for table in document.tables)
            or "No structural tables detected."
        )
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
        processing_started_perf = time.perf_counter()
        processing_started_at = utc_now_iso()
        stage_durations_ms: dict[str, float] = {}
        queued_event = next(
            (
                event
                for event in reversed(document.audit_events)
                if event.event_type in {"document_queued", "document_reprocess_queued"}
            ),
            None,
        )
        queued_at = queued_event.created_at if queued_event else None
        queued_duration_ms: float | None = None
        if queued_at:
            try:
                queued_duration_ms = max(
                    0.0,
                    (
                        datetime.fromisoformat(processing_started_at)
                        - datetime.fromisoformat(queued_at)
                    ).total_seconds()
                    * 1000,
                )
            except (TypeError, ValueError):
                queued_duration_ms = None
        document.metadata["pipeline_version"] = CURRENT_PIPELINE_VERSION
        document.metadata["processing_timing"] = {
            "duration_unit": "milliseconds",
            "queued_at": queued_at,
            "processing_started_at": processing_started_at,
            "processing_completed_at": None,
            "queued_duration_ms": (
                round(queued_duration_ms, 3) if queued_duration_ms is not None else None
            ),
            "processing_duration_ms": None,
            "stage_durations_ms": stage_durations_ms,
            "pipeline_version": CURRENT_PIPELINE_VERSION,
            "reliable": False,
            "exclusion_reason": "processing_incomplete",
        }
        self._transition(
            document,
            VisionProcessingStatus.preprocessing,
            "Started quality analysis and preprocessing.",
        )

        ingestion_started = time.perf_counter()
        text, pages, parse_evidence, stored_path = self.ingestion.parse(filename, content)
        stage_durations_ms["ingestion"] = round(
            (time.perf_counter() - ingestion_started) * 1000,
            3,
        )
        parse_confidence = parse_evidence[0].confidence if parse_evidence else 0.5
        document.parsed_text = text
        document.original_storage_path = stored_path

        preprocessing_started = time.perf_counter()
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
        stage_durations_ms["preprocessing"] = round(
            (time.perf_counter() - preprocessing_started) * 1000,
            3,
        )

        ocr_started = time.perf_counter()
        ocr_adapter = (
            self.paddle_ocr if self.paddle_ocr.status().status == "active" else self.local_ocr
        )
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
        document.ocr_lines = ocr_line_reconstructor.reconstruct(document.ocr_words)
        if not document.parsed_text.strip() and document.ocr_lines:
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
        self._transition(
            document, VisionProcessingStatus.ocr_completed, "Completed OCR with page evidence."
        )
        stage_durations_ms["ocr"] = round(
            (time.perf_counter() - ocr_started) * 1000,
            3,
        )

        layout_started = time.perf_counter()
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
        stage_durations_ms["layout"] = round(
            (time.perf_counter() - layout_started) * 1000,
            3,
        )

        classification_started = time.perf_counter()
        document_type_hint = self._document_type_hint(metadata)
        if document_type_hint is not None:
            document.document_type = document_type_hint
            document.metadata["classification"] = {
                "predicted_type": document_type_hint.value,
                "parent_category": None,
                "confidence": None,
                "classifier": "user-selected-document-type",
                "signals": [
                    {
                        "type": "user_selection",
                        "value": document_type_hint.value,
                        "weight": None,
                        "candidate_type": document_type_hint.value,
                    }
                ],
                "alternatives": [],
                "user_override": document_type_hint.value,
            }
            document.model_runs.append(
                self._model_run(
                    "document_classification",
                    "reviewer-selected-document-type",
                    "v1",
                    "completed",
                    None,
                    duration_ms=(time.perf_counter() - classification_started) * 1000,
                    document=document,
                    notes=[
                        "The explicit upload selection was used instead of model classification."
                    ],
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
                statistical_type, classification_confidence = self.layoutlmv3_classifier.classify(
                    document
                )
                classification = hybrid_document_classifier.classify(
                    text,
                    filename,
                    lines=document.ocr_lines,
                    layout_regions=document.layout_regions,
                    statistical_type=statistical_type,
                    statistical_confidence=classification_confidence,
                )
                document.document_type = classification.predicted_type
                document.metadata["classification"] = classification.as_dict()
                document.model_runs.append(
                    self._model_run(
                        "document_classification",
                        self.layoutlmv3_classifier.adapter_id,
                        self.layoutlmv3_classifier.model_version,
                        "completed",
                        classification.confidence,
                        duration_ms=(time.perf_counter() - classification_started) * 1000,
                        document=document,
                        metrics={"signals": len(classification.signals)},
                    )
                )
            except Exception as exc:
                classification = hybrid_document_classifier.classify(
                    text,
                    filename,
                    lines=document.ocr_lines,
                    layout_regions=document.layout_regions,
                )
                document.document_type = classification.predicted_type
                document.metadata["classification"] = classification.as_dict()
                document.model_runs.append(
                    self._model_run(
                        "document_classification",
                        self.layoutlmv3_classifier.adapter_id,
                        self.layoutlmv3_classifier.model_version,
                        "failed_fallback_used",
                        classification.confidence,
                        duration_ms=(time.perf_counter() - classification_started) * 1000,
                        document=document,
                        notes=[str(exc)],
                    )
                )
        else:
            classification = hybrid_document_classifier.classify(
                text,
                filename,
                lines=document.ocr_lines,
                layout_regions=document.layout_regions,
            )
            document.document_type = classification.predicted_type
            document.metadata["classification"] = classification.as_dict()
            document.model_runs.append(
                self._model_run(
                    "document_classification",
                    hybrid_document_classifier.adapter_id,
                    hybrid_document_classifier.model_version,
                    "completed",
                    classification.confidence,
                    duration_ms=(time.perf_counter() - classification_started) * 1000,
                    document=document,
                    metrics={"signals": len(classification.signals)},
                    notes=[
                        "Confidence is derived from configured rule weights, not learned accuracy."
                    ],
                )
            )
        stage_durations_ms["classification"] = round(
            (time.perf_counter() - classification_started) * 1000,
            3,
        )

        extraction_started = time.perf_counter()
        extraction_errors: list[str] = []
        extraction_adapter_name = deterministic_schema_extractor.adapter_id
        extraction_adapter_version = deterministic_schema_extractor.model_version
        try:
            if self.layoutlmv3_extractor.status().status == "active":
                learned_fields = self.layoutlmv3_extractor.extract_fields(document)
                learned_names = {field.name for field in learned_fields}
                supplemental_fields = [
                    field
                    for field in self._extract_fields(document)
                    if field.name not in learned_names
                ]
                document.extracted_fields = [*learned_fields, *supplemental_fields]
                extraction_adapter_name = self.layoutlmv3_extractor.adapter_id
                extraction_adapter_version = self.layoutlmv3_extractor.model_version
            else:
                document.extracted_fields = self._extract_fields(document)
        except Exception as exc:
            extraction_errors.append(f"{type(exc).__name__}: {exc}")
            document.extracted_fields = []
        try:
            self._discover_generic_intelligence(document)
            self._resolve_semantic_entities(document)
        except Exception as exc:
            extraction_errors.append(f"{type(exc).__name__}: {exc}")
        extraction_runtime_ms = (time.perf_counter() - extraction_started) * 1000
        vision_analytics_service.refresh_document(document)
        extraction_outcome = (
            VisionExtractionOutcome.extraction_failed
            if extraction_errors
            else self._extraction_outcome(document)
        )
        extraction_diagnostics = dict(document.metadata.get("extraction_diagnostics") or {})
        schema_selection = dict(document.metadata.get("schema_selection") or {})
        document.metadata["extraction"] = {
            "stage": "extraction",
            "status": extraction_outcome.value,
            "document_type": document.document_type.value,
            "schema": schema_selection.get("selected_schema"),
            "fields_extracted": len(document.extracted_fields),
            "required_fields_missing": len(document.field_completeness.missing_required_fields),
            "extractor": f"{deterministic_schema_extractor.adapter_id}-{deterministic_schema_extractor.model_version}",
            "runtime_ms": round(extraction_runtime_ms, 3),
            "pipeline_version": document.metadata["pipeline_version"],
            "errors": extraction_errors,
            "ocr_retrieval_preserved": bool(document.ocr_lines),
        }
        if document.document_type == VisionDocumentType.contract and not extraction_errors:
            try:
                self._run_contract_extraction(document)
            except Exception as exc:
                extraction_errors.append(f"{type(exc).__name__}: {exc}")
                extraction_outcome = VisionExtractionOutcome.extraction_failed
                document.metadata["extraction"]["status"] = extraction_outcome.value
                document.metadata["extraction"]["errors"] = extraction_errors
        stage_durations_ms["extraction"] = round(extraction_runtime_ms, 3)

        visual_analysis_started = time.perf_counter()
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
                "failed_ocr_retrieval_preserved" if extraction_errors else "completed",
                self._mean_field_confidence(document),
                duration_ms=extraction_runtime_ms,
                document=document,
                input_schema="ocr-lines-v1",
                output_schema="extracted-fields-v1",
                selected_schema=str(schema_selection.get("selected_schema") or "generic_document"),
                metrics={
                    key: extraction_diagnostics.get(key)
                    for key in (
                        "ocr_words_inspected",
                        "ocr_lines_inspected",
                        "labels_detected",
                        "fields_extracted",
                    )
                },
                result_status=extraction_outcome.value,
                notes=[
                    *extraction_errors,
                    *(
                        ["OCR evidence chunking, indexing, and grounded retrieval continued."]
                        if extraction_errors
                        else []
                    ),
                    "Confidence is uncalibrated and therefore cannot imply model accuracy."
                    if not bool(self.confidence_policy.get("calibration", {}).get("calibrated"))
                    else "Confidence calibration artefact is active.",
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
        document.language_analysis = vision_language_analysis_service.analyze(document.ocr_words)
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
        stage_durations_ms["visual_and_language_analysis"] = round(
            (time.perf_counter() - visual_analysis_started) * 1000,
            3,
        )
        self._audit(
            document,
            "extraction_outcome",
            (
                "Structured extraction failed; OCR evidence indexing and retrieval continued."
                if extraction_errors
                else f"Extraction completed: {len(document.extracted_fields)} fields extracted "
                f"using {schema_selection.get('selected_schema', 'generic_document')} schema."
                if document.extracted_fields
                else "Extraction completed with no source-backed fields. Generic schema fallback was attempted."
            ),
            metadata=document.metadata["extraction"],
        )
        self._transition(
            document,
            VisionProcessingStatus.extraction_completed,
            (
                f"Extraction completed - {len(document.extracted_fields)} fields extracted."
                if document.extracted_fields
                else "Extraction completed - no fields found."
            ),
        )

        forensic_screening_started = time.perf_counter()
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
        stage_durations_ms["forensic_screening"] = round(
            (time.perf_counter() - forensic_screening_started) * 1000,
            3,
        )

        validation_started = time.perf_counter()
        validation_risks = self._validation_risks(document, metadata)
        if extraction_errors:
            validation_risks.append(
                self._risk(
                    document,
                    title="Structured Extraction Failure",
                    severity=VisionRiskSeverity.high,
                    category="processing",
                    description=(
                        "Structured extraction failed, but OCR evidence retrieval and indexing "
                        "continued so source-backed review remains available."
                    ),
                    recommended_action="Review OCR evidence and retry structured extraction.",
                    source_text="; ".join(extraction_errors),
                )
            )
        validation_risks.extend(self._generic_semantic_risks(document, metadata))
        validation_risks.extend(self._contract_risks(document))
        self._enrich_field_confidence(document, validation_risks)
        vision_analytics_service.refresh_document(document)
        stage_durations_ms["validation"] = round(
            (time.perf_counter() - validation_started) * 1000,
            3,
        )

        risk_screening_started = time.perf_counter()
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
        self._transition(
            document,
            VisionProcessingStatus.validation_completed,
            "Completed validation and risk screening.",
        )
        stage_durations_ms["risk_screening"] = round(
            (time.perf_counter() - risk_screening_started) * 1000,
            3,
        )

        chunking_started = time.perf_counter()
        document.evidence_chunks = self._chunks(document)
        self._enrich_field_confidence(document, document.risk_indicators)
        vision_analytics_service.refresh_document(document)
        stage_durations_ms["chunking"] = round(
            (time.perf_counter() - chunking_started) * 1000,
            3,
        )

        graph_started = time.perf_counter()
        document.graph = self._graph(document)
        self.persistence.upsert_graph(
            document.tenant_id,
            document.document_id,
            document.graph,
        )
        stage_durations_ms["graph_projection"] = round(
            (time.perf_counter() - graph_started) * 1000,
            3,
        )

        indexing_started = time.perf_counter()
        self._index_document(document)
        stage_durations_ms["embedding_and_indexing"] = round(
            (time.perf_counter() - indexing_started) * 1000,
            3,
        )
        self._transition(
            document,
            VisionProcessingStatus.indexed,
            "Indexed structure-aware chunks and graph projection.",
        )

        review_routing_started = time.perf_counter()
        previous_review_id = document.metadata.pop("_reprocess_review_id", None)
        if self._requires_review(document):
            document.review_task = self._create_review_task(
                document,
                review_id=str(previous_review_id) if previous_review_id else None,
            )
            self.review_tasks[document.review_task.review_id] = document.review_task
            self._transition(
                document, VisionProcessingStatus.review_required, "Document routed to human review."
            )
        else:
            self._transition(
                document,
                VisionProcessingStatus.completed,
                "Document completed without mandatory review.",
            )
        document.metadata["pipeline_status"] = VisionProcessingStatus.completed.value
        document.metadata["decision_status"] = document.status.value
        document.metadata["status_summary"] = (
            "Processing completed - awaiting human review"
            if document.status == VisionProcessingStatus.review_required
            else "Processing completed"
        )
        stage_durations_ms["review_routing"] = round(
            (time.perf_counter() - review_routing_started) * 1000,
            3,
        )
        processing_completed_at = utc_now_iso()
        document.metadata["processing_timing"] = {
            "duration_unit": "milliseconds",
            "queued_at": queued_at,
            "processing_started_at": processing_started_at,
            "processing_completed_at": processing_completed_at,
            "queued_duration_ms": (
                round(queued_duration_ms, 3) if queued_duration_ms is not None else None
            ),
            "processing_duration_ms": round(
                (time.perf_counter() - processing_started_perf) * 1000,
                3,
            ),
            "stage_durations_ms": stage_durations_ms,
            "pipeline_version": CURRENT_PIPELINE_VERSION,
            "reliable": True,
            "exclusion_reason": None,
        }
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
            self._remove_review(previous_review_id)
        self.retrieval_index.remove_document(document.document_id)
        document.status = VisionProcessingStatus.queued
        document.parsed_text = ""
        document.page_count = 1
        document.page_artifacts = []
        document.quality = []
        document.ocr_words = []
        document.ocr_lines = []
        document.layout_regions = []
        document.tables = []
        document.visual_detections = []
        document.tamper_analysis = []
        document.generic_fields = []
        document.typed_entities = []
        document.extracted_fields = []
        document.verified_facts = []
        document.schema_profile = type(document.schema_profile)()
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
            "fax": "fax_transmission",
            "facsimile": "fax_transmission",
            "businessform": "business_form",
            "general_form": "business_form",
            "financialreport": "financial_report",
            "generalreport": "general_report",
        }
        try:
            return VisionDocumentType(aliases.get(normalized, normalized))
        except ValueError:
            return None

    def _classify_document(self, text: str, filename: str) -> VisionDocumentType:
        return hybrid_document_classifier.classify(text, filename).predicted_type

    def _extraction_outcome(
        self,
        document: VisionDocumentRecord,
    ) -> VisionExtractionOutcome:
        selection = document.metadata.get("schema_selection") or {}
        if document.extracted_fields:
            if selection.get("fallback_used"):
                return VisionExtractionOutcome.schema_fallback_used
            observed = [
                field.confidence
                for field in document.extracted_fields
                if field.confidence is not None
            ]
            if observed and min(observed) < self._mandatory_review_threshold(document):
                return VisionExtractionOutcome.low_confidence_fields
            return VisionExtractionOutcome.success_with_fields
        if not selection.get("selected_schema"):
            return VisionExtractionOutcome.schema_unavailable
        if document.model_routing.get("fallback_required"):
            return VisionExtractionOutcome.fallback_required
        return VisionExtractionOutcome.completed_no_fields

    def _extract_fields(self, document: VisionDocumentRecord) -> list[ExtractedVisionField]:
        fields = deterministic_schema_extractor.extract(document)
        if document.document_type == VisionDocumentType.contract:
            fields.extend(self._clause_fields(document))
        return self._dedupe_fields(fields)

    def _discover_generic_intelligence(self, document: VisionDocumentRecord) -> None:
        """Retain schema-independent OCR facts alongside deterministic extraction.

        The generic layer is deliberately additive: registered-schema fields remain
        authoritative, while unseen labels and reusable typed values stay available
        to review, retrieval, graph, and open-set document workflows.
        """
        result = generic_field_discovery_engine.discover(
            document.ocr_lines,
            document.ocr_words,
            document_type=document.document_type.value,
        )
        registered_schema = vision_schema_registry.get(document.document_type)
        reviewed_aliases = self._feedback_aliases(document.tenant_id)
        embedding_label_matches, embedding_mapping_diagnostics = self._embedding_schema_labels(
            [observed.raw_label for observed in result.fields],
            registered_schema,
        )
        schema_names = (
            {item.name for item in registered_schema.fields}
            if registered_schema is not None
            else set()
        )
        evidence_by_field_id: dict[str, VisionEvidence] = {}
        generic_fields: list[VisionGenericDiscoveredField] = []
        rejected_candidates: list[dict[str, Any]] = []
        rejected_field_ids: set[str] = set()
        for observed in result.fields:
            if self._generic_observation_in_table(document, observed):
                rejected_candidates.append(
                    self._generic_rejection_record(observed, "inside_structured_table")
                )
                rejected_field_ids.add(observed.field_id)
                continue
            semantic_type = self._vision_semantic_type(observed.semantic_type)
            registered_label = self._registered_schema_label(
                observed.raw_label,
                registered_schema,
            )
            reviewed_label = reviewed_aliases.get(self._normalize_token(observed.raw_label))
            embedding_label = embedding_label_matches.get(observed.raw_label)
            normalized_label = (
                registered_label
                or reviewed_label
                or (embedding_label[0] if embedding_label else None)
                or observed.normalized_label
            )
            semantic_mapping_method = (
                "registered_schema_alias"
                if registered_label
                else "reviewer_feedback_alias"
                if reviewed_label
                else "bge_m3_embedding_similarity"
                if embedding_label
                else observed.label_mapping_method
            )
            semantic_mapping_confidence = (
                1.0
                if registered_label or reviewed_label
                else embedding_label[1]
                if embedding_label
                else observed.label_mapping_confidence
            )
            rejection_reason = self._generic_observation_rejection_reason(
                document,
                observed,
                normalized_label=normalized_label,
                registered_schema=registered_schema,
                semantic_type=semantic_type,
            )
            if rejection_reason is not None:
                rejected_candidates.append(
                    self._generic_rejection_record(observed, rejection_reason)
                )
                rejected_field_ids.add(observed.field_id)
                continue
            evidence = VisionEvidence(
                evidence_id=f"vis-ev-{observed.field_id}",
                document_id=document.document_id,
                page_number=observed.page,
                source_text=f"{observed.raw_label}: {observed.raw_value}",
                bounding_box=VisionBoundingBox(**observed.evidence_bbox.as_dict()),
                model_confidence=observed.confidence,
                model_name=generic_field_discovery_engine.adapter_id,
                model_version=generic_field_discovery_engine.model_version,
                processing_timestamp=utc_now_iso(),
                metadata={
                    "raw_label": observed.raw_label,
                    "normalized_label": normalized_label,
                    "generic_engine_normalized_label": observed.normalized_label,
                    "label_mapping_method": observed.label_mapping_method,
                    "label_mapping_confidence": observed.label_mapping_confidence,
                    "semantic_mapping_method": semantic_mapping_method,
                    "semantic_mapping_confidence": semantic_mapping_confidence,
                    "extraction_strategy": observed.extraction_strategy,
                    "source_line_ids": list(observed.source_line_ids),
                    "label_word_ids": list(observed.label_word_ids),
                    "value_word_ids": list(observed.value_word_ids),
                },
            )
            evidence_by_field_id[observed.field_id] = evidence
            generic_fields.append(
                VisionGenericDiscoveredField(
                    field_id=observed.field_id,
                    raw_label=observed.raw_label,
                    normalized_label=normalized_label,
                    raw_value=observed.raw_value,
                    normalized_value=observed.normalized_value,
                    page=observed.page,
                    label_bbox=VisionBoundingBox(**observed.label_bbox.as_dict()),
                    value_bbox=VisionBoundingBox(**observed.value_bbox.as_dict()),
                    confidence=observed.confidence,
                    source_engine=observed.source_engine,
                    semantic_type_if_known=semantic_type,
                    semantic_mapping_confidence=(
                        observed.semantic_confidence if semantic_type is not None else None
                    ),
                    evidence=evidence,
                    review_required=observed.review_required,
                    metadata={
                        "canonical_label": observed.canonical_label,
                        "label_mapping_method": observed.label_mapping_method,
                        "label_mapping_confidence": observed.label_mapping_confidence,
                        "semantic_mapping_method": semantic_mapping_method,
                        "semantic_mapping_confidence": semantic_mapping_confidence,
                        "extraction_strategy": observed.extraction_strategy,
                    },
                )
            )
        document.generic_fields = generic_fields

        entities: list[TypedVisionEntity] = []
        for observed in result.entities:
            if observed.linked_field_id in rejected_field_ids:
                continue
            semantic_type = self._vision_semantic_type(observed.semantic_type)
            if semantic_type is None or semantic_type == VisionSemanticType.text:
                continue
            linked_evidence = evidence_by_field_id.get(observed.linked_field_id or "")
            entities.append(
                TypedVisionEntity(
                    entity_id=observed.entity_id,
                    raw_value=observed.raw_value,
                    normalized_value=observed.normalized_value,
                    semantic_type=semantic_type,
                    confidence=observed.confidence,
                    context={
                        "page": observed.page,
                        "bounding_box": observed.bounding_box.as_dict(),
                        "contextual_label": observed.contextual_label,
                        "source_engine": observed.source_engine,
                        "source_line_ids": list(observed.source_line_ids),
                        "word_ids": list(observed.word_ids),
                    },
                    evidence_ids=(
                        [linked_evidence.evidence_id] if linked_evidence is not None else []
                    ),
                    review_required=observed.confidence < 0.8,
                )
            )
        document.typed_entities = entities
        document.metadata["generic_discovery"] = result.as_dict()["diagnostics"]
        document.metadata["generic_discovery"]["generic_fields_retained"] = len(
            document.generic_fields
        )
        document.metadata["generic_discovery"]["candidates_rejected"] = len(rejected_candidates)
        document.metadata["generic_discovery"]["rejected_candidates"] = rejected_candidates
        document.metadata["generic_discovery"]["embedding_label_mapping"] = (
            embedding_mapping_diagnostics
        )
        supplemental = [
            field for field in document.generic_fields if field.normalized_label not in schema_names
        ]
        if schema_names and supplemental:
            document.schema_profile.mode = "B"
            document.schema_profile.selection_source = "registered_plus_generic"
        elif not schema_names:
            document.schema_profile.mode = "C"
            document.schema_profile.schema_status = "generic"
            document.schema_profile.selection_source = "open_set_generic"

        # Mirror only out-of-schema discoveries into the established field contract
        # so legacy consumers gain Mode B/C structure without duplicating Mode A fields.
        existing_names = {field.name for field in document.extracted_fields}
        for field in supplemental:
            if field.normalized_label in existing_names:
                continue
            document.extracted_fields.append(
                ExtractedVisionField(
                    field_id=f"field-{field.field_id}",
                    name=field.normalized_label,
                    display_name=field.raw_label,
                    value=field.raw_value,
                    raw_value=field.raw_value,
                    normalized_value=field.normalized_value,
                    validation_status=(
                        "valid" if field.semantic_type_if_known is not None else "not_evaluated"
                    ),
                    confidence=field.confidence,
                    evidence=field.evidence,
                    model_name=generic_field_discovery_engine.adapter_id,
                    model_version=generic_field_discovery_engine.model_version,
                    extractor="generic-open-set",
                    required=False,
                )
            )
            existing_names.add(field.normalized_label)
        document.extracted_fields = self._dedupe_fields(document.extracted_fields)

    def _vision_semantic_type(self, value: object) -> VisionSemanticType | None:
        name = str(getattr(value, "name", "")).lower()
        return VisionSemanticType.__members__.get(name)

    def _resolve_semantic_entities(self, document: VisionDocumentRecord) -> None:
        """Deduplicate observations and record only conservative tenant matches."""
        self._load_all_documents(document.tenant_id)
        corpus = [item for item in self.documents.values() if item.tenant_id == document.tenant_id]
        report = conservative_entity_resolver.resolve(
            document,
            corpus,
            reviewed_value_aliases=self.feedback_dataset.reviewed_value_aliases(document.tenant_id),
        )
        document.typed_entities = list(report.entities)
        document.metadata["entity_resolution"] = report.as_dict()

    def _semantic_type_for_reviewed_field(
        self,
        document: VisionDocumentRecord,
        field: ExtractedVisionField,
    ) -> VisionSemanticType | None:
        generic = next(
            (
                item
                for item in document.generic_fields
                if item.evidence.evidence_id == field.evidence.evidence_id
            ),
            None,
        )
        return semantic_type_for_name(field.name) or (
            generic.semantic_type_if_known if generic else None
        )

    def _apply_field_review_to_generic_layers(
        self,
        document: VisionDocumentRecord,
        field: ExtractedVisionField,
        *,
        semantic_type: VisionSemanticType | None,
    ) -> None:
        approved = field.reviewer_status in {
            ReviewerStatus.accepted,
            ReviewerStatus.edited,
        }
        for generic in document.generic_fields:
            if generic.evidence.evidence_id != field.evidence.evidence_id:
                continue
            generic.evidence.reviewer_status = field.reviewer_status
            generic.review_required = not approved
            generic.metadata["reviewed_field_name"] = field.name
            generic.metadata["reviewed_value"] = field.reviewed_value
            if approved:
                generic.normalized_label = field.name
                generic.normalized_value = field.normalized_value or generic.normalized_value
                generic.semantic_type_if_known = semantic_type or generic.semantic_type_if_known
        for index, entity in enumerate(document.typed_entities):
            if field.evidence.evidence_id not in entity.evidence_ids:
                continue
            context = dict(entity.context)
            context["reviewed_field_name"] = field.name
            context["reviewer_status"] = field.reviewer_status.value
            document.typed_entities[index] = entity.model_copy(
                update={
                    "normalized_value": (
                        field.normalized_value or entity.normalized_value
                        if approved
                        else entity.normalized_value
                    ),
                    "semantic_type": semantic_type or entity.semantic_type,
                    "review_required": not approved,
                    "context": context,
                }
            )

    def _generic_semantic_risks(
        self,
        document: VisionDocumentRecord,
        metadata: dict[str, Any],
    ) -> list[VisionRiskIndicator]:
        self._load_all_documents(document.tenant_id)
        corpus = [item for item in self.documents.values() if item.tenant_id == document.tenant_id]
        findings = generic_semantic_risk_engine.evaluate(document, metadata, corpus)
        return [self._semantic_risk_indicator(document, finding) for finding in findings]

    def _semantic_risk_indicator(
        self,
        document: VisionDocumentRecord,
        finding: SemanticRiskFinding,
    ) -> VisionRiskIndicator:
        indicator = self._risk(
            document,
            title=finding.title,
            severity=finding.severity,
            category=finding.category,
            description=finding.description,
            recommended_action=finding.recommended_action,
            source_text=finding.source_text,
            field_names=list(finding.affected_fields),
            evidence=list(finding.evidence),
            rule_id=finding.rule_id,
        )
        indicator.confidence = finding.confidence
        indicator.model_name = "generic-semantic-risk-engine"
        indicator.model_version = generic_semantic_risk_engine.model_version
        indicator.model_or_rule = finding.rule_id
        return indicator

    def _generic_observation_in_table(
        self,
        document: VisionDocumentRecord,
        observed: object,
    ) -> bool:
        box = observed.evidence_bbox
        area = max((box.x2 - box.x1) * (box.y2 - box.y1), 1.0)
        for table in document.tables:
            if table.page_number != observed.page:
                continue
            table_box = table.bounding_box
            overlap_width = max(0.0, min(box.x2, table_box.x2) - max(box.x1, table_box.x1))
            overlap_height = max(0.0, min(box.y2, table_box.y2) - max(box.y1, table_box.y1))
            if overlap_width * overlap_height / area >= 0.8:
                return True
        return False

    def _generic_observation_rejection_reason(
        self,
        document: VisionDocumentRecord,
        observed: object,
        *,
        normalized_label: str,
        registered_schema: object,
        semantic_type: VisionSemanticType | None,
    ) -> str | None:
        """Reject only demonstrable OCR pair noise, preserving uncertain real labels.

        Weak geometry strategies can accidentally split initials or the value-bearing
        tail of a known schema line into a second field.  Explicit delimiters and
        semantically typed values remain eligible even when their labels are short.
        """
        if getattr(observed, "label_mapping_method", "") != "open_set":
            return None
        if self._registered_schema_label(observed.raw_label, registered_schema):
            return None

        strategy = str(getattr(observed, "extraction_strategy", ""))
        weak_geometry = strategy in {
            "same_line_geometry_gap",
            "same_line_whitespace",
            "separate_lines_horizontal",
            "label_above_value",
        }
        raw_label = str(getattr(observed, "raw_label", ""))
        raw_value = str(getattr(observed, "raw_value", ""))
        label_tokens = re.findall(r"[a-z0-9]+", raw_label.casefold())
        value_alnum = re.sub(r"[^a-z0-9]+", "", raw_value.casefold())
        untyped_text = semantic_type in {None, VisionSemanticType.text}

        if weak_geometry and untyped_text:
            if label_tokens and all(len(token) <= 2 for token in label_tokens):
                return "ocr_label_fragment"
            if len(value_alnum) <= 1:
                return "single_character_geometry_value"

        if (
            weak_geometry
            and normalized_label == getattr(observed, "normalized_label", "")
            and self._generic_label_contains_schema_prefix(raw_label, registered_schema)
            and self._generic_observation_overlaps_extracted_field(document, observed)
        ):
            return "known_schema_line_value_leakage"
        return None

    def _generic_label_contains_schema_prefix(
        self,
        raw_label: str,
        registered_schema: object,
    ) -> bool:
        if registered_schema is None:
            return False
        observed_words = re.findall(r"[a-z0-9]+", raw_label.casefold())
        for field in getattr(registered_schema, "fields", ()):
            candidates = (
                field.name,
                getattr(field, "display_name", ""),
                *getattr(field, "aliases", ()),
            )
            for candidate in candidates:
                candidate_words = re.findall(r"[a-z0-9]+", str(candidate).casefold())
                if (
                    len(candidate_words) >= 2
                    and len(observed_words) > len(candidate_words)
                    and observed_words[: len(candidate_words)] == candidate_words
                ):
                    return True
        return False

    def _generic_observation_overlaps_extracted_field(
        self,
        document: VisionDocumentRecord,
        observed: object,
    ) -> bool:
        observed_box = observed.evidence_bbox
        observed_area = max(
            (observed_box.x2 - observed_box.x1) * (observed_box.y2 - observed_box.y1),
            1.0,
        )
        for field in document.extracted_fields:
            evidence_box = field.evidence.bounding_box
            if evidence_box is None or field.evidence.page_number != observed.page:
                continue
            overlap_width = max(
                0.0,
                min(observed_box.x2, evidence_box.x2) - max(observed_box.x1, evidence_box.x1),
            )
            overlap_height = max(
                0.0,
                min(observed_box.y2, evidence_box.y2) - max(observed_box.y1, evidence_box.y1),
            )
            if overlap_width * overlap_height / observed_area >= 0.55:
                return True
        return False

    def _generic_rejection_record(self, observed: object, reason: str) -> dict[str, Any]:
        return {
            "field_id": str(getattr(observed, "field_id", "")),
            "raw_label": str(getattr(observed, "raw_label", "")),
            "normalized_label": str(getattr(observed, "normalized_label", "")),
            "raw_value": str(getattr(observed, "raw_value", "")),
            "confidence": float(getattr(observed, "confidence", 0.0)),
            "extraction_strategy": str(getattr(observed, "extraction_strategy", "unknown")),
            "reason": reason,
        }

    def _registered_schema_label(self, raw_label: str, schema: object) -> str | None:
        if schema is None:
            return None
        observed = self._normalize_token(raw_label)
        for field in getattr(schema, "fields", ()):
            candidates = (
                field.name,
                getattr(field, "display_name", ""),
                *getattr(field, "aliases", ()),
            )
            if observed and observed in {
                self._normalize_token(candidate) for candidate in candidates if candidate
            }:
                return field.name
        return None

    def _embedding_schema_labels(
        self,
        raw_labels: list[str],
        schema: object,
    ) -> tuple[dict[str, tuple[str, float]], dict[str, Any]]:
        """Conservatively map unseen labels with BGE when that runtime is active."""
        embedding = self.retrieval_index.embedding
        if (
            schema is None
            or getattr(embedding, "adapter_id", "") != "bge-m3-embedding"
            or not hasattr(embedding, "embed_many")
        ):
            return {}, {"status": "not_used", "reason": "bge_m3_not_active"}

        unique_labels = list(dict.fromkeys(label.strip() for label in raw_labels if label.strip()))
        fields = list(getattr(schema, "fields", ()))
        if not unique_labels or not fields:
            return {}, {"status": "not_used", "reason": "no_labels_or_schema_fields"}

        schema_id = str(getattr(schema, "name", "anonymous-schema"))
        try:
            cached = self._semantic_label_embedding_cache.get(schema_id)
            if cached is None:
                candidate_texts = [
                    (
                        "canonical document field: "
                        f"{field.display_name}. aliases: "
                        + ", ".join((field.name, *getattr(field, "aliases", ())))
                    )
                    for field in fields
                ]
                candidate_vectors = embedding.embed_many(candidate_texts)
                cached = {
                    field.name: vector.dense
                    for field, vector in zip(fields, candidate_vectors, strict=True)
                }
                self._semantic_label_embedding_cache[schema_id] = cached
            query_vectors = embedding.embed_many(
                [f"document field label: {label}" for label in unique_labels]
            )
        except Exception as exc:
            return {}, {
                "status": "failed_fallback_used",
                "reason": f"{type(exc).__name__}: embedding label mapping unavailable",
            }

        matches: dict[str, tuple[str, float]] = {}
        for label, vector in zip(unique_labels, query_vectors, strict=True):
            ranked = sorted(
                (
                    (self._vector_cosine(vector.dense, candidate), field_name)
                    for field_name, candidate in cached.items()
                ),
                reverse=True,
            )
            if not ranked:
                continue
            best_score, best_name = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else 0.0
            if best_score >= 0.84 and best_score - second_score >= 0.04:
                matches[label] = (best_name, round(best_score, 4))
        return matches, {
            "status": "completed",
            "engine": "bge-m3-embedding",
            "threshold": 0.84,
            "ambiguity_margin": 0.04,
            "labels_considered": len(unique_labels),
            "labels_mapped": len(matches),
        }

    def _vector_cosine(self, left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = sum(value * value for value in left) ** 0.5
        right_norm = sum(value * value for value in right) ** 0.5
        return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0

    def _refresh_verified_facts(
        self,
        document: VisionDocumentRecord,
        *,
        reviewer: str | None = None,
    ) -> None:
        semantic_by_label = {
            field.normalized_label: field.semantic_type_if_known
            for field in document.generic_fields
            if field.semantic_type_if_known is not None
        }
        document.verified_facts = [
            VerifiedVisionFact(
                fact_id=f"fact-{field.field_id}",
                field_id=field.field_id,
                name=field.name,
                value=field.reviewed_value or field.value,
                normalized_value=field.normalized_value,
                semantic_type=semantic_by_label.get(field.name),
                evidence_id=field.evidence.evidence_id,
                reviewer_status=field.reviewer_status,
                reviewer=reviewer,
                verified_at=utc_now_iso(),
            )
            for field in document.extracted_fields
            if field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
        ]

    def _record_active_learning_correction(
        self,
        document: VisionDocumentRecord,
        field: ExtractedVisionField,
        *,
        original_field_name: str,
        original_value: str,
        comment: str,
        reviewer: str,
        reviewed_at: str,
    ) -> None:
        generic = next(
            (
                item
                for item in document.generic_fields
                if item.evidence.evidence_id == field.evidence.evidence_id
                or item.normalized_label == original_field_name
            ),
            None,
        )
        raw_label = generic.raw_label if generic else self._source_label(field)
        semantic_type = self._semantic_type_for_reviewed_field(document, field)
        record = {
            "correction_id": f"correction-{uuid4().hex[:12]}",
            "tenant_id": document.tenant_id,
            "document_id": document.document_id,
            "document_type": document.document_type.value,
            "field_id": field.field_id,
            "raw_label": raw_label,
            "original_field_name": original_field_name,
            "corrected_field_name": field.name,
            "original_value": original_value,
            "corrected_value": field.reviewed_value or field.value,
            "normalized_corrected_value": field.normalized_value,
            "semantic_type": semantic_type.value if semantic_type else None,
            "source_text": field.evidence.source_text,
            "evidence_id": field.evidence.evidence_id,
            "page": field.evidence.page_number,
            "bounding_box": (
                field.evidence.bounding_box.model_dump(mode="json")
                if field.evidence.bounding_box
                else None
            ),
            "extraction_method": field.extractor,
            "layout_relation": (generic.metadata.get("extraction_strategy") if generic else None),
            "source_line_ids": list(field.evidence.metadata.get("source_line_ids") or []),
            "schema_mode": document.schema_profile.mode,
            "schema_status": document.schema_profile.schema_status,
            "schema_id": document.schema_profile.schema_id,
            "processing_version": document.metadata.get(
                "pipeline_version",
                CURRENT_PIPELINE_VERSION,
            ),
            "reviewer": reviewer,
            "reviewer_comment": comment,
            "reviewed_at": reviewed_at,
            "status": field.reviewer_status.value,
            "dataset_usage": "controlled_alias_and_training_export",
        }
        self.feedback_dataset.append(record)

    def _feedback_aliases(self, tenant_id: str) -> dict[str, str]:
        return self.feedback_dataset.approved_label_aliases(
            tenant_id,
            normalizer=self._normalize_token,
        )

    def _active_learning_path(self, tenant_id: str) -> Path:
        return self.feedback_dataset.path_for(tenant_id)

    def _source_label(self, field: ExtractedVisionField) -> str:
        source = field.evidence.source_text.strip()
        matched = re.match(r"^\s*([^:#=]{1,100}?)\s*[:#=]", source)
        return matched.group(1).strip() if matched else field.display_name

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
                    fields.append(
                        self._field(document, name, snippet or keyword, snippet or keyword)
                    )
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
        approved_bank = metadata.get("approved_bank_identifier") or metadata.get(
            "approved_bank_account"
        )
        bank_identifier = fields.get("bank_account_number") or fields.get("IFSC_or_bank_code")
        if (
            approved_bank
            and bank_identifier
            and self._normalize(approved_bank) != self._normalize(bank_identifier)
        ):
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
        if required_signature and not any(
            item.detection_class == "signature" for item in document.visual_detections
        ):
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
        if (
            contract_terms
            and payment_terms
            and self._normalize(contract_terms) != self._normalize(payment_terms)
        ):
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
                region
                for region in analysis.suspicious_regions
                if region.screening_score >= threshold
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
        self._load_all_documents(document.tenant_id)
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
                                metadata={
                                    "matched_document_id": existing_id,
                                    "algorithm": "SHA-256",
                                },
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
            measured = [value for value in (dhash_similarity, dino_similarity) if value is not None]
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
                similarity, matches, compared = self._structured_duplicate_similarity(
                    document, existing
                )
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
            if not any(
                candidate.matched_document_id == related.document_id for candidate in candidates
            ):
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
                        severity=VisionRiskSeverity.high
                        if candidate.status == "exact_duplicate"
                        else VisionRiskSeverity.medium,
                        category="duplicate_detection",
                        description="The document contains suspicious indicators and requires human review.",
                        recommended_action="Compare duplicate match before approval.",
                        source_text=candidate.reason,
                        extra_evidence=candidate.evidence,
                    )
                )
        return risks

    def _run_anomaly_models(self, document: VisionDocumentRecord) -> None:
        self._load_all_documents(document.tenant_id)
        corpus = [item for item in self.documents.values() if item.tenant_id == document.tenant_id]
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
                notes=[
                    "XGBoost prediction is uncalibrated until validation calibration is supplied."
                ],
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
                    evidence=[self._word_evidence(document, word) for word in low_words[:8]],
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
        schema_selection = document.metadata.get("schema_selection") or {}
        if document.field_completeness.schema_available and schema_selection.get("fallback_used"):
            selected_schema = str(schema_selection.get("selected_schema") or "generic_document")
            warnings.append(
                self._risk(
                    document,
                    title="Generic Schema Fallback Used",
                    severity=VisionRiskSeverity.info,
                    category="processing_warning",
                    description=(
                        "Using generic key-value schema."
                        if selected_schema == "generic_key_value_form"
                        else f"Using {selected_schema.replace('_', ' ')} schema."
                    ),
                    recommended_action="Review extracted source evidence for the generic fields.",
                    source_text=selected_schema,
                    rule_id="processing_warning.schema_fallback_used",
                )
            )
        elif not document.field_completeness.schema_available:
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
        common_metadata = {
            "filename": document.filename,
            "pipeline_version": document.metadata.get("pipeline_version", "legacy"),
            "extractor_version": deterministic_schema_extractor.model_version,
            "verification_status": ReviewerStatus.unverified.value,
        }
        for line in document.ocr_lines:
            line_chunk = VisionEvidenceChunk(
                chunk_id=self._chunk_id(
                    document,
                    line.page,
                    "ocr_line",
                    line.text,
                ),
                document_id=document.document_id,
                tenant_id=document.tenant_id,
                page_number=line.page,
                section=line.line_id,
                document_type=document.document_type,
                vendor_id=self._field_map(document).get("vendor_name"),
                bounding_boxes=[line.bounding_box],
                source_type="ocr_line",
                model_name="ocr-line-retriever",
                model_version="v1",
                text=line.text,
                confidence=line.mean_confidence or 0.0,
                metadata={
                    **common_metadata,
                    "word_ids": line.word_ids,
                    "block_ids": line.block_ids,
                },
            )
            chunks.append(line_chunk)
            if line.metadata.get("key_value_candidate"):
                chunks.append(
                    line_chunk.model_copy(
                        update={
                            "chunk_id": self._chunk_id(
                                document,
                                line.page,
                                "key_value",
                                line.text,
                            ),
                            "section": "key_value",
                            "source_type": "key_value",
                        }
                    )
                )
        lines_by_page: dict[int, list[Any]] = {}
        for line in document.ocr_lines:
            lines_by_page.setdefault(line.page, []).append(line)
        for page_number, page_lines in lines_by_page.items():
            page_text = "\n".join(
                line.text for line in sorted(page_lines, key=lambda item: item.reading_order)
            ).strip()
            if not page_text:
                continue
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(document, page_number, "page_text", page_text),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=page_number,
                    section=f"page_{page_number}",
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=[self._union_box([line.bounding_box for line in page_lines])],
                    source_type="page_text",
                    model_name="ocr-page-chunker",
                    model_version="v1",
                    text=page_text[:6000],
                    confidence=round(
                        sum((line.mean_confidence or 0.0) for line in page_lines) / len(page_lines),
                        4,
                    ),
                    metadata=common_metadata,
                )
            )
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
                    metadata=common_metadata,
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
                    bounding_boxes=[field.evidence.bounding_box]
                    if field.evidence.bounding_box
                    else [],
                    source_type="field",
                    review_status=field.reviewer_status,
                    model_name=field.model_name,
                    model_version=field.model_version,
                    text=f"{field.name}: {field.reviewed_value or field.value}",
                    confidence=field.confidence or 0.0,
                    metadata={
                        **common_metadata,
                        "evidence_id": field.evidence.evidence_id,
                        "verification_status": field.reviewer_status.value,
                        "schema_field": field.schema_field,
                    },
                )
            )
        for field in document.generic_fields:
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(
                        document,
                        field.page,
                        f"generic_field:{field.normalized_label}",
                        field.normalized_value,
                    ),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=field.page,
                    section=field.normalized_label,
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=[field.evidence.bounding_box]
                    if field.evidence.bounding_box
                    else [],
                    source_type="generic_field",
                    model_name=generic_field_discovery_engine.adapter_id,
                    model_version=generic_field_discovery_engine.model_version,
                    text=f"{field.raw_label}: {field.raw_value}",
                    confidence=field.confidence,
                    metadata={
                        **common_metadata,
                        "evidence_id": field.evidence.evidence_id,
                        "raw_label": field.raw_label,
                        "normalized_label": field.normalized_label,
                        "semantic_type": (
                            field.semantic_type_if_known.value
                            if field.semantic_type_if_known is not None
                            else None
                        ),
                        "review_required": field.review_required,
                    },
                )
            )
        for entity in document.typed_entities:
            page_number = max(1, int(entity.context.get("page", 1)))
            raw_box = entity.context.get("bounding_box")
            entity_box = VisionBoundingBox(**raw_box) if isinstance(raw_box, dict) else None
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(
                        document,
                        page_number,
                        f"typed_entity:{entity.semantic_type.value}",
                        entity.normalized_value,
                    ),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=page_number,
                    section=str(entity.context.get("contextual_label") or "typed_entity"),
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=[entity_box] if entity_box is not None else [],
                    source_type="typed_entity",
                    model_name=str(
                        entity.context.get("source_engine")
                        or generic_field_discovery_engine.adapter_id
                    ),
                    model_version=generic_field_discovery_engine.model_version,
                    text=(
                        f"{entity.context.get('contextual_label')}: {entity.raw_value}"
                        if entity.context.get("contextual_label")
                        else entity.raw_value
                    ),
                    confidence=entity.confidence,
                    metadata={
                        **common_metadata,
                        "entity_id": entity.entity_id,
                        "semantic_type": entity.semantic_type.value,
                        "evidence_ids": entity.evidence_ids,
                        "review_required": entity.review_required,
                    },
                )
            )
        evidence_by_id = {
            field.evidence.evidence_id: field.evidence for field in document.extracted_fields
        }
        for fact in document.verified_facts:
            evidence = evidence_by_id.get(fact.evidence_id)
            chunks.append(
                VisionEvidenceChunk(
                    chunk_id=self._chunk_id(
                        document,
                        evidence.page_number if evidence else 1,
                        f"verified_fact:{fact.name}",
                        fact.value,
                    ),
                    document_id=document.document_id,
                    tenant_id=document.tenant_id,
                    page_number=evidence.page_number if evidence else 1,
                    section=fact.name,
                    document_type=document.document_type,
                    vendor_id=self._field_map(document).get("vendor_name"),
                    bounding_boxes=(
                        [evidence.bounding_box] if evidence and evidence.bounding_box else []
                    ),
                    source_type="verified_fact",
                    review_status=fact.reviewer_status,
                    model_name="human-review",
                    model_version="v1",
                    text=f"{fact.name}: {fact.value}",
                    confidence=1.0,
                    metadata={
                        **common_metadata,
                        "evidence_id": fact.evidence_id,
                        "verification_status": fact.reviewer_status.value,
                        "verified_at": fact.verified_at,
                        "reviewer": fact.reviewer,
                    },
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
                    metadata=common_metadata,
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
            (f"{document.document_id}|{page_number}|{source_type}|{self._normalize(text)}").encode()
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
                (node_id, document.document_id) if reverse else (document.document_id, node_id)
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
                    "bounding_boxes": [field.evidence.bounding_box.model_dump(mode="json")]
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
                        if evidence_field.reviewer_status
                        in {ReviewerStatus.accepted, ReviewerStatus.edited}
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
                    if evidence_field.reviewer_status
                    in {ReviewerStatus.accepted, ReviewerStatus.edited}
                    else "proposed",
                    "source_document": document.document_id,
                    "source_page": evidence_field.evidence.page_number,
                    "evidence_id": evidence_field.evidence.evidence_id,
                    "evidence_text": evidence_field.evidence.source_text,
                    "bounding_boxes": [evidence_field.evidence.bounding_box.model_dump(mode="json")]
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
        reasons.extend(
            risk.title
            for risk in document.risk_indicators
            if risk.severity in {VisionRiskSeverity.high, VisionRiskSeverity.critical}
        )
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
                "available" if document.confidence_summary.confidence_available else "not_available"
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
            metadata={
                "review_id": task.review_id,
                "priority": severity.value,
                "reasons": task.reasons,
            },
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
        if any(
            risk.severity in {VisionRiskSeverity.high, VisionRiskSeverity.critical}
            for risk in document.risk_indicators
        ):
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
                field
                for field in document.extracted_fields
                if not field_names or field.name in field_names
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
        resolved_category = self._indicator_category(category)
        resolved_rule_id = rule_id or self._rule_id(category, title)
        risk_identity = hash_document(
            "|".join(
                (
                    document.document_id,
                    resolved_rule_id,
                    resolved_category,
                    title.strip().lower(),
                    severity.value,
                    ",".join(sorted(field_names or [])),
                    self._normalize(source_text),
                )
            ).encode()
        )
        return VisionRiskIndicator(
            risk_id=f"vis-risk-{risk_identity[:12]}",
            title=title,
            severity=severity,
            category=resolved_category,
            description=description,
            recommended_action=recommended_action,
            evidence=evidence_items,
            rule_id=resolved_rule_id,
            affected_fields=field_names or [],
        )

    def _field(
        self, document: VisionDocumentRecord, name: str, value: str, source_text: str
    ) -> ExtractedVisionField:
        box = self._box_for_text(value, document.ocr_words) or self._box_for_text(
            source_text, document.ocr_words
        )
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
                if (
                    schema_field := (
                        vision_schema_registry.get(document.document_type).field(name)
                        if vision_schema_registry.get(document.document_type)
                        else None
                    )
                )
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
        resolved_source = (
            source_text or document.parsed_text[:400] or "No source text available."
        )[:700]
        resolved_box = bounding_box or self._box_for_text(
            source_text,
            document.ocr_words,
        )
        box_identity = (
            ",".join(
                f"{value:.3f}"
                for value in (
                    resolved_box.x1,
                    resolved_box.y1,
                    resolved_box.x2,
                    resolved_box.y2,
                )
            )
            if resolved_box
            else "no-box"
        )
        evidence_identity = hash_document(
            (
                f"{document.document_id}|{resolved_page}|{model_name}|"
                f"{self._normalize(resolved_source)}|{box_identity}"
            ).encode()
        )
        return VisionEvidence(
            evidence_id=f"vis-ev-{evidence_identity[:12]}",
            document_id=document.document_id,
            page_number=resolved_page,
            source_text=resolved_source,
            bounding_box=resolved_box,
            visual_region=f"page_{resolved_page}",
            model_confidence=max(0.0, min(1.0, confidence)),
            model_name=model_name,
            model_version="v1",
            processing_timestamp=utc_now_iso(),
            metadata=metadata or {},
        )

    def _word_evidence(self, document: VisionDocumentRecord, word: OCRWord) -> VisionEvidence:
        box_identity = ",".join(
            f"{value:.3f}"
            for value in (
                word.bounding_box.x1,
                word.bounding_box.y1,
                word.bounding_box.x2,
                word.bounding_box.y2,
            )
        )
        evidence_identity = hash_document(
            (
                f"{document.document_id}|{word.page}|{word.reading_order}|"
                f"{word.model_name}|{self._normalize(word.text)}|{box_identity}"
            ).encode()
        )
        return VisionEvidence(
            evidence_id=f"vis-ev-{evidence_identity[:12]}",
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

    def _transition(
        self, document: VisionDocumentRecord, status: VisionProcessingStatus, message: str
    ) -> None:
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
        duration_ms: float | None = None,
        document: VisionDocumentRecord | None = None,
        input_schema: str | None = None,
        output_schema: str | None = None,
        selected_schema: str | None = None,
        metrics: dict[str, Any] | None = None,
        result_status: str | None = None,
        reason: str | None = None,
    ) -> ModelRunRecord:
        now = utc_now_iso()
        runtime_ms = round(max(0.0, duration_ms or 0.0), 3)
        return ModelRunRecord(
            stage=stage,
            model_name=model_name,
            model_version=model_version,
            status=status,
            started_at=now,
            completed_at=now,
            confidence=confidence,
            processing_duration_ms=runtime_ms,
            runtime_ms=runtime_ms,
            source_document=document.document_id if document else None,
            source_pages=list(range(1, document.page_count + 1)) if document else [],
            routing_reason=routing_reason,
            notes=notes or [],
            input_schema=input_schema,
            output_schema=output_schema,
            document_type=document.document_type if document else None,
            selected_schema=selected_schema,
            metrics=metrics or {},
            result_status=result_status,
            reason=reason,
        )

    def _field_matches(
        self, question: str, documents: list[VisionDocumentRecord]
    ) -> list[ExtractedVisionField]:
        question_tokens = set(re.findall(r"[a-z0-9_]+", question))
        requested_names = self._requested_field_names(question)
        matches = []
        for document in documents:
            for field in document.extracted_fields:
                name_tokens = set(field.name.lower().split("_"))
                if (
                    field.name in requested_names
                    or name_tokens & question_tokens
                    or field.name.lower() in question
                ):
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
            extraction_status=field.reviewer_status.value,
            evidence_type="structured_field",
        )

    def _requested_field_names(self, question: str) -> set[str]:
        lower = question.lower()
        aliases = {
            "organisation_name": ("organisation", "organization", "company name"),
            "address": ("address",),
            "contact_name": ("contact name", "contact person", "contact"),
            "phone_number": ("phone", "telephone", "contact number"),
            "recipient_phone": ("phone", "telephone", "contact number"),
            "sender_phone": ("sender phone", "from phone"),
            "fax_number": ("fax", "fax number", "facsimile", "telefax"),
            "recipient_fax": ("fax", "fax number", "facsimile", "telefax"),
            "sender_fax": ("sender fax", "from fax"),
            "visible_fax_numbers": ("fax", "fax number", "facsimile", "telefax"),
            "visible_phone_numbers": ("phone", "telephone", "contact number"),
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
            "date": ("date", "transmission date"),
            "sample_number": ("sample number", "sample no", "sample id"),
            "product_type": ("product type", "type of product", "type of cigarette", "sample type"),
            "batch_size": ("batch size", "lot size", "quantity"),
            "requested_by": ("original request", "requested by", "requester", "requestor"),
            "specifications_written_by": (
                "sample specifications",
                "specifications written by",
                "specifications by",
                "who wrote",
            ),
            "visible_dates": ("date", "visible date"),
            "recipient_name": ("recipient", "message to", "sent to"),
            "sender_name": ("sender", "sent by", "from"),
            "recipient_company": ("company", "organisation", "organization"),
            "company_name": ("company", "organisation", "organization"),
            "client_number": ("client number", "client no", "reference number"),
            "reference_number": ("reference number", "reference no", "ref no"),
            "document_title": ("document title", "title"),
            "expiry_date": ("expiry date", "expiration date"),
            "governing_law": ("governing law",),
            "termination_clause": ("termination", "termination clause"),
            "notice_period": ("notice period",),
            "liability_clause": ("liability clause",),
            "confidentiality_clause": ("confidentiality", "non-disclosure"),
            "indemnification_clause": ("indemnification", "indemnity"),
        }
        return {name for name, terms in aliases.items() if any(term in lower for term in terms)}

    def _logical_requested_field(self, question: str) -> str | None:
        lower = question.lower()
        terms = (
            ("sample_number", ("sample number", "sample no", "sample id")),
            (
                "product_type",
                ("product type", "type of product", "type of cigarette", "sample type"),
            ),
            ("batch_size", ("batch size", "lot size", "quantity")),
            ("requested_by", ("original request", "requested by", "requester", "requestor")),
            (
                "specifications_written_by",
                (
                    "sample specifications",
                    "specifications written by",
                    "specifications by",
                    "who wrote",
                ),
            ),
            ("fax_number", ("fax", "facsimile", "telefax")),
            ("phone_number", ("phone", "telephone", "contact number")),
            ("client_number", ("client number", "client no")),
            ("reference_number", ("reference number", "reference no", "ref no")),
            ("recipient_name", ("recipient", "message to", "sent to")),
            ("sender_name", ("sender", "sent by")),
            ("company", ("company", "organisation", "organization")),
            ("date", ("date", "dated")),
            ("email", ("email", "e-mail")),
        )
        return next(
            (name for name, aliases in terms if any(alias in lower for alias in aliases)),
            None,
        )

    def _ocr_lookup(
        self,
        question: str,
        documents: list[VisionDocumentRecord],
        top_k: int,
    ) -> list[tuple[VisionDocumentRecord, Any, str]]:
        logical = self._logical_requested_field(question)
        if logical is None:
            return []
        value_patterns: dict[str, str] = {
            "fax_number": r"(?i)\b(?:fax|facsimile|telefax)(?:\s+(?:no\.?|number))?\s*[:#=-]?\s*((?:\+?\d[\d()./ -]{6,}\d))",
            "phone_number": r"(?i)\b(?:phone|telephone|tel|contact\s+number)(?:\s+(?:no\.?|number))?\s*[:#=-]?\s*((?:\+?\d[\d()./ -]{6,}\d))",
            "client_number": r"(?i)\b(?:client)(?:\s+(?:no\.?|number))?\s*[:#=-]?\s*([A-Z0-9][A-Z0-9._/-]{2,39})",
            "reference_number": r"(?i)\b(?:reference|ref)(?:\s+(?:no\.?|number))?\s*[:#=-]?\s*([A-Z0-9][A-Z0-9._/-]{2,39})",
            "recipient_name": r"(?i)^(?:message\s+to|recipient|attention|attn|to)\s*[:#=-]?\s*(\S.+)$",
            "sender_name": r"(?i)^(?:sender|sent\s+by|from)\s*[:#=-]?\s*(\S.+)$",
            "company": r"(?i)^(?:company|organisation|organization|firm)\s*[:#=-]?\s*(\S.+)$",
            "date": r"(?i)\b(?:date|dated|transmission\s+date)\s*[:#=-]?\s*((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            "email": r"(?i)\b([^\s@]+@[^\s@]+\.[^\s@]+)\b",
            "sample_number": r"(?i)\b(?:sample\s+(?:no\.?|number|id))\s*[:#=-]?\s*([A-Z0-9][A-Z0-9._/-]{1,39})",
            "product_type": r"(?i)\b(?:type\s+of\s+(?:cigarette|product)|product\s+type|sample\s+type|item\s+type)\s*[:#=-]?\s*(\S.+)$",
            "batch_size": r"(?i)\b(?:batch\s+size|lot\s+size|quantity)\s*[:#=-]?\s*(\d+(?:\.\d+)?\s*[A-Za-z.]+)",
            "requested_by": r"(?i)\b(?:original\s+request\s+made\s+by|requested\s+by|requester|requestor)\s*[:#=-]?\s*(\S.+)$",
            "specifications_written_by": r"(?i)\b(?:sample\s+specifications\s+written\s+by|specifications\s+(?:written\s+)?by|prepared\s+by)\s*[:#=-]?\s*(\S.+)$",
        }
        pattern = re.compile(value_patterns[logical])
        matches: list[tuple[VisionDocumentRecord, Any, str]] = []
        seen: set[tuple[str, str]] = set()
        for document in documents:
            if not document.ocr_lines:
                document.ocr_lines = ocr_line_reconstructor.reconstruct(document.ocr_words)
            lines = sorted(document.ocr_lines, key=lambda item: item.reading_order)
            for index, line in enumerate(lines):
                match = pattern.search(line.text)
                value = match.group(1).strip(" :#=-") if match else ""
                evidence_line = line
                if not value and self._line_has_requested_label(logical, line.text):
                    candidate = lines[index + 1] if index + 1 < len(lines) else None
                    if candidate and candidate.page == line.page:
                        raw_match = re.search(
                            r"(?:\+?\d[\d()./ -]{6,}\d|[A-Z0-9][A-Z0-9._/-]{2,39})",
                            candidate.text,
                            re.I,
                        )
                        if raw_match:
                            value = raw_match.group(0).strip()
                            evidence_line = candidate.model_copy(
                                update={
                                    "text": f"{line.text} {candidate.text}",
                                    "bounding_box": self._union_box(
                                        [line.bounding_box, candidate.bounding_box]
                                    ),
                                    "mean_confidence": round(
                                        (
                                            (line.mean_confidence or 0.0)
                                            + (candidate.mean_confidence or 0.0)
                                        )
                                        / 2,
                                        4,
                                    ),
                                }
                            )
                if not value:
                    continue
                if logical in {"phone_number", "fax_number"}:
                    validation = semantic_value_registry.validate(
                        SemanticValueType.PHONE,
                        value,
                        SemanticContext(label=line.text),
                    )
                    if not validation.valid:
                        continue
                key = (document.document_id, self._normalize(value))
                if key in seen:
                    continue
                seen.add(key)
                matches.append((document, evidence_line, value))
        matches.sort(
            key=lambda item: item[1].mean_confidence or 0.0,
            reverse=True,
        )
        return matches[:top_k]

    def _line_has_requested_label(self, logical: str, text: str) -> bool:
        labels = {
            "fax_number": ("fax", "facsimile", "telefax"),
            "phone_number": ("phone", "telephone", "tel"),
            "client_number": ("client",),
            "reference_number": ("reference", "ref"),
            "recipient_name": ("message to", "recipient", "attention", "attn", "to"),
            "sender_name": ("sender", "sent by", "from"),
            "company": ("company", "organisation", "organization", "firm"),
            "date": ("date", "dated"),
            "email": ("email", "e-mail"),
            "sample_number": ("sample no", "sample number", "sample id"),
            "product_type": ("type of cigarette", "product type", "sample type", "item type"),
            "batch_size": ("batch size", "lot size", "quantity"),
            "requested_by": ("original request made by", "requested by", "requester", "requestor"),
            "specifications_written_by": (
                "sample specifications written by",
                "specifications by",
                "prepared by",
            ),
        }
        normalized = text.lower()
        return any(label in normalized for label in labels.get(logical, ()))

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
        advanced_embedding = self.retrieval_index.embedding.adapter_id != "local-hashing-embedding"
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
            name for name in compared if self._normalize(left[name]) == self._normalize(right[name])
        ]
        denominator = sum(weights[name] for name in compared)
        score = sum(weights[name] for name in matches) / denominator
        return score, matches, len(compared)

    def _related_document(
        self,
        document: VisionDocumentRecord,
    ) -> tuple[VisionDocumentRecord, float] | None:
        self._load_all_documents(document.tenant_id)
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
            "conversations",
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
            dict(value) for value in audit_payload.get("events", []) if isinstance(value, dict)
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
        self._document_tenant_index[document.document_id] = document.tenant_id
        self.persistence.save_document(document)
        self._write_json(
            self._state_file("documents", document.document_id),
            document.model_dump(mode="json"),
        )

    def _load_document(
        self,
        document_id: str,
        *,
        tenant_id: str | None = None,
    ) -> VisionDocumentRecord | None:
        payload = self._read_json(self._state_file("documents", document_id))
        if not payload:
            return None
        payload_tenant = str(payload.get("tenant_id") or "tenant-local")
        self._document_tenant_index[document_id] = payload_tenant
        if tenant_id and payload_tenant != tenant_id:
            return None
        cached = self.documents.get(document_id)
        if (
            cached is not None
            and cached.tenant_id == payload_tenant
            and cached.updated_at == payload.get("updated_at")
            and cached.data_version == int(payload.get("data_version", 1))
        ):
            return cached
        try:
            document = VisionDocumentRecord.model_validate(payload)
        except ValueError:
            return None
        self.documents[document.document_id] = document
        if document.review_task:
            self.review_tasks[document.review_task.review_id] = document.review_task
        self._index_document(document)
        return document

    def _load_all_documents(self, tenant_id: str | None = None) -> None:
        index_changed = False
        for path in (self._state_root / "documents").glob("*.json"):
            document_id = path.stem
            cached = self.documents.get(document_id)
            if cached is not None:
                self._document_tenant_index[document_id] = cached.tenant_id
                if tenant_id and cached.tenant_id != tenant_id:
                    continue
                self._load_document(document_id, tenant_id=tenant_id)
                continue
            known_tenant = self._document_tenant_index.get(document_id)
            if tenant_id and known_tenant and known_tenant != tenant_id:
                continue
            previous_tenant = known_tenant
            self._load_document(document_id, tenant_id=tenant_id)
            index_changed = index_changed or (
                previous_tenant != self._document_tenant_index.get(document_id)
            )
        if index_changed:
            self._persist_indexes()

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
        self._write_json(
            self._state_file("reviews", review.review_id), review.model_dump(mode="json")
        )
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
        self._write_json(
            pending_dir / "metadata.json", {"filename": filename, "metadata": metadata}
        )

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

    def _mark_derived_artifacts_stale(
        self,
        document: VisionDocumentRecord,
        *,
        actor: str,
    ) -> dict[str, list[str]]:
        """Invalidate outputs whose evidence snapshot predates a reprocess.

        The records remain durable for audit/version history, but normal listings
        no longer present them as current evidence-backed outputs.
        """
        stale_narratives: list[str] = []
        stale_reports: list[str] = []
        for narrative in self.list_finance_narratives(document.tenant_id):
            source_ids = narrative.source_document_ids or narrative.document_ids
            if document.document_id not in source_ids:
                continue
            narrative.status = "stale"
            self._persist_finance_narrative(narrative)
            stale_narratives.append(narrative.narrative_id)
        for report in self.list_reports(document.tenant_id):
            if document.document_id not in report.document_ids:
                continue
            report.status = "stale"
            self._persist_report_record(report)
            stale_reports.append(report.report_id)
        if stale_narratives or stale_reports:
            self._audit(
                document,
                "derived_artifacts_invalidated",
                "Marked evidence-derived narratives and reports stale before reprocessing.",
                actor=actor,
                metadata={
                    "narrative_ids": stale_narratives,
                    "report_ids": stale_reports,
                    "source_data_version": document.data_version,
                },
            )
        return {
            "narrative_ids": stale_narratives,
            "report_ids": stale_reports,
        }

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
                    if field.reviewer_status in {ReviewerStatus.accepted, ReviewerStatus.edited}
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
                "document_tenants": self._document_tenant_index,
            },
        )

    def _load_indexes(self) -> None:
        payload = self._read_json(self._state_root / "indexes.json") or {}
        self._idempotency_index.update(dict(payload.get("idempotency") or {}))
        self._hash_index.update(dict(payload.get("hash") or {}))
        self._semantic_index.update(dict(payload.get("semantic") or {}))
        self._document_tenant_index.update(dict(payload.get("document_tenants") or {}))

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
        risk_evidence_ids = {evidence.evidence_id for risk in risks for evidence in risk.evidence}
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
                components["donut_agreement"] = 0.0 if field.name in donut_disagreements else 1.0
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
                        normalized_value and normalized_value in self._normalize(chunk.text)
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
            field.calibrated_confidence = field.confidence if field.calibrated else None
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
            sum(components[key] * weight for key, weight in active_weights.items()) / denominator
            if denominator
            else 0.0
        )
        return round(max(0.0, min(1.0, score)), 4)

    def _mean_word_confidence(self, document: VisionDocumentRecord) -> float | None:
        if not document.ocr_words:
            return None
        return round(
            sum(word.confidence for word in document.ocr_words) / len(document.ocr_words), 3
        )

    def _text_from_ocr(self, words: list[OCRWord]) -> str:
        lines = ocr_line_reconstructor.reconstruct(words)
        pages: dict[int, list[str]] = {}
        for line in lines:
            pages.setdefault(line.page, []).append(line.text)
        return "\n\n".join("\n".join(pages[page_number]) for page_number in sorted(pages))

    def _mean_field_confidence(self, document: VisionDocumentRecord) -> float | None:
        if not document.extracted_fields:
            return None
        values = [
            field.confidence for field in document.extracted_fields if field.confidence is not None
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
        return round(
            sum(region.confidence for region in document.layout_regions)
            / len(document.layout_regions),
            3,
        )

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
            item.severity for item in [*document.risk_indicators, *document.processing_warnings]
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
        prefix = (
            "cross_document" if category == "cross_document" else self._indicator_category(category)
        )
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
        formats = (
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%b %d, %Y",
            "%B %d, %Y",
        )
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
