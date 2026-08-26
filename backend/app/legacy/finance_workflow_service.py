import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.security import hash_document
from app.legacy.finance_evaluation_service import FinanceEvaluationService
from app.legacy.finance_extraction_service import FinanceExtractionService
from app.legacy.finance_narrative_service import FinanceNarrativeService
from app.legacy.vector_store import FinanceVectorStore
from app.models.schemas import (
    DemoCase,
    EvaluationResult,
    ExtractedFinanceFields,
    FinanceDocument,
    FinanceDocumentType,
    FinanceNarrative,
    FinanceNarrativeRequest,
    FinanceWorkflowStatus,
    HumanReview,
    HumanReviewAcceptance,
    HumanReviewRequest,
    RiskFlag,
    RiskLevel,
    VectorRecord,
)
from app.services.ingestion import DocumentIngestionService


class FinanceReviewGuardrailError(ValueError):
    """Raised when a reviewer action violates finance safety guardrails."""


class FinanceProject6Workflow:
    """End-to-end NVIDIA Project 6 workflow for finance document intelligence."""

    def __init__(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[3]
        self.ingestion = DocumentIngestionService()
        self.extractor = FinanceExtractionService()
        self.narratives = FinanceNarrativeService()
        self.evaluator = FinanceEvaluationService()
        self.vector_store = FinanceVectorStore()
        self.documents: dict[str, FinanceDocument] = {}
        self.extractions: dict[str, ExtractedFinanceFields] = {}
        self.generated_narratives: dict[str, FinanceNarrative] = {}
        self.reviews: dict[str, HumanReview] = {}
        self.evaluations: dict[str, EvaluationResult] = {}
        self.latency_by_document: dict[str, dict[str, float]] = {}
        self.demo_cases = self._demo_cases()

    def upload(
        self,
        filename: str,
        content: bytes,
        document_type: FinanceDocumentType | None = None,
        uploaded_by: str = "analyst",
        metadata: dict[str, Any] | None = None,
    ) -> FinanceDocument:
        started = time.perf_counter()
        text, pages, _, stored_path = self.ingestion.parse(filename, content)
        document = FinanceDocument(
            file_name=filename,
            document_type=document_type or FinanceDocumentType.unknown,
            uploaded_by=uploaded_by,
            upload_timestamp=self._now(),
            text=text,
            pages=pages,
            stored_path=stored_path,
            checksum=hash_document(content),
            metadata=metadata or {},
        )
        selected_type = document.document_type
        detected_type = self.extractor.detect_type(document)
        document.metadata["user_selected_document_type"] = selected_type.value
        document.metadata["auto_detected_document_type"] = detected_type.value
        document.document_type = self._resolve_document_type(selected_type, detected_type, document.metadata)
        if document.document_type == FinanceDocumentType.bank_change_request:
            document.risk_level = RiskLevel.high
        ground_truth_path = self._ground_truth_for_upload(document.file_name, document.document_type)
        if ground_truth_path and not document.metadata.get("ground_truth_path"):
            document.metadata["ground_truth_path"] = ground_truth_path
        self.documents[document.document_id] = document
        chunks = self._chunk_document(document)
        document.chunk_ids = [record.chunk_id for record in chunks]
        vector_started = time.perf_counter()
        self.vector_store.upsert_many(chunks)
        self._latency(document.document_id, "upload_processing_latency", started)
        self._latency(document.document_id, "embedding_generation_latency", vector_started)
        self._latency(document.document_id, "vector_storage_latency", vector_started)
        return document

    def extract(self, document_id: str) -> ExtractedFinanceFields:
        document = self._document(document_id)
        started = time.perf_counter()
        extraction = self.extractor.extract(document)
        document.document_type = extraction.document_type
        document.status = FinanceWorkflowStatus.extracted
        document.review_status = FinanceWorkflowStatus.extracted
        document.risk_level = self._risk_level(extraction.risk_flags)
        self.extractions[document_id] = extraction
        vector_started = time.perf_counter()
        self.vector_store.upsert(self._record_from_extraction(document, extraction))
        self.vector_store.upsert_many(self._records_from_risk_flags(document, extraction.risk_flags))
        self._latency(document_id, "extraction_latency", started)
        self._latency(document_id, "vector_storage_latency", vector_started)
        return extraction

    def generate_narrative(self, request: FinanceNarrativeRequest) -> FinanceNarrative:
        document = self._document(request.document_id)
        extraction = self.extractions.get(request.document_id) or self.extract(request.document_id)
        started = time.perf_counter()
        narrative = self.narratives.generate(document, extraction, request.narrative_type)
        document.status = FinanceWorkflowStatus.narrative_generated
        document.review_status = FinanceWorkflowStatus.narrative_generated
        document.risk_level = self._risk_level(narrative.risk_flags)
        self.generated_narratives[request.document_id] = narrative
        vector_started = time.perf_counter()
        self.vector_store.upsert(self._record_from_narrative(document, narrative))
        self._latency(request.document_id, "narrative_generation_latency", started)
        self._latency(request.document_id, "vector_storage_latency", vector_started)
        return narrative

    def send_to_review(self, document_id: str) -> HumanReview:
        document = self._document(document_id)
        narrative = self.generated_narratives.get(document_id) or self.generate_narrative(
            FinanceNarrativeRequest(document_id=document_id)
        )
        review = self.reviews.get(document_id) or HumanReview(
            document_id=document_id,
            status=FinanceWorkflowStatus.pending_review,
            ai_generated_narrative=narrative.narrative,
        )
        review.status = FinanceWorkflowStatus.pending_review
        review.review_decision = "pending"
        document.status = FinanceWorkflowStatus.pending_review
        document.review_status = FinanceWorkflowStatus.pending_review
        self.reviews[document_id] = review
        self.vector_store.upsert(self._record_from_review(document, review, "human_review_queue"))
        return review

    def review(self, document_id: str, request: HumanReviewRequest, action: str = "update") -> HumanReview:
        document = self._document(document_id)
        started = time.perf_counter()
        review = self.reviews.get(document_id) or self.send_to_review(document_id)
        extraction = self.extractions.get(document_id)
        narrative = self.generated_narratives.get(document_id)
        if request.corrected_extraction and extraction:
            extraction.corrected_extraction = request.corrected_extraction
            review.corrected_extraction = request.corrected_extraction
            review.correction_count += self._correction_count(extraction.original_extraction, request.corrected_extraction)

        if action == "approve":
            if self._approval_requires_verification(document, extraction, narrative) and not self._has_verification_proof(
                request.reviewer_comments
            ):
                raise FinanceReviewGuardrailError(
                    "Approval blocked: high-risk or bank-change finance documents require escalation, correction, "
                    "or reviewer comments documenting independent out-of-band verification."
                )
            review.status = FinanceWorkflowStatus.approved
            review.review_decision = "approved"
            review.acceptance_status = request.acceptance_status or self._acceptance_for_corrections(review.correction_count)
        elif action == "reject":
            review.status = FinanceWorkflowStatus.rejected
            review.review_decision = "rejected"
            review.acceptance_status = HumanReviewAcceptance.rejected
        elif action == "escalate":
            review.status = FinanceWorkflowStatus.escalated
            review.review_decision = "escalated"
            review.acceptance_status = HumanReviewAcceptance.escalated
            review.escalation_reason = request.escalation_reason or request.reviewer_comments or "High-risk finance review required."
        elif action in {"request_correction", "needs_correction"} or request.decision == "request_correction":
            review.status = FinanceWorkflowStatus.needs_correction
            review.review_decision = "needs_correction"
            review.acceptance_status = request.acceptance_status or HumanReviewAcceptance.accepted_with_major_changes
        else:
            review.review_decision = request.decision
            if request.acceptance_status:
                review.acceptance_status = request.acceptance_status

        review.reviewer_name = request.reviewer_name
        review.reviewer_id = request.reviewer_id
        review.reviewer_comments = request.reviewer_comments
        review.final_approved_narrative = request.final_approved_narrative or (
            narrative.narrative if narrative and review.status == FinanceWorkflowStatus.approved else review.final_approved_narrative
        )
        review.review_timestamp = self._now()
        review.history.append(
            {
                "action": action,
                "decision": review.review_decision,
                "status": review.status.value,
                "reviewer": request.reviewer_name,
                "timestamp": review.review_timestamp,
                "comments": request.reviewer_comments,
            }
        )
        document.status = review.status
        document.review_status = review.status
        self.reviews[document_id] = review
        self.vector_store.upsert(self._record_from_review(document, review, "human_review_decision"))
        if review.final_approved_narrative:
            self.vector_store.upsert(self._record_from_final_narrative(document, review))
        self._latency(document_id, "review_submission_latency", started)
        self.latency_by_document.setdefault(document_id, {})["total_end_to_end_latency"] = round(
            sum(self.latency_by_document.get(document_id, {}).values()),
            4,
        )
        return review

    def evaluate(self, document_id: str | None = None) -> EvaluationResult:
        ground_truth = self._ground_truth_by_document()
        evaluation = self.evaluator.evaluate(
            self.extractions,
            self.generated_narratives,
            self.reviews,
            self.vector_store,
            ground_truth_by_document=ground_truth,
            latency_by_document=self.latency_by_document,
            document_id=document_id,
        )
        self.evaluations[document_id or "all"] = evaluation
        if document_id:
            document = self._document(document_id)
            self.vector_store.upsert(self._record_from_evaluation(document, evaluation))
        return evaluation

    def metrics(self) -> dict[str, Any]:
        evaluation = self.evaluations.get("all") or self.evaluate()
        return {
            "documents_processed": len(self.documents),
            "pending_reviews": len([review for review in self.reviews.values() if review.status == FinanceWorkflowStatus.pending_review]),
            "completed_reviews": len([review for review in self.reviews.values() if review.status != FinanceWorkflowStatus.pending_review]),
            "high_risk_documents": len([document for document in self.documents.values() if document.risk_level in {RiskLevel.high, RiskLevel.critical}]),
            "vector_records": self.vector_store.facets()["record_count"],
            "extraction_accuracy": evaluation.extraction_accuracy,
            "hallucination": evaluation.hallucination,
            "latency": evaluation.latency,
            "human_review_acceptance": evaluation.human_review_acceptance,
            "retrieval": evaluation.retrieval,
            "safety": evaluation.safety,
            "cost": evaluation.cost,
        }

    def run_demo_case(self, case_id: str) -> dict[str, Any]:
        demo = next((item for item in self.demo_cases if item.case_id == case_id), None)
        if not demo:
            raise ValueError("demo case not found")
        path = self.repo_root / demo.file_path
        document = self.upload(
            path.name,
            path.read_bytes(),
            document_type=demo.document_type,
            uploaded_by="demo",
            metadata={"demo_case_id": demo.case_id, "ground_truth_path": demo.ground_truth_path},
        )
        extraction = self.extract(document.document_id)
        narrative = self.generate_narrative(FinanceNarrativeRequest(document_id=document.document_id))
        review = self.send_to_review(document.document_id)
        if demo.document_type == FinanceDocumentType.bank_change_request:
            review = self.review(
                document.document_id,
                HumanReviewRequest(
                    reviewer_name="Demo Finance Reviewer",
                    decision="escalate",
                    reviewer_comments="Bank-change requests require out-of-band verification.",
                    escalation_reason="Vendor bank details changed with payment risk indicators.",
                ),
                action="escalate",
            )
        elif document.risk_level in {RiskLevel.high, RiskLevel.critical}:
            review = self.review(
                document.document_id,
                HumanReviewRequest(
                    reviewer_name="Demo Finance Reviewer",
                    decision="request_correction",
                    reviewer_comments="Risk flags require correction or supporting evidence before approval.",
                ),
                action="request_correction",
            )
        else:
            review = self.review(
                document.document_id,
                HumanReviewRequest(
                    reviewer_name="Demo Finance Reviewer",
                    decision="approve",
                    reviewer_comments="Fields and narrative accepted for demo workflow.",
                ),
                action="approve",
            )
        evaluation = self.evaluate(document.document_id)
        return {
            "demo_case": demo.model_dump(mode="json"),
            "document": document.model_dump(mode="json"),
            "extraction": extraction.model_dump(mode="json"),
            "narrative": narrative.model_dump(mode="json"),
            "review": review.model_dump(mode="json"),
            "evaluation": evaluation.model_dump(mode="json"),
            "vector_verification": {
                "document_records": len(self.vector_store.document_records(document.document_id)),
                "search_hits": self.vector_store.search("finance risk payment", document_id=document.document_id),
            },
        }

    def pending_reviews(self) -> list[dict[str, Any]]:
        return [self._review_payload(review) for review in self.reviews.values() if review.status == FinanceWorkflowStatus.pending_review]

    def completed_reviews(self) -> list[dict[str, Any]]:
        return [self._review_payload(review) for review in self.reviews.values() if review.status != FinanceWorkflowStatus.pending_review]

    def _review_payload(self, review: HumanReview) -> dict[str, Any]:
        document = self.documents.get(review.document_id)
        return {
            "document": document.model_dump(mode="json") if document else None,
            "extraction": self.extractions.get(review.document_id).model_dump(mode="json")
            if review.document_id in self.extractions
            else None,
            "narrative": self.generated_narratives.get(review.document_id).model_dump(mode="json")
            if review.document_id in self.generated_narratives
            else None,
            "review": review.model_dump(mode="json"),
        }

    def _chunk_document(self, document: FinanceDocument) -> list[VectorRecord]:
        normalized = " ".join(document.text.split()) or "No text extracted. OCR review required."
        chunks = []
        window = 720
        for index in range(0, len(normalized), window):
            text = normalized[index : index + window]
            if not text:
                continue
            chunks.append(
                VectorRecord(
                    document_id=document.document_id,
                    document_type=document.document_type,
                    file_name=document.file_name,
                    uploaded_by=document.uploaded_by,
                    upload_timestamp=document.upload_timestamp,
                    page_number=len(chunks) + 1,
                    chunk_id=f"{document.document_id}-chunk-{len(chunks) + 1}",
                    review_status=document.review_status,
                    risk_level=document.risk_level,
                    source_type="document_chunk",
                    created_at=self._now(),
                    text=text,
                    metadata={
                        "checksum": document.checksum,
                        "stored_path": document.stored_path,
                        "source_traceability": "raw_document_text",
                    },
                )
            )
        return chunks or [
            VectorRecord(
                document_id=document.document_id,
                document_type=document.document_type,
                file_name=document.file_name,
                uploaded_by=document.uploaded_by,
                upload_timestamp=document.upload_timestamp,
                page_number=1,
                chunk_id=f"{document.document_id}-chunk-1",
                review_status=document.review_status,
                risk_level=document.risk_level,
                source_type="document_chunk",
                created_at=self._now(),
                text="No text extracted. OCR review required.",
            )
        ]

    def _record_from_extraction(self, document: FinanceDocument, extraction: ExtractedFinanceFields) -> VectorRecord:
        return self._vector_record(
            document,
            "extracted_finance_json",
            json.dumps(extraction.model_dump(mode="json"), sort_keys=True),
            extraction_version=extraction.extraction_version,
        )

    def _records_from_risk_flags(self, document: FinanceDocument, flags: list[RiskFlag]) -> list[VectorRecord]:
        return [
            self._vector_record(
                document,
                "risk_flag",
                f"{flag.title}. {flag.description} Recommended action: {flag.recommended_action}",
                risk_level=flag.severity,
                metadata={"flag_id": flag.flag_id, "category": flag.category},
            )
            for flag in flags
        ]

    def _record_from_narrative(self, document: FinanceDocument, narrative: FinanceNarrative) -> VectorRecord:
        return self._vector_record(
            document,
            "finance_narrative",
            narrative.narrative,
            risk_level=self._risk_level(narrative.risk_flags),
            metadata={
                "narrative_id": narrative.narrative_id,
                "narrative_type": narrative.narrative_type.value,
                "groundedness_score": narrative.groundedness_score,
                "unsupported_claims": narrative.unsupported_claims,
            },
        )

    def _record_from_review(self, document: FinanceDocument, review: HumanReview, source_type: str) -> VectorRecord:
        return self._vector_record(
            document,
            source_type,
            " ".join(
                part
                for part in [
                    review.review_decision,
                    review.reviewer_comments,
                    review.escalation_reason,
                    review.final_approved_narrative,
                ]
                if part
            )
            or "Pending human review.",
            metadata={"review_id": review.review_id, "acceptance_status": review.acceptance_status.value if review.acceptance_status else None},
        )

    def _record_from_final_narrative(self, document: FinanceDocument, review: HumanReview) -> VectorRecord:
        return self._vector_record(
            document,
            "final_approved_narrative",
            review.final_approved_narrative or "",
            metadata={"review_id": review.review_id},
        )

    def _record_from_evaluation(self, document: FinanceDocument, evaluation: EvaluationResult) -> VectorRecord:
        return self._vector_record(
            document,
            "evaluation_result",
            json.dumps(evaluation.model_dump(mode="json"), sort_keys=True),
            metadata={"evaluation_id": evaluation.evaluation_id},
        )

    def _vector_record(
        self,
        document: FinanceDocument,
        source_type: str,
        text: str,
        risk_level: RiskLevel | None = None,
        extraction_version: str = "finance-rules-v1",
        metadata: dict[str, Any] | None = None,
    ) -> VectorRecord:
        return VectorRecord(
            document_id=document.document_id,
            document_type=document.document_type,
            file_name=document.file_name,
            uploaded_by=document.uploaded_by,
            upload_timestamp=document.upload_timestamp,
            page_number=1,
            chunk_id=f"{document.document_id}-{source_type}-{int(time.time() * 1000)}",
            extraction_version=extraction_version,
            review_status=document.review_status,
            risk_level=risk_level or document.risk_level,
            source_type=source_type,
            created_at=self._now(),
            text=text,
            metadata=metadata or {},
        )

    def _ground_truth_by_document(self) -> dict[str, dict[str, Any]]:
        result = {}
        for document in self.documents.values():
            path_value = document.metadata.get("ground_truth_path")
            if not path_value:
                continue
            path = self.repo_root / path_value
            if not path.exists():
                continue
            try:
                result[document.document_id] = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
        return result

    def _demo_cases(self) -> list[DemoCase]:
        return [
            DemoCase(
                case_id="invoice-demo",
                title="Invoice Demo",
                scenario="Extract invoice payment values, PO reference, tax, terms, and bank details.",
                document_type=FinanceDocumentType.invoice,
                file_path="demo_data/invoice/invoice_demo.txt",
                ground_truth_path="demo_data/ground_truth/invoice_demo.json",
                expected_risk_level=RiskLevel.medium,
            ),
            DemoCase(
                case_id="purchase-order-demo",
                title="Purchase Order Demo",
                scenario="Extract PO supplier, amount, delivery, line item, and approval status.",
                document_type=FinanceDocumentType.purchase_order,
                file_path="demo_data/purchase_order/purchase_order_demo.txt",
                ground_truth_path="demo_data/ground_truth/purchase_order_demo.json",
                expected_risk_level=RiskLevel.low,
            ),
            DemoCase(
                case_id="contract-demo",
                title="Contract Demo",
                scenario="Extract payment clauses, obligations, renewal terms, and missing/risky clauses.",
                document_type=FinanceDocumentType.contract,
                file_path="demo_data/contract/contract_demo.txt",
                ground_truth_path="demo_data/ground_truth/contract_demo.json",
                expected_risk_level=RiskLevel.medium,
            ),
            DemoCase(
                case_id="bank-change-risk-demo",
                title="Bank-Change Risk Demo",
                scenario="Detect vendor bank-change risk, mismatch indicators, and required escalation.",
                document_type=FinanceDocumentType.bank_change_request,
                file_path="demo_data/bank_change_risk/bank_change_risk_demo.txt",
                ground_truth_path="demo_data/ground_truth/bank_change_risk_demo.json",
                expected_risk_level=RiskLevel.high,
            ),
        ]

    def _document(self, document_id: str) -> FinanceDocument:
        document = self.documents.get(document_id)
        if not document:
            raise ValueError("finance document not found")
        return document

    def _risk_level(self, flags: list[RiskFlag]) -> RiskLevel:
        severities = {flag.severity for flag in flags}
        if RiskLevel.critical in severities:
            return RiskLevel.critical
        if RiskLevel.high in severities:
            return RiskLevel.high
        if RiskLevel.medium in severities:
            return RiskLevel.medium
        return RiskLevel.low

    def _acceptance_for_corrections(self, correction_count: int) -> HumanReviewAcceptance:
        if correction_count == 0:
            return HumanReviewAcceptance.accepted_without_changes
        if correction_count <= 2:
            return HumanReviewAcceptance.accepted_with_minor_changes
        return HumanReviewAcceptance.accepted_with_major_changes

    def _resolve_document_type(
        self,
        selected_type: FinanceDocumentType,
        detected_type: FinanceDocumentType,
        metadata: dict[str, Any],
    ) -> FinanceDocumentType:
        if detected_type == FinanceDocumentType.unknown:
            return selected_type
        if selected_type == FinanceDocumentType.unknown:
            metadata["document_type_resolution"] = "auto_detected"
            return detected_type
        if detected_type == FinanceDocumentType.bank_change_request and selected_type != detected_type:
            metadata["document_type_resolution"] = "overridden_by_bank_change_guardrail"
            metadata["document_type_warning"] = (
                f"User selected {selected_type.value}, but bank-change risk language was detected."
            )
            return detected_type
        if selected_type != detected_type:
            metadata["document_type_resolution"] = "user_selected_with_detection_warning"
            metadata["document_type_warning"] = f"Auto-detection suggested {detected_type.value}; reviewer should confirm."
        return selected_type

    def _ground_truth_for_upload(self, filename: str, document_type: FinanceDocumentType) -> str | None:
        filename_key = filename.lower().replace(" ", "_")
        by_filename = {
            "invoice_demo.txt": "demo_data/ground_truth/invoice_demo.json",
            "purchase_order_demo.txt": "demo_data/ground_truth/purchase_order_demo.json",
            "contract_demo.txt": "demo_data/ground_truth/contract_demo.json",
            "bank_change_risk_demo.txt": "demo_data/ground_truth/bank_change_risk_demo.json",
        }
        if filename_key in by_filename:
            return by_filename[filename_key]
        if "bank_change" in filename_key and document_type == FinanceDocumentType.bank_change_request:
            return "demo_data/ground_truth/bank_change_risk_demo.json"
        return None

    def _approval_requires_verification(
        self,
        document: FinanceDocument,
        extraction: ExtractedFinanceFields | None,
        narrative: FinanceNarrative | None,
    ) -> bool:
        if document.document_type == FinanceDocumentType.bank_change_request:
            return True
        if document.risk_level in {RiskLevel.high, RiskLevel.critical}:
            return True
        flags = [*(extraction.risk_flags if extraction else []), *(narrative.risk_flags if narrative else [])]
        return any(flag.severity in {RiskLevel.high, RiskLevel.critical} for flag in flags)

    def _has_verification_proof(self, comments: str) -> bool:
        lowered = comments.lower()
        verification_terms = ("verified", "verification", "validated", "confirmed")
        trusted_channel_terms = ("out-of-band", "trusted", "callback", "call back", "independent", "registered contact")
        return any(term in lowered for term in verification_terms) and any(term in lowered for term in trusted_channel_terms)

    def _correction_count(self, original: dict[str, Any], corrected: dict[str, Any]) -> int:
        keys = set(original) | set(corrected)
        return sum(1 for key in keys if original.get(key) != corrected.get(key))

    def _latency(self, document_id: str, key: str, started: float) -> None:
        self.latency_by_document.setdefault(document_id, {})[key] = round(time.perf_counter() - started, 4)

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()


finance_workflow = FinanceProject6Workflow()
