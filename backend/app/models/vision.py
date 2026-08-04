from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


class VisionProcessingStatus(str, Enum):
    uploaded = "UPLOADED"
    queued = "QUEUED"
    preprocessing = "PREPROCESSING"
    ocr_completed = "OCR_COMPLETED"
    extraction_completed = "EXTRACTION_COMPLETED"
    validation_completed = "VALIDATION_COMPLETED"
    indexed = "INDEXED"
    review_required = "REVIEW_REQUIRED"
    under_review = "UNDER_REVIEW"
    approved = "APPROVED"
    rejected = "REJECTED"
    confirmed_error = "CONFIRMED_ERROR"
    confirmed_fraud = "CONFIRMED_FRAUD"
    completed = "COMPLETED"


class ReviewerStatus(str, Enum):
    unverified = "unverified"
    accepted = "accepted"
    edited = "edited"
    rejected = "rejected"
    confirmed_error = "confirmed_error"
    confirmed_fraud = "confirmed_fraud"


class VisionDocumentType(str, Enum):
    invoice = "invoice"
    purchase_order = "purchase_order"
    contract = "contract"
    receipt = "receipt"
    bank_document = "bank_document"
    insurance_document = "insurance_document"
    form = "form"
    report = "report"
    unknown = "unknown"


class VisionRiskSeverity(str, Enum):
    info = "info"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class VisionReportType(str, Enum):
    document_intelligence = "document_intelligence_report"
    risk_screening = "risk_screening_report"
    contract_review = "contract_review_report"
    vendor_summary = "vendor_summary"
    finance_narrative = "finance_narrative_report"
    model_evaluation = "model_evaluation_report"


class VisionFinanceNarrativeType(str, Enum):
    monthly_invoice_summary = "monthly_invoice_summary"
    vendor_payment_summary = "vendor_payment_summary"
    tax_summary = "tax_summary"
    outstanding_payment_summary = "outstanding_payment_summary"
    anomaly_summary = "anomaly_summary"
    approval_status_summary = "approval_status_summary"
    contract_exposure_summary = "contract_exposure_summary"


class VisionBoundingBox(BaseModel):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self):
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("Bounding-box maximum coordinates must not precede minimum coordinates")
        return self


class VisionEvidence(BaseModel):
    evidence_id: str = Field(default_factory=lambda: f"vis-ev-{uuid4().hex[:12]}")
    document_id: str
    page_number: int = Field(default=1, ge=1)
    source_text: str
    bounding_box: VisionBoundingBox | None = None
    visual_region: str | None = None
    model_confidence: float = Field(default=0.0, ge=0, le=1)
    model_name: str
    model_version: str
    processing_timestamp: str
    reviewer_status: ReviewerStatus = ReviewerStatus.unverified
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageQualityReport(BaseModel):
    page_number: int = Field(default=1, ge=1)
    width: int = Field(default=1000, ge=1)
    height: int = Field(default=1400, ge=1)
    dpi_estimate: int = Field(default=150, ge=1)
    rotation_degrees: int = 0
    blur_score: float = Field(default=0.0, ge=0)
    contrast_score: float = Field(default=0.0, ge=0, le=1)
    shadow_risk: float = Field(default=0.0, ge=0, le=1)
    image_quality_score: float = Field(default=0.0, ge=0, le=1)
    preprocessing_steps: list[str] = Field(default_factory=list)
    processed_image_path: str | None = None
    notes: list[str] = Field(default_factory=list)


class VisionPageArtifact(BaseModel):
    page_number: int = Field(default=1, ge=1)
    original_image_path: str
    processed_image_path: str
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    dpi: int = Field(ge=1)
    source_kind: str
    native_text_available: bool = False
    crop_box: VisionBoundingBox | None = None


class OCRWord(BaseModel):
    word_id: str = Field(default_factory=lambda: f"word-{uuid4().hex[:10]}")
    text: str
    confidence: float = Field(ge=0, le=1)
    polygon: list[list[float]]
    bounding_box: VisionBoundingBox
    page: int = Field(default=1, ge=1)
    line_id: str
    block_id: str
    reading_order: int = Field(ge=1)
    review_required: bool = False
    model_name: str = "local-ocr-word-adapter"
    model_version: str = "v1"


class LayoutRegion(BaseModel):
    region_id: str = Field(default_factory=lambda: f"region-{uuid4().hex[:10]}")
    region_type: str
    page_number: int = Field(default=1, ge=1)
    bounding_box: VisionBoundingBox
    text: str = ""
    confidence: float = Field(default=0.0, ge=0, le=1)
    reading_order: int = Field(ge=1)
    model_name: str = "local-layout-adapter"
    model_version: str = "v1"


class TableCell(BaseModel):
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    text: str
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    bounding_box: VisionBoundingBox


class TableResult(BaseModel):
    table_id: str = Field(default_factory=lambda: f"table-{uuid4().hex[:10]}")
    page_number: int = Field(default=1, ge=1)
    markdown: str
    html: str
    rows: list[list[str]]
    cells: list[TableCell]
    bounding_box: VisionBoundingBox
    confidence: float = Field(default=0.0, ge=0, le=1)
    model_name: str = "local-table-structure-adapter"
    model_version: str = "v1"


class VisualDetection(BaseModel):
    detection_id: str = Field(default_factory=lambda: f"det-{uuid4().hex[:10]}")
    detection_class: str
    page_number: int = Field(default=1, ge=1)
    bounding_box: VisionBoundingBox
    confidence: float = Field(default=0.0, ge=0, le=1)
    cropped_evidence_path: str | None = None
    model_name: str = "local-visual-cue-detector"
    model_version: str = "v1"
    reviewer_status: ReviewerStatus = ReviewerStatus.unverified
    notes: list[str] = Field(default_factory=list)


class TamperRegion(BaseModel):
    region_id: str = Field(default_factory=lambda: f"tamper-region-{uuid4().hex[:10]}")
    manipulation_class: str = "unclassified_visual_inconsistency"
    bounding_box: VisionBoundingBox
    screening_score: float = Field(default=0.0, ge=0, le=1)
    cropped_evidence_path: str | None = None


class TamperAnalysisResult(BaseModel):
    analysis_id: str = Field(default_factory=lambda: f"tamper-{uuid4().hex[:10]}")
    page_number: int = Field(default=1, ge=1)
    screening_score: float = Field(default=0.0, ge=0, le=1)
    tampering_probability: float | None = Field(default=None, ge=0, le=1)
    calibrated: bool = False
    heatmap_path: str | None = None
    suspicious_regions: list[TamperRegion] = Field(default_factory=list)
    explanation: str
    model_name: str
    model_version: str
    reviewer_status: ReviewerStatus = ReviewerStatus.unverified


class ExtractedVisionField(BaseModel):
    field_id: str = Field(default_factory=lambda: f"field-{uuid4().hex[:10]}")
    name: str
    value: str
    normalized_value: str | None = None
    confidence: float = Field(default=0.0, ge=0, le=1)
    evidence: VisionEvidence
    model_name: str = "local-schema-extraction-adapter"
    model_version: str = "v1"
    target_model: str | None = None
    confidence_components: dict[str, float] = Field(default_factory=dict)
    calibrated: bool = False
    reviewer_status: ReviewerStatus = ReviewerStatus.unverified
    reviewed_value: str | None = None


class VisionRiskIndicator(BaseModel):
    risk_id: str = Field(default_factory=lambda: f"vis-risk-{uuid4().hex[:10]}")
    title: str
    severity: VisionRiskSeverity = VisionRiskSeverity.medium
    category: str
    description: str
    recommended_action: str
    evidence: list[VisionEvidence] = Field(default_factory=list)
    reviewer_status: ReviewerStatus = ReviewerStatus.unverified
    model_name: str = "local-risk-rule-engine"
    model_version: str = "v1"


class DuplicateCandidate(BaseModel):
    duplicate_id: str = Field(default_factory=lambda: f"dup-{uuid4().hex[:10]}")
    status: str
    matched_document_id: str | None = None
    matched_filename: str | None = None
    similarity: float = Field(default=0.0, ge=0, le=1)
    evidence: list[VisionEvidence] = Field(default_factory=list)
    reason: str


class AnomalyAnalysisResult(BaseModel):
    analysis_id: str = Field(default_factory=lambda: f"anomaly-{uuid4().hex[:10]}")
    layer: str
    status: str
    anomaly_score: float | None = Field(default=None, ge=0, le=1)
    supervised_probability: float | None = Field(default=None, ge=0, le=1)
    calibrated: bool = False
    threshold: float | None = Field(default=None, ge=0, le=1)
    sample_count: int = Field(default=0, ge=0)
    features: dict[str, float | None] = Field(default_factory=dict)
    explanations: list[dict[str, Any]] = Field(default_factory=list)
    model_name: str
    model_version: str


class ModelRunRecord(BaseModel):
    run_id: str = Field(default_factory=lambda: f"model-run-{uuid4().hex[:10]}")
    stage: str
    model_name: str
    model_version: str
    status: str
    started_at: str
    completed_at: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    routing_reason: str | None = None
    notes: list[str] = Field(default_factory=list)


class VisionAuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"vis-aud-{uuid4().hex[:12]}")
    document_id: str
    actor: str = "system"
    event_type: str
    message: str
    created_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisionEvidenceChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: f"vis-chk-{uuid4().hex[:10]}")
    document_id: str
    tenant_id: str = "tenant-local"
    page_number: int = Field(default=1, ge=1)
    section: str = "body"
    document_type: VisionDocumentType = VisionDocumentType.unknown
    vendor_id: str | None = None
    bounding_boxes: list[VisionBoundingBox] = Field(default_factory=list)
    review_status: ReviewerStatus = ReviewerStatus.unverified
    source_type: str = "paragraph"
    model_name: str = "structure-aware-chunker"
    model_version: str = "local-structure-chunker-v1"
    text: str
    confidence: float = Field(default=0.0, ge=0, le=1)


class VisionGraphProjection(BaseModel):
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    evidence_policy: str = "Edges are deterministic or proposed and require evidence."


class VisionReviewTask(BaseModel):
    review_id: str = Field(default_factory=lambda: f"review-{uuid4().hex[:10]}")
    document_id: str
    status: VisionProcessingStatus = VisionProcessingStatus.review_required
    priority: VisionRiskSeverity = VisionRiskSeverity.medium
    reasons: list[str] = Field(default_factory=list)
    assigned_role: str = "REVIEWER"
    reviewer_comments: str = ""
    created_at: str
    updated_at: str
    history: list[VisionAuditEvent] = Field(default_factory=list)


class VisionDocumentRecord(BaseModel):
    document_id: str = Field(default_factory=lambda: f"vision-doc-{uuid4().hex[:10]}")
    case_id: str
    filename: str
    tenant_id: str = "tenant-local"
    status: VisionProcessingStatus = VisionProcessingStatus.uploaded
    document_type: VisionDocumentType = VisionDocumentType.unknown
    file_hash: str
    perceptual_hash: str | None = None
    visual_embedding: list[float] | None = None
    visual_embedding_model: str | None = None
    original_storage_path: str | None = None
    parsed_text: str = ""
    page_count: int = 1
    page_artifacts: list[VisionPageArtifact] = Field(default_factory=list)
    quality: list[PageQualityReport] = Field(default_factory=list)
    ocr_words: list[OCRWord] = Field(default_factory=list)
    layout_regions: list[LayoutRegion] = Field(default_factory=list)
    tables: list[TableResult] = Field(default_factory=list)
    visual_detections: list[VisualDetection] = Field(default_factory=list)
    tamper_analysis: list[TamperAnalysisResult] = Field(default_factory=list)
    extracted_fields: list[ExtractedVisionField] = Field(default_factory=list)
    risk_indicators: list[VisionRiskIndicator] = Field(default_factory=list)
    duplicate_candidates: list[DuplicateCandidate] = Field(default_factory=list)
    anomaly_analysis: list[AnomalyAnalysisResult] = Field(default_factory=list)
    evidence_chunks: list[VisionEvidenceChunk] = Field(default_factory=list)
    graph: VisionGraphProjection = Field(default_factory=VisionGraphProjection)
    model_runs: list[ModelRunRecord] = Field(default_factory=list)
    model_routing: dict[str, Any] = Field(default_factory=dict)
    review_task: VisionReviewTask | None = None
    audit_events: list[VisionAuditEvent] = Field(default_factory=list)
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisionUploadResponse(BaseModel):
    case_id: str
    job_id: str
    document_ids: list[str]
    status: VisionProcessingStatus
    status_url: str


class VisionFieldReviewRequest(BaseModel):
    reviewed_value: str | None = None
    reviewer_status: ReviewerStatus = ReviewerStatus.edited
    comment: str = ""


class VisionReviewDecisionRequest(BaseModel):
    decision: VisionProcessingStatus
    comment: str = ""


class VisionAskRequest(BaseModel):
    question: str
    document_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=12)


class VisionAskCitation(BaseModel):
    document_id: str
    page_number: int
    source_text: str
    bounding_boxes: list[VisionBoundingBox] = Field(default_factory=list)
    chunk_id: str | None = None
    evidence_id: str | None = None
    retrieval_score: float | None = Field(default=None, ge=0, le=1)
    retrieval_backend: str | None = None
    model_confidence: float = Field(default=0.0, ge=0, le=1)
    model_name: str = "local-hybrid-retriever"
    model_version: str = "v1"
    processing_timestamp: str | None = None
    reviewer_status: ReviewerStatus = ReviewerStatus.unverified


class VisionAskResponse(BaseModel):
    question_id: str | None = None
    answer: str
    answer_status: str
    query_type: str = "semantic_question"
    citations: list[VisionAskCitation] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)
    reasoning_summary: str = ""
    requires_review: bool = False
    generator_name: str = "grounded-evidence-composer"
    generator_version: str = "v1"


class VisionCalculationTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: f"calc-{uuid4().hex[:12]}")
    label: str
    value: str
    currency: str | None = None
    source_document_ids: list[str] = Field(default_factory=list)
    source_evidence_ids: list[str] = Field(default_factory=list)
    date_range: dict[str, str | None] = Field(default_factory=dict)
    calculation_function: str
    verification_status: str = "verified"


class VisionFinanceNarrativeRequest(BaseModel):
    narrative_type: VisionFinanceNarrativeType
    vendor_name: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    document_ids: list[str] = Field(default_factory=list)


class VisionFinanceNarrative(BaseModel):
    narrative_id: str = Field(default_factory=lambda: f"narrative-{uuid4().hex[:12]}")
    tenant_id: str
    narrative_type: VisionFinanceNarrativeType
    narrative: str
    calculations: list[VisionCalculationTrace] = Field(default_factory=list)
    citations: list[VisionAskCitation] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    all_values_verified: bool = True
    answer_status: str = "supported"
    created_at: str
    generated_by: str


class VisionReportRequest(BaseModel):
    report_type: VisionReportType = VisionReportType.document_intelligence
    document_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    vendor_name: str | None = None
    narrative_id: str | None = None
    evaluation: dict[str, Any] | None = None


class VisionReportRecord(BaseModel):
    report_id: str = Field(default_factory=lambda: f"report-{uuid4().hex[:12]}")
    tenant_id: str
    report_type: VisionReportType
    title: str
    document_ids: list[str] = Field(default_factory=list)
    content: dict[str, Any]
    generated_by: str
    created_at: str
    object_uri: str | None = None
