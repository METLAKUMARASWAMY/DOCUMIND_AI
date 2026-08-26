from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field, model_validator


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


class VisionExtractionOutcome(str, Enum):
    success_with_fields = "SUCCESS_WITH_FIELDS"
    completed_no_fields = "COMPLETED_NO_FIELDS"
    schema_fallback_used = "SCHEMA_FALLBACK_USED"
    schema_unavailable = "SCHEMA_UNAVAILABLE"
    low_confidence_fields = "LOW_CONFIDENCE_FIELDS"
    extraction_failed = "EXTRACTION_FAILED"
    fallback_required = "FALLBACK_REQUIRED"


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
    bank_confirmation = "bank_confirmation"
    fax_transmission = "fax_transmission"
    business_form = "business_form"
    legal_form = "legal_form"
    insurance_document = "insurance_document"
    financial_report = "financial_report"
    general_report = "general_report"
    correspondence = "correspondence"
    letter = "letter"
    statement = "statement"
    sample_form = "sample_form"
    product_specification = "product_specification"
    email = "email"
    generic_form = "generic_form"
    # Retained for records created before the v3 taxonomy migration.
    form = "form"
    report = "report"
    unknown = "unknown"


class VisionRiskSeverity(str, Enum):
    info = "info"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class VisionIndicatorCategory(str, Enum):
    processing_warning = "processing_warning"
    data_quality = "data_quality"
    business_risk = "business_risk"
    visual_risk = "visual_risk"


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
            raise ValueError(
                "Bounding-box maximum coordinates must not precede minimum coordinates"
            )
        return self


class VisionSemanticType(str, Enum):
    date = "DATE"
    datetime = "DATETIME"
    person = "PERSON"
    organisation = "ORGANISATION"
    money = "MONEY"
    number = "NUMBER"
    quantity = "QUANTITY"
    percentage = "PERCENTAGE"
    phone = "PHONE"
    email = "EMAIL"
    address = "ADDRESS"
    bank_account = "BANK_ACCOUNT"
    identifier = "IDENTIFIER"
    document_reference = "DOCUMENT_REFERENCE"
    currency = "CURRENCY"
    product = "PRODUCT"
    text = "TEXT"
    boolean = "BOOLEAN"
    url = "URL"


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


class GenericDiscoveredField(BaseModel):
    """A schema-independent label/value observation backed by OCR geometry."""

    field_id: str = Field(default_factory=lambda: f"generic-field-{uuid4().hex[:12]}")
    raw_label: str
    normalized_label: str
    raw_value: str
    normalized_value: str
    page: int = Field(default=1, ge=1)
    label_bbox: VisionBoundingBox
    value_bbox: VisionBoundingBox
    confidence: float = Field(ge=0, le=1)
    source_engine: str
    semantic_type_if_known: VisionSemanticType | None = None
    semantic_mapping_confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: VisionEvidence
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    review_required: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class TypedVisionEntity(BaseModel):
    """A reusable semantic interpretation; extraction remains independently auditable."""

    entity_id: str = Field(default_factory=lambda: f"entity-{uuid4().hex[:12]}")
    raw_value: str
    normalized_value: str
    semantic_type: VisionSemanticType
    confidence: float = Field(ge=0, le=1)
    validator_valid: bool = True
    validation_reasons: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    alternatives: list[dict[str, Any]] = Field(default_factory=list)
    conflict_ids: list[str] = Field(default_factory=list)
    review_required: bool = False


class VerifiedVisionFact(BaseModel):
    fact_id: str = Field(default_factory=lambda: f"fact-{uuid4().hex[:12]}")
    field_id: str
    name: str
    value: str
    normalized_value: str | None = None
    semantic_type: VisionSemanticType | None = None
    evidence_id: str
    reviewer_status: ReviewerStatus
    reviewer: str | None = None
    verified_at: str | None = None


class VisionSchemaProfile(BaseModel):
    mode: str = Field(default="C", pattern="^[ABC]$")
    schema_status: str = Field(default="generic", pattern="^(registered|inferred|generic)$")
    schema_id: str | None = None
    schema_version: str = "v1"
    selection_source: str = "generic"


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
    enhanced_image_path: str | None = None
    grayscale_image_path: str | None = None
    binary_image_path: str | None = None
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
    preprocessing_variant: str = "unknown"
    ocr_pass: str = "primary"


class OCRLineRecord(BaseModel):
    line_id: str
    text: str
    page: int = Field(default=1, ge=1)
    bounding_box: VisionBoundingBox
    mean_confidence: float | None = Field(default=None, ge=0, le=1)
    word_ids: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    reading_order: int = Field(ge=1)
    reconstruction_method: str = "geometry-line-reconstructor-v1"
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    structure_status: str = Field(
        default="structured",
        pattern="^(structured|table_uncertain)$",
    )
    uncertainty_reasons: list[str] = Field(default_factory=list)
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
    schema_field: str | None = None
    display_name: str = ""
    value: str
    raw_value: str | None = None
    normalized_value: str | None = None
    validation_status: str = "not_evaluated"
    validation_reason: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    extraction_confidence: float | None = Field(default=None, ge=0, le=1)
    validation_confidence: float | None = Field(default=None, ge=0, le=1)
    agreement_confidence: float | None = Field(default=None, ge=0, le=1)
    calibrated_confidence: float | None = Field(default=None, ge=0, le=1)
    combined_confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: VisionEvidence
    model_name: str = "local-schema-extraction-adapter"
    model_version: str = "v1"
    target_model: str | None = None
    confidence_components: dict[str, float | None] = Field(default_factory=dict)
    calibrated: bool = False
    extractor: str = "deterministic-fallback"
    verification_status: str = "unverified"
    required: bool = False
    reviewer_status: ReviewerStatus = ReviewerStatus.unverified
    reviewed_value: str | None = None

    @model_validator(mode="after")
    def sync_field_contract(self):
        self.schema_field = self.schema_field or self.name
        self.raw_value = self.raw_value if self.raw_value is not None else self.value
        if not self.display_name:
            self.display_name = self.name.replace("_", " ").title()
        if self.combined_confidence is None:
            self.combined_confidence = self.confidence
        if self.confidence is None:
            self.confidence = self.combined_confidence
        if self.ocr_confidence is None:
            self.ocr_confidence = self.confidence_components.get("ocr_confidence")
        if self.extraction_confidence is None:
            self.extraction_confidence = self.confidence_components.get("extraction_confidence")
        if self.calibrated_confidence is None and self.calibrated:
            self.calibrated_confidence = self.confidence
        self.verification_status = self.reviewer_status.value
        return self


class VisionRiskIndicator(BaseModel):
    risk_id: str = Field(default_factory=lambda: f"vis-risk-{uuid4().hex[:10]}")
    indicator_id: str | None = None
    title: str
    severity: VisionRiskSeverity = VisionRiskSeverity.medium
    category: str
    description: str
    recommended_action: str
    rule_id: str = "local-rule"
    evidence: list[VisionEvidence] = Field(default_factory=list)
    affected_fields: list[str] = Field(default_factory=list)
    page_numbers: list[int] = Field(default_factory=list)
    bounding_boxes: list[VisionBoundingBox] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    review_status: str = "pending"
    reviewer_status: ReviewerStatus = ReviewerStatus.unverified
    model_name: str = "local-risk-rule-engine"
    model_version: str = "v1"
    model_or_rule: str = ""

    @model_validator(mode="after")
    def sync_indicator_contract(self):
        self.indicator_id = self.indicator_id or self.risk_id
        if not self.page_numbers:
            self.page_numbers = sorted({item.page_number for item in self.evidence})
        if not self.bounding_boxes:
            self.bounding_boxes = [
                item.bounding_box for item in self.evidence if item.bounding_box is not None
            ]
        if self.confidence is None and self.evidence:
            values = [
                item.model_confidence for item in self.evidence if item.model_confidence is not None
            ]
            self.confidence = max(values) if values else None
        self.review_status = self.reviewer_status.value
        self.model_or_rule = self.model_or_rule or self.rule_id or self.model_name
        return self


class VisionConfidenceSummary(BaseModel):
    mean_ocr_confidence: float | None = Field(default=None, ge=0, le=1)
    mean_extraction_confidence: float | None = Field(default=None, ge=0, le=1)
    mean_field_confidence: float | None = Field(default=None, ge=0, le=1)
    mean_calibrated_confidence: float | None = Field(default=None, ge=0, le=1)
    approved_field_confidence: float | None = Field(default=None, ge=0, le=1)
    ocr_word_count: int = Field(default=0, ge=0)
    field_count: int = Field(default=0, ge=0)
    approved_field_count: int = Field(default=0, ge=0)
    confidence_available: bool = False
    calculation_version: str = "vision-confidence-v2"


class VisionFieldCompleteness(BaseModel):
    schema_available: bool = False
    document_type: VisionDocumentType = VisionDocumentType.unknown
    required_fields: list[str] = Field(default_factory=list)
    optional_fields: list[str] = Field(default_factory=list)
    extracted_fields: list[str] = Field(default_factory=list)
    missing_required_fields: list[str] = Field(default_factory=list)
    invalid_fields: list[str] = Field(default_factory=list)
    disputed_fields: list[str] = Field(default_factory=list)
    manually_resolved_fields: list[str] = Field(default_factory=list)
    required_matched_fields: list[str] = Field(default_factory=list)
    optional_extracted_fields: list[str] = Field(default_factory=list)
    duplicate_fields: list[str] = Field(default_factory=list)
    required_expected_count: int | None = Field(default=None, ge=0)
    required_matched_count: int | None = Field(default=None, ge=0)
    required_missing_count: int | None = Field(default=None, ge=0)
    optional_extracted_count: int = Field(default=0, ge=0)
    total_extracted_count: int = Field(default=0, ge=0)
    duplicates_rejected_count: int = Field(default=0, ge=0)
    disputed_count: int = Field(default=0, ge=0)
    # Backward-compatible names retained for existing API clients.
    expected_required_count: int | None = Field(default=None, ge=0)
    extracted_count: int = Field(default=0, ge=0)
    unresolved_required_count: int | None = Field(default=None, ge=0)
    exact_schema: str | None = None
    fallback_schema: str | None = None
    generic_schema: str | None = None
    selected_schema: str | None = None
    schema_source: str = "unavailable"
    schema_path: list[str] = Field(default_factory=list)
    fallback_used: bool = False


class VisionLanguageAnalysis(BaseModel):
    detected_languages: list[dict[str, Any]] = Field(default_factory=list)
    valid_character_count: int = Field(default=0, ge=0)
    secondary_language_ratio: float = Field(default=0.0, ge=0, le=1)
    secondary_language_characters: int = Field(default=0, ge=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    sample_text: str = ""
    mixed_language_routing: bool = False
    decision_reason: str = "insufficient_clean_text"
    primary_language: str | None = None
    secondary_languages: list[str] = Field(default_factory=list)
    detector_version: str = "language-router-v3"


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
    model_run_id: str | None = None
    model_id: str | None = None
    stage: str
    model_name: str
    model_version: str
    status: str
    started_at: str
    completed_at: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    checkpoint: str | None = None
    package_version: str | None = None
    device: str = "cpu"
    processing_duration_ms: float | None = Field(default=None, ge=0)
    runtime_ms: float | None = Field(default=None, ge=0)
    source_document: str | None = None
    source_pages: list[int] = Field(default_factory=list)
    routing_reason: str | None = None
    notes: list[str] = Field(default_factory=list)
    input_schema: str | None = None
    output_schema: str | None = None
    document_type: VisionDocumentType | None = None
    selected_schema: str | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    result_status: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def populate_provenance_aliases(self):
        self.model_run_id = self.model_run_id or self.run_id
        version_suffix = f"-{self.model_version}"
        self.model_id = self.model_id or (
            self.model_name
            if self.model_name.lower().endswith(version_suffix.lower())
            else f"{self.model_name}{version_suffix}"
        )
        if self.runtime_ms is None:
            self.runtime_ms = self.processing_duration_ms
        if self.processing_duration_ms is None:
            self.processing_duration_ms = self.runtime_ms
        return self


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
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisionGraphProjection(BaseModel):
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)
    evidence_policy: str = "Edges are deterministic or proposed and require evidence."
    schema_version: str = "documind-evidence-graph-v2"
    preferred_backend: str = "neo4j"
    active_backend: str = "local_evidence_graph"
    legend: dict[str, Any] = Field(default_factory=dict)


class VisionReviewTask(BaseModel):
    review_id: str = Field(default_factory=lambda: f"review-{uuid4().hex[:10]}")
    document_id: str
    status: VisionProcessingStatus = VisionProcessingStatus.review_required
    priority: VisionRiskSeverity = VisionRiskSeverity.medium
    reasons: list[str] = Field(default_factory=list)
    reason_labels: list[str] = Field(default_factory=list)
    assigned_role: str = "REVIEWER"
    assigned_reviewer: str | None = None
    thumbnail_url: str | None = None
    filename: str | None = None
    document_type: VisionDocumentType | None = None
    page_count: int | None = Field(default=None, ge=0)
    confidence_status: str = "not_available"
    confidence_value: float | None = Field(default=None, ge=0, le=1)
    expected_required_fields: int | None = Field(default=None, ge=0)
    unresolved_required_fields: int | None = Field(default=None, ge=0)
    reviewer_comments: str = ""
    created_at: str
    updated_at: str
    history: list[VisionAuditEvent] = Field(default_factory=list)

    @model_validator(mode="after")
    def enrich_queue_contract(self):
        reason_map = {
            "ocr_mean_confidence_below_threshold": "Low OCR confidence",
            "mixed_languages_detected": "Multiple-language uncertainty",
            "reading_order_confidence_below_threshold": "Uncertain reading order",
            "complex_table_or_chart_detected": "Complex table or chart",
            "model_outputs_conflict": "Model output disagreement",
            "manual_advanced_requested": "Advanced parsing requested",
            "field_confidence_requires_review": "Field confidence requires review",
        }
        if not self.reason_labels:
            self.reason_labels = [
                (
                    f"Missing required field: {reason.split(':', 1)[1].replace('_', ' ')}"
                    if reason.startswith("missing_required_field:")
                    else reason_map.get(reason, reason.replace("_", " ").capitalize())
                )
                for reason in self.reasons
            ]
        self.thumbnail_url = self.thumbnail_url or (
            f"/documents/{self.document_id}/pages/1/image?variant=original"
        )
        return self


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
    ocr_lines: list[OCRLineRecord] = Field(default_factory=list)
    layout_regions: list[LayoutRegion] = Field(default_factory=list)
    tables: list[TableResult] = Field(default_factory=list)
    visual_detections: list[VisualDetection] = Field(default_factory=list)
    tamper_analysis: list[TamperAnalysisResult] = Field(default_factory=list)
    generic_fields: list[GenericDiscoveredField] = Field(default_factory=list)
    typed_entities: list[TypedVisionEntity] = Field(default_factory=list)
    extracted_fields: list[ExtractedVisionField] = Field(default_factory=list)
    verified_facts: list[VerifiedVisionFact] = Field(default_factory=list)
    schema_profile: VisionSchemaProfile = Field(default_factory=VisionSchemaProfile)
    confidence_summary: VisionConfidenceSummary = Field(default_factory=VisionConfidenceSummary)
    field_completeness: VisionFieldCompleteness = Field(default_factory=VisionFieldCompleteness)
    language_analysis: VisionLanguageAnalysis = Field(default_factory=VisionLanguageAnalysis)
    processing_warnings: list[VisionRiskIndicator] = Field(default_factory=list)
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
    data_version: int = Field(default=1, ge=1)
    analytics_cache_version: int = Field(default=1, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VisionUploadResponse(BaseModel):
    case_id: str
    job_id: str
    document_ids: list[str]
    status: VisionProcessingStatus
    status_url: str


class VisionFieldReviewRequest(BaseModel):
    reviewed_value: str | None = None
    reviewed_field_name: str | None = Field(
        default=None,
        pattern="^[a-z][a-z0-9_]{0,79}$",
    )
    reviewer_status: ReviewerStatus = ReviewerStatus.edited
    comment: str = ""


class VisionModelRequestedStateRequest(BaseModel):
    requested: bool
    profile: str | None = Field(default=None, pattern="^(?i:lite|standard|full)$")
    notes: str = Field(default="", max_length=1000)


class VisionModelLegacyPatchRequest(BaseModel):
    enabled: bool | None = None
    requested: bool | None = None
    profile: str | None = Field(default=None, pattern="^(?i:lite|standard|full)$")
    notes: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def require_requested_value(self):
        if self.enabled is None and self.requested is None:
            raise ValueError("enabled or requested is required")
        return self


class VisionRuleOverrideRequest(BaseModel):
    enabled: bool | None = None
    threshold: float | None = None
    notes: str = Field(default="", max_length=1000)


class VisionReviewDecisionRequest(BaseModel):
    decision: VisionProcessingStatus
    comment: str = ""
    approval_reason: str = ""
    manual_verification_completed: bool = False
    confirmed_fields: dict[str, str] = Field(default_factory=dict)
    confirmation_acknowledged: bool = False
    evidence_ids: list[str] = Field(default_factory=list)
    decision_rationale: str = ""
    automated_indicators_acknowledged: bool = False
    reauthentication_token: str | None = None
    second_reviewer_email: str | None = None
    second_reviewer_approved: bool = False


class VisionClarificationOption(BaseModel):
    option_id: str
    label: str
    resolved_question: str
    query_type: str


class VisionRetrievalTrace(BaseModel):
    trace_id: str = Field(default_factory=lambda: f"retrieval-{uuid4().hex[:12]}")
    sources_checked: list[str] = Field(default_factory=list)
    candidate_counts: dict[str, int] = Field(default_factory=dict)
    selected_evidence: list[str] = Field(default_factory=list)
    retrieval_duration_ms: float = Field(default=0.0, ge=0)
    retriever_versions: dict[str, str] = Field(default_factory=dict)


class VisionConversationState(BaseModel):
    conversation_id: str
    user_id: str
    organisation_id: str
    selected_scope: dict[str, Any] = Field(default_factory=dict)
    pending_clarification: dict[str, Any] = Field(default_factory=dict)
    last_query_type: str = ""
    last_resolved_question: str = ""
    last_answer: dict[str, Any] = Field(default_factory=dict)
    turns: list[dict[str, Any]] = Field(default_factory=list)
    expires_at: str
    updated_at: str


class VisionAskRequest(BaseModel):
    question: str
    conversation_id: str | None = Field(default=None, max_length=200)
    clarification_option: str | None = Field(default=None, max_length=100)
    document_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    scope: str = Field(
        default="all_accessible",
        pattern="^(current_document|selected_documents|approved_documents|all_accessible)$",
    )
    top_k: int = Field(default=5, ge=1, le=12)

    @model_validator(mode="after")
    def validate_scope_selection(self):
        if self.document_id:
            return self
        if self.scope == "current_document":
            raise ValueError("document_id is required for current_document scope")
        if self.scope == "selected_documents" and not self.document_ids:
            raise ValueError("document_ids are required for selected_documents scope")
        return self


class VisionAskCitation(BaseModel):
    document_id: str
    filename: str | None = None
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
    extraction_status: str = "unverified"
    evidence_type: str = "document_evidence"

    @computed_field
    @property
    def page(self) -> int:
        return self.page_number


class VisionAskCalculationOperation(str, Enum):
    sum = "sum"
    count_verified_invoices = "count_verified_invoices"
    invoice_po_difference = "invoice_po_difference"
    percentage_variance = "percentage_variance"
    date_interval = "date_interval"


class VisionAskCalculationInput(BaseModel):
    document_id: str
    document_type: VisionDocumentType
    role: str
    field_name: str
    raw_value: str
    normalized_value: str
    value_type: str
    currency: str | None = None
    evidence_id: str | None = None
    page_number: int | None = Field(default=None, ge=1)
    reviewer_status: ReviewerStatus


class VisionAskCalculationTrace(BaseModel):
    operation: VisionAskCalculationOperation
    inputs: list[VisionAskCalculationInput] = Field(default_factory=list)
    result: str | None = None
    result_type: str
    currency: str | None = None
    unit: str | None = None
    formula: str
    calculation_version: str = "deterministic-calculation-v1"


class VisionAnswerDegradation(BaseModel):
    """One disclosed reduction in answer quality.

    Fallbacks are allowed; undisclosed fallbacks are not. Every place that
    silently substituted a weaker result now records why it did so.
    """

    component: str
    stage: str
    detail: str
    impact: str = "reduced_quality"


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
    retriever: str = "local-hybrid"
    retrieval_path: str = "not_used"
    scope: dict[str, Any] = Field(default_factory=dict)
    broader_scope_available: bool = False
    conversation_id: str | None = None
    resolved_question: str | None = None
    clarification_options: list[VisionClarificationOption] = Field(default_factory=list)
    retrieval_trace_id: str | None = None
    retrieval_trace: VisionRetrievalTrace | None = None
    reranker: str = "not_used"
    graph_paths: list[dict[str, Any]] = Field(default_factory=list)
    calculation_trace: VisionAskCalculationTrace | None = None
    #: Disclosed degradations that shaped this answer. A fallback is never silent:
    #: if a generator, reranker or vector backend was skipped or rejected, it is
    #: named here and surfaced in the UI. Empty means nothing was degraded.
    degradations: list[VisionAnswerDegradation] = Field(default_factory=list)

    @computed_field
    @property
    def generator(self) -> str:
        return f"{self.generator_name}:{self.generator_version}"

    @computed_field
    @property
    def degraded(self) -> bool:
        return bool(self.degradations)


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
    generate_new_version: bool = False


class VisionFinanceNarrative(BaseModel):
    narrative_id: str = Field(default_factory=lambda: f"narrative-{uuid4().hex[:12]}")
    tenant_id: str
    narrative_type: VisionFinanceNarrativeType
    narrative: str
    calculations: list[VisionCalculationTrace] = Field(default_factory=list)
    citations: list[VisionAskCitation] = Field(default_factory=list)
    # Contributors are the records counted/summed in the narrative. Source IDs
    # additionally include conservatively excluded duplicates so versioning,
    # staleness, and audit remain tied to every reviewed input.
    document_ids: list[str] = Field(default_factory=list)
    source_document_ids: list[str] = Field(default_factory=list)
    excluded_document_ids: list[str] = Field(default_factory=list)
    all_values_verified: bool = True
    answer_status: str = "supported"
    created_at: str
    generated_by: str
    identity_hash: str = ""
    version: int = Field(default=1, ge=1)
    source_data_version: str = ""
    supersedes: str | None = None
    status: str = "active"
    request_scope: dict[str, Any] = Field(default_factory=dict)


class VisionReportRequest(BaseModel):
    report_type: VisionReportType = VisionReportType.document_intelligence
    document_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    vendor_name: str | None = None
    narrative_id: str | None = None
    evaluation: dict[str, Any] | None = None
    generate_new_version: bool = False


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
    identity_hash: str = ""
    version: int = Field(default=1, ge=1)
    source_data_version: str = ""
    supersedes: str | None = None
    status: str = "active"
    template_version: str = "vision-report-v2"
    request_scope: dict[str, Any] = Field(default_factory=dict)
