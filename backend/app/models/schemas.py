from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentType(str, Enum):
    invoice = "invoice"
    contract = "contract"
    purchase_order = "purchase_order"
    bank_change_request = "bank_change_request"
    bank_statement = "bank_statement"
    compliance_policy = "compliance_policy"
    medical_report = "medical_report"
    legal_document = "legal_document"
    resume = "resume"
    email = "email"
    image = "image"
    unknown = "unknown"


class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ComplianceFramework(str, Enum):
    gdpr = "GDPR"
    hipaa = "HIPAA"
    pci = "PCI-DSS"
    soc2 = "SOC2"
    iso27001 = "ISO27001"
    custom = "custom"


class UserRole(str, Enum):
    admin = "ADMIN"
    processor = "PROCESSOR"
    reviewer = "REVIEWER"
    auditor = "AUDITOR"
    viewer = "VIEWER"

    @classmethod
    def _missing_(cls, value):
        legacy_roles = {
            "admin": cls.admin,
            "processor": cls.processor,
            "risk_analyst": cls.processor,
            "reviewer": cls.reviewer,
            "compliance_officer": cls.reviewer,
            "finance_reviewer": cls.reviewer,
            "legal_reviewer": cls.reviewer,
            "auditor": cls.auditor,
            "viewer": cls.viewer,
        }
        return legacy_roles.get(str(value).strip().lower())


class WorkflowAction(str, Enum):
    approve = "approve"
    reject = "reject"
    request_changes = "request_changes"
    escalate_to_legal = "escalate_to_legal"
    escalate_to_finance = "escalate_to_finance"
    escalate_to_compliance = "escalate_to_compliance"
    hold_payment = "hold_payment"


class Evidence(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    evidence_id: str = Field(default_factory=lambda: f"ev-{uuid4().hex[:12]}")
    chunk_id: str | None = None
    document_id: str
    source_document: str
    page_number: int = 1
    text: str = ""
    snippet: str = ""
    extraction_method: str = "local_rule"
    confidence: float = Field(default=0.0, ge=0, le=1)
    reasoning: str = ""
    linked_findings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def sync_text_and_snippet(self):
        if not self.text and self.snippet:
            self.text = self.snippet
        if not self.snippet and self.text:
            self.snippet = self.text[:420]
        return self


class EvidenceChunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: f"chk-{uuid4().hex[:12]}")
    case_id: str
    document_id: str
    source_document: str
    page_number: int = 1
    text: str
    extraction_method: str = "chunking"
    confidence: float = Field(default=0.72, ge=0, le=1)
    linked_findings: list[str] = Field(default_factory=list)
    embedding_model: str = "local-hashing-v1"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedField(BaseModel):
    name: str
    value: str
    normalized_value: str | None = None
    confidence: float = Field(ge=0, le=1)
    extraction_method: str = "regex_rule"
    evidence: Evidence


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    document_type: DocumentType
    checksum: str
    text: str
    pages: int = 1
    stored_path: str | None = None
    tenant_id: str = "tenant-local"
    metadata: dict[str, Any] = Field(default_factory=dict)
    extracted_fields: list[ExtractedField] = Field(default_factory=list)


class ComplianceFinding(BaseModel):
    finding_id: str
    title: str
    severity: RiskLevel
    status: str = "open"
    framework: str = ComplianceFramework.custom.value
    clause_or_policy: str = "custom"
    evidence: list[Evidence]
    recommendation: str
    approval_blocking: bool = False
    score_impact: int = Field(default=0, ge=0, le=100)


class FraudSignal(BaseModel):
    signal_id: str
    title: str
    severity: RiskLevel
    probability: float = Field(ge=0, le=1)
    status: str = "open"
    evidence: list[Evidence]
    recommendation: str
    score_impact: int = Field(default=0, ge=0, le=100)


class CrossDocumentCheck(BaseModel):
    check_id: str
    title: str
    status: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[Evidence]
    left_value: str | None = None
    right_value: str | None = None
    recommendation: str | None = None


class AgentDecision(BaseModel):
    agent_name: str = ""
    agent: str = ""
    decision: str
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    reasoning_summary: str = ""
    reasoning_chain: list[str] = Field(default_factory=list)
    next_action: str = ""
    status: str = "completed"

    @model_validator(mode="after")
    def sync_agent_fields(self):
        if not self.agent_name and self.agent:
            self.agent_name = self.agent
        if not self.agent and self.agent_name:
            self.agent = self.agent_name
        if not self.evidence_ids and self.evidence:
            self.evidence_ids = [item.evidence_id for item in self.evidence]
        if not self.reasoning_summary and self.reasoning_chain:
            self.reasoning_summary = self.reasoning_chain[0]
        return self


class WorkflowState(BaseModel):
    case_id: str
    status: str
    assigned_team: str
    next_action: str
    recommended_action: WorkflowAction = WorkflowAction.request_changes
    allowed_actions: list[WorkflowAction] = Field(default_factory=list)
    approval_required: bool
    approval_blocked: bool = False
    blocked_reasons: list[str] = Field(default_factory=list)


class AuditEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"aud-{uuid4().hex[:12]}")
    case_id: str
    event_type: str
    actor: str = "system"
    message: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class AnalysisResponse(BaseModel):
    case_id: str
    tenant_id: str = "tenant-local"
    industry: str | None = None
    company: str | None = None
    vendor: str | None = None
    documents: list[DocumentRecord]
    entities: list[dict[str, Any]] = Field(default_factory=list)
    evidence_chunks: list[EvidenceChunk] = Field(default_factory=list)
    executive_summary: str
    compliance_score: int = Field(ge=0, le=100)
    risk_score: int = Field(ge=0, le=100)
    fraud_probability: float = Field(ge=0, le=1)
    fraud_confidence: str = "medium"
    compliance_findings: list[ComplianceFinding]
    fraud_signals: list[FraudSignal]
    cross_document_checks: list[CrossDocumentCheck]
    agent_decisions: list[AgentDecision]
    workflow: WorkflowState
    report_url: str
    knowledge_graph: dict[str, Any]
    graph_nodes: list[dict[str, Any]] = Field(default_factory=list)
    graph_edges: list[dict[str, Any]] = Field(default_factory=list)
    reports: dict[str, Any] = Field(default_factory=dict)
    scores: dict[str, Any] = Field(default_factory=dict)
    audit_events: list[AuditEvent] = Field(default_factory=list)


CaseAnalysis = AnalysisResponse


class DemoRunRequest(BaseModel):
    case_id: str | None = None


class ApiEnvelope(BaseModel):
    success: bool
    case_id: str | None = None
    data: Any = None
    errors: list[str] = Field(default_factory=list)


class TextAnalysisRequest(BaseModel):
    filename: str = "sample-document.txt"
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    question: str
    case_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=12)


class QueryCitation(BaseModel):
    document_id: str
    source_document: str
    page_number: int
    chunk_id: str
    evidence_id: str | None = None
    evidence_text: str
    snippet: str
    score: float
    confidence: float = Field(ge=0, le=1)
    retrieval_method: str = "hybrid"
    linked_findings: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    citations: list[QueryCitation]
    confidence: float
    reasoning: list[str]


class ModelRoutingPreviewRequest(BaseModel):
    ocr_mean_confidence: float = Field(default=0.84, ge=0, le=1)
    reading_order_confidence: float = Field(default=0.82, ge=0, le=1)
    has_complex_table: bool = False
    has_chart: bool = False
    mixed_languages: bool = False
    model_outputs_conflict: bool = False
    manual_advanced: bool = False
    model_comparison_mode: bool = False


class WorkflowDecisionRequest(BaseModel):
    case_id: str
    approver: str
    decision: WorkflowAction
    comment: str | None = None


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    role: UserRole = UserRole.viewer
    organisation_name: str = "Local Organisation"


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthUser(BaseModel):
    user_id: str
    name: str
    email: str
    role: UserRole
    tenant_id: str = "tenant-local"
    active: bool = True


class AuthTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_expires_in: int
    user: AuthUser


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class ForgotPasswordRequest(BaseModel):
    email: str


class DashboardMetrics(BaseModel):
    documents_processed: int
    open_cases: int
    average_compliance_score: int
    high_risk_cases: int
    fraud_alerts: int
    sla_breach_probability: float
    evidence_count: int = 0
    open_findings: int = 0
    approval_blocked_cases: int = 0
    average_risk_score: int = 0


class FinanceDocumentType(str, Enum):
    invoice = "invoice"
    purchase_order = "purchase_order"
    contract = "contract"
    bank_change_request = "bank_change_request"
    unknown = "unknown"


class FinanceWorkflowStatus(str, Enum):
    uploaded = "Uploaded"
    extracted = "Extracted"
    narrative_generated = "Narrative Generated"
    pending_review = "Pending Review"
    approved = "Approved"
    needs_correction = "Needs Correction"
    rejected = "Rejected"
    escalated = "Escalated"


class HumanReviewAcceptance(str, Enum):
    accepted_without_changes = "accepted_without_changes"
    accepted_with_minor_changes = "accepted_with_minor_changes"
    accepted_with_major_changes = "accepted_with_major_changes"
    rejected = "rejected"
    escalated = "escalated"


class FinanceNarrativeType(str, Enum):
    executive_summary = "Executive Summary"
    payment_summary = "Payment Summary"
    risk_summary = "Risk Summary"
    compliance_summary = "Compliance Summary"
    exception_summary = "Exception Summary"
    reviewer_notes_draft = "Reviewer Notes Draft"


class SourceReference(BaseModel):
    document_id: str
    source_document: str
    page_number: int = 1
    chunk_id: str
    field_name: str | None = None
    evidence_text: str
    confidence: float = Field(default=0.0, ge=0, le=1)


class RiskFlag(BaseModel):
    flag_id: str = Field(default_factory=lambda: f"risk-{uuid4().hex[:10]}")
    title: str
    severity: RiskLevel = RiskLevel.medium
    category: str = "finance"
    description: str
    recommended_action: str
    source_references: list[SourceReference] = Field(default_factory=list)


class FinanceDocument(BaseModel):
    document_id: str = Field(default_factory=lambda: f"fin-doc-{uuid4().hex[:10]}")
    document_type: FinanceDocumentType = FinanceDocumentType.unknown
    file_name: str
    uploaded_by: str = "analyst"
    upload_timestamp: str
    text: str = ""
    pages: int = 1
    stored_path: str | None = None
    status: FinanceWorkflowStatus = FinanceWorkflowStatus.uploaded
    review_status: FinanceWorkflowStatus = FinanceWorkflowStatus.uploaded
    risk_level: RiskLevel = RiskLevel.low
    checksum: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_ids: list[str] = Field(default_factory=list)


class ExtractedFinanceFields(BaseModel):
    document_id: str
    document_type: FinanceDocumentType
    extraction_version: str = "finance-rules-v1"
    fields: dict[str, Any] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0, le=1)
    source_references: list[SourceReference] = Field(default_factory=list)
    original_extraction: dict[str, Any] = Field(default_factory=dict)
    corrected_extraction: dict[str, Any] | None = None
    created_at: str


class FinanceNarrative(BaseModel):
    narrative_id: str = Field(default_factory=lambda: f"fin-nar-{uuid4().hex[:10]}")
    document_id: str
    narrative_type: FinanceNarrativeType = FinanceNarrativeType.executive_summary
    narrative: str
    confidence_score: float = Field(default=0.0, ge=0, le=1)
    groundedness_score: float = Field(default=0.0, ge=0, le=1)
    unsupported_claims: list[str] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    source_references: list[SourceReference] = Field(default_factory=list)
    reviewer_recommendation: str
    status: FinanceWorkflowStatus = FinanceWorkflowStatus.narrative_generated
    created_at: str


class HumanReview(BaseModel):
    review_id: str = Field(default_factory=lambda: f"fin-rev-{uuid4().hex[:10]}")
    document_id: str
    status: FinanceWorkflowStatus = FinanceWorkflowStatus.pending_review
    review_decision: str = "pending"
    acceptance_status: HumanReviewAcceptance | None = None
    corrected_extraction: dict[str, Any] | None = None
    ai_generated_narrative: str | None = None
    final_approved_narrative: str | None = None
    reviewer_comments: str = ""
    reviewer_name: str = "Finance Reviewer"
    reviewer_id: str | None = None
    review_timestamp: str | None = None
    correction_count: int = 0
    escalation_reason: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    evaluation_id: str = Field(default_factory=lambda: f"fin-eval-{uuid4().hex[:10]}")
    document_id: str | None = None
    case_id: str | None = None
    extraction_accuracy: dict[str, float] = Field(default_factory=dict)
    hallucination: dict[str, float] = Field(default_factory=dict)
    latency: dict[str, float] = Field(default_factory=dict)
    human_review_acceptance: dict[str, float] = Field(default_factory=dict)
    retrieval: dict[str, float] = Field(default_factory=dict)
    safety: dict[str, float] = Field(default_factory=dict)
    cost: dict[str, Any] = Field(default_factory=dict)
    document_results: list[dict[str, Any]] = Field(default_factory=list)
    created_at: str


class VectorRecord(BaseModel):
    record_id: str = Field(default_factory=lambda: f"vec-{uuid4().hex[:12]}")
    document_id: str
    document_type: FinanceDocumentType
    file_name: str
    uploaded_by: str = "analyst"
    upload_timestamp: str
    page_number: int = 1
    chunk_id: str
    extraction_version: str = "finance-rules-v1"
    review_status: FinanceWorkflowStatus = FinanceWorkflowStatus.uploaded
    risk_level: RiskLevel = RiskLevel.low
    source_type: str = "document_chunk"
    created_at: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DemoCase(BaseModel):
    case_id: str
    title: str
    scenario: str
    document_type: FinanceDocumentType
    file_path: str
    ground_truth_path: str | None = None
    expected_risk_level: RiskLevel = RiskLevel.low


class FinanceExtractionRequest(BaseModel):
    document_id: str


class FinanceNarrativeRequest(BaseModel):
    document_id: str
    narrative_type: FinanceNarrativeType = FinanceNarrativeType.executive_summary


class HumanReviewRequest(BaseModel):
    reviewer_name: str = "Finance Reviewer"
    reviewer_id: str | None = None
    decision: str = "request_correction"
    acceptance_status: HumanReviewAcceptance | None = None
    corrected_extraction: dict[str, Any] | None = None
    final_approved_narrative: str | None = None
    reviewer_comments: str = ""
    escalation_reason: str | None = None


class EvaluationRunRequest(BaseModel):
    document_id: str | None = None
    include_demo_ground_truth: bool = True
