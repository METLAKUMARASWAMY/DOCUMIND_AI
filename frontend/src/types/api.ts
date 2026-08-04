export type UserRole = "ADMIN" | "PROCESSOR" | "REVIEWER" | "AUDITOR" | "VIEWER";

export interface AuthUser {
  user_id: string;
  name: string;
  email: string;
  role: UserRole;
  tenant_id: string;
  active: boolean;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface Evidence {
  evidence_id: string;
  document_id: string;
  page_number: number;
  source_text: string;
  bounding_box?: BoundingBox | null;
  model_confidence: number;
  model_name: string;
  model_version: string;
  processing_timestamp: string;
  reviewer_status: string;
  metadata?: Record<string, unknown>;
}

export interface OCRWord {
  word_id: string;
  text: string;
  confidence: number;
  polygon: number[][];
  bounding_box: BoundingBox;
  page: number;
  line_id: string;
  block_id: string;
  reading_order: number;
  review_required: boolean;
  model_name: string;
  model_version: string;
}

export interface LayoutRegion {
  region_id: string;
  region_type: string;
  page_number: number;
  bounding_box: BoundingBox;
  text: string;
  confidence: number;
  reading_order: number;
  model_name: string;
  model_version: string;
}

export interface VisualDetection {
  detection_id: string;
  detection_class: string;
  page_number: number;
  bounding_box: BoundingBox;
  confidence: number;
  cropped_evidence_path?: string | null;
  model_name: string;
  model_version: string;
  reviewer_status: string;
  notes: string[];
}

export interface TamperRegion {
  region_id: string;
  manipulation_class: string;
  bounding_box: BoundingBox;
  screening_score: number;
  cropped_evidence_path?: string | null;
}

export interface TamperAnalysis {
  analysis_id: string;
  page_number: number;
  screening_score: number;
  tampering_probability?: number | null;
  calibrated: boolean;
  heatmap_path?: string | null;
  suspicious_regions: TamperRegion[];
  explanation: string;
  model_name: string;
  model_version: string;
  reviewer_status: string;
}

export interface ExtractedField {
  field_id: string;
  name: string;
  schema_field?: string | null;
  display_name?: string | null;
  value: string;
  normalized_value?: string | null;
  confidence: number | null;
  ocr_confidence?: number | null;
  extraction_confidence?: number | null;
  validation_confidence?: number | null;
  agreement_confidence?: number | null;
  calibrated_confidence?: number | null;
  combined_confidence?: number | null;
  evidence: Evidence;
  model_name: string;
  model_version: string;
  target_model?: string | null;
  confidence_components: Record<string, number | null>;
  calibrated: boolean;
  extractor?: string;
  verification_status?: string;
  required?: boolean;
  reviewer_status: string;
  reviewed_value?: string | null;
}

export interface RiskIndicator {
  risk_id: string;
  indicator_id?: string | null;
  title: string;
  severity: "info" | "low" | "medium" | "high" | "critical";
  category: string;
  description: string;
  recommended_action: string;
  rule_id?: string;
  affected_fields?: string[];
  page_numbers?: number[];
  bounding_boxes?: BoundingBox[];
  confidence?: number | null;
  review_status?: string;
  evidence: Evidence[];
  reviewer_status: string;
  model_name: string;
  model_version: string;
  model_or_rule?: string;
}

export interface DuplicateCandidate {
  duplicate_id: string;
  status: string;
  matched_document_id?: string | null;
  matched_filename?: string | null;
  similarity: number;
  evidence: Evidence[];
  reason: string;
}

export interface TableResult {
  table_id: string;
  page_number: number;
  markdown: string;
  html: string;
  rows: string[][];
  bounding_box: BoundingBox;
  confidence: number;
  model_name: string;
  model_version: string;
}

export interface PageArtifact {
  page_number: number;
  original_image_path: string;
  processed_image_path: string;
  enhanced_image_path?: string | null;
  grayscale_image_path?: string | null;
  binary_image_path?: string | null;
  width: number;
  height: number;
  dpi: number;
  source_kind: string;
  native_text_available: boolean;
}

export interface AuditEvent {
  event_id: string;
  document_id: string;
  actor: string;
  event_type: string;
  message: string;
  created_at: string;
  metadata: Record<string, unknown>;
}

export interface ReviewTask {
  review_id: string;
  document_id: string;
  status: string;
  priority: string;
  reasons: string[];
  reason_labels: string[];
  assigned_role: string;
  assigned_reviewer?: string | null;
  thumbnail_url?: string | null;
  filename?: string | null;
  document_type?: string | null;
  page_count?: number | null;
  confidence_status: string;
  confidence_value?: number | null;
  expected_required_fields?: number | null;
  unresolved_required_fields?: number | null;
  reviewer_comments: string;
  created_at: string;
  updated_at: string;
  history: AuditEvent[];
}

export interface ReviewQueueResponse {
  total: number;
  unfiltered_total: number;
  page: number;
  page_size: number;
  total_pages: number;
  reviews: ReviewTask[];
  facets: {
    document_types: string[];
    priorities: string[];
    reasons: Array<{ value: string; label: string }>;
    assignees: string[];
    statuses: string[];
  };
}

export interface ReviewQueueFilters {
  search?: string;
  document_type?: string;
  priority?: string;
  reason?: string;
  assignee?: string;
  status?: string;
  sort_by?: string;
  sort_direction?: string;
  page?: number;
  page_size?: number;
}

export interface ApprovalPolicyBlocker {
  code: string;
  message: string;
  overrideable: boolean;
  affected_fields: string[];
  evidence_ids: string[];
}

export interface ApprovalPolicy {
  allowed: boolean;
  mode: "ordinary" | "manual_override" | "blocked";
  blockers: ApprovalPolicyBlocker[];
  observed_conditions: ApprovalPolicyBlocker[];
  manual_override_available: boolean;
  manual_override_required: boolean;
  required_confirmed_fields: string[];
  required_evidence_ids: string[];
  confidence_threshold: number;
  message: string;
}

export interface ReviewDecisionPayload {
  comment: string;
  approval_reason?: string;
  manual_verification_completed?: boolean;
  confirmed_fields?: Record<string, string>;
  confirmation_acknowledged?: boolean;
  evidence_ids?: string[];
  decision_rationale?: string;
  automated_indicators_acknowledged?: boolean;
  reauthentication_token?: string;
}

export interface GraphProjection {
  nodes: GraphNode[];
  edges: GraphEdge[];
  evidence_policy: string;
  schema_version: string;
  preferred_backend: string;
  active_backend: string;
}

export interface GraphNode extends Record<string, unknown> {
  id: string;
  type: string;
  label: string;
  display_label?: string;
  document_id?: string;
  source_page?: number;
  evidence_id?: string | null;
  bounding_boxes?: BoundingBox[];
  status?: string;
  review_status?: string;
}

export interface GraphEdge extends Record<string, unknown> {
  id: string;
  source: string;
  target: string;
  type: string;
  label?: string;
  document_id?: string;
  source_page?: number;
  evidence_id?: string | null;
  evidence_text?: string;
  bounding_boxes?: BoundingBox[];
  confidence?: number | null;
  status?: "verified" | "proposed";
  verification_eligible?: boolean;
  source_linked?: boolean;
}

export interface VisionDocument {
  document_id: string;
  case_id: string;
  filename: string;
  tenant_id: string;
  status: string;
  document_type: string;
  file_hash: string;
  perceptual_hash?: string | null;
  visual_embedding_model?: string | null;
  original_storage_path?: string | null;
  parsed_text: string;
  page_count: number;
  page_artifacts: PageArtifact[];
  quality: Array<Record<string, unknown>>;
  ocr_words: OCRWord[];
  layout_regions: LayoutRegion[];
  tables: TableResult[];
  visual_detections: VisualDetection[];
  tamper_analysis: TamperAnalysis[];
  extracted_fields: ExtractedField[];
  confidence_summary: {
    mean_ocr_confidence: number | null;
    mean_extraction_confidence: number | null;
    mean_field_confidence: number | null;
    mean_calibrated_confidence: number | null;
    approved_field_confidence: number | null;
    ocr_word_count: number;
    field_count: number;
    approved_field_count: number;
    confidence_available: boolean;
    calculation_version: string;
  };
  field_completeness: {
    schema_available: boolean;
    document_type: string;
    required_fields: string[];
    optional_fields: string[];
    extracted_fields: string[];
    missing_required_fields: string[];
    invalid_fields: string[];
    disputed_fields: string[];
    manually_resolved_fields: string[];
    expected_required_count: number | null;
    extracted_count: number;
    unresolved_required_count: number | null;
  };
  processing_warnings: RiskIndicator[];
  risk_indicators: RiskIndicator[];
  language_analysis: {
    detected_languages: Array<{
      language: string;
      script: string;
      character_count: number;
      ratio: number;
      confidence: number | null;
    }>;
    valid_character_count: number;
    secondary_language_ratio: number;
    secondary_language_characters: number;
    confidence: number | null;
    sample_text: string;
    mixed_language_routing: boolean;
    decision_reason: string;
  };
  duplicate_candidates: DuplicateCandidate[];
  anomaly_analysis: Array<Record<string, unknown>>;
  graph: GraphProjection;
  model_runs: Array<Record<string, unknown>>;
  model_routing: Record<string, unknown>;
  review_task?: ReviewTask | null;
  audit_events: AuditEvent[];
  created_at: string;
  updated_at: string;
  metadata: Record<string, unknown>;
}

export interface Citation {
  document_id: string;
  filename?: string | null;
  page_number: number;
  source_text: string;
  bounding_boxes: BoundingBox[];
  chunk_id?: string | null;
  evidence_id?: string | null;
  retrieval_score?: number | null;
  model_confidence: number;
  reviewer_status: string;
}

export interface AskResponse {
  question_id: string;
  answer: string;
  answer_status: "supported" | "partially_supported" | "not_found" | "clarification_needed";
  query_type: string;
  citations: Citation[];
  confidence: number;
  reasoning_summary: string;
  requires_review: boolean;
  generator_name: string;
  generator_version: string;
  retriever: string;
}

export interface FinanceNarrative {
  narrative_id: string;
  tenant_id: string;
  narrative_type: string;
  narrative: string;
  calculations: Array<{
    trace_id: string;
    label: string;
    value: string;
    currency?: string | null;
    source_document_ids: string[];
    source_evidence_ids: string[];
    date_range: Record<string, string | null>;
    calculation_function: string;
    verification_status: string;
  }>;
  citations: Citation[];
  document_ids: string[];
  all_values_verified: boolean;
  answer_status: string;
  created_at: string;
  generated_by: string;
  identity_hash: string;
  version: number;
  source_data_version: string;
  supersedes?: string | null;
  status: string;
  request_scope: Record<string, unknown>;
}

export interface ReportRecord {
  report_id: string;
  tenant_id: string;
  report_type: string;
  title: string;
  document_ids: string[];
  content: Record<string, unknown>;
  generated_by: string;
  created_at: string;
  object_uri?: string | null;
  identity_hash: string;
  version: number;
  source_data_version: string;
  supersedes?: string | null;
  status: string;
  template_version: string;
  request_scope: Record<string, unknown>;
}

export interface ApiEnvelope<T> {
  success: boolean;
  case_id?: string | null;
  data: T;
  errors: unknown[];
}
