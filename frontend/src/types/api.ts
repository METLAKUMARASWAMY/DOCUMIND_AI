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
  value: string;
  normalized_value?: string | null;
  confidence: number;
  evidence: Evidence;
  model_name: string;
  model_version: string;
  target_model?: string | null;
  confidence_components: Record<string, number>;
  calibrated: boolean;
  reviewer_status: string;
  reviewed_value?: string | null;
}

export interface RiskIndicator {
  risk_id: string;
  title: string;
  severity: "info" | "low" | "medium" | "high" | "critical";
  category: string;
  description: string;
  recommended_action: string;
  evidence: Evidence[];
  reviewer_status: string;
  model_name: string;
  model_version: string;
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
  assigned_role: string;
  reviewer_comments: string;
  created_at: string;
  updated_at: string;
  history: AuditEvent[];
}

export interface GraphProjection {
  nodes: Array<Record<string, unknown> & { id: string }>;
  edges: Array<Record<string, unknown> & { id: string; source: string; target: string; type: string }>;
  evidence_policy: string;
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
  risk_indicators: RiskIndicator[];
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
  answer_status: "supported" | "partially_supported" | "not_found";
  query_type: string;
  citations: Citation[];
  confidence: number;
  reasoning_summary: string;
  requires_review: boolean;
  generator_name: string;
  generator_version: string;
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
}

export interface ApiEnvelope<T> {
  success: boolean;
  case_id?: string | null;
  data: T;
  errors: unknown[];
}
