BEGIN;

ALTER TABLE extracted_fields
  ADD COLUMN IF NOT EXISTS raw_ocr TEXT,
  ADD COLUMN IF NOT EXISTS corrected_ocr TEXT,
  ADD COLUMN IF NOT EXISTS correction_method TEXT,
  ADD COLUMN IF NOT EXISTS correction_confidence NUMERIC(5,4),
  ADD COLUMN IF NOT EXISTS ocr_candidates JSONB NOT NULL DEFAULT '[]',
  ADD COLUMN IF NOT EXISTS label_bounding_box JSONB,
  ADD COLUMN IF NOT EXISTS value_bounding_box JSONB,
  ADD COLUMN IF NOT EXISTS candidate_status TEXT NOT NULL DEFAULT 'selected',
  ADD COLUMN IF NOT EXISTS rejection_reason TEXT;

CREATE TABLE IF NOT EXISTS vision_field_candidate_resolutions (
  tenant_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  candidate_id TEXT NOT NULL,
  schema_field TEXT NOT NULL,
  evidence_fingerprint TEXT NOT NULL,
  score_components JSONB NOT NULL DEFAULT '{}',
  resolution TEXT NOT NULL
    CHECK (resolution IN ('selected', 'rejected_duplicate', 'rejected_conflict', 'disputed')),
  reason TEXT NOT NULL,
  extractor_version TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, document_id, candidate_id)
);

CREATE TABLE IF NOT EXISTS vision_processing_timings (
  tenant_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  pipeline_version TEXT NOT NULL,
  queued_at TIMESTAMPTZ,
  processing_started_at TIMESTAMPTZ,
  processing_completed_at TIMESTAMPTZ,
  queued_duration_ms BIGINT CHECK (queued_duration_ms >= 0),
  processing_duration_ms BIGINT CHECK (processing_duration_ms >= 0),
  stage_durations_ms JSONB NOT NULL DEFAULT '{}',
  reliable BOOLEAN NOT NULL DEFAULT false,
  exclusion_reason TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, document_id, pipeline_version)
);

CREATE TABLE IF NOT EXISTS vision_conversations (
  tenant_id TEXT NOT NULL,
  user_id TEXT NOT NULL,
  conversation_id TEXT NOT NULL,
  selected_scope JSONB NOT NULL DEFAULT '{}',
  pending_clarification JSONB NOT NULL DEFAULT '{}',
  last_query_type TEXT,
  turns JSONB NOT NULL DEFAULT '[]',
  expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, conversation_id)
);

CREATE TABLE IF NOT EXISTS vision_retrieval_traces (
  tenant_id TEXT NOT NULL,
  trace_id TEXT NOT NULL,
  conversation_id TEXT,
  question_id TEXT,
  sources_checked JSONB NOT NULL DEFAULT '[]',
  candidate_counts JSONB NOT NULL DEFAULT '{}',
  selected_evidence JSONB NOT NULL DEFAULT '[]',
  retriever_versions JSONB NOT NULL DEFAULT '{}',
  retrieval_duration_ms BIGINT NOT NULL CHECK (retrieval_duration_ms >= 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, trace_id)
);

CREATE INDEX IF NOT EXISTS idx_field_candidate_evidence
  ON vision_field_candidate_resolutions(
    tenant_id,
    document_id,
    evidence_fingerprint,
    resolution
  );
CREATE INDEX IF NOT EXISTS idx_processing_timing_pipeline
  ON vision_processing_timings(tenant_id, pipeline_version, reliable);
CREATE INDEX IF NOT EXISTS idx_conversation_expiry
  ON vision_conversations(tenant_id, user_id, expires_at);
CREATE INDEX IF NOT EXISTS idx_retrieval_trace_question
  ON vision_retrieval_traces(tenant_id, question_id, created_at DESC);

COMMIT;
