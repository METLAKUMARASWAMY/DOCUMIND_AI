BEGIN;

ALTER TABLE document_pages
  ALTER COLUMN confidence DROP NOT NULL,
  ALTER COLUMN confidence DROP DEFAULT;

ALTER TABLE evidence
  ALTER COLUMN confidence DROP NOT NULL,
  ALTER COLUMN confidence DROP DEFAULT;

ALTER TABLE extracted_fields
  ALTER COLUMN confidence DROP NOT NULL,
  ALTER COLUMN confidence DROP DEFAULT;

ALTER TABLE extracted_fields
  ADD COLUMN IF NOT EXISTS schema_field TEXT,
  ADD COLUMN IF NOT EXISTS display_name TEXT,
  ADD COLUMN IF NOT EXISTS ocr_confidence NUMERIC(5,4),
  ADD COLUMN IF NOT EXISTS extraction_confidence NUMERIC(5,4),
  ADD COLUMN IF NOT EXISTS validation_confidence NUMERIC(5,4),
  ADD COLUMN IF NOT EXISTS agreement_confidence NUMERIC(5,4),
  ADD COLUMN IF NOT EXISTS calibrated_confidence NUMERIC(5,4),
  ADD COLUMN IF NOT EXISTS combined_confidence NUMERIC(5,4),
  ADD COLUMN IF NOT EXISTS extractor_name TEXT,
  ADD COLUMN IF NOT EXISTS verification_status TEXT NOT NULL DEFAULT 'unverified',
  ADD COLUMN IF NOT EXISTS required BOOLEAN NOT NULL DEFAULT false,
  ADD COLUMN IF NOT EXISTS model_provenance JSONB NOT NULL DEFAULT '{}';

CREATE TABLE IF NOT EXISTS vision_document_completeness (
  tenant_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  document_type TEXT NOT NULL,
  schema_available BOOLEAN NOT NULL DEFAULT false,
  required_fields JSONB NOT NULL DEFAULT '[]',
  optional_fields JSONB NOT NULL DEFAULT '[]',
  missing_required_fields JSONB NOT NULL DEFAULT '[]',
  invalid_fields JSONB NOT NULL DEFAULT '[]',
  disputed_fields JSONB NOT NULL DEFAULT '[]',
  manually_resolved_fields JSONB NOT NULL DEFAULT '[]',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, document_id)
);

CREATE TABLE IF NOT EXISTS vision_processing_warnings (
  tenant_id TEXT NOT NULL,
  warning_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  title TEXT NOT NULL,
  severity TEXT NOT NULL,
  rule_id TEXT NOT NULL,
  evidence JSONB NOT NULL DEFAULT '[]',
  recommended_action TEXT NOT NULL,
  review_status TEXT NOT NULL DEFAULT 'pending',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, warning_id)
);

CREATE TABLE IF NOT EXISTS vision_risk_indicators (
  tenant_id TEXT NOT NULL,
  indicator_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  category TEXT NOT NULL
    CHECK (category IN ('data_quality', 'business_risk', 'visual_risk')),
  title TEXT NOT NULL,
  severity TEXT NOT NULL,
  rule_id TEXT NOT NULL,
  evidence JSONB NOT NULL DEFAULT '[]',
  affected_fields JSONB NOT NULL DEFAULT '[]',
  page_numbers JSONB NOT NULL DEFAULT '[]',
  bounding_boxes JSONB NOT NULL DEFAULT '[]',
  confidence NUMERIC(5,4),
  recommended_action TEXT NOT NULL,
  review_status TEXT NOT NULL DEFAULT 'pending',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, indicator_id)
);

CREATE TABLE IF NOT EXISTS vision_model_runtime_status (
  tenant_id TEXT NOT NULL DEFAULT 'global',
  model_id TEXT NOT NULL,
  requested BOOLEAN NOT NULL DEFAULT false,
  runtime_status TEXT NOT NULL DEFAULT 'disabled'
    CHECK (runtime_status IN ('active', 'unavailable', 'initialising', 'error', 'disabled')),
  health_status TEXT NOT NULL DEFAULT 'unknown'
    CHECK (health_status IN ('healthy', 'unhealthy', 'unknown')),
  last_smoke_test TIMESTAMPTZ,
  last_error TEXT,
  checkpoint TEXT,
  package_version TEXT,
  device TEXT,
  smoke_evidence JSONB NOT NULL DEFAULT '{}',
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, model_id)
);

CREATE TABLE IF NOT EXISTS vision_approval_overrides (
  tenant_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  review_id TEXT NOT NULL,
  reviewer_id TEXT NOT NULL,
  approval_reason TEXT NOT NULL,
  reviewer_comment TEXT NOT NULL,
  manual_verification_completed BOOLEAN NOT NULL,
  confirmed_fields JSONB NOT NULL DEFAULT '{}',
  evidence_ids JSONB NOT NULL DEFAULT '[]',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, review_id, created_at)
);

CREATE TABLE IF NOT EXISTS vision_fraud_confirmations (
  tenant_id TEXT NOT NULL,
  document_id TEXT NOT NULL,
  review_id TEXT NOT NULL,
  reviewer_id TEXT NOT NULL,
  evidence_ids JSONB NOT NULL,
  decision_rationale TEXT NOT NULL,
  reviewer_comment TEXT NOT NULL,
  reauthenticated_at TIMESTAMPTZ NOT NULL,
  automated_indicators_acknowledged BOOLEAN NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, review_id, created_at)
);

ALTER TABLE reports
  ADD COLUMN IF NOT EXISTS identity_hash TEXT,
  ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1,
  ADD COLUMN IF NOT EXISTS source_data_version TEXT,
  ADD COLUMN IF NOT EXISTS supersedes UUID REFERENCES reports(id) ON DELETE SET NULL,
  ADD COLUMN IF NOT EXISTS template_version TEXT NOT NULL DEFAULT 'vision-report-v2';

CREATE TABLE IF NOT EXISTS finance_narratives (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  identity_hash TEXT NOT NULL,
  narrative_type TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  source_data_version TEXT NOT NULL,
  source_snapshot JSONB NOT NULL DEFAULT '{}',
  content JSONB NOT NULL,
  generated_by UUID REFERENCES users(id) ON DELETE SET NULL,
  supersedes UUID REFERENCES finance_narratives(id) ON DELETE SET NULL,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_reports_tenant_identity_version
  ON reports(tenant_id, identity_hash, version)
  WHERE identity_hash IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_narratives_tenant_identity_version
  ON finance_narratives(tenant_id, identity_hash, version);
CREATE INDEX IF NOT EXISTS idx_vision_warning_document
  ON vision_processing_warnings(tenant_id, document_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_vision_risk_document
  ON vision_risk_indicators(tenant_id, document_id, category, severity);
CREATE INDEX IF NOT EXISTS idx_vision_completeness_type
  ON vision_document_completeness(tenant_id, document_type);
CREATE INDEX IF NOT EXISTS idx_vision_model_runtime
  ON vision_model_runtime_status(runtime_status, health_status);

COMMIT;
