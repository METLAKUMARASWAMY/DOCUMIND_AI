CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS tenants (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name TEXT NOT NULL,
  slug TEXT NOT NULL UNIQUE,
  data_region TEXT NOT NULL DEFAULT 'us',
  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active', 'suspended', 'deleted')),
  retention_days INTEGER NOT NULL DEFAULT 2555
    CHECK (retention_days > 0),
  settings JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS permissions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  code TEXT NOT NULL UNIQUE,
  resource TEXT NOT NULL,
  action TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS roles (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  is_system BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, name)
);

CREATE TABLE IF NOT EXISTS role_permissions (
  role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
  permission_id UUID NOT NULL REFERENCES permissions(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (role_id, permission_id)
);

CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  external_subject TEXT NOT NULL,
  email TEXT NOT NULL,
  display_name TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active', 'invited', 'disabled', 'deleted')),
  last_login_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, external_subject),
  UNIQUE (tenant_id, email)
);

CREATE TABLE IF NOT EXISTS user_roles (
  user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  role_id UUID NOT NULL REFERENCES roles(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (user_id, role_id)
);

CREATE TABLE IF NOT EXISTS cases (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_key TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'open'
    CHECK (status IN ('open', 'in_review', 'approved', 'rejected', 'escalated', 'closed')),
  priority TEXT NOT NULL DEFAULT 'medium'
    CHECK (priority IN ('low', 'medium', 'high', 'critical')),
  owner_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  legal_hold BOOLEAN NOT NULL DEFAULT false,
  compliance_score INTEGER NOT NULL DEFAULT 100
    CHECK (compliance_score BETWEEN 0 AND 100),
  risk_score INTEGER NOT NULL DEFAULT 0
    CHECK (risk_score BETWEEN 0 AND 100),
  fraud_probability NUMERIC(5,4) NOT NULL DEFAULT 0
    CHECK (fraud_probability BETWEEN 0 AND 1),
  sla_due_at TIMESTAMPTZ,
  metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, case_key)
);

CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  document_type TEXT NOT NULL DEFAULT 'unknown'
    CHECK (
      document_type IN (
        'invoice',
        'contract',
        'purchase_order',
        'bank_statement',
        'compliance_policy',
        'policy',
        'medical_report',
        'legal_document',
        'resume',
        'email',
        'audit_report',
        'insurance_claim',
        'vendor_agreement',
        'unknown'
      )
    ),
  title TEXT NOT NULL DEFAULT '',
  current_version_number INTEGER NOT NULL DEFAULT 1
    CHECK (current_version_number > 0),
  status TEXT NOT NULL DEFAULT 'uploaded'
    CHECK (status IN ('uploaded', 'processing', 'ready', 'failed', 'archived')),
  sensitivity_label TEXT NOT NULL DEFAULT 'internal'
    CHECK (sensitivity_label IN ('public', 'internal', 'confidential', 'restricted')),
  created_by UUID REFERENCES users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS document_versions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  version_number INTEGER NOT NULL CHECK (version_number > 0),
  filename TEXT NOT NULL,
  mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
  file_size_bytes BIGINT NOT NULL DEFAULT 0 CHECK (file_size_bytes >= 0),
  checksum_sha256 TEXT NOT NULL,
  object_uri TEXT,
  original_language TEXT NOT NULL DEFAULT 'unknown',
  page_count INTEGER NOT NULL DEFAULT 1 CHECK (page_count >= 0),
  parser_status TEXT NOT NULL DEFAULT 'queued'
    CHECK (parser_status IN ('queued', 'processing', 'completed', 'failed', 'skipped')),
  ocr_status TEXT NOT NULL DEFAULT 'queued'
    CHECK (ocr_status IN ('queued', 'processing', 'completed', 'failed', 'skipped')),
  classification_status TEXT NOT NULL DEFAULT 'queued'
    CHECK (classification_status IN ('queued', 'processing', 'completed', 'failed', 'skipped')),
  extraction_status TEXT NOT NULL DEFAULT 'queued'
    CHECK (extraction_status IN ('queued', 'processing', 'completed', 'failed', 'skipped')),
  embedding_status TEXT NOT NULL DEFAULT 'queued'
    CHECK (embedding_status IN ('queued', 'processing', 'completed', 'failed', 'skipped')),
  metadata JSONB NOT NULL DEFAULT '{}',
  uploaded_by UUID REFERENCES users(id) ON DELETE SET NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (document_id, version_number)
);

CREATE TABLE IF NOT EXISTS document_pages (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  document_version_id UUID NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
  page_number INTEGER NOT NULL CHECK (page_number > 0),
  text TEXT NOT NULL DEFAULT '',
  layout_json JSONB NOT NULL DEFAULT '{}',
  width_points NUMERIC(10,2),
  height_points NUMERIC(10,2),
  confidence NUMERIC(5,4) NOT NULL DEFAULT 0
    CHECK (confidence BETWEEN 0 AND 1),
  bounding_regions JSONB NOT NULL DEFAULT '[]',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (document_version_id, page_number)
);

CREATE TABLE IF NOT EXISTS document_chunks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  document_version_id UUID NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
  page_start INTEGER NOT NULL CHECK (page_start > 0),
  page_end INTEGER NOT NULL CHECK (page_end >= page_start),
  chunk_text TEXT NOT NULL,
  token_count INTEGER NOT NULL DEFAULT 0 CHECK (token_count >= 0),
  checksum_sha256 TEXT NOT NULL,
  sensitivity_label TEXT NOT NULL DEFAULT 'internal'
    CHECK (sensitivity_label IN ('public', 'internal', 'confidential', 'restricted')),
  metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (document_version_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS document_embeddings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  chunk_id UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
  provider TEXT NOT NULL DEFAULT 'local',
  model_name TEXT NOT NULL,
  embedding_dimension INTEGER NOT NULL CHECK (embedding_dimension > 0),
  embedding_vector DOUBLE PRECISION[],
  vector_store_provider TEXT NOT NULL DEFAULT 'postgres'
    CHECK (vector_store_provider IN ('postgres', 'pgvector', 'chroma', 'faiss')),
  vector_collection TEXT,
  vector_id TEXT,
  model_metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (chunk_id, provider, model_name)
);

CREATE TABLE IF NOT EXISTS evidence (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
  document_version_id UUID REFERENCES document_versions(id) ON DELETE SET NULL,
  chunk_id UUID REFERENCES document_chunks(id) ON DELETE SET NULL,
  source_document TEXT NOT NULL,
  page_number INTEGER NOT NULL DEFAULT 1 CHECK (page_number > 0),
  highlighted_text TEXT NOT NULL,
  bounding_boxes JSONB NOT NULL DEFAULT '[]',
  confidence NUMERIC(5,4) NOT NULL DEFAULT 0
    CHECK (confidence BETWEEN 0 AND 1),
  extraction_method TEXT NOT NULL DEFAULT 'unknown',
  model_version TEXT,
  reasoning TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS extracted_fields (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
  document_version_id UUID REFERENCES document_versions(id) ON DELETE SET NULL,
  evidence_id UUID REFERENCES evidence(id) ON DELETE SET NULL,
  field_name TEXT NOT NULL,
  value TEXT NOT NULL,
  normalized_value TEXT,
  value_type TEXT NOT NULL DEFAULT 'text'
    CHECK (value_type IN ('text', 'number', 'money', 'date', 'entity', 'boolean', 'json')),
  confidence NUMERIC(5,4) NOT NULL DEFAULT 0
    CHECK (confidence BETWEEN 0 AND 1),
  extractor_name TEXT NOT NULL DEFAULT 'local',
  model_version TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS compliance_findings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  rule_code TEXT NOT NULL,
  title TEXT NOT NULL,
  framework TEXT NOT NULL DEFAULT 'internal'
    CHECK (framework IN ('internal', 'GDPR', 'HIPAA', 'PCI-DSS', 'SOC2', 'ISO27001')),
  control_id TEXT,
  severity TEXT NOT NULL
    CHECK (severity IN ('low', 'medium', 'high', 'critical')),
  status TEXT NOT NULL DEFAULT 'open'
    CHECK (status IN ('open', 'in_review', 'accepted', 'remediated', 'false_positive')),
  score_impact INTEGER NOT NULL DEFAULT 0
    CHECK (score_impact BETWEEN 0 AND 100),
  recommendation TEXT NOT NULL,
  evidence_snapshot JSONB NOT NULL DEFAULT '[]',
  rule_pack_version TEXT NOT NULL DEFAULT 'local-dev',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS fraud_findings (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  signal_type TEXT NOT NULL
    CHECK (
      signal_type IN (
        'duplicate_invoice',
        'amount_mismatch',
        'vendor_anomaly',
        'payment_anomaly',
        'fake_document',
        'suspicious_transaction',
        'other'
      )
    ),
  title TEXT NOT NULL,
  severity TEXT NOT NULL
    CHECK (severity IN ('low', 'medium', 'high', 'critical')),
  status TEXT NOT NULL DEFAULT 'open'
    CHECK (status IN ('open', 'in_review', 'accepted', 'remediated', 'false_positive')),
  probability NUMERIC(5,4) NOT NULL DEFAULT 0
    CHECK (probability BETWEEN 0 AND 1),
  amount_at_risk NUMERIC(18,2),
  currency CHAR(3),
  recommendation TEXT NOT NULL,
  evidence_snapshot JSONB NOT NULL DEFAULT '[]',
  model_version TEXT NOT NULL DEFAULT 'local-dev',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cross_document_checks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  check_type TEXT NOT NULL
    CHECK (
      check_type IN (
        'invoice_po_amount',
        'contract_invoice_terms',
        'vendor_bank_match',
        'policy_compliance',
        'other'
      )
    ),
  title TEXT NOT NULL,
  status TEXT NOT NULL
    CHECK (status IN ('matched', 'mismatch', 'missing_document', 'needs_review', 'not_applicable')),
  confidence NUMERIC(5,4) NOT NULL DEFAULT 0
    CHECK (confidence BETWEEN 0 AND 1),
  compared_entities JSONB NOT NULL DEFAULT '[]',
  evidence_snapshot JSONB NOT NULL DEFAULT '[]',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agent_decisions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  agent_name TEXT NOT NULL,
  decision TEXT NOT NULL,
  confidence NUMERIC(5,4) NOT NULL DEFAULT 0
    CHECK (confidence BETWEEN 0 AND 1),
  evidence_snapshot JSONB NOT NULL DEFAULT '[]',
  reasoning_chain JSONB NOT NULL DEFAULT '[]',
  model_version TEXT NOT NULL DEFAULT 'local-dev',
  prompt_version TEXT NOT NULL DEFAULT 'local-dev',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS workflow_tasks (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'assigned', 'in_progress', 'blocked', 'completed', 'cancelled')),
  assigned_team TEXT NOT NULL,
  assignee_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  next_action TEXT NOT NULL,
  approval_required BOOLEAN NOT NULL DEFAULT true,
  due_at TIMESTAMPTZ,
  metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS approvals (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  workflow_task_id UUID REFERENCES workflow_tasks(id) ON DELETE SET NULL,
  approval_type TEXT NOT NULL
    CHECK (approval_type IN ('finance', 'legal', 'compliance', 'executive', 'audit', 'risk')),
  status TEXT NOT NULL DEFAULT 'requested'
    CHECK (status IN ('requested', 'approved', 'rejected', 'changes_requested', 'escalated', 'cancelled')),
  requested_by UUID REFERENCES users(id) ON DELETE SET NULL,
  approver_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  decision TEXT,
  comment TEXT,
  due_at TIMESTAMPTZ,
  decided_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS reports (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  report_type TEXT NOT NULL
    CHECK (report_type IN ('executive', 'compliance', 'fraud', 'risk', 'audit', 'case')),
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued', 'processing', 'completed', 'failed')),
  object_uri TEXT,
  content_hash TEXT,
  generated_by UUID REFERENCES users(id) ON DELETE SET NULL,
  evidence_snapshot JSONB NOT NULL DEFAULT '[]',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS notifications (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  recipient_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  case_id UUID REFERENCES cases(id) ON DELETE CASCADE,
  notification_type TEXT NOT NULL
    CHECK (
      notification_type IN (
        'assignment',
        'escalation',
        'sla_risk',
        'approval',
        'report',
        'system'
      )
    ),
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  channel TEXT NOT NULL DEFAULT 'in_app'
    CHECK (channel IN ('in_app', 'email', 'webhook')),
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued', 'sent', 'delivered', 'read', 'failed')),
  delivered_at TIMESTAMPTZ,
  read_at TIMESTAMPTZ,
  metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS audit_logs (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID REFERENCES tenants(id) ON DELETE SET NULL,
  actor_user_id UUID REFERENCES users(id) ON DELETE SET NULL,
  action TEXT NOT NULL,
  resource_type TEXT NOT NULL,
  resource_id TEXT NOT NULL,
  outcome TEXT NOT NULL DEFAULT 'success'
    CHECK (outcome IN ('success', 'failure', 'denied')),
  ip_address INET,
  user_agent TEXT,
  correlation_id TEXT,
  metadata JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_roles_tenant ON roles(tenant_id);
CREATE INDEX IF NOT EXISTS idx_role_permissions_permission ON role_permissions(permission_id);
CREATE INDEX IF NOT EXISTS idx_user_roles_role ON user_roles(role_id);
CREATE INDEX IF NOT EXISTS idx_users_tenant_status ON users(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_users_tenant_email_lower ON users(tenant_id, lower(email));

CREATE INDEX IF NOT EXISTS idx_cases_tenant_status ON cases(tenant_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cases_tenant_priority ON cases(tenant_id, priority, risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_cases_owner ON cases(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_cases_metadata_gin ON cases USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_documents_tenant_case ON documents(tenant_id, case_id);
CREATE INDEX IF NOT EXISTS idx_documents_case ON documents(case_id);
CREATE INDEX IF NOT EXISTS idx_documents_type_status ON documents(tenant_id, document_type, status);
CREATE INDEX IF NOT EXISTS idx_document_versions_document ON document_versions(document_id, version_number DESC);
CREATE INDEX IF NOT EXISTS idx_document_versions_checksum ON document_versions(tenant_id, checksum_sha256);
CREATE INDEX IF NOT EXISTS idx_document_versions_metadata_gin ON document_versions USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_document_versions_ocr_queue
  ON document_versions(tenant_id, ocr_status, created_at)
  WHERE ocr_status IN ('queued', 'processing', 'failed');
CREATE INDEX IF NOT EXISTS idx_document_versions_extraction_queue
  ON document_versions(tenant_id, extraction_status, created_at)
  WHERE extraction_status IN ('queued', 'processing', 'failed');
CREATE INDEX IF NOT EXISTS idx_document_versions_embedding_queue
  ON document_versions(tenant_id, embedding_status, created_at)
  WHERE embedding_status IN ('queued', 'processing', 'failed');

CREATE INDEX IF NOT EXISTS idx_document_pages_version ON document_pages(document_version_id, page_number);
CREATE INDEX IF NOT EXISTS idx_document_pages_layout_gin ON document_pages USING GIN (layout_json);

CREATE INDEX IF NOT EXISTS idx_document_chunks_case ON document_chunks(tenant_id, case_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_version ON document_chunks(document_version_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata_gin ON document_chunks USING GIN (metadata);
CREATE INDEX IF NOT EXISTS idx_document_chunks_text_fts
  ON document_chunks USING GIN (to_tsvector('english', chunk_text));
CREATE INDEX IF NOT EXISTS idx_document_embeddings_chunk ON document_embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_document_embeddings_vector_ref
  ON document_embeddings(vector_store_provider, vector_collection, vector_id);

CREATE INDEX IF NOT EXISTS idx_evidence_case ON evidence(tenant_id, case_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_evidence_document_page ON evidence(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_evidence_chunk ON evidence(chunk_id);
CREATE INDEX IF NOT EXISTS idx_evidence_bounding_boxes_gin ON evidence USING GIN (bounding_boxes);
CREATE INDEX IF NOT EXISTS idx_extracted_fields_case_name ON extracted_fields(tenant_id, case_id, field_name);
CREATE INDEX IF NOT EXISTS idx_extracted_fields_document ON extracted_fields(document_id);

CREATE INDEX IF NOT EXISTS idx_compliance_findings_case ON compliance_findings(tenant_id, case_id);
CREATE INDEX IF NOT EXISTS idx_compliance_findings_status ON compliance_findings(tenant_id, status, severity);
CREATE INDEX IF NOT EXISTS idx_compliance_findings_framework ON compliance_findings(tenant_id, framework, control_id);
CREATE INDEX IF NOT EXISTS idx_compliance_findings_evidence_gin
  ON compliance_findings USING GIN (evidence_snapshot);

CREATE INDEX IF NOT EXISTS idx_fraud_findings_case ON fraud_findings(tenant_id, case_id);
CREATE INDEX IF NOT EXISTS idx_fraud_findings_probability ON fraud_findings(tenant_id, probability DESC);
CREATE INDEX IF NOT EXISTS idx_fraud_findings_signal ON fraud_findings(tenant_id, signal_type, status);
CREATE INDEX IF NOT EXISTS idx_fraud_findings_evidence_gin ON fraud_findings USING GIN (evidence_snapshot);

CREATE INDEX IF NOT EXISTS idx_cross_document_checks_case ON cross_document_checks(tenant_id, case_id);
CREATE INDEX IF NOT EXISTS idx_cross_document_checks_type_status
  ON cross_document_checks(tenant_id, check_type, status);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_case ON agent_decisions(tenant_id, case_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_decisions_agent ON agent_decisions(tenant_id, agent_name);

CREATE INDEX IF NOT EXISTS idx_workflow_tasks_case ON workflow_tasks(tenant_id, case_id);
CREATE INDEX IF NOT EXISTS idx_workflow_tasks_assignee ON workflow_tasks(assignee_user_id, status);
CREATE INDEX IF NOT EXISTS idx_workflow_tasks_due ON workflow_tasks(tenant_id, due_at)
  WHERE status NOT IN ('completed', 'cancelled');
CREATE INDEX IF NOT EXISTS idx_approvals_case ON approvals(tenant_id, case_id);
CREATE INDEX IF NOT EXISTS idx_approvals_approver ON approvals(approver_user_id, status);
CREATE INDEX IF NOT EXISTS idx_approvals_due ON approvals(tenant_id, due_at)
  WHERE status IN ('requested', 'escalated');

CREATE INDEX IF NOT EXISTS idx_reports_case ON reports(tenant_id, case_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(tenant_id, status, created_at);
CREATE INDEX IF NOT EXISTS idx_notifications_recipient ON notifications(recipient_user_id, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_notifications_case ON notifications(tenant_id, case_id);
CREATE INDEX IF NOT EXISTS idx_notifications_metadata_gin ON notifications USING GIN (metadata);

CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_created ON audit_logs(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_actor_created ON audit_logs(actor_user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_correlation ON audit_logs(correlation_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_metadata_gin ON audit_logs USING GIN (metadata);

CREATE TABLE IF NOT EXISTS graph_nodes (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  graph_node_key TEXT NOT NULL,
  label TEXT NOT NULL,
  properties JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, case_id, graph_node_key)
);

CREATE TABLE IF NOT EXISTS graph_edges (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  case_id UUID NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
  source_node_key TEXT NOT NULL,
  target_node_key TEXT NOT NULL,
  relationship_type TEXT NOT NULL,
  properties JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_graph_nodes_case_label ON graph_nodes(tenant_id, case_id, label);
CREATE INDEX IF NOT EXISTS idx_graph_edges_case_type ON graph_edges(tenant_id, case_id, relationship_type);
CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(tenant_id, case_id, source_node_key);
CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(tenant_id, case_id, target_node_key);
