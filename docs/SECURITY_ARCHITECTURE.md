# Security Architecture

## Identity and Access

- OIDC/SAML SSO integration.
- Tenant-scoped RBAC for analyst, approver, legal reviewer, compliance reviewer, auditor, and admin roles.
- Attribute-based access for region, business unit, document type, and legal hold.

## Data Protection

- TLS everywhere.
- AES-256 encryption at rest for PostgreSQL, object storage, vector stores, and graph stores.
- Column-level encryption for PII, account numbers, tax IDs, and health data.
- Secrets managed with cloud KMS, Vault, or Kubernetes secrets backed by external secret operators.

## Auditability

- Immutable audit events for upload, view, query, report generation, approval, export, model decision, and admin actions.
- Store model version, prompt version, retrieval set, evidence IDs, and confidence for every decision.
- Support legal hold and retention policies.

## AI Safety Controls

- Prompt injection filtering for retrieved content.
- Strict citation requirement for answers.
- Human approval for high-risk findings and payment actions.
- PII redaction in logs.
- Model allowlist and tenant-specific data boundaries.

## Threat Model Highlights

- Malicious document content: isolate parsing workers and sanitize text before LLM prompts.
- Data exfiltration through chat: enforce tenant-scoped retrieval filters and citation checks.
- Duplicate or forged documents: checksum, graph linkage, and anomaly detection.
- Insider misuse: RBAC, audit review, export controls, and just-in-time admin access.

