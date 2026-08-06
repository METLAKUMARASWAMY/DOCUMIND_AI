"""SQLAlchemy-ready model placeholders.

The runnable MVP uses in-process repositories for deterministic local tests. The table
contract is implemented in `infra/docker/postgres-init.sql`; these names mirror that
schema so repository methods can be moved to SQLAlchemy without changing services.
"""

DATABASE_TABLES = [
    "tenants",
    "users",
    "roles",
    "cases",
    "documents",
    "document_versions",
    "document_pages",
    "document_chunks",
    "document_embeddings",
    "evidence",
    "extracted_fields",
    "compliance_findings",
    "fraud_findings",
    "cross_document_checks",
    "agent_decisions",
    "workflow_tasks",
    "approvals",
    "audit_logs",
    "vision_field_candidate_resolutions",
    "vision_processing_timings",
    "vision_conversations",
    "vision_retrieval_traces",
]
