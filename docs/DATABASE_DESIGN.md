# Database Design

Phase 3 source of truth: [Phase 3 Database Design](phases/PHASE_03_DATABASE_DESIGN.md)

PostgreSQL stores operational metadata, document lineage, evidence, workflow state, approvals, notifications, audit logs, and report records. Large binary documents stay in object storage, with hashes and URIs stored in PostgreSQL.

## Core Table Groups

- Identity and access: `tenants`, `users`, `roles`, `permissions`, `role_permissions`, `user_roles`.
- Case and document lineage: `cases`, `documents`, `document_versions`, `document_pages`.
- Retrieval and AI artifacts: `document_chunks`, `document_embeddings`, `evidence`, `extracted_fields`, `agent_decisions`.
- Risk, compliance, and fraud: `compliance_findings`, `fraud_findings`, `cross_document_checks`.
- Workflow and operations: `workflow_tasks`, `approvals`, `reports`, `notifications`.
- Governance: `audit_logs`.

## Runnable Schema

The local Docker schema is defined in:

```text
infra/docker/postgres-init.sql
```

It is compatible with the default PostgreSQL Docker image. Production deployments should convert the accepted schema into Alembic migrations and enable tenant row-level security, PITR backups, encrypted storage, and managed secrets.

## Vector Data

The schema stores chunk metadata and embedding metadata in PostgreSQL while supporting multiple vector execution options:

- ChromaDB collections per tenant and environment.
- pgvector in PostgreSQL for deployments that enable the `vector` extension.
- FAISS for isolated offline deployments.

For pgvector, migrate `document_embeddings.embedding_vector` to `vector(n)` and add an HNSW or IVFFlat index after selecting the embedding dimension.

## Retention

- Store original documents according to tenant retention policy.
- Store checksums for duplicate and fraud checks when legally allowed.
- Maintain legal hold flags at case and document levels.
- Retain audit logs according to regulatory and contractual requirements.
- Encrypt sensitive columns and all object storage artifacts in production.
