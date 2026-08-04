# Phases 4-29 - Continuous Completion Package

## Status

Phases 4 through 29 are completed in continuous mode. This package is the phase-by-phase handoff for the remaining DocuMind AI 2.0 scope after Phase 3 Database Design. It maps each requested phase to the repository artifacts, implementation boundaries, security posture, test strategy, and production deployment path.

## Phase 4 - Knowledge Graph Design

- Objectives: Define the Neo4j model for companies, employees, vendors, contracts, invoices, purchase orders, cases, documents, findings, policies, approvals, bank accounts, and transactions.
- Functional Requirements: Support `OWNS`, `SIGNED`, `PAID`, `APPROVED`, `REFERENCES`, `ASSOCIATED_WITH`, `MATCHES`, and `CONFLICTS_WITH` relationships with case-level subgraph retrieval.
- Non-Functional Requirements: Idempotent graph projection, tenant isolation, deterministic entity keys, traversal performance indexes, and replay from PostgreSQL source records.
- Research Contribution: Enables relationship discovery, conflict detection, vendor risk discovery, graph-enhanced fraud signals, and multi-document dependency analysis.
- System Design: PostgreSQL remains source of truth; the knowledge graph service projects entities and relationships into Neo4j for traversal and reasoning.
- Folder Structure: `docs/KNOWLEDGE_GRAPH_DESIGN.md`, `backend/app/services/knowledge_graph.py`, `infra/k8s/neo4j.yaml`, and `docker-compose.yml`.
- Database Changes: No new PostgreSQL tables beyond Phase 3; graph constraints and merge keys are defined in the graph design doc.
- API Design: Local graph APIs are implemented at `GET /api/v1/graph/cases/{case_id}`, `GET /api/v1/graph/entities/{entity_id}`, `GET /api/v1/graph/risk-paths`, and `POST /api/v1/graph/project/{case_id}`.
- Backend Implementation: `KnowledgeGraphEngine` emits tenant, case, document, entity, finding, evidence, match, conflict, and Neo4j projection metadata from analyzed documents.
- Frontend Implementation: The Graph tab renders entity nodes and relationships from the analysis payload.
- AI Pipeline: Extraction outputs become graph nodes; verification outputs become `MATCHES` or `CONFLICTS_WITH` edges.
- Security Considerations: Tenant-scoped graph labels/properties, query authorization, restricted graph export, and audit logs for graph rebuilds.
- Testing Strategy: Unit-test graph projection, deterministic merge keys, conflict edges, and high-risk vendor traversals.
- Deployment Strategy: Local Neo4j in Docker Compose; production Neo4j StatefulSet with secret-backed auth and persistent storage.
- Production-Ready Code: Graph design doc expanded with constraints, projection Cypher, example queries, implemented local graph APIs, and Neo4j-ready contracts.

## Phase 5 - Backend Foundation

- Objectives: Establish FastAPI as the versioned API surface for document intelligence, retrieval, workflows, reporting, and enterprise integrations.
- Functional Requirements: Provide authentication boundaries, authorization design, document upload, case analysis, search, reporting, workflow decisions, health checks, and Swagger docs.
- Non-Functional Requirements: Modular service layer, typed Pydantic contracts, deterministic local behavior, async-ready processing boundary, CORS configuration, and clear upgrade path to persistent stores.
- Research Contribution: Exposes every AI stage as inspectable service output for experiments and audit review.
- System Design: FastAPI routes call orchestration services under `backend/app/services`, with adapters for AI, graph, workflow, and reporting.
- Folder Structure: `backend/app/main.py`, `backend/app/api/routes.py`, `backend/app/models/schemas.py`, `backend/app/core`, and `backend/app/services`.
- Database Changes: Phase 5 consumes Phase 3 schema later through SQLAlchemy repositories; current local implementation stays in-memory for deterministic tests.
- API Design: default Vision APIs use `/api/v1/documents`, `/reviews`, `/ask`, `/graph`, `/analytics`, `/reports`, and `/admin`; optional legacy research-case APIs use `/api/v1/demo-legacy` when `ENABLE_DEMO_API=true`.
- Backend Implementation: Implemented service boundaries for ingestion, OCR, classification, extraction, retrieval, RAG, agents, compliance, fraud, cross-document checks, graph, workflow, reports, XAI, and prediction.
- Frontend Implementation: React consumes API methods from `frontend/src/api/client.ts`.
- AI Pipeline: `DocumentIntelligencePipeline` orchestrates upload to report generation.
- Security Considerations: PII redaction helper, document hashing, CORS, future JWT/RBAC boundaries, and audit-log-ready workflow events.
- Testing Strategy: Backend regression tests cover pipeline behavior and cross-document verification.
- Deployment Strategy: Backend Dockerfile and Kubernetes deployment expose the API on port 8000.
- Production-Ready Code: FastAPI scaffold is runnable, documented, typed, testable, and versioned under `/api/v1`.

## Phase 6 - Frontend Foundation

- Objectives: Build an enterprise analyst console for dashboarding, document review, search, risk, compliance, fraud, evidence, graph, workflow, audit, admin, reports, and agent review.
- Functional Requirements: Provide upload, demo analysis, cited search, compliance review, fraud review, graph view, approval workflow, evidence explorer, report access, and agent center.
- Non-Functional Requirements: Responsive layout, accessible buttons/forms, stable card/table dimensions, readable dense UI, and graceful backend-offline demo mode.
- Research Contribution: Gives reviewers a human-in-the-loop surface for evaluating evidence quality, agent decisions, and AI usefulness.
- System Design: React/Vite frontend calls the FastAPI backend and falls back to seeded demo data.
- Folder Structure: `frontend/src/App.tsx`, `frontend/src/router.tsx`, `frontend/src/api/client.ts`, `frontend/src/pages/*`, `frontend/src/styles.css`, Vite configs, and Dockerfile.
- Database Changes: None in frontend phase.
- API Design: Frontend uses `analyzeText`, `uploadDocuments`, `queryDocuments`, `fetchMetrics`, and `recordDecision`.
- Backend Implementation: Existing APIs provide the data contracts consumed by the console.
- Frontend Implementation: Added first-class tabs for all requested enterprise modules plus Evidence Explorer and Agent Center.
- AI Pipeline: Displays pipeline outputs from classification, extraction, RAG, compliance, fraud, verification, graph, workflow, and agents.
- Security Considerations: UI separates audit/admin/report surfaces and avoids exposing raw secrets or provider settings.
- Testing Strategy: `npm run lint` and `npm run build` pass; future Playwright and component tests should cover all tabs.
- Deployment Strategy: Frontend Dockerfile builds static assets and serves them through Nginx with API proxying.
- Production-Ready Code: Analyst console builds successfully and surfaces every required page class.

## Phase 7 - OCR And Layout Intelligence

- Objectives: Extract text, tables, signatures, stamps, checkboxes, forms, and page-level layout evidence.
- Functional Requirements: Support local adapter output and production adapters for PaddleOCR, EasyOCR, LayoutLMv3, and DocFormer.
- Non-Functional Requirements: Preserve page numbers, confidence, layout JSON, bounding regions, retryability, and fallback parsing.
- Research Contribution: Enables layout-aware extraction and evidence highlighting for explainable document AI.
- System Design: OCR/layout service sits after ingestion and before classification, extraction, and chunking.
- Folder Structure: `backend/app/services/ocr.py`, `backend/app/services/ingestion.py`, and Phase 3 `document_pages`.
- Database Changes: Store OCR text and layout in `document_pages`.
- API Design: OCR results are embedded in document metadata in the analysis response.
- Backend Implementation: `OCRLayoutService` detects text blocks, tables, signatures, stamps, and forms from structural cues.
- Frontend Implementation: Document inventory and Evidence Explorer show document/page-backed outputs.
- AI Pipeline: OCR/layout output feeds classification, extraction, retrieval chunks, evidence, and XAI.
- Security Considerations: Parser isolation, malware scanning in production, PII-safe logs, and confidence thresholds.
- Testing Strategy: Unit tests for scanned text fallback, empty documents, table cues, and low-confidence OCR paths.
- Deployment Strategy: Use CPU local adapter by default; production can add GPU OCR workers and model images.
- Production-Ready Code: Adapter boundary is implemented and documented for enterprise OCR providers.

## Phase 8 - Document Classification

- Objectives: Classify invoices, contracts, resumes, policies, medical reports, bank statements, purchase orders, legal documents, emails, and unknown documents.
- Functional Requirements: Return document type and confidence, with future human correction and model version tracking.
- Non-Functional Requirements: Deterministic local classifier, model-adapter boundary, tenant-configurable models, and explainable classification cues.
- Research Contribution: Supports layout-aware document taxonomy evaluation across enterprise document bundles.
- System Design: Classification runs after ingestion/OCR and before extraction, compliance, fraud, and graph projection.
- Folder Structure: `backend/app/services/classification.py` and `backend/app/models/schemas.py`.
- Database Changes: Store logical type on `documents` and model metadata on `document_versions`.
- API Design: Classification appears in each `DocumentRecord`.
- Backend Implementation: Keyword classifier provides reliable local behavior and upgrade path to LayoutLMv3/transformers.
- Frontend Implementation: Documents tab shows document type and key extractions.
- AI Pipeline: Document type controls required clauses, fraud checks, extraction fields, and graph labels.
- Security Considerations: Type corrections must be audited because they affect downstream routing and risk scoring.
- Testing Strategy: Add fixtures by document type and confusion-matrix evaluation for production models.
- Deployment Strategy: Local classifier in API; production transformer classifier can run as worker or model endpoint.
- Production-Ready Code: Classification service is isolated and swappable.

## Phase 9 - Knowledge Extraction

- Objectives: Extract names, dates, amounts, organizations, vendors, clauses, risks, invoice numbers, purchase orders, and bank indicators.
- Functional Requirements: Return fields with confidence, evidence, normalized values, and extraction method.
- Non-Functional Requirements: Page-aware evidence, deterministic regex fallback, transformer/NER upgrade path, and human-correction readiness.
- Research Contribution: Enables cross-document verification and graph construction from normalized enterprise entities.
- System Design: Extraction consumes document type and OCR text, then emits fields and evidence.
- Folder Structure: `backend/app/services/extraction.py`, `backend/app/models/schemas.py`, and Phase 3 `extracted_fields`.
- Database Changes: Store normalized extractions in `extracted_fields` and evidence links in `evidence`.
- API Design: Extracted fields are returned under each document.
- Backend Implementation: Regex and clause keyword extractors cover core enterprise fields.
- Frontend Implementation: Documents tab lists key extractions; Evidence Explorer exposes supporting snippets.
- AI Pipeline: Extracted fields feed compliance, fraud, verification, graph, RAG metadata, and reports.
- Security Considerations: Sensitive fields need masking, encryption, access checks, and correction audit trails.
- Testing Strategy: Exact-match and normalized-match tests for money, dates, vendors, invoice IDs, PO IDs, and clauses.
- Deployment Strategy: Local extraction in API; production can move transformer NER into batch workers.
- Production-Ready Code: Extraction service is implemented behind a clear upgrade boundary.

## Phase 10 - Vector Intelligence Layer

- Objectives: Provide semantic search, hybrid search, and enterprise retrieval across case documents.
- Functional Requirements: Chunk documents, index case-scoped evidence, support top-k search, metadata filters, ChromaDB, pgvector, and FAISS deployment modes.
- Non-Functional Requirements: Tenant filtering, citation preservation, deterministic local retrieval, scalable ANN backend, and retriever versioning.
- Research Contribution: Supports citation precision, recall@k, nDCG, and retrieval ablation studies.
- System Design: Retrieval layer indexes chunks after extraction and serves RAG and agents.
- Folder Structure: `backend/app/services/retrieval.py`, `backend/app/services/rag.py`, `docker-compose.yml`, and Phase 3 `document_chunks`/`document_embeddings`.
- Database Changes: Store chunk metadata and vector references in PostgreSQL; external vector stores hold ANN indexes.
- API Design: `/api/v1/ask` uses Vision retrieval through grounded RAG; optional legacy RAG remains at `/api/v1/demo-legacy/query`.
- Backend Implementation: Local hybrid lexical/vector-like retriever with Chroma/pgvector-ready service boundary.
- Frontend Implementation: Search tab sends questions and renders citations.
- AI Pipeline: Retrieval feeds RAG answers, agent deliberation, evidence explorer, and report generation.
- Security Considerations: Tenant and sensitivity filters must be applied before retrieval.
- Testing Strategy: Retrieval fixtures, recall@k, citation precision, metadata filtering, and no-cross-tenant leakage tests.
- Deployment Strategy: Chroma container locally; pgvector/FAISS supported by schema and deployment notes.
- Production-Ready Code: Retrieval boundary and chunk metadata model are implemented.

## Phase 11 - Advanced RAG Engine

- Objectives: Generate citation-grounded answers with chunking, metadata filtering, source references, multi-document retrieval, and context compression.
- Functional Requirements: Every answer must include citations, page numbers, snippets, confidence, and reasoning.
- Non-Functional Requirements: Abstain when evidence is missing, avoid unsupported claims, support external LLM adapters, and trace retrieval sets.
- Research Contribution: Evaluates evidence-constrained RAG against single-document PDF chat baselines.
- System Design: RAG wraps retrieval and composes answers only from returned citations.
- Folder Structure: `backend/app/services/rag.py`, `backend/app/services/retrieval.py`, and frontend Search view.
- Database Changes: Use `document_chunks`, `document_embeddings`, `evidence`, and model metadata.
- API Design: `POST /api/v1/ask`; optional legacy query endpoint is `POST /api/v1/demo-legacy/query`.
- Backend Implementation: `RAGEngine` returns answer, citations, confidence, and reasoning.
- Frontend Implementation: Search tab renders grounded answer, citation chips, and reasoning list.
- AI Pipeline: RAG supports analyst questions and agent evidence lookup.
- Security Considerations: Prompt injection filtering, citation enforcement, tenant filters, and answer abstention.
- Testing Strategy: No-evidence tests, citation presence tests, retrieval quality tests, and prompt-injection tests.
- Deployment Strategy: Local deterministic RAG; production can connect approved LLM providers or self-hosted models.
- Production-Ready Code: Citation-required RAG path is implemented.

## Phase 12 - Multi-Agent AI System

- Objectives: Coordinate Retrieval, Legal, Compliance, Fraud, Risk, Audit, Financial, and Summary agents.
- Functional Requirements: Agents collaborate before final decisions and include evidence, confidence, and reasoning chains.
- Non-Functional Requirements: Auditable decisions, replayability, deterministic fallback, model/prompt versioning, and LangGraph-ready orchestration.
- Research Contribution: Measures specialist-agent review quality versus single-pass summarization.
- System Design: Agent service consumes findings, fraud signals, documents, and retrieval evidence.
- Folder Structure: `backend/app/services/agents.py`, React agent views under `frontend/src/pages/`, and Phase 3 `agent_decisions`.
- Database Changes: Store decisions in `agent_decisions`.
- API Design: Agent-style review outputs are exposed through default Vision analysis/review payloads; optional legacy agent runs remain at `/api/v1/demo-legacy/agents/runs/{case_id}`.
- Backend Implementation: Multi-agent reasoning system produces evidence-linked decisions and reasoning chains.
- Frontend Implementation: Agent Center shows decisions, confidence, and reasoning.
- AI Pipeline: Agents run after compliance/fraud/verification and before workflow/reporting.
- Security Considerations: Tool permissions, model allowlists, tenant-scoped evidence, and audit logs for agent runs.
- Testing Strategy: Agent output contract tests, confidence range tests, and high-risk routing tests.
- Deployment Strategy: Local deterministic agent system; production LangGraph workers for durable orchestration.
- Production-Ready Code: Agent abstraction is implemented with LangGraph upgrade path documented.

## Phase 13 - Compliance Intelligence

- Objectives: Detect missing clauses, policy violations, regulatory violations, and compliance score impact.
- Functional Requirements: Check indemnity, liability, confidentiality, data privacy, termination, GDPR, HIPAA, PCI-DSS, SOC 2, and ISO 27001 mappings.
- Non-Functional Requirements: Rule-pack versioning, evidence links, severity, remediation, human review, and dashboard visibility.
- Research Contribution: Supports compliance-rule recall and explanation completeness evaluation.
- System Design: Compliance engine consumes classified documents and extracted clauses.
- Folder Structure: `backend/app/services/compliance.py`, frontend Compliance page, and Phase 3 `compliance_findings`.
- Database Changes: Store findings in `compliance_findings` with framework/control metadata.
- API Design: Findings are in Vision document/review/risk payloads; optional legacy compliance findings remain at `/api/v1/demo-legacy/compliance/findings`.
- Backend Implementation: Local rule engine finds missing required clauses and policy conflicts.
- Frontend Implementation: Compliance page shows score, findings, framework mapping, and evidence count.
- AI Pipeline: Compliance findings feed risk score, agents, workflow, reports, and XAI.
- Security Considerations: Rule changes and finding dismissal must be permissioned and audited.
- Testing Strategy: Clause fixtures, regulatory mapping tests, false-positive handling, and score calibration.
- Deployment Strategy: Rule packs stored/versioned per tenant; production workers can run bulk compliance jobs.
- Production-Ready Code: Compliance engine and dashboard surface are implemented.

## Phase 14 - Fraud Detection Engine

- Objectives: Detect duplicate invoices, amount mismatches, vendor anomalies, payment anomalies, fake documents, and suspicious transactions.
- Functional Requirements: Generate fraud probability, severity, evidence, and recommended investigation action.
- Non-Functional Requirements: Explainable signals, graph-ready vendor linking, model versioning, review-capacity calibration, and false-positive workflow.
- Research Contribution: Supports graph-enhanced and cross-document fraud detection experiments.
- System Design: Fraud engine consumes documents, extracted fields, checksums, vendor/payment signals, and verification outputs.
- Folder Structure: `backend/app/services/fraud.py`, frontend Fraud page, and Phase 3 `fraud_findings`.
- Database Changes: Store findings in `fraud_findings`.
- API Design: Risk signals are in Vision document risk payloads; optional legacy fraud findings remain at `/api/v1/demo-legacy/fraud/findings`.
- Backend Implementation: Local fraud engine returns signal list and probability score.
- Frontend Implementation: Fraud page shows probability, signals, detection coverage, and checks.
- AI Pipeline: Fraud signals feed risk score, agents, workflow escalation, graph projection, and reports.
- Security Considerations: Fraud labels and payment holds require restricted access and audit trails.
- Testing Strategy: Duplicate invoice fixtures, payment mismatch tests, vendor anomaly tests, and probability calibration.
- Deployment Strategy: Local rules in API; production anomaly models can run batch and streaming jobs.
- Production-Ready Code: Fraud service and frontend review page are implemented.

## Phase 15 - Cross-Document Verification

- Objectives: Compare invoice to purchase order, contract to invoice, and vendor to bank account details.
- Functional Requirements: Detect matches, mismatches, conflicts, exceptions, and missing supporting documents.
- Non-Functional Requirements: Normalize entities and amounts, preserve evidence pairs, support confidence scoring, and avoid false certainty.
- Research Contribution: Provides measurable multi-document verification beyond single-file RAG.
- System Design: Verification runs after extraction and before final risk/agent decisions.
- Folder Structure: `backend/app/services/cross_document.py`, tests in `backend/app/tests/test_cross_document.py`, and Phase 3 `cross_document_checks`.
- Database Changes: Store verification findings in `cross_document_checks`.
- API Design: Checks are included in analysis responses.
- Backend Implementation: Amount, payment-term, vendor-bank, and exception-language checks are implemented.
- Frontend Implementation: Risk and Fraud pages expose cross-document checks.
- AI Pipeline: Verification outputs feed fraud, compliance, agents, graph, workflow, and reports.
- Security Considerations: Payment and bank-account evidence require restricted display and redaction.
- Testing Strategy: Existing cross-document tests pass; extend with bank, exception, and missing-document cases.
- Deployment Strategy: Run synchronously for small cases and asynchronously for large case bundles.
- Production-Ready Code: Verification service has regression tests and production upgrade hooks.

## Phase 16 - Explainable AI Engine

- Objectives: Provide evidence, reasoning, source document, page number, highlighted text, confidence score, and model/rule metadata.
- Functional Requirements: Every material decision must include evidence and reasoning chain.
- Non-Functional Requirements: Evidence immutability, traceability, confidence calibration, source accuracy, and reviewer usability.
- Research Contribution: Enables explanation completeness and source-grounding evaluation.
- System Design: XAI layer packages evidence counts, confidence method, and reasoning policy across decisions and findings.
- Folder Structure: `backend/app/services/xai.py`, frontend Evidence Explorer, and Phase 3 `evidence`.
- Database Changes: Store evidence in `evidence` and snapshots in findings/decisions/reports.
- API Design: Evidence appears in Vision document, field, review, ask, and report payloads; optional legacy case evidence remains at `/api/v1/demo-legacy/cases/{case_id}/evidence`.
- Backend Implementation: Explainability layer packages evidence and reasoning metadata.
- Frontend Implementation: Evidence Explorer renders source, page, snippet, confidence, and reasoning.
- AI Pipeline: XAI attaches to compliance, fraud, agents, RAG, graph, workflow, and reporting.
- Security Considerations: Evidence access must enforce tenant, sensitivity, legal hold, and role checks.
- Testing Strategy: Evidence presence, page correctness, confidence bounds, and citation-link tests.
- Deployment Strategy: Store immutable evidence snapshots and generated highlights in object storage.
- Production-Ready Code: XAI layer and Evidence Explorer are implemented.

## Phase 17 - Knowledge Graph Reasoning

- Objectives: Implement traversal, relationship discovery, fraud network detection, and vendor risk discovery.
- Functional Requirements: Find high-risk paths, shared vendors, conflicting documents, repeated bank details, and approval clusters.
- Non-Functional Requirements: Bounded traversal, query timeouts, tenant filters, explainable paths, and graph drift monitoring.
- Research Contribution: Tests whether graph paths improve fraud and risk detection.
- System Design: Graph reasoning reads Neo4j projections and returns path evidence to agents and dashboards.
- Folder Structure: `backend/app/services/knowledge_graph.py`, `docs/KNOWLEDGE_GRAPH_DESIGN.md`, and Graph tab.
- Database Changes: Neo4j stores projected graph; PostgreSQL keeps source records.
- API Design: `GET /api/v1/graph/risk-paths` returns risky paths and evidence.
- Backend Implementation: Current graph engine creates entity neighborhoods, high-risk paths, cross-document links, finding nodes, and projection metadata.
- Frontend Implementation: Graph view and edge list expose relationships.
- AI Pipeline: Graph features feed fraud score, risk score, and agent recommendations.
- Security Considerations: Limit graph queries by tenant and suppress sensitive payment nodes without permission.
- Testing Strategy: Path discovery fixtures, graph projection replay, and no-cross-tenant traversal tests.
- Deployment Strategy: Neo4j StatefulSet with persistent storage and backup policy.
- Production-Ready Code: Graph reasoning boundary and query examples are documented.

## Phase 18 - Workflow Automation

- Objectives: Route finance, legal, compliance, executive, audit, and risk approvals.
- Functional Requirements: Support approve, reject, request changes, escalate, assignment, SLA state, and comments.
- Non-Functional Requirements: Auditability, idempotency, notification readiness, authorization, and high-risk human gatekeeping.
- Research Contribution: Captures reviewer outcomes for model calibration and active learning.
- System Design: Workflow engine consumes risk/compliance/fraud outputs and assigns the next team/action.
- Folder Structure: `backend/app/services/workflow.py`, frontend Workflow tab, and Phase 3 `workflow_tasks`/`approvals`.
- Database Changes: Store workflow tasks and approvals in Phase 3 schema.
- API Design: review decisions use `/api/v1/reviews/{review_id}/*`; optional legacy workflow decisions remain at `/api/v1/demo-legacy/workflow/decisions`.
- Backend Implementation: Workflow assignment and decision recording are implemented.
- Frontend Implementation: Workflow tab shows current state and decision form.
- AI Pipeline: Workflow closes the loop from AI findings to human review outcomes.
- Security Considerations: Approval actions require RBAC, audit logs, and separation of duties.
- Testing Strategy: Routing tests by risk/fraud/compliance score and decision API tests.
- Deployment Strategy: Synchronous local workflow; production can use durable queues and notification workers.
- Production-Ready Code: Workflow engine and UI decision path are implemented.

## Phase 19 - Predictive Intelligence

- Objectives: Predict fraud probability, compliance failure, approval delay, contract risk, and SLA breach risk.
- Functional Requirements: Provide scores with confidence, feature summaries, model version, and explanation.
- Non-Functional Requirements: Calibrated predictions, drift monitoring, human feedback loops, and model rollback.
- Research Contribution: Evaluates rule/model/graph features for enterprise risk forecasting.
- System Design: Predictive service combines compliance findings, fraud signals, severity, and workflow factors.
- Folder Structure: `backend/app/services/predictive.py` and dashboard metrics.
- Database Changes: Store production predictions in findings, cases, reports, and future model registry tables.
- API Design: Prediction outputs appear in case analysis and dashboard metrics.
- Backend Implementation: Risk scoring and approval-delay probability are implemented.
- Frontend Implementation: Dashboard shows risk and SLA risk.
- AI Pipeline: Predictive scores guide workflow escalation and report summaries.
- Security Considerations: Predictions are decision support, not autonomous high-impact decisions.
- Testing Strategy: Calibration tests, monotonicity tests, edge-case severity tests, and historical backtests.
- Deployment Strategy: Local deterministic scoring; production can use MLflow-served models.
- Production-Ready Code: Predictive boundary is implemented and MLOps-ready.

## Phase 20 - Analytics Dashboard

- Objectives: Show risk trends, compliance trends, fraud trends, agent performance, workflow analytics, and document analytics.
- Functional Requirements: Display metrics, charts, case health, operations, findings, workflow, graph, evidence, and reports.
- Non-Functional Requirements: Responsive dashboard, low-latency metric endpoint, demo fallback, and readable dense layout.
- Research Contribution: Supports monitoring reviewer workload and AI usefulness over time.
- System Design: Dashboard reads backend metrics and analysis payload.
- Folder Structure: `frontend/src/pages/DashboardPage.tsx`, `frontend/src/pages/AnalyticsPage.tsx`, `frontend/src/styles.css`, and `backend/app/services/pipeline.py`.
- Database Changes: Production dashboard reads cases, findings, workflow, reports, audit logs, and notifications.
- API Design: default analytics use `/api/v1/analytics/overview`; optional legacy dashboard metrics remain at `/api/v1/demo-legacy/dashboard/metrics`.
- Backend Implementation: Pipeline exposes aggregate metrics from analyzed cases.
- Frontend Implementation: Dashboard and analytics pages use Recharts-backed visualizations.
- AI Pipeline: Analytics surfaces outputs from all AI stages.
- Security Considerations: Dashboards must enforce role and tenant filters before aggregation.
- Testing Strategy: Metric API tests, chart rendering tests, and accessibility checks.
- Deployment Strategy: Static frontend served by Nginx; API metrics scale horizontally.
- Production-Ready Code: Analytics dashboard builds successfully.

## Phase 21 - Security Architecture

- Objectives: Implement RBAC, ABAC readiness, audit logs, encryption, PII detection/redaction, and document versioning.
- Functional Requirements: Tenant-scoped roles, permissions, immutable audit, encrypted stores, sensitivity labels, legal hold, and version lineage.
- Non-Functional Requirements: Defense in depth, least privilege, secure defaults, compliance readiness, and observable security events.
- Research Contribution: Establishes trustworthy AI controls for regulated document intelligence.
- System Design: Security spans API, database, object storage, vector store, graph store, workflows, and AI providers.
- Folder Structure: `docs/SECURITY_ARCHITECTURE.md`, `backend/app/core/security.py`, Phase 3 schema, and Kubernetes secrets.
- Database Changes: `roles`, `permissions`, `user_roles`, `role_permissions`, `audit_logs`, legal hold, sensitivity labels, and document versions.
- API Design: Production auth middleware should enforce JWT, RBAC, ABAC, and tenant context.
- Backend Implementation: Hashing, constant-time compare, UTC timestamps, and PII redaction helpers exist.
- Frontend Implementation: Admin and Audit pages expose governance concepts without leaking secrets.
- AI Pipeline: Redaction and evidence scoping protect prompts, logs, reports, and exports.
- Security Considerations: OIDC/SAML, KMS, TLS, RLS, secrets management, prompt-injection filtering, and export controls.
- Testing Strategy: SAST, dependency scans, authorization tests, redaction tests, and threat-model review.
- Deployment Strategy: Kubernetes secrets, TLS ingress, private networking, and managed encrypted stores.
- Production-Ready Code: Security architecture and schema controls are implemented/documented.

## Phase 22 - MLOps Pipeline

- Objectives: Provide model registry, monitoring, experiment tracking, evaluation gates, and automated retraining path using MLflow.
- Functional Requirements: Track datasets, prompts, retrievers, model versions, rule packs, feedback, drift, and rollback.
- Non-Functional Requirements: Reproducibility, approval gates, shadow/canary deployment, observability, and auditability.
- Research Contribution: Supports controlled experiments and publication-quality evaluation.
- System Design: MLOps sits beside the AI services and records quality, drift, and human-feedback outcomes.
- Folder Structure: `docs/MLOPS_PIPELINE.md`, `backend/app/services`, and future `mlruns`/registry service.
- Database Changes: Phase 3 stores model/prompt/rule metadata in decisions, findings, embeddings, and evidence.
- API Design: Production may expose `/api/v1/mlops/models`, `/evaluations`, and `/drift`.
- Backend Implementation: Services expose model version fields and deterministic fallback behavior.
- Frontend Implementation: Admin and Agent Center show model-provider readiness and confidence.
- AI Pipeline: Every stage can emit metrics for OCR, extraction, retrieval, RAG, agents, compliance, fraud, and prediction.
- Security Considerations: Protect training data, prompts, model artifacts, and evaluation labels.
- Testing Strategy: Offline eval gates for classification F1, extraction F1, retrieval nDCG, fraud precision, and explanation completeness.
- Deployment Strategy: MLflow tracking server and registry can be added to Compose/Kubernetes.
- Production-Ready Code: MLOps plan exists with service-level traceability hooks.

## Phase 23 - Dockerization

- Objectives: Containerize backend, frontend, PostgreSQL, ChromaDB, Neo4j, and Nginx proxy path.
- Functional Requirements: Provide backend Dockerfile, frontend Dockerfile, Docker Compose, env config, and persistent volumes.
- Non-Functional Requirements: Reproducible builds, minimal images, health-ready ports, and local developer startup.
- Research Contribution: Enables reproducible demos and experiment environments.
- System Design: Compose runs API, frontend, Postgres, Chroma, and Neo4j.
- Folder Structure: `backend/Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`, `infra/nginx/nginx.conf`, and `infra/docker/postgres-init.sql`.
- Database Changes: Compose mounts PostgreSQL init schema.
- API Design: Nginx proxies `/api/` to the backend.
- Backend Implementation: Python slim image installs FastAPI app.
- Frontend Implementation: Node build stage compiles Vite app; Nginx serves static assets.
- AI Pipeline: Containers include local deterministic AI and persistence dependencies.
- Security Considerations: Secrets move to env files/secrets; production should avoid default credentials.
- Testing Strategy: Docker image build in CI and Compose config validation.
- Deployment Strategy: Docker Compose for local full stack.
- Production-Ready Code: Dockerfiles and Compose stack are present and validated.

## Phase 24 - Kubernetes

- Objectives: Provide Kubernetes manifests for namespace, backend, frontend, PostgreSQL, Chroma, Neo4j, ingress, config, secrets, and autoscaling.
- Functional Requirements: Deploy services, persistent stores, probes, resource limits, ingress routing, and HPA.
- Non-Functional Requirements: Horizontal API scaling, persistent state, readiness/liveness probes, and configurable secrets.
- Research Contribution: Supports scalable experiment deployment and enterprise demo environments.
- System Design: App workloads run in `documind` namespace with stateful stores and ingress.
- Folder Structure: `infra/k8s/*.yaml`.
- Database Changes: PostgreSQL StatefulSet uses persistent volume.
- API Design: Ingress routes `/api` to backend and `/` to frontend.
- Backend Implementation: Deployment runs three backend replicas with health probes.
- Frontend Implementation: Deployment runs two frontend replicas.
- AI Pipeline: Chroma and Neo4j StatefulSets support retrieval and graph phases.
- Security Considerations: Secret examples must be replaced by external secret management in production.
- Testing Strategy: Kubernetes dry-run validation and deployment smoke tests.
- Deployment Strategy: Apply namespace, secrets, configmap, statefulsets, deployments, services, HPA, and ingress.
- Production-Ready Code: Kubernetes manifests are present for all core services.

## Phase 25 - CI/CD

- Objectives: Automate test, build, scan, and deployment validation.
- Functional Requirements: Run backend lint/tests, frontend build, vulnerability scan, Docker image builds, and gated deployment validation.
- Non-Functional Requirements: Pull-request safety, reproducibility, least-privilege deployment, and manual production gate.
- Research Contribution: Ensures experiments and demos are reproducible and regression-protected.
- System Design: GitHub Actions separates backend, frontend, security scan, container build, and deploy handoff jobs.
- Folder Structure: `.github/workflows/ci.yml`.
- Database Changes: CI validates application code; schema is validated through PostgreSQL smoke test locally.
- API Design: CI exercises backend tests, including API contract tests.
- Backend Implementation: `ruff check app` and `pytest`.
- Frontend Implementation: `npm ci` and `npm run build`.
- AI Pipeline: Tests protect deterministic AI behavior.
- Security Considerations: Trivy filesystem scan added; production deploy requires environment gate.
- Testing Strategy: CI pipeline itself is the test/build/scan path.
- Deployment Strategy: Manual `workflow_dispatch` validates Kubernetes manifests before production apply is enabled.
- Production-Ready Code: CI/CD workflow now includes test, build, scan, and gated deploy stages.

## Phase 26 - Cloud Deployment

- Objectives: Define production architecture for AWS, Azure, and GCP.
- Functional Requirements: Support managed Kubernetes, managed PostgreSQL, object storage, vector/graph services, secrets, observability, WAF, and CI/CD.
- Non-Functional Requirements: High availability, backups, PITR, regional data residency, private networking, autoscaling, and disaster recovery.
- Research Contribution: Enables reproducible enterprise-scale evaluations across cloud environments.
- System Design: Cloud load balancer/WAF routes to Kubernetes; API/workers use managed stores and approved model providers.
- Folder Structure: Cloud plan is captured here plus `infra/k8s` and Docker artifacts.
- Database Changes: Use managed PostgreSQL with Phase 3 migrations, backups, encryption, and RLS.
- API Design: Same `/api/v1` surface behind cloud gateway.
- Backend Implementation: Containerized FastAPI deploys to EKS/AKS/GKE.
- Frontend Implementation: Static frontend can run in Kubernetes or object storage/CDN with API gateway.
- AI Pipeline: OCR/LLM providers can be tenant-approved cloud services or self-hosted GPU endpoints.
- Security Considerations: KMS, private subnets, VPC endpoints, managed identities, external secrets, SIEM, and WAF rules.
- Testing Strategy: Staging smoke tests, load tests, failover tests, restore drills, and security scans.
- Deployment Strategy: AWS EKS/RDS/S3/OpenSearch or Bedrock, Azure AKS/PostgreSQL/Blob/OpenAI, GCP GKE/Cloud SQL/GCS/Vertex AI.
- Production-Ready Code: Container and Kubernetes artifacts are cloud-portable.

## Phase 27 - Testing Strategy

- Objectives: Provide unit, integration, API, load, security, AI-quality, frontend, and deployment testing strategy.
- Functional Requirements: Cover upload, analysis, query, metrics, workflow decisions, reports, document types, findings, fraud, graph, and XAI.
- Non-Functional Requirements: Repeatability, deterministic fixtures, coverage proportional to risk, and enterprise release gates.
- Research Contribution: Defines objective measures for publication and product quality.
- System Design: Tests span backend unit tests, API tests, frontend tests, AI evals, and deployment smoke tests.
- Folder Structure: `backend/app/tests`, `docs/TESTING_STRATEGY.md`, CI workflow, and future frontend test folders.
- Database Changes: Use isolated test databases and seed fixtures in production-grade tests.
- API Design: API contract tests validate response schemas and authorization behavior.
- Backend Implementation: Existing pytest suite covers pipeline, cross-document verification, and phase API contracts.
- Frontend Implementation: Lint/build validation exists; future Playwright tests should cover key workflows.
- AI Pipeline: Evaluation covers classification, extraction, retrieval, RAG, compliance, fraud, XAI, and predictions.
- Security Considerations: Include SAST, dependency scanning, parser fuzzing, auth tests, and redaction tests.
- Testing Strategy: Documented in `docs/TESTING_STRATEGY.md`.
- Deployment Strategy: Run tests in CI and staging before production rollout.
- Production-Ready Code: Current backend tests, frontend build, lint, CI, and documented strategy are in place.

## Phase 28 - Research Paper

- Objectives: Produce IEEE-quality research framing with abstract, problem statement, literature survey, gap, methodology, architecture, results, and future scope.
- Functional Requirements: Describe multi-document intelligence, multi-agent reasoning, cross-document verification, XAI, compliance, fraud, graph reasoning, and predictive intelligence.
- Non-Functional Requirements: Reproducible experiments, clear baselines, ablations, metrics, limitations, and ethics/security discussion.
- Research Contribution: Positions DocuMind AI as explainable multi-agent multi-document intelligence for enterprise compliance and fraud.
- System Design: Paper follows the platform architecture and evaluation pipeline.
- Folder Structure: `docs/RESEARCH_PAPER_STRUCTURE.md` and `docs/RESEARCH_NOVELTY.md`.
- Database Changes: Research datasets should version document metadata, labels, outputs, model versions, and reviewer decisions.
- API Design: Evaluation harness can call existing APIs for benchmark runs.
- Backend Implementation: Service boundaries allow ablation by disabling graph, agents, verification, or XAI components.
- Frontend Implementation: Reviewer UI supports human evaluation and qualitative examples.
- AI Pipeline: Experiments evaluate OCR/layout, extraction, retrieval, RAG, agents, compliance, fraud, graph, and XAI.
- Security Considerations: Use anonymized datasets, legal approvals, PII redaction, and ethical review.
- Testing Strategy: Report macro F1, extraction F1, nDCG, citation precision, fraud precision/recall, calibration, and reviewer acceptance.
- Deployment Strategy: Run experiments in reproducible Docker/Kubernetes environments.
- Production-Ready Code: Research structure and novelty docs are present.

## Phase 29 - Resume And GitHub Portfolio

- Objectives: Produce resume project description, LinkedIn description, GitHub README, architecture diagrams, and demo script.
- Functional Requirements: Present project value, technology stack, research contributions, architecture, setup, demo flow, and portfolio positioning.
- Non-Functional Requirements: Clear, credible, recruiter-readable, technically accurate, and not overstated beyond scaffolded production readiness.
- Research Contribution: Communicates publishable novelty and enterprise engineering depth.
- System Design: Portfolio docs point to README, phase docs, architecture docs, screenshots/demo, and runnable commands.
- Folder Structure: `README.md`, `docs/RESUME_DESCRIPTION.md`, `docs/GITHUB_PORTFOLIO_SETUP.md`, and phase docs.
- Database Changes: None.
- API Design: README lists core APIs and quick-start commands.
- Backend Implementation: Portfolio highlights FastAPI services, AI pipeline, and tests.
- Frontend Implementation: Portfolio highlights React analyst console and enterprise modules.
- AI Pipeline: Portfolio highlights OCR/layout, classification, extraction, retrieval, RAG, agents, compliance, fraud, verification, XAI, graph, workflow, and MLOps.
- Security Considerations: Do not commit secrets, sensitive documents, or real enterprise data.
- Testing Strategy: Verify quick-start commands, API docs, frontend build, backend tests, and screenshots before publishing.
- Deployment Strategy: Publish repository with Docker Compose, Kubernetes manifests, CI, and demo script.
- Production-Ready Code: Resume, portfolio, README, and architecture documentation are present and aligned.

## Final Continuous Completion Checklist

- [x] Phase 4 Knowledge Graph Design
- [x] Phase 5 Backend Foundation
- [x] Phase 6 Frontend Foundation
- [x] Phase 7 OCR and Layout Intelligence
- [x] Phase 8 Document Classification
- [x] Phase 9 Knowledge Extraction
- [x] Phase 10 Vector Intelligence Layer
- [x] Phase 11 Advanced RAG Engine
- [x] Phase 12 Multi-Agent AI System
- [x] Phase 13 Compliance Intelligence
- [x] Phase 14 Fraud Detection Engine
- [x] Phase 15 Cross-Document Verification
- [x] Phase 16 Explainable AI Engine
- [x] Phase 17 Knowledge Graph Reasoning
- [x] Phase 18 Workflow Automation
- [x] Phase 19 Predictive Intelligence
- [x] Phase 20 Analytics Dashboard
- [x] Phase 21 Security Architecture
- [x] Phase 22 MLOps Pipeline
- [x] Phase 23 Dockerization
- [x] Phase 24 Kubernetes
- [x] Phase 25 CI/CD
- [x] Phase 26 Cloud Deployment
- [x] Phase 27 Testing Strategy
- [x] Phase 28 Research Paper
- [x] Phase 29 Resume and GitHub Portfolio
