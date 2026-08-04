# Architecture

## System Architecture
React dashboard communicates with FastAPI APIs. The backend loads dataset labels or uploaded files, builds a single `CaseAnalysis` object, and serves every page from that object.

## Frontend Architecture
The frontend has pages for Login, Signup, Dashboard, Documents, Search, Risk, Compliance, Fraud, Evidence, Graph, Workflow, Audit, Admin, Reports, and Agent Center. The app shell stores the local auth token, attaches bearer authorization to API calls, displays the user role badge, and redirects unauthenticated users to login. Each business page calls backend APIs and renders loading, empty, and error states.

## Backend Architecture
Core services include dataset loader, local auth, ingestion, OCR fallback, classification, extraction, chunking, embedding, retrieval, RAG, compliance, fraud, graph, workflow, report, audit, and scoring.

## AI Pipeline
`upload/run demo -> parse -> classify -> extract entities -> chunk -> index -> retrieve -> analyze findings -> score -> graph -> workflow -> report`

## RAG Pipeline
Dataset evidence chunks are indexed with local hashed embeddings and keyword fallback. Search returns document name, page, chunk ID, confidence, and evidence text.

Payment and payout intent detection routes queries such as "expected payout", "payment amount", and "invoice amount" to a structured answer that combines invoice amount, PO amount, payment terms, due date, approval status, blockers, and citations.

## Auth Design
Local college-demo authentication uses Argon2id password hashing, persisted local user/session state, JWT access and refresh tokens, refresh rotation, logout token revocation, bootstrap scripts for privileged demo users, and role labels. No external identity provider or paid API is required.

## Multi-Agent Pipeline
Document Understanding, Retrieval, Legal, Financial, Compliance, Risk, and Summary agents return decisions, confidence, evidence IDs, reasoning, next actions, and status.

## Database/Design Explanation
The MVP uses an in-memory repository plus dataset-backed labels for local demo reliability. The structure is compatible with PostgreSQL tables for cases, documents, evidence, findings, graph edges, workflows, and audit events.

## Knowledge Graph Design
Nodes include Case, Document, Vendor, Company, Invoice, PurchaseOrder, Clause, Finding, FraudSignal, Risk, and Workflow. Edges include CASE_CONTAINS_DOCUMENT, DOCUMENT_MENTIONS_ENTITY, INVOICE_REFERENCES_PO, INVOICE_ISSUED_BY_VENDOR, CONTRACT_MISSING_CLAUSE, FINDING_SUPPORTED_BY_EVIDENCE, CASE_HAS_RISK, and CASE_HAS_WORKFLOW_ACTION.
