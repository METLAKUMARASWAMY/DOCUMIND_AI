# DocuMind AI

DocuMind AI is an enterprise document intelligence platform for banks, insurers, legal teams, healthcare organizations, manufacturers, government teams, and finance operations. It combines OCR/layout analysis, document classification, extraction, hybrid retrieval, multi-agent reasoning, compliance intelligence, fraud detection, cross-document verification, explainability, workflow automation, reporting, and knowledge graph modeling.

The repository is intentionally production-shaped: a FastAPI backend, React analyst console, deployment assets, CI, test strategy, security architecture, MLOps plan, research framing, and portfolio material are included.

## Phase-Driven Delivery

The DocuMind AI 2.0 brief is being handled phase by phase. Current phase status:

- Phase 1, Software Requirements Specification: complete.
- Phase 2, System Architecture: complete.
- Phase 3, Database Design: complete.
- Phases 4-29, remaining product, AI, security, deployment, research, and portfolio scope: complete.

Phase 1 source of truth: [Phase 1 SRS](docs/phases/PHASE_01_SRS.md)
Phase 2 source of truth: [Phase 2 System Architecture](docs/phases/PHASE_02_SYSTEM_ARCHITECTURE.md)
Phase 3 source of truth: [Phase 3 Database Design](docs/phases/PHASE_03_DATABASE_DESIGN.md)
Phases 4-29 source of truth: [Continuous Completion Package](docs/phases/PHASE_04_TO_29_COMPLETION.md)

## What Is Implemented

- FastAPI backend with dataset-backed demo cases, upload, text analysis, retrieval QA, metrics, workflow decision, and report endpoints.
- Local deterministic AI pipeline with file storage, OCR/layout fallback, classification, entity extraction, evidence chunking, local hashed embeddings, hybrid retrieval/reranking, citation-grounded RAG, multi-agent decisions, compliance checks, fraud signals, cross-document validation, XAI, reports, scoring, workflow blocking, audit trail, and knowledge graph projection.
- Runtime model registry with selective Vision AI routing metadata for PP-OCRv5, PP-StructureV3, LayoutLMv3, PaddleOCR-VL, Donut, YOLO, BGE-M3, Qdrant, Neo4j, and Qwen target adapters. Heavy models are reported as disabled, unavailable, or adapter-pending unless explicitly enabled, importable, and wired into the pipeline.
- Optional Vision infrastructure readiness gates for MongoDB, Redis/Celery, MinIO, Qdrant, and Neo4j, with secret-redacted configuration and local fallbacks kept explicit.
- DOCUMIND VISION AI local execution path with quality reports, word-level OCR boxes, layout regions, table structure, visual cue detection, field-level evidence, confidence routing, duplicate detection, risk indicators, review tasks, evidence-grounded asking, JSON/PDF reports, and active-learning correction export.
- Vision processing can run through FastAPI background tasks locally or through the Docker Compose Celery worker with Redis enabled; queued job state is persisted under `var/vision_state` so API and worker processes can share work before MongoDB is enabled.
- React/Vite analyst console using MUI, Recharts, custom CSS, and lucide icons, with backend-driven dashboard, documents, search, risk, compliance, fraud, evidence, graph, workflow, audit, admin, reports, and agent pages.
- NVIDIA Project 6 finance workflow with invoice/PO/contract/bank-change extraction, local VectorDB storage, grounded finance narratives, human review, evaluation metrics, and demo cases.
- PostgreSQL schema, Neo4j graph model, Chroma/pgvector-ready retrieval design.
- Docker Compose, Nginx, Kubernetes manifests, CI workflow, tests, and documentation.

## Repository Layout

```text
backend/                 FastAPI application and AI service layer
frontend/                React/Vite analyst dashboard
dataset/                 Synthetic DocuMind AI cases, documents, labels, evidence, RAG questions
demo-data/               Legacy seed contract, invoice, PO, and bank-change email documents
docs/                    SRS, architecture, API, research, security, MLOps, portfolio docs
infra/docker/            Database init SQL
infra/k8s/               Kubernetes manifests
infra/nginx/             Nginx reverse proxy config
.github/workflows/       CI pipeline
scripts/                 Utility scripts
```

## Quick Start

Backend:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DOCUMIND_DATASET_PATH="/Users/kumar/Downloads/DocuMind AI/dataset/documind_ai_synthetic_dataset"
uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` or `http://localhost:5174` depending on the Vite port shown in the terminal. Local demo accounts are seeded automatically outside production:

```text
admin@documind.ai / Admin@123
compliance@documind.ai / Compliance@123
finance@documind.ai / Finance@123
legal@documind.ai / Legal@123
risk@documind.ai / Risk@123
auditor@documind.ai / Auditor@123
```

You can also sign up from the UI to create a local viewer account, or create a custom admin/reviewer account for protected review and admin screens:

```bash
python scripts/bootstrap_auth_users.py --role ADMIN --email admin@documind.local
python scripts/bootstrap_auth_users.py --role REVIEWER --email reviewer@documind.local
```

Full stack:

```bash
docker compose up --build
```

The Compose profile starts `backend`, `worker`, `frontend`, Redis, MongoDB, Qdrant, MinIO, Neo4j, Chroma, and PostgreSQL. The backend and worker share `./var:/app/var`; Vision jobs use Celery/Redis in Compose and the local background-task fallback outside `ENABLE_REDIS=true`.

Seed the Vision AI demo after the backend is running:

```bash
python scripts/seed_vision_demo.py
```

## Core API

Default health and auth:

- `GET /health/live`
- `GET /health/ready`
- `POST /api/v1/auth/register`
- `POST /api/v1/auth/login`
- `GET /api/v1/auth/me`
- `POST /api/v1/auth/logout`

Default Vision API:

- `POST /api/v1/documents/upload`
- `GET /api/v1/documents`
- `GET /api/v1/documents/{document_id}`
- `GET /api/v1/documents/{document_id}/status`
- `GET /api/v1/documents/{document_id}/pages`
- `GET /api/v1/documents/{document_id}/fields`
- `GET /api/v1/documents/{document_id}/tables`
- `GET /api/v1/documents/{document_id}/risk`
- `GET /api/v1/reviews`
- `POST /api/v1/ask`
- `GET /api/v1/graph/document/{document_id}`
- `POST /api/v1/graph/query`
- `GET /api/v1/analytics/overview`
- `GET /api/v1/admin/models`
- `GET /api/v1/vision/infrastructure/readiness`
- `POST /api/v1/vision/evaluation/run`
- `GET /api/v1/models/comparison/{document_id}`
- `POST /api/v1/reports/generate`
- `GET /api/v1/reports/{report_id}?format=json|md|pdf`
- `GET /api/v1/reports/{report_id}/download`
- `GET /api/v1/vision/active-learning/export`

Optional legacy research-case and NVIDIA Project 6 demo routes are mounted only when `ENABLE_DEMO_API=true`; their base path is `/api/v1/demo-legacy`.

## Demo Behavior

The dataset case `case-df5ea07cd6` in `dataset/documind_ai_synthetic_dataset` produces:

- Missing indemnity clause with high severity and approval blocking.
- PO/invoice amount and PO reference verification.
- Vendor bank account change and urgent wire anomaly.
- Manual override risk.
- Backend-computed compliance score, fraud probability, risk score, evidence, and workflow metrics.
- Risk score of 100/100 with workflow recommendation `hold_payment`, not default approval.
- Search answers with document, page, chunk ID, confidence, and evidence text.
- Markdown, JSON, and PDF reports generated from backend case data.

## NVIDIA Project 6 Finance Workflow

DocuMind AI includes a dedicated implementation of **NVIDIA Project 6: Finance Document-Extraction and Narrative-Generation Workflow with Human Review**.

Frontend pages:

- Finance
- Human Review
- Evaluation
- Demo Cases
- VectorDB

Final-polish guardrails:

- Finance uploads default to auto-detect. If a bank-change request is accidentally uploaded as an invoice, the backend overrides it to `bank_change_request` and records both the selected and detected types.
- Bank-change and high-risk finance documents cannot be approved with a casual click. Approval requires reviewer comments documenting independent out-of-band verification; otherwise reviewers should escalate or request correction.
- Reviewer-facing evidence and VectorDB cards mask account numbers while preserving source traceability.
- Known demo filenames automatically attach ground-truth files for meaningful evaluation metrics.

Demo API:

```bash
ENABLE_DEMO_API=true uvicorn app.main:app --reload --port 8000
curl -X POST http://localhost:8000/api/v1/demo-legacy/finance/demo-cases/invoice-demo/run
curl -X POST http://localhost:8000/api/v1/demo-legacy/finance/demo-cases/bank-change-risk-demo/run
```

Report: [NVIDIA Project 6 Report](docs/NVIDIA_Project6_Report.md)

## Testing

```bash
cd backend
python -m pytest
ruff check app

cd ../frontend
npm run lint
npm run build
```

## Production AI Upgrade Path

The local pipeline is built behind service boundaries so enterprise models can be swapped in without changing the API:

- OCR/layout: PP-OCRv5, PP-StructureV3, PP-DocLayoutV3, PaddleOCR-VL fallback, EasyOCR, Tesseract, DocFormer.
- Visual extraction: LayoutLMv3 for schema-specific classification/extraction, Donut for OCR-free comparison, YOLO-family detectors for signatures, stamps, checkboxes, logos, and altered-region candidates.
- Retrieval: BGE-M3, BGE reranking, Qdrant, ChromaDB, FAISS, pgvector, hybrid BM25 plus embeddings.
- RAG and agents: Qwen/local transformers, Gemini, OpenAI-compatible models, LangGraph, CrewAI, AutoGen.
- Knowledge graph: Neo4j with entity resolution, proposed relationship review, and temporal transaction edges.
- MLOps: drift monitoring, model registry, prompt/version control, evaluation gates, and audit dashboards.

See [DOCUMIND VISION AI Migration Plan](docs/VISION_AI_MIGRATION_PLAN.md) for the repository assessment, target architecture, phased implementation plan, and model-routing commands.

## Documentation Index

- [Phase 1 SRS](docs/phases/PHASE_01_SRS.md)
- [Phase 2 System Architecture](docs/phases/PHASE_02_SYSTEM_ARCHITECTURE.md)
- [Phase 3 Database Design](docs/phases/PHASE_03_DATABASE_DESIGN.md)
- [Phases 4-29 Continuous Completion Package](docs/phases/PHASE_04_TO_29_COMPLETION.md)
- [Phase Completion Audit](docs/PHASE_COMPLETION_AUDIT.md)
- [SRS Index](docs/SRS.md)
- [Research Novelty](docs/RESEARCH_NOVELTY.md)
- [System Architecture](docs/ARCHITECTURE.md)
- [Database Design](docs/DATABASE_DESIGN.md)
- [Knowledge Graph Design](docs/KNOWLEDGE_GRAPH_DESIGN.md)
- [API Design](docs/API_DESIGN.md)
- [Security Architecture](docs/SECURITY_ARCHITECTURE.md)
- [MLOps Pipeline](docs/MLOPS_PIPELINE.md)
- [DOCUMIND VISION AI Migration Plan](docs/VISION_AI_MIGRATION_PLAN.md)
- [DOCUMIND VISION AI Completion Audit](docs/VISION_AI_COMPLETION_AUDIT.md)
- [DOCUMIND VISION AI Dataset Card](docs/dataset_card.md)
- [Local Vision Pipeline Model Card](docs/model_cards/local_vision_pipeline.md)
- [Target Model Stack Card](docs/model_cards/target_model_stack.md)
- [Testing Strategy](docs/TESTING_STRATEGY.md)
- [Research Paper Structure](docs/RESEARCH_PAPER_STRUCTURE.md)
- [Resume Description](docs/RESUME_DESCRIPTION.md)
- [GitHub Portfolio Setup](docs/GITHUB_PORTFOLIO_SETUP.md)

## Current Scope

This is a complete enterprise scaffold with runnable local intelligence and clean production integration points. For regulated production deployment, connect approved OCR/LLM providers, enable persistent stores, configure identity/RBAC, add human evaluation sets, run threat modeling, and complete organization-specific compliance controls.
