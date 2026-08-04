# DocuMind AI - Submission README

## Project Title
DocuMind AI - Intelligent Enterprise Document Intelligence Platform

## Objective
Build an end-to-end enterprise document intelligence MVP that ingests multiple business documents, extracts evidence, performs grounded RAG, detects compliance/fraud issues, verifies cross-document consistency, builds a knowledge graph, recommends workflow actions, and generates reports.

## Dataset Used
Primary dataset: `dataset/documind_ai_synthetic_dataset`  
Validated local path: `/Users/kumar/Downloads/DocuMind AI/dataset/documind_ai_synthetic_dataset`

Labels used: cases, documents, entities, evidence chunks, compliance findings, fraud findings, cross-document checks, workflow decisions, agent decisions, knowledge graph edges, NER records, classification records, and RAG questions.

## Modules
- Document ingestion and checksum validation
- Classification and entity extraction
- Evidence chunking and retrieval
- Compliance engine
- Fraud detection engine
- Cross-document verification
- Multi-agent analysis
- Knowledge graph projection
- Workflow approval routing
- Audit trail and reports
- Local login, signup, logout, role badges, and protected reviewer actions
- React analyst dashboard

## Tech Stack
FastAPI, Pydantic, React, Vite, TypeScript, MUI, Recharts, lucide icons, local hybrid retrieval, deterministic scoring, SHA-256 file checksums, Argon2id password hashing, JWT bearer tokens, Docker, Kubernetes manifests, GitHub Actions.

## Setup Commands
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export DOCUMIND_DATASET_PATH="/Users/kumar/Downloads/DocuMind AI/dataset/documind_ai_synthetic_dataset"
export DOCUMIND_AUTH_SECRET="change-me-local-demo-secret"
uvicorn app.main:app --reload --port 8000
```

```bash
cd frontend
npm install
npm run dev
```

## Local Login
Local demo accounts are seeded automatically outside production:

```text
admin@documind.ai / Admin@123
compliance@documind.ai / Compliance@123
finance@documind.ai / Finance@123
legal@documind.ai / Legal@123
risk@documind.ai / Risk@123
auditor@documind.ai / Auditor@123
```

You can also create users through the signup page. For custom protected reviewer/admin screens, bootstrap a local role account:

```bash
python scripts/bootstrap_auth_users.py --role ADMIN --email admin@documind.local
python scripts/bootstrap_auth_users.py --role REVIEWER --email reviewer@documind.local
```

The dashboard is protected. Login stores a local bearer token, shows the reviewer role badge, and enables protected workflow decisions and report export.

## API Summary
Default endpoints include `/health/live`, `/health/ready`, `/api/v1/auth/login`, `/api/v1/auth/register`, `/api/v1/auth/me`, `/api/v1/auth/logout`, `/api/v1/documents`, `/api/v1/reviews`, `/api/v1/ask`, `/api/v1/graph`, `/api/v1/analytics`, `/api/v1/reports`, and `/api/v1/admin`. Optional legacy case/finance demo endpoints are under `/api/v1/demo-legacy` when `ENABLE_DEMO_API=true`.

## Screenshots Section
Use `submission/screenshots/README.md` as the screenshot capture checklist. Required screens include Login, Signup, Dashboard, Documents, Search payout query, Risk, Compliance, Fraud, Evidence, Graph, Workflow, Audit, Reports, Agent Center, and Admin.

## Final Features
Vision upload/analyze flows produce evidence-linked fields, risk indicators, review tasks, graph data, finance narratives, and reports. Optional legacy demo mode loads `case-df5ea07cd6`; dashboard/report/risk scores match the seeded dataset expectations.
