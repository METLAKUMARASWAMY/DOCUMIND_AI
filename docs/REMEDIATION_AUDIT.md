# DocuMind Vision AI Remediation Audit

Audit date: 2026-08-04

Final implementation and acceptance evidence: [Final Acceptance Report](FINAL_ACCEPTANCE_REPORT.md).

## Baseline

- Checkpoint: `backup/pre-remediation-20260804` at commit `09ab28c`.
- Worktree at audit start: clean, branch `main`.
- Backend: FastAPI `>=0.115`, Pydantic Settings, Python `>=3.11`.
- Frontend: React 18.3, Vite 7.3, TypeScript 5.7, MUI 6.4, React Query 5.66.
- Local persistence: JSON state and local files under ignored `var/`; optional MongoDB, MinIO, Redis, Qdrant, and Neo4j connectors.
- Worker: Celery task and Redis dispatcher with FastAPI background-task fallback.
- Database assets: a PostgreSQL bootstrap schema and optional MongoDB collections. No executable versioned migration existed before this remediation.
- Baseline verification: 52 backend tests passed; 1 frontend test passed; frontend lint and production build passed.
- Runtime records observed without mutation: 84 root-state documents and 182 backend-state documents; existing uploads and reviewer history are preserved.

## Runtime Truth

The local parser, Tesseract executable, geometry layout, OpenCV visual cues, local forensics, Isolation Forest, local hashing retrieval, local vector index, local graph projection, and grounded evidence composer are available. Docker is installed but its daemon was not running during the audit.

The current Python runtime is missing `pymongo`, `redis`, `qdrant-client`, `neo4j`, `minio`, `celery`, `paddleocr`, `transformers`, `FlagEmbedding`, and `xgboost`. No project checkpoint was found for LayoutLMv3, Donut, DINOv2, YOLO, DeBERTa, SegFormer, or XGBoost. These target models and external stores must remain unavailable until a smoke inference or connectivity check succeeds.

## Directory Map

```text
backend/app/api/            FastAPI auth, finance, Vision, and top-level alias routes
backend/app/models/         Pydantic API and persisted Vision contracts
backend/app/services/       Legacy services and the connected Vision pipeline
backend/app/services/vision adapters, pipeline, persistence, retrieval, reports
backend/app/workers/        Celery application and Vision processing task
backend/app/tests/          Vision/API/infrastructure tests
backend/tests/              legacy workflow and service tests
frontend/src/api/           authenticated HTTP client
frontend/src/components/    shell, document canvas, feedback, status controls
frontend/src/pages/         operational product pages
infra/docker/               baseline PostgreSQL schema
infra/migrations/           versioned remediation migrations
ml/                         annotation, preparation, training, and evaluation code
dataset/                    synthetic cases and labels
scripts/                    bootstrap and demo utilities
var/                        ignored uploads, artifacts, state, reviews, reports
```

## Mandatory Backlog Mapping

| # | Defect | Current fault | Primary repair surface |
| --- | --- | --- | --- |
| 1 | Zero fields | Regex-only extraction lacks a schema registry and spatial association | `schema_registry.py`, deterministic extractor, `service.py`, viewer |
| 2 | Zero confidence | No-field means are represented as `0.0` | `analytics.py`, Vision models, dashboard/library/viewer |
| 3 | Unsafe approval | API records approval without policy checks | approval policy, Vision routes, review workspace |
| 4 | Unresolved count | UI counts only returned fields | schema completeness service and review APIs/UI |
| 5 | Dashboard mismatch | Independent frontend calculations use different denominators | shared analytics service and consumers |
| 6 | Unreadable viewer | Processed image and several overlays default on | page variants, canvas controls, overlay tests |
| 7 | Template Ask | Router has no greeting, help, ambiguity, or risk classes | query router, Ask service/UI, citation navigation |
| 8 | Placeholder graph | Graph is field-limited and UI lacks evidence interactions | graph builder/routes/UI |
| 9 | Language false positives | Any non-ASCII character triggers mixed-language routing | cleaned language analysis and routing evidence |
| 10 | Weak risk detail | Processing warnings are counted as risk | separate warning collection and normalized risk schema |
| 11 | Fraud safeguards | Ordinary reviewers can call confirmation with a comment only | admin-only action, reauthentication, evidence gate, audit |
| 12 | Duplicate artefacts | Narrative/report generation always creates a new ID | identity hashes, versioning, unique indexes, UI |
| 13 | Queue readability | Review tasks expose document IDs without filename context | enriched review task/API and queue filters |
| 14 | Analytics context | Charts omit legends, denominators, dates, and consistency | analytics aggregation endpoints and frontend charts |

## Broken Or Missing API Contracts

The initial API omitted dedicated OCR, layout, missing-field, visual-element, and warning endpoints; graph relationship verification; model smoke-test and health endpoints; and several requested analytics views. Review decision bodies were too small to carry approval overrides or fraud evidence. These contracts are included in the remediation plan and integration tests.

## Data Model Inconsistencies

- Missing confidence and true zero confidence were conflated.
- Expected schema fields were not persisted or returned.
- Processing quality issues and business/visual risks shared one list.
- Report and narrative resources lacked identity hash, version, and source snapshot fields.
- Admin model toggles stored requested intent separately in memory but did not expose runtime health or smoke evidence.
- Review rows lacked filename, page count, confidence status, assignee, and schema counts.

## Preserved Functionality

Upload validation and storage, local parsing, Tesseract/native-text OCR, geometry layout, deterministic finance calculations, evidence chunks, local retrieval, review history, reports, audit records, authentication, tenant checks, feature-gated adapters, and all existing stored documents are preserved.

## Delivered Modification Surface

Backend changes are scoped to Vision models, configuration, registry/routing, persistence, routes, new schema/confidence/approval/language/graph/smoke services, and focused tests. Frontend changes cover the API client/types, dashboard, library, viewer/canvas, review queue/workspace, Ask, graph, analytics, narratives, reports, model lab, admin settings, component tests, and live Playwright acceptance. Infrastructure changes include a transactional migration, fresh-Compose migration mount, secure secret examples, and preparation/smoke/benchmark scripts.

## Final Audit Result

- All fourteen mandatory backlog repairs pass backend, frontend, and relevant live-browser checks.
- Backend regression result: 78 passed. Frontend result: 15 unit/component tests passed, lint passed, production build passed, and live Playwright acceptance passed with no browser errors.
- Runtime model result: 11 passed, 16 truthfully blocked, 0 failed. Requested model intent cannot override verified health.
- Local backend and frontend startup pass. Compose interpolation passes; container execution remains blocked by the stopped Docker daemon.
- LayoutLMv3 preparation remains blocked because the 60 existing text records have no page images, dimensions, or OCR word boxes. No geometry, model output, or metric was fabricated.
