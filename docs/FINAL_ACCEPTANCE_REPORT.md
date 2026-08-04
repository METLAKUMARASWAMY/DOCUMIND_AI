# DocuMind AI Final Acceptance Report

Acceptance date: 2026-08-04

Overall result: **CORE REMEDIATION COMPLETE; EXTERNAL VALIDATION PARTIALLY BLOCKED**.

All fourteen mandatory product defects are repaired and covered by passing automated tests. The local application, browser workflow, model truthfulness controls, and security checks pass. Full Docker runtime execution is blocked because the host Docker daemon is not running. Sixteen optional advanced/external model adapters remain truthfully blocked by missing checkpoints, packages, services, or credentials; the eleven local adapters below completed real smoke inference.

## Backlog Evidence

| # | Status | Files changed | Tests and result | Screenshot or API evidence | Remaining limitation |
| --- | --- | --- | --- | --- | --- |
| 1. Zero fields | COMPLETE | `schema_registry.py`, `deterministic_extraction.py`, `service.py`, viewer pages | `test_supported_form_and_invoice_extract_real_boxed_schema_fields`: PASS | `GET /api/v1/documents/{id}/fields`; [viewer overlays](screenshots/document-viewer-overlays.png) | Deterministic local extraction is the truthful fallback until a project-trained layout checkpoint passes smoke testing. |
| 2. Zero confidence | COMPLETE | `analytics.py`, Vision schemas, dashboard/library/viewer pages | `test_unknown_document_uses_unavailable_confidence_and_schema_message_contract`: PASS | `GET /api/v1/documents/{id}`; [document library](screenshots/document-library.png) | Confidence is unavailable when evidence is absent; no calibrated ML score is invented. |
| 3. Unsafe approval | COMPLETE | `approval_policy.py`, Vision routes/service, `ReviewWorkspacePage.tsx` | Phase 3 approval-policy tests: PASS | `POST /api/v1/reviews/{id}/approve`; [review workspace](screenshots/review-workspace.png) | A manual override remains a privileged human decision and requires recorded evidence. |
| 4. Unresolved count | COMPLETE | `schema_registry.py`, `analytics.py`, service and review UI | Phase 2 and consistency tests: PASS | `GET /api/v1/documents/{id}/missing-fields`; [review queue](screenshots/review-queue.png) | Unknown document types intentionally report that no schema is available. |
| 5. Dashboard mismatch | COMPLETE | `analytics.py`, dashboard/library/queue/analytics pages | `test_dashboard_library_queue_risk_and_report_totals_are_consistent`: PASS | `GET /api/v1/analytics/overview`; [dashboard](screenshots/dashboard-desktop.png) | Counts describe the current tenant snapshot, not a cross-tenant warehouse. |
| 6. Unreadable viewer | COMPLETE | `DocumentCanvas.tsx`, `DocumentViewerPage.tsx`, `styles.css` | component overlay tests and Playwright: PASS | [readable mode](screenshots/document-viewer-readable.png), [mobile](screenshots/document-viewer-mobile.png) | Very large PDFs still depend on browser PDF rendering and the bundled worker. |
| 7. Template Ask | COMPLETE | `query_router.py`, service, Ask page and citation navigation | Phase 5 query tests and Playwright citation journey: PASS | `POST /api/v1/ai/ask`; [grounded answer](screenshots/ask-documents-citations.png) | The local generator is evidence-bound and deliberately declines unsupported answers. |
| 8. Placeholder graph | COMPLETE | `graph_builder.py`, persistence/service/routes, `GraphExplorerPage.tsx` | Phase 6 graph tests and graph projection component tests: PASS | `GET /api/v1/graph/document/{id}`; [knowledge graph](screenshots/knowledge-graph.png) | Local persisted projection is used while Neo4j is disabled. |
| 9. Language false positives | COMPLETE | `language_analysis.py`, preprocessing/service | Phase 7 English-noise and mixed-language tests: PASS | document analysis language evidence | Language routing needs a minimum cleaned character count and secondary-language ratio. |
| 10. Weak risk detail | COMPLETE | `language_analysis.py`, `analytics.py`, service and risk UI contracts | `test_processing_warnings_and_business_risks_have_distinct_counted_contracts`: PASS | `GET /api/v1/documents/{id}/risk` and `/warnings`; [analytics](screenshots/analytics.png) | Risk rules are deterministic unless an optional trained model is verified. |
| 11. Fraud safeguards | COMPLETE | `approval_policy.py`, auth/vision routes, review UI | `test_confirm_fraud_requires_admin_reauthentication_evidence_and_rationale`: PASS | `POST /api/v1/reviews/{id}/confirm-fraud`; durable admin audit | Optional two-reviewer enforcement is controlled by `FRAUD_SECOND_REVIEWER_REQUIRED`. |
| 12. Duplicate artefacts | COMPLETE | `finance_narratives.py`, `reporting.py`, persistence/service/routes and pages | finance narrative/report identity, versioning, and tenant tests: PASS | narratives/reports APIs return existing identity unless a new version is requested | Local JSON persistence is the default; SQL uniqueness is supplied for PostgreSQL deployments. |
| 13. Queue readability | COMPLETE | service, Vision schemas, `ReviewQueuePage.tsx` | `test_review_queue_returns_readable_metadata_filters_and_pagination`: PASS | `GET /api/v1/reviews`; [review queue](screenshots/review-queue.png) | Pagination is local-state backed unless an external repository is enabled. |
| 14. Analytics context | COMPLETE | `analytics.py`, analytics aliases, dashboard and analytics pages | consistency and Phase 9 API tests: PASS | all six analytics endpoints; [analytics](screenshots/analytics.png) | Local measurements are operational summaries, not long-term BI aggregates. |

## 1. Updated Directory Tree

```text
backend/
  app/api/                     authenticated API and exact alias contracts
  app/models/                  typed request, response, and persisted models
  app/services/vision/         extraction, policy, graph, analytics, runtime health
  app/tests/                   integration and remediation regression tests
frontend/
  e2e/                         live Playwright acceptance journey
  src/components/              evidence canvas and controls
  src/pages/                   connected operational workspaces
  src/utils/                   graph and review policy helpers/tests
infra/
  docker/                      PostgreSQL base schema
  migrations/                  versioned remediation migration
  k8s/                         secret-safe deployment manifests
ml/
  annotation/                  LayoutLMv3 word-box contract
  configs/                     training configuration example
  training/                    existing model training entry points
scripts/                       secure bootstrap, smoke, data audit, benchmark
docs/
  screenshots/                 14 live-service acceptance screenshots
  FINAL_ACCEPTANCE_REPORT.md   this report
```

Runtime uploads, state, generated reports, credentials, and browser traces remain under ignored `.env` or `var/` paths.

## 2. Modified Files

The remediation changes the following tracked or newly added source groups. Screenshot files are listed individually in section 18.

```text
.env.example
README.md
docker-compose.yml
backend/.env.example
backend/app/api/auth_routes.py
backend/app/api/vision_alias_routes.py
backend/app/api/vision_routes.py
backend/app/core/config.py
backend/app/models/schemas.py
backend/app/models/vision.py
backend/app/services/auth.py
backend/app/services/model_registry.py
backend/app/services/vision/analytics.py
backend/app/services/vision/anomaly_adapters.py
backend/app/services/vision/approval_policy.py
backend/app/services/vision/checkpoint_validation.py
backend/app/services/vision/contract_adapters.py
backend/app/services/vision/deterministic_extraction.py
backend/app/services/vision/extraction_adapters.py
backend/app/services/vision/finance_narratives.py
backend/app/services/vision/generator_adapters.py
backend/app/services/vision/graph_builder.py
backend/app/services/vision/language_analysis.py
backend/app/services/vision/model_smoke.py
backend/app/services/vision/persistence.py
backend/app/services/vision/preprocessing.py
backend/app/services/vision/query_router.py
backend/app/services/vision/reporting.py
backend/app/services/vision/retrieval_adapters.py
backend/app/services/vision/runtime_health.py
backend/app/services/vision/schema_registry.py
backend/app/services/vision/service.py
backend/app/services/vision/tamper_adapters.py
backend/app/services/vision/visual_adapters.py
backend/app/tests/auth_helpers.py
backend/app/tests/test_model_registry.py
backend/app/tests/test_vision_ai.py
backend/app/tests/test_vision_finance_reports_tenants.py
backend/app/tests/test_vision_remediation_phase2.py
backend/app/tests/test_vision_remediation_phase3.py
backend/app/tests/test_vision_remediation_phase5.py
backend/app/tests/test_vision_remediation_phase6.py
backend/app/tests/test_vision_remediation_phase7.py
backend/app/tests/test_vision_remediation_phase8.py
backend/app/tests/test_vision_remediation_phase9.py
backend/tests/test_auth.py
frontend/e2e/remediation.spec.ts
frontend/playwright.config.ts
frontend/src/api/client.ts
frontend/src/components/DocumentCanvas.test.ts
frontend/src/components/DocumentCanvas.tsx
frontend/src/pages/AdminSettingsPage.tsx
frontend/src/pages/AnalyticsPage.tsx
frontend/src/pages/AskDocumentsPage.tsx
frontend/src/pages/DashboardPage.tsx
frontend/src/pages/DocumentLibraryPage.tsx
frontend/src/pages/DocumentViewerPage.tsx
frontend/src/pages/FinanceNarrativesPage.tsx
frontend/src/pages/GraphExplorerPage.tsx
frontend/src/pages/ModelLabPage.tsx
frontend/src/pages/ReportsPage.tsx
frontend/src/pages/ReviewQueuePage.tsx
frontend/src/pages/ReviewWorkspacePage.tsx
frontend/src/styles.css
frontend/src/types/api.ts
frontend/src/utils/graphProjection.test.ts
frontend/src/utils/graphProjection.ts
frontend/src/utils/reviewPolicy.test.ts
frontend/src/utils/reviewPolicy.ts
frontend/vite.config.js
infra/k8s/postgres.yaml
infra/k8s/secret.example.yaml
infra/migrations/002_vision_remediation.sql
k8s/secret-example.yaml
ml/annotation/layoutlmv3_token_record.schema.json
ml/configs/layoutlmv3_token.example.yml
ml/training/README.md
scripts/benchmark_local_workflow.py
scripts/bootstrap_auth_users.py
scripts/generate_local_env.py
scripts/prepare_layoutlmv3_dataset.py
scripts/seed_demo.py
scripts/seed_vision_demo.py
scripts/smoke_test_models.py
submission/DEPLOYMENT.md
submission/README.md
docs/MODEL_RUNTIME_VERIFICATION.md
docs/REMEDIATION_AUDIT.md
docs/screenshots/README.md
docs/FINAL_ACCEPTANCE_REPORT.md
```

## 3. Database Migration Commands

Fresh Compose databases apply `001-init.sql` and `002-vision-remediation.sql` in order. For an existing Postgres volume, run:

```bash
docker compose exec -T postgres sh -c \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1' \
  < infra/migrations/002_vision_remediation.sql
```

For an independently managed database:

```bash
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 \
  -f infra/migrations/002_vision_remediation.sql
```

The migration is transactional and adds nullable confidence, provenance, completeness, warnings, risk, model runtime state, approval/fraud evidence, idempotent artefact identity/versioning, and supporting indexes. Tenant IDs participate in repository filters and relevant uniqueness keys.

## 4. Dependency Installation Commands

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r backend/requirements.txt

cd frontend
npm ci
npx playwright install chromium
```

Optional adapters must be installed only with their matching checkpoint or service configuration; installing a package alone does not make a model active.

## 5. Environment Variables

Generate ignored local secrets once:

```bash
python3 scripts/generate_local_env.py
```

The complete templates are [`.env.example`](../.env.example) and [`backend/.env.example`](../backend/.env.example). Important groups are:

- Core/security: `APP_ENV`, `DOCUMIND_AUTH_SECRET`, `DOCUMIND_AUTH_STATE_DIR`, `MAX_UPLOAD_BYTES`, `UPLOAD_DIR`.
- Data/services: `DATABASE_URL`, `MONGODB_URI`, `REDIS_URL`, `QDRANT_URL`, `MINIO_ENDPOINT`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `CHROMA_HOST`, `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`.
- Service gates: `ENABLE_CHROMA`, `ENABLE_NEO4J`, `ENABLE_MONGODB`, `ENABLE_REDIS`, `ENABLE_MINIO`, `ENABLE_QDRANT`.
- Model gates: `MODEL_PROFILE`, `ENABLE_HEAVY_MODELS`, `ENABLE_PADDLEOCR`, `ENABLE_PADDLEOCR_STRUCTURE`, `ENABLE_PADDLEOCR_VL`, `ENABLE_LAYOUTLMV3`, `ENABLE_DONUT`, `ENABLE_YOLO26`, `ENABLE_DINOV2`, `ENABLE_BGE_M3`, `ENABLE_QWEN_LOCAL`, `ENABLE_EXTERNAL_LLM`.
- Assets: `LAYOUTLMV3_CHECKPOINT`, `LAYOUTLMV3_CLASSIFICATION_CHECKPOINT`, `DONUT_CHECKPOINT`, `YOLO26_CHECKPOINT`, `DINOV2_CHECKPOINT`, `CONTRACT_CHECKPOINT`, `TAMPER_CHECKPOINT`, `XGBOOST_RISK_CHECKPOINT`, `BGE_M3_CHECKPOINT`, `BGE_RERANKER_CHECKPOINT`, `QWEN_CHECKPOINT`.
- Providers: `EXTERNAL_LLM_BASE_URL`, `EXTERNAL_LLM_API_KEY`, `EXTERNAL_LLM_MODEL`, `BGE_ALLOW_DOWNLOAD`, `QWEN_ALLOW_DOWNLOAD`.
- Routing/policy: OCR, reading-order, and language thresholds; `REVIEW_REAUTHENTICATION_TTL_SECONDS`; `FRAUD_SECOND_REVIEWER_REQUIRED`.
- Verification only: `DOCUMIND_DEMO_EMAIL`, `DOCUMIND_DEMO_PASSWORD`, `DOCUMIND_E2E_EMAIL`, `DOCUMIND_E2E_PASSWORD`, `DOCUMIND_BENCHMARK_EMAIL`, `DOCUMIND_BENCHMARK_PASSWORD`.

No real secret is committed. Compose refuses to interpolate missing service secrets.

## 6. Model Preparation Commands

Audit the existing text-only records:

```bash
.venv/bin/python scripts/prepare_layoutlmv3_dataset.py --audit-only
```

Current result: 60 records inspected, 0 valid layout records. Every record lacks `page_id`, a rendered page image, positive page dimensions, and OCR words with absolute boxes and labels. This is an intentional failing readiness gate.

After annotating records against `ml/annotation/layoutlmv3_token_record.schema.json`:

```bash
.venv/bin/python scripts/prepare_layoutlmv3_dataset.py \
  --annotations /path/to/word-box-annotations.jsonl \
  --output ml/prepared/layoutlmv3

.venv/bin/python -m ml.training.layoutlmv3.train_token_classification \
  --config ml/configs/layoutlmv3_token.example.yml \
  --validate-only

.venv/bin/python -m ml.training.layoutlmv3.train_token_classification \
  --config ml/configs/layoutlmv3_token.example.yml
```

Document-level train/validation/test splitting is mandatory to prevent page leakage.

## 7. Model Checkpoint Requirements

| Adapter | Requirement before activation |
| --- | --- |
| LayoutLMv3 extraction/classification | Project token/layout checkpoint, field labels, adjacent `run_manifest.json`, document-level leakage audit, held-out metrics |
| Donut | Project document checkpoint and manifest matching the configured task |
| DINOv2 | Local image-embedding checkpoint plus compatible `torch`/`transformers` runtime |
| YOLO26 | Project-trained visual-element detector and manifest with held-out metrics |
| DeBERTa CUAD | Fine-tuned contract extraction checkpoint and manifest |
| SegFormer forensics | Fine-tuned document-tamper checkpoint and manifest |
| XGBoost risk | Labelled project risk corpus, trained checkpoint, feature contract, and held-out metrics |
| PaddleOCR variants | Compatible PaddleOCR/PaddlePaddle runtime and explicit enable flags |
| BGE-M3/reranker | `FlagEmbedding` plus local assets, or explicit download consent |
| Qwen | `transformers` plus local model assets, or explicit download consent |
| Qdrant/Neo4j | Enabled service, client package, credentials where needed, connectivity, and successful probe |
| External generator | Explicit base URL, API key, model ID, and successful evidence-bound smoke response |

Checkpoint validators reject generic base models for project extraction tasks and invalidate prior smoke evidence when configuration changes.

## 8. Local Run Commands

Backend shell:

```bash
python3 scripts/generate_local_env.py  # omit when ignored .env files already exist
python3 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
cd backend
../.venv/bin/uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Frontend shell:

```bash
cd frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5174
```

Live local URLs validated during acceptance: `http://127.0.0.1:8000/health/live` and `http://127.0.0.1:5174`.

## 9. Docker Compose Commands

```bash
python3 scripts/generate_local_env.py  # first run only
docker compose config --quiet
docker compose up --build
docker compose ps
docker compose down
```

`docker compose config --quiet` passes. Container execution is **BLOCKED in this environment** because the Docker daemon is unavailable; no container runtime success is claimed.

## 10. Test Commands

```bash
cd backend
../.venv/bin/ruff check app tests ../scripts
../.venv/bin/pytest -q
../.venv/bin/pip check

cd ../frontend
npm test -- --run
npm run lint
npm run build
npm audit

DOCUMIND_E2E_EMAIL="$DOCUMIND_DEMO_EMAIL" \
DOCUMIND_E2E_PASSWORD="$DOCUMIND_DEMO_PASSWORD" \
npx playwright test

cd ..
.venv/bin/python scripts/smoke_test_models.py
git diff --check
docker compose config --quiet
```

Use a securely bootstrapped credential in the environment for Playwright; do not place it in source or shell scripts.

## 11. Test Results

| Gate | Result |
| --- | --- |
| Backend Pytest | PASS: 78 tests, 36 deprecation warnings, 53.43 s |
| Backend Ruff | PASS |
| Frontend Vitest | PASS: 4 files, 15 tests |
| Frontend ESLint | PASS: zero warnings |
| Frontend production build | PASS: 3,191 modules, 3.55 s |
| Playwright live-service E2E | PASS: 1 comprehensive desktop/mobile journey, 14 screenshots, zero browser errors |
| Python dependency check | PASS: no broken requirements |
| NPM dependency audit | PASS: 0 vulnerabilities |
| Git whitespace check | PASS |
| Compose interpolation/config | PASS |
| Backend live health | PASS: `{"status":"ok"}` |
| Frontend live response | PASS: HTTP 200 |

Observed local loopback timings, not an SLA or load test: analysis 1,396.882 ms; ten document reads median 1.886 ms and max/p95-nearest-rank 2.583 ms; grounded question 1.982 ms; analytics overview 23.662 ms.

## 12. Model Smoke-Test Results

Command:

```bash
.venv/bin/python scripts/smoke_test_models.py
```

Latest result at `2026-08-04T08:56:10Z`: **27 total, 11 passed, 16 blocked, 0 failed**.

Blocked models and exact current reasons:

- PP-OCRv5 mobile: heavy/PaddleOCR flags are disabled.
- PP-StructureV3 / PP-DocLayoutV3 and PaddleOCR-VL: compatible PaddleOCR runtime and enable flags are absent.
- LayoutLMv3, Donut, YOLO26, DeBERTa, SegFormer, XGBoost: no valid project checkpoint is configured.
- DINOv2: local checkpoint and compatible runtime are absent.
- BGE-M3 and reranker: `FlagEmbedding` and local assets or explicit download consent are absent.
- Qdrant and Neo4j: services are disabled/not connected.
- Qwen: local assets/runtime or explicit download consent are absent.
- External provider: URL, API key, and model are intentionally unconfigured.

Every blocked record includes its preparation command, required environment variables, missing packages/checkpoint state, and `not_run` inference status. There are no fabricated outputs.

## 13. Active Model Registry With Runtime Evidence

Only these adapters are `active` because initialization, one real inference, and output-schema validation passed for the current configuration fingerprint:

| Active adapter | Runtime evidence |
| --- | --- |
| `local-parser-v1` | parsed the smoke document into the page contract |
| `Tesseract-5` | executable OCR inference returned word/page output |
| `local-layout-adapter-v1` | geometry inference returned validated layout regions |
| `deterministic-schema-extractor-v2` | returned boxed schema fields with provenance |
| `OpenCV-visual-cues-v1` | returned validated visual-element output |
| `ELA-noise-forensics-v1` | returned local forensics analysis |
| `IsolationForest` | executed real inference and correctly returned the minimum-corpus gate |
| `local-hashing-v1` | returned finite 768-dimensional dense and sparse features |
| `local-vector-store-v1` | indexed and retrieved the smoke evidence chunk |
| `local-graph-projection-v1` | projected 12 nodes and 14 source-linked edges |
| `grounded-template-generator-v1` | produced a supported answer with an evidence citation |

Admin Settings exposes requested intent separately from verified runtime health. A toggle cannot manufacture `active` status.

## 14. API Endpoint Summary

All required contracts are present under `/api/v1`; `/docs` and `/openapi.json` expose the typed schema.

- Documents: upload, list, detail, delete, reprocess, status/events, pages/image/source, OCR, layout, fields, missing fields, tables, visual elements, warnings, risk, and related documents.
- Reviews: list/detail, field correction, high-confidence acceptance, approve, reject, escalate, confirm error, and evidence-gated confirm fraud.
- Ask: `/ai/ask`, response retrieval, feedback, and compatibility `/ask` routes.
- Graph: document/entity projection, query, and relationship verification.
- Models: registry, health, smoke test, requested-state mutation, comparison, rules, and durable admin audit.
- Analytics: overview, documents, risk, reviews, models, RAG, and vendor views.
- Artefacts: narrative generate/list/detail and report generate/list/detail/download/archive.
- Auth: register, login, refresh rotation, current user, reauthenticate, logout, and generic forgot-password response.

Phase 9 integration tests verify the exact required paths, role and tenant checks, service execution, and durable mutation/download audit events.

## 15. Demo Credentials Through a Secure Seed Script

No privileged default user or static password exists. Create a credential and record its one-time output in a password manager:

```bash
.venv/bin/python scripts/bootstrap_auth_users.py \
  --role ADMIN \
  --email admin@documind.local
```

To supply a password without a command-line argument, set `DOCUMIND_BOOTSTRAP_PASSWORD` in the process environment. Public signup always creates a viewer. The historical demo admin password is tested as invalid in a fresh auth state.

## 16. Demo Dataset Preparation

With the backend running and the generated admin credential held only in the environment:

```bash
export DOCUMIND_DEMO_EMAIL=admin@documind.local
export DOCUMIND_DEMO_PASSWORD='<password-manager value>'
.venv/bin/python scripts/seed_vision_demo.py
unset DOCUMIND_DEMO_PASSWORD
```

The live E2E setup also calls `POST /api/v1/demo/vision/seed`, then analyzes a uniquely named invoice through the real API. The repository's synthetic source data stays immutable; tenant-specific runtime records are written under ignored `var/` state.

## 17. Complete Demo Walkthrough

1. Start the backend and frontend from section 8, then sign in with the securely bootstrapped admin.
2. Open Dashboard and compare its document, confidence, unresolved, risk, and review totals with the Library and Analytics pages.
3. Open `documind-demo-invoice.txt`; switch Original, Enhanced, and Forensic modes, then toggle fields, OCR, layout, visual, and risk overlays independently.
4. Select a field and confirm the page scrolls to its exact evidence box. Clear overlays and reset zoom to restore a readable page.
5. Open the review queue, filter by type/priority/reason, and open a filename-based task. Attempt unsafe approval to see the policy requirements; use manual override only with the required comment, confirmations, and evidence.
6. Ask `hi`, an ambiguous question, `What is the total amount?`, and an absent-field question. Confirm greeting, clarification, cited support, and `not_found` behavior. Click the citation to reopen the source region.
7. Open Knowledge Graph, select the invoice, inspect typed entities/edges and source evidence, then verify only a proposed relationship with evidence.
8. Open Analytics and compare displayed denominators, date context, warning/risk separation, model status, and citation coverage.
9. Generate a finance narrative and report twice to observe identity reuse; request a new version to increment it, then download the report.
10. Open Model Lab and Admin Settings. Compare only real outputs; run smoke tests and observe unavailable adapters remain blocked despite requested-state changes.

## 18. Screenshots of Repaired Modules

- [Dashboard desktop](screenshots/dashboard-desktop.png)
- [Dashboard mobile](screenshots/dashboard-mobile.png)
- [Document library](screenshots/document-library.png)
- [Readable document viewer](screenshots/document-viewer-readable.png)
- [Independent document overlays](screenshots/document-viewer-overlays.png)
- [Document viewer mobile](screenshots/document-viewer-mobile.png)
- [Review queue](screenshots/review-queue.png)
- [Review workspace](screenshots/review-workspace.png)
- [Ask Documents citations](screenshots/ask-documents-citations.png)
- [Citation source navigation](screenshots/citation-source-navigation.png)
- [Knowledge graph](screenshots/knowledge-graph.png)
- [Analytics](screenshots/analytics.png)
- [Model laboratory](screenshots/model-laboratory.png)
- [Admin model runtime](screenshots/admin-model-runtime.png)

All 14 PNGs were captured by Playwright against the live local FastAPI service with no mocked API responses. The [screenshot manifest](screenshots/README.md) records how to reproduce them.

## 19. Remaining Genuine Limitations

- **BLOCKED - Docker runtime:** the host Docker daemon is unavailable. Compose configuration passes and fresh-schema mounts are wired, but containers and the SQL migration were not executed here.
- **BLOCKED - LayoutLMv3 data/checkpoint:** 60 text records contain no rendered images or word boxes, so training is correctly refused. The annotation schema, audit/preparation tool, config, validation, and training commands are ready.
- **BLOCKED - advanced models:** 16 adapters lack genuine checkpoints, packages, services, or credentials. They return no fake outputs and cannot become active without a passing smoke fingerprint.
- **Local persistence:** JSON/local-file repositories are the validated default. PostgreSQL, MongoDB, MinIO, Qdrant, and Neo4j production paths need runtime integration testing once their services are available.
- **Performance:** the benchmark is single-process loopback only. The production build includes a roughly 1.38 MB PDF worker and larger graph/chart chunks; production profiling and load tests remain deployment work.
- **Warnings:** tests emit one Starlette/httpx compatibility warning and Pillow `getdata` deprecation warnings; no warning caused a behavioral failure.

## 20. Final Acceptance Checklist

- [x] COMPLETE - Supported readable documents produce structured fields.
- [x] COMPLETE - No-field documents display confidence as unavailable, not fake zero.
- [x] COMPLETE - No-evidence approval is blocked.
- [x] COMPLETE - Unresolved-field count is schema-aware.
- [x] COMPLETE - Dashboard confidence matches document data.
- [x] COMPLETE - Document viewer is readable.
- [x] COMPLETE - Overlays are independent and accurate.
- [x] COMPLETE - Ask Documents handles greetings.
- [x] COMPLETE - Ask Documents requests clarification for ambiguous questions.
- [x] COMPLETE - Supported answers contain citations.
- [x] COMPLETE - Citation click opens source evidence.
- [x] COMPLETE - Knowledge graph contains real entities and edges.
- [x] COMPLETE - Language routing does not trigger on OCR noise.
- [x] COMPLETE - Processing warnings are separated from risk indicators.
- [x] COMPLETE - Risk indicators contain evidence and recommendations.
- [x] COMPLETE - Confirm Fraud is role-restricted and evidence-gated.
- [x] COMPLETE - Duplicate narratives are prevented.
- [x] COMPLETE - Duplicate reports are prevented.
- [x] COMPLETE - Review queue displays filenames and readable reasons.
- [x] COMPLETE - Analytics include legends and denominators.
- [x] COMPLETE - Dashboard, library, queue, and analytics counts agree.
- [x] COMPLETE - Model active status is based on runtime verification.
- [x] COMPLETE - No fake model outputs exist.
- [x] COMPLETE - No fake metrics exist.
- [x] COMPLETE - Visible controls are functional or clearly disabled.
- [x] COMPLETE - Browser E2E and API integration journeys pass.
- [ ] BLOCKED - Docker/local startup instructions: local startup passes and Compose config passes, but Docker execution cannot be proven until the daemon runs.
- [x] COMPLETE - Existing working features remain functional under the full regression suite.

Acceptance tally: **27 complete, 1 externally blocked**. This report does not claim full Docker or advanced-model completion.
