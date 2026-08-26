# DocuMind Vision AI — final release verification

Verification date: 2026-08-13 (Asia/Kolkata)  
Repository: `/Users/kumar/Downloads/DocuMind AI`  
Branch: `main`

This report is the current implementation and runtime source of truth. It
supersedes runtime-status claims in older August 2026 reports. The remediation
goal is complete for the verified local architecture and fallbacks described
below; optional models without legitimate checkpoints, labels, or safe host
resources remain explicitly blocked rather than being represented as active.

## 1. Repository changes made

The existing FastAPI/React application was strengthened in place.

- Backend API/contracts: `backend/app/api/vision_alias_routes.py`,
  `backend/app/api/vision_routes.py`, and `backend/app/models/vision.py`.
- Authentication/runtime truth: `backend/app/services/auth.py` and
  `backend/app/services/model_registry.py`.
- OCR, layout, classification, and extraction:
  `backend/app/services/vision/ocr_adapters.py`, `ocr_lines.py`,
  `layout_adapters.py`, `document_classification.py`, and
  `deterministic_extraction.py`.
- Open-set semantics: new `generic_field_discovery.py`, `semantic_values.py`,
  `semantic_facts.py`, `semantic_risk.py`, `entity_resolution.py`, and
  `reviewer_feedback.py`.
- Configurable schemas: new `schema_config.py`, `schemas/__init__.py`, and
  `schemas/catalog.v1.json`; refactored `schema_registry.py`; package data in
  `backend/pyproject.toml`.
- Retrieval, GraphRAG, persistence, conversations, finance, reporting, and
  orchestration: `retrieval_adapters.py`, `graph_builder.py`, `persistence.py`,
  `query_router.py`, `finance_narratives.py`, `reporting.py`, and `service.py`.
- Durable conversation/evidence migration:
  `infra/migrations/004_conversation_evidence_state.sql`.
- Runtime/deployment/security: `.env.example`, `.github/workflows/ci.yml`,
  `.gitignore`, `docker-compose.yml`, `backend/requirements/base.txt`,
  `backend/requirements/ai-core.txt`, `scripts/bootstrap_ai_core.sh`,
  `scripts/generate_local_env.py`, and `scripts/migrate_postgres.py`.
- Frontend intelligence/evidence/calculation UI:
  `frontend/src/components/GenericIntelligencePanel.tsx`,
  `DocumentCanvas.tsx`, `frontend/src/pages/AskDocumentsPage.tsx`,
  `DocumentViewerPage.tsx`, `ReviewWorkspacePage.tsx`, `frontend/src/api/client.ts`,
  `frontend/src/types/api.ts`, and `frontend/src/styles.css`.
- Frontend verification/contracts:
  `frontend/e2e/core-intelligence-v4.spec.ts`, `e2e/remediation.spec.ts`,
  `playwright.config.ts`, `package.json`, `package-lock.json`,
  `src/components/DocumentCanvas.test.ts`,
  `src/components/GenericIntelligencePanel.test.tsx`, and
  `src/pages/AskDocumentsPage.test.tsx`.
- Backend tests changed or added:
  `backend/app/tests/test_ai_core_runtime_v4.py`,
  `test_ask_deterministic_calculations.py`,
  `test_deterministic_extraction_v3_fixture.py`,
  `test_document_variation_acceptance.py`,
  `test_finance_duplicate_invoice_exclusion.py`,
  `test_generalization_regressions.py`, `test_geometry_layout_tables.py`,
  `test_postgres_integration_v4.py`, `test_schema_configuration_v1.py`,
  `test_semantic_risk_feedback_resolution.py`,
  `test_vision_finance_reports_tenants.py`,
  `test_vision_postgresql_persistence_v4.py`,
  `test_vision_rag_graphrag_v4.py`, `test_vision_remediation_phase2.py`,
  `test_vision_remediation_phase6.py`, and
  `test_vision_repository_worker.py`, plus `backend/tests/test_auth.py` and
  `backend/tests/test_generic_field_discovery.py`.
- Stable regression images are
  `backend/var/test_fixtures/00093726.png` (SHA-256
  `be0934b66626af43bc8e1826d2aa1858a1efd59019209e2af048217902cb3654`)
  and `87093315_87093318.png` (SHA-256
  `3522b6f82fd8af414dabc099b286e5f4d956d698e9bbf8d21fb2f28ee8219916`).
- Nineteen browser-evidence PNGs under `docs/screenshots/` were refreshed:
  `admin-model-runtime.png`, `analytics.png`, `ask-documents-citations.png`,
  `citation-source-navigation.png`, `core-v4-00093726-fields.png`,
  `core-v4-admin-models.png`, `core-v4-analytics.png`,
  `core-v4-ask-clarification.png`, `core-v4-graphrag-path.png`,
  `dashboard-desktop.png`, `dashboard-mobile.png`, `document-library.png`,
  `document-viewer-mobile.png`, `document-viewer-overlays.png`,
  `document-viewer-readable.png`, `knowledge-graph.png`,
  `model-laboratory.png`, `review-queue.png`, and `review-workspace.png`.
  `README.md`, the prior bounded handoff
  `docs/REMAINING_WORK_2026-08-12.md`, and
  `docs/FINAL_RELEASE_VERIFICATION_2026-08-13.md` document setup and
  verification boundaries.

At the final report synchronization checkpoint, `git status --short
--untracked-files=all` showed 100 paths: 73 tracked modifications and 27
untracked paths. Four are protected user-owned inputs—`new.txt`, `new2.txt`,
`MERGE.TXT`, and `docs/Big Monster.txt`—and were not edited by this completion
work. Runtime uploads, credentials, indexes, and temporary acceptance state are
not source artefacts and must not be committed.

## 2. Bugs fixed

- Calendar dates such as `3/14/90`, `2/15/90`, and ISO dates cannot validate as
  PHONE values; legitimate slash-separated fax values such as `910/535-7707`
  remain valid and normalize correctly.
- Filename semantics no longer influence classification. Weak or ambiguous
  evidence abstains to unknown/mode C, while a real statement heading still
  selects the registered statement schema.
- The same evidence cannot silently populate incompatible semantic fields.
- OCR line reconstruction rejects damaging tall-box/adjacent-row merges, and
  multi-column key/value lines retain independent block/line provenance.
- The `00093726.png` date is not fabricated into duplicate document/form title
  fields; low-quality fragments are rejected with diagnostics.
- Review completeness uses one backend contract for expected, matched, missing,
  optional, total, disputed, and duplicate-rejected counts.
- Model smoke evidence expires and configuration/endpoint changes invalidate
  stale ACTIVE state.
- API and Celery worker refresh shared durable state rather than returning a
  pre-worker cached document.
- Structured-extraction failure preserves OCR chunks, retrieval, and grounded
  Ask while recording the failed extraction stage and fallback.
- Reprocessing invalidates dependent narratives/reports, including outputs that
  cite an excluded duplicate source, while retaining history.
- Duplicate invoice selection is vendor-scoped, approved-first, complete-field
  aware, deterministic, and conflict-safe for every aggregate input.
- Narrative “New version” round-trips the original duplicate-inclusive document
  scope instead of losing excluded-source lineage.
- Document reports include OCR/layout, graph, and limitations payloads.
- Successful document deletion is audited only after persistence deletion
  succeeds; a forced failure cannot emit a false success event.

## 3. Semantic extraction improvements

- The pipeline supports schema mode A (known schema), B (known schema plus new
  fields), and C (generic/inferred schema).
- A strict, versioned JSON catalog defines 18 schemas, including two generic
  profiles, and 23 reusable catalog value-kind definitions mapped onto the 19
  semantic entity types below. It declares fields, aliases, required status,
  classification hints, validation/normalisation, modules, and graph semantics.
  Invalid or incompatible configuration fails explicitly.
- Generic discovery supports colon, delimiter, whitespace-gap,
  label-above-value, label-left-of-value, and aligned-geometry strategies while
  retaining raw labels/values, page, boxes, word IDs, confidence, strategy, and
  review state.
- The semantic registry contains 19 exact kinds: DATE, DATETIME, PERSON,
  ORGANISATION, MONEY, NUMBER, QUANTITY, PERCENTAGE, PHONE, EMAIL, ADDRESS,
  BANK_ACCOUNT, IDENTIFIER, DOCUMENT_REFERENCE, CURRENCY, PRODUCT, TEXT,
  BOOLEAN, and URL.
- Context-aware validation uses label, surrounding text, document type, and
  section. Uncertain mappings remain generic/reviewable rather than being
  forced.
- OCR words/lines, layout regions, generic fields, typed entities, registered
  fields, tables, and verified facts remain distinct evidence layers.
- Reviewer corrections are append-only, tenant-scoped, integrity-checked, and
  preserve the original extraction, corrected value, context, evidence,
  relationship, reviewer, and document type. Conflicts abstain; no silent
  production retraining occurs.
- BGE similarity is available for conservative generic-label mapping when the
  real embedding adapter is active; ambiguity/margin guardrails prevent forced
  matches.
- Generalisation evidence covers renamed inputs, alias permutations, generated
  date/phone invariants, rotated PNG, blurred/low-contrast/compressed JPEG,
  two-page TIFF, two-page scanned PDF, multi-column key/value geometry, and an
  untouched ten-page research-paper holdout that completed the full mode-C
  service pipeline.

## 4. Regression test results

`87093315_87093318.png` was reprocessed as
`vision-doc-dceb6f12e3` and remains a mode-A `business_form` with eight
source-backed fields, 71 evidence chunks, and an 18-node/25-edge graph:

| Field | Raw value | Normalized value |
| --- | --- | --- |
| `sample_number` | `1194-90` | `1194-90` |
| `date` | `3/14/90` | `1990-03-14` |
| `product_type` | `100 mm Filter` | product text |
| `batch_size` | `47.5 lbs.` | quantity preserved |
| `requested_by` | `J.H. Bell` | person preserved |
| `request_date` | `2/15/90` | `1990-02-15` |
| `purpose` | `Cigarette Modification B-451` | text preserved |
| `specifications_written_by` | `C.W. Lassiter` | person preserved |

Every field has source evidence; neither date becomes PHONE, and classification
is unchanged when the source is renamed.

`00093726.png` was reprocessed as `vision-doc-35d6390941` and remains a mode-A
`business_form` with seven legitimate fields, 93 chunks, and a 19-node/26-edge
graph: `date`, `sample_number`, `product_type`, `batch_size`, `requested_by`,
optional `request_date`, and `specifications_written_by`. The completeness
contract is 6 required expected, 6 matched, 0 missing, 1 optional, and 7 total;
the three observed OCR/geometry fragments do not become fields.

Prior snapshots/reviews were archived at
`backend/var/vision_state/archive/reprocess/20260813T043511.923827Z/`. Current
durable data versions are 4 and 9 respectively; versions are read from state,
not inferred from filenames.

## 5. OCR/layout status

| Runtime | Status | Verified boundary |
| --- | --- | --- |
| Tesseract 5 | **ACTIVE** | Real local OCR text, confidence, word IDs, and geometry smoke passed. |
| Adaptive OCR consensus | **ACTIVE** | Original/processed/grayscale/enhanced candidates are scored with stored provenance. |
| PP-OCRv5 mobile | **ACTIVE** | Cached CPU adapter performed real OCR with geometry in the opt-in AI profile. |
| Local geometry layout/table adapter | **ACTIVE** | Aligned/literal tables, prose guard, sparse/merged/empty-cell geometry, and multi-column key/value tests passed. |
| PP-StructureV3 / PP-DocLayoutV3 | **ACTIVE** | Cached CPU adapter performed real layout inference in the opt-in AI profile. |
| PaddleOCR-VL 1.6 | **DISABLED** | No current deployment/smoke path is enabled; baseline OCR/layout remains operational. |

The robust variation matrix passed for rotated scans, low
contrast/blur/compression, PNG/JPEG/TIFF, scanned multi-page PDFs, and a genuine
ten-page PDF. Sparse or merged geometry preserves blank cells and exposes
`table_uncertain` plus machine-readable uncertainty reasons for human review;
confident aligned tables expose `structured`. Handwriting and multilingual
accuracy are not claimed because no labelled evaluation corpus was supplied.
Complex real merged-cell tables remain a corpus-level evaluation limitation;
the implementation does not manufacture confident structure when geometry is
insufficient.

## 6. Model activation status

ACTIVE means the named implementation initialized, executed real inference or
its real external lifecycle, and passed output-schema validation in the stated
profile. An opt-in cached model is not claimed to be loaded by every default
Uvicorn process.

| Model/runtime | Status | Reason |
| --- | --- | --- |
| `local-parser-v1` | **ACTIVE** | Fresh ingestion smoke passed. |
| Tesseract 5 | **ACTIVE** | Fresh OCR smoke passed. |
| `local-layout-adapter-v1` | **ACTIVE** | Fresh layout smoke passed. |
| PP-OCRv5 mobile | **ACTIVE** | Cached real CPU inference passed. |
| PP-StructureV3 / PP-DocLayoutV3 | **ACTIVE** | Cached real CPU inference passed. |
| `deterministic-schema-extractor-v3` | **ACTIVE** | Fresh source-backed extraction smoke passed. |
| OpenCV visual cues | **ACTIVE** | Fresh visual-cue smoke passed. |
| ELA/noise forensics | **ACTIVE** | Fresh measured-forensics smoke passed. |
| Isolation Forest | **ACTIVE** | Fresh anomaly smoke passed. |
| `local-hashing-v1` | **ACTIVE** | Fresh embedding smoke passed and remains the local fallback. |
| BAAI BGE-M3 | **ACTIVE** | Cached real 1024-dimensional dense+sparse embedding passed. |
| BGE reranker v2-m3 | **ACTIVE** | Cached exact real reranking passed. |
| `local-vector-store-v1` | **ACTIVE** | Fresh storage/retrieval lifecycle passed. |
| Qdrant | **ACTIVE** | Live vector/filter/replacement/deletion lifecycle passed. |
| `local-graph-projection-v1` | **ACTIVE** | Fresh graph projection passed. |
| Neo4j | **ACTIVE** | Live authenticated query/path/lifecycle passed. |
| Grounded template generator | **ACTIVE** | Fresh evidence-only generation passed. |
| LayoutLMv3 extraction/classification | **BLOCKED_BY_LABELS** | No project-labelled token dataset, compatible label map, or fine-tuned task checkpoint. |
| Donut | **BLOCKED_BY_CHECKPOINT** | No suitable project task checkpoint. |
| YOLO visual detector | **BLOCKED_BY_CHECKPOINT** | No project detector weights/labels. |
| DINOv2 visual similarity | **BLOCKED_BY_CHECKPOINT** | No approved local project asset. |
| DeBERTa CUAD | **BLOCKED_BY_CHECKPOINT** | No project clause checkpoint/integration asset. |
| SegFormer forensics | **BLOCKED_BY_CHECKPOINT** | No trained project tamper checkpoint. |
| XGBoost supervised risk | **BLOCKED_BY_LABELS** | No labelled project dataset/trained risk model. |
| Qwen 3.5 9B | **BLOCKED_BY_RESOURCES** | No local checkpoint; downloading/running it is unsafe on the current 16-GiB, disk-constrained host. |
| External LLM provider | **DISABLED** | No approved endpoint/model/credential was supplied. |

Final strict artefacts are `var/ai-core/verification-2026-08-13-final.json`
(13/13) and `verification-2026-08-13-heavy-final.json` (17/17 cached opt-in
profile), both with zero failed or blocked enabled targets.

## 7. Qdrant/BGE retrieval status

- Qdrant 1.18 is **ACTIVE**, healthy, authenticated where configured, and bound
  to loopback ports 6333/6334.
- BGE-M3 produced 1024-dimensional dense vectors and sparse terms; the exact
  BGE reranker remained compatible after Transformers 5.5.4.
- Collections encode embedding identity/dimension, including
  `vision_evidence_chunks__bge_m3__d1024_v1`, preventing silent dimension
  mixing with `local_hashing_v1/d768` data.
- A final isolated acceptance used the real dataset source
  `dataset/documind_ai_synthetic_dataset/documents/case-0003/invoice.txt` as
  `vision-doc-90ea79ff84`: 40 chunks, 1024 dense dimensions, and sparse vectors
  on all 40 chunks. A semantic Ask response was supported with five citations
  and a reranker trace.
- Tenant/document scoping, exact replacement, and scoped deletion were
  exercised; deletion left no matching hits and `external_error` was null.
- Generic fields, typed entities, verified facts, tables, OCR lines, and layout
  evidence enter the retrieval corpus. Local hashing/vector retrieval remains
  **FALLBACK_ACTIVE** if BGE or Qdrant is deliberately disabled/unavailable.
- The isolated runtime root was destroyed after verification; no test points,
  credentials, or temporary state were retained.

## 8. Neo4j/GraphRAG status

- Docker Neo4j 5.22 community is **ACTIVE**, healthy, authenticated, and bound
  to loopback ports 7474/7687.
- Generic facts project as
  `Document-[:HAS_ATTRIBUTE]->Attribute`; recognized high-confidence facts are
  promoted to bounded semantic relationships while retaining tenant, document,
  evidence, and processing provenance.
- Entity resolution is conservative: strong identifiers/reviewer-approved
  aliases may resolve across documents; similar names alone remain proposed or
  document-scoped.
- Two bounded regression rebuild passes were identical: 18 nodes/25 edges for
  `87093315_87093318.png` and 19 nodes/26 edges for `00093726.png`, with zero
  duplicate node or edge IDs.
- The final isolated acceptance returned two live `HAS_ATTRIBUTE` paths. Ask
  used `neo4j-evidence-path-v1`, exposed `HAS_ATTRIBUTE`, preserved tenant
  isolation, and cleaned its tenant nodes from 26 to 0 afterward.
- A live relationship path also supports
  `Invoice-[:REFERENCES]->PurchaseOrder` where the evidence contains a verified
  PO reference. Historical unkeyed local nodes are legacy data, not evidence
  for the bounded current projection and were not destructively rewritten.

## 9. Ask Documents/RAG status

- Retrieval prioritizes reviewed facts, high-confidence registered/generic
  fields, tables, exact identifiers, OCR lines, layout regions, vector results,
  graph paths, and visual/risk evidence.
- Explicit scope never silently broadens. Tenant, user, conversation,
  clarification, and citation state persist across service reloads.
- “Is this safe?” correctly asks for clarification. Selecting suspicious
  indicators and following with “Why?”, “Show the evidence”, or “What page?”
  reuses the original context and citations.
- Unsupported or absent fields return `not_found`/insufficient evidence. In the
  real invoice acceptance, missing bank evidence returned `not_found` rather
  than being inferred.
- A forced schema-and-generic extraction failure retained OCR chunks and
  answered a source-backed OCR question, proving structured extraction failure
  does not imply RAG failure.
- Deterministic Ask calculations now support Decimal sums, verified-invoice
  counts, linked invoice/PO differences, percentage variance, and date
  intervals. Responses carry typed operation/input/result traces and evidence
  citations; ambiguity, missing/contradictory currency, and zero-denominator
  guardrails abstain rather than guess. Six focused backend and two focused UI
  tests passed.
- In final isolated acceptance, all 10/10 persisted Ask citations resolved to
  existing evidence after a fresh service reload. A repository invariant also
  reloads and resolves structured, GraphRAG, semantic, and OCR-fallback
  citations from persisted question state. Analytics independently measured
  8/8 eligible supported answers with valid citations (coverage 1.0).
- Definitive post-freeze production-preview run `20260813121920-28415` passed
  both Playwright journeys in 30.3 seconds (core 4.0 seconds; remediation 9.0
  seconds; 31-second wall time) against a fresh FastAPI process, solo Celery
  worker, and production Vite preview. `00093726.png` became
  `vision-doc-6466952166`, REVIEW_REQUIRED with exactly seven fields and
  `data_version=1`; Celery job `vision-job-8fb55504e3` completed in 7.839
  seconds with `errors=[]`. Auth, upload, cited Ask/clarification/GraphRAG,
  analytics, reports, review, evidence overlays, responsive routes, and model
  presentation passed.

## 10. Risk and review status

- Automated output says suspicious indicator, inconsistency, or human
  verification required; it does not make a final fraud determination.
- Generic semantic facts drive account mismatch, date-sequence, amount
  arithmetic, and tenant-scoped duplicate-identifier rules with citations.
  Ambiguous dates abstain and account values are masked in explanations/exports.
- Required-field counts and approval blockers are backend-calculated.
  Approval requires appropriate role, comment, and evidence.
- “Confirm fraud” remains human/admin-only, reauthentication/evidence/rationale
  gated, tenant-scoped, and audited.
- Feedback persists across service restarts/reprocessing, conflicting feedback
  abstains, and no automatic retraining occurs.
- The JSONL feedback store is appropriate for this single-host capstone; a
  multi-host deployment should move it to a shared transactional store.

## 11. Finance/report status

- Finance narratives use only approved/completed documents and reviewed values,
  perform monetary arithmetic with `Decimal`, group currency explicitly, retain
  source calculations/citations, and mask sensitive account values.
- Invoice de-duplication requires a verified canonical invoice number and
  verified vendor identity. It is applied only to aggregate narratives,
  prefers an approved/more-complete source deterministically, cites every
  excluded source, and records contributor/source/excluded IDs.
- Conflicts in total, subtotal, tax, outstanding amount, currency, invoice date,
  PO reference, payment status, or status fail closed for review.
- Duplicate-inclusive request scope survives the “New version” Pydantic/API
  round trip. Reprocessing either a contributor or excluded source marks its
  narrative/report stale while retaining the prior version for audit.
- Backend JSON/Markdown/PDF reports use deterministic identities for the same
  source snapshot and include document summary, classification, fields/review,
  OCR/layout, tables, risks, graph relationships, model/runtime trace,
  limitations, evidence, and audit history.
- Final acceptance used the real current-dataset file
  `dataset/documind_ai_synthetic_dataset/documents/case-0003/invoice.txt`.
  Isolated document `vision-doc-67aaeaff92`, job `vision-job-e17f7d7047`, and
  review `review-a8f8ac869f` reached APPROVED with invoice `INV-2260`, vendor
  `MedSecure Systems`, due date `2026-02-15`, total `USD 64,000.00`, currency
  `USD`, and reviewer-corrected PO `PO-2260`.
- Narrative `narrative-ac3434bdd92f` reports one verified invoice and
  `$64,000.00 (USD 64000.00)`, with two calculation traces, ten citations, and
  `all_values_verified=true`. Report `report-ed1f002d4788` preserves that same
  source document/snapshot.
- The source fixture contains no invoice-date or bank evidence. The acceptance
  invoice date `2026-01-16` is explicitly a manual reviewer confirmation; bank
  fields remain null and Ask returns `not_found`. Neither is represented as
  machine-extracted source evidence.

## 12. Analytics status

- Processing duration ends at the terminal pipeline event and excludes later
  review time. Invalid/legacy/implausible timing is excluded rather than mixed
  with seconds/milliseconds or epoch timestamps.
- Final isolated analytics contained two documents: one APPROVED and one
  REVIEW_REQUIRED. The controlled failure was visible with one error; an
  actor-audited retry moved QUEUED with zero errors and then COMPLETED with zero
  errors. `job_retry_queued` audit count was one.
- Measured run counts were `failure_count=1` and `success_count=1`; processing
  mean/median were 1.435 seconds and P95 was 1.482 seconds. Citation coverage
  was 1.0 (8/8), with one narrative and one report. Evaluation metrics were not
  fabricated from operational inference counts (`fake_metrics=false`).
- Historical job `vision-job-c2c7f1248d` remains preserved in legacy state as
  a PREPROCESSING record without invented root-cause metadata. The controlled
  current failure/retry path demonstrates the fixed behavior without rewriting
  historical evidence.
- Very large accumulated single-host JSON state can still incur synchronous
  reload/reindex latency; PostgreSQL-backed/shared production scaling remains
  the recommended deployment hardening path.

## 13. Security/audit status

- Authentication covers registration, login, rotating refresh tokens, logout,
  generic password reset, password policy, security headers, and explicit
  privileged bootstrapping.
- Tenant isolation applies to documents, vectors, graph facts, conversations,
  reviews, feedback, narratives, reports, and audits. Privileged mutations are
  backend-authorized and actor/tenant-audited.
- Login success/failure audit records use an email hash—never an email,
  password, token, or credential. Retry, deletion, narrative, review, report,
  admin/model, and fraud-confirmation actions have tested audit boundaries.
- Compose publishes only `127.0.0.1` ports. Secrets/provider/database/Neo4j
  credentials stay backend-side; the built frontend contains no matching secret
  or private backend URI and emits no source maps.
- Final Trivy artefacts report 0 HIGH, 0 CRITICAL, and 0 secret findings; CI
  blocks HIGH/CRITICAL results.
- Final pip-audit artefacts report root 87 packages/0 vulnerabilities, backend
  86/0, and AI 163/0. `pip check` passes in all three environments.
- `npm audit` reports 0 vulnerabilities across 460 dependencies. The verified
  post-freeze browser run passed 7/7 security-header assertions and recorded no
  browser error, traceback, HTTP 4xx/5xx, or backend/worker/frontend error
  entry.

## 14. Tests executed and results

| Gate | Final observed result |
| --- | --- |
| Backend full suite | **218 passed**, 80 warnings, 203.82 s |
| Ruff (`app`, `tests`, `../scripts`) | Passed |
| Last classifier/fax/delete focused regression | 17 passed; Ruff passed |
| Deterministic Ask calculations | 6 focused backend tests passed |
| Persisted cross-retriever citation invariant | Passed after fresh service/question reload |
| Geometry tables and multi-column layout | 6 passed; structured/uncertain states and empty-cell provenance verified |
| Frontend unit suite | **7/7 files, 23/23 tests**, 2.48 s |
| Ask calculation UI | 2 focused tests passed |
| Frontend lint | Passed |
| Frontend production build | Passed, 3,193 modules in 3.15 s |
| Definitive post-freeze production-preview Playwright | 2 passed in 30.3 s (31 s wall); 7/7 security headers; clean browser/API/worker/frontend logs |
| Strict core model verifier | 13/13 passed, 0 failed/blocked enabled targets |
| Strict cached heavy/offline verifier | 17/17 passed, 0 failed/blocked enabled targets |
| Real BGE-M3 -> Qdrant -> reranker acceptance | Passed; 40 chunks, scoped replacement/deletion, 5 cited hits |
| Neo4j bounded rebuild and generic GraphRAG | Passed; stable two-pass IDs, 2 live `HAS_ATTRIBUTE` paths, tenant cleanup |
| Real current-dataset invoice/narrative/report | Passed with source limitations explicitly preserved |
| Citation resolution | 10/10 persisted citations; analytics 8/8 eligible answers |
| PostgreSQL migrations | 001–004 idempotent; PostgreSQL 16.14; 37 tables |
| PostgreSQL/Redis/Qdrant/Neo4j health/read-write | All passed; Redis PONG, SQL `SELECT 1`, Qdrant ready/lifecycle, Neo4j authenticated query |
| Trivy filesystem/secrets | 0 HIGH, 0 CRITICAL, 0 secrets |
| pip-audit / pip check | 0 vulnerabilities in all three environments; all checks passed |
| npm audit | 0 vulnerabilities across 460 dependencies |
| Compose config/loopback publication | Passed |
| `git diff --check` | Passed |

Representative documents actually processed/reprocessed were the two stable
business-form fixtures, the untouched ten-page research-paper holdout, the real
case-0003 invoice acceptance documents `vision-doc-67aaeaff92` and
`vision-doc-90ea79ff84`, generated rotated/low-quality image variants, and
generated two-page TIFF/PDF scans. The post-freeze browser run produced 19
isolated screenshots and zero failure artefacts. Its processes were stopped,
temporary root destroyed, Redis DB 15 flushed, credentials discarded, and the
shared state left untouched.

## 15. Remaining genuine blockers

These do not block the verified local application and fallbacks:

- LayoutLMv3 lacks project labels/label map and a task-fine-tuned checkpoint.
- Donut, YOLO, DINOv2, DeBERTa CUAD, and SegFormer lack approved compatible
  project checkpoints; YOLO/SegFormer also require project labels/training.
- Supervised XGBoost risk lacks labelled project data and a trained checkpoint.
- Qwen 3.5 9B lacks a local checkpoint, and the current 16-GiB Apple Silicon
  host plus constrained disk cannot safely host an unreviewed multi-gigabyte
  activation. Grounded template generation remains ACTIVE.
- An external LLM remains intentionally DISABLED because no approved provider
  endpoint/model/credential was supplied.

Honest evaluation boundaries: handwriting/multilingual accuracy and complex
real merged-cell tables were not established by a labelled representative
corpus. The real invoice fixture contains no source invoice date or bank
details, so those values were correctly left absent unless a reviewer supplied
them. MongoDB, MinIO, and Chroma remain deliberately disabled optional profiles
because they are not active sources of truth.

## 16. Exact commands to start the completed project

First-time local setup:

```bash
cd "/Users/kumar/Downloads/DocuMind AI"
python scripts/generate_local_env.py
python -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install \
  -r backend/requirements/base.txt \
  -r backend/requirements/dev.txt
(cd frontend && npm ci)
docker compose up -d postgres redis qdrant neo4j
./.venv/bin/python scripts/migrate_postgres.py
./.venv/bin/python scripts/migrate_postgres.py --execute
```

`generate_local_env.py` refuses to overwrite an existing `.env` or
`backend/.env`. Do not use `--force` without reviewing and backing up existing
ignored configuration.

Start the host application in four terminals:

```bash
# Terminal 1: infrastructure
cd "/Users/kumar/Downloads/DocuMind AI"
docker compose up -d postgres redis qdrant neo4j
docker compose ps
```

```bash
# Terminal 2: backend
cd "/Users/kumar/Downloads/DocuMind AI/backend"
../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

```bash
# Terminal 3: worker (required when ENABLE_REDIS=true)
cd "/Users/kumar/Downloads/DocuMind AI/backend"
../.venv/bin/celery -A app.workers.vision_tasks:celery_app worker \
  --loglevel=INFO -Q vision-processing --pool=solo
```

```bash
# Terminal 4: frontend
cd "/Users/kumar/Downloads/DocuMind AI/frontend"
npm run dev -- --host 127.0.0.1 --port 5174
```

Health checks:

```bash
curl --fail http://127.0.0.1:8000/health/live
curl --fail http://127.0.0.1:8000/health/ready
curl --fail http://127.0.0.1:6333/readyz
```

Start the default containerized stack instead:

```bash
cd "/Users/kumar/Downloads/DocuMind AI"
ENABLE_NEO4J=true ENABLE_QDRANT=true docker compose up --build
```

The default stack is backend, worker, frontend, PostgreSQL, Redis, Qdrant, and
Neo4j. Start unused optional adapters only when deliberately testing them:

```bash
docker compose --profile optional up --build
```

Verify cached AI adapters without downloading models:

```bash
cd "/Users/kumar/Downloads/DocuMind AI"
scripts/bootstrap_ai_core.sh \
  --skip-install \
  --skip-services \
  --enable-cached-bge \
  --enable-cached-paddle \
  --summary var/ai-core/bootstrap-offline-current.json \
  --strict
```

Never use `docker compose down -v` as routine startup or troubleshooting; the
named volumes contain persistent state.
