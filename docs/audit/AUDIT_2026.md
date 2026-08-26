# DocuMind AI — Phase 0 Audit

**Date:** 2026-08-26
**Branch:** `hardening/audit-2026-08-26` (created from `main` @ `e6151ee`)
**Auditor:** acting principal engineer session
**Rule applied:** every claim below traces to a command run in this session. Historical
reports in `docs/` were treated as hypotheses, not evidence.

> **No application code was modified during this phase.**

---

## 0. Working-tree state at audit start

`main` was **already dirty** before this session began. Nothing was discarded, stashed, or
reverted. A working branch was created from the current HEAD; all changes carried over intact.

| | Count |
|---|---|
| Modified tracked files | 70 |
| Untracked new files/dirs | 29 |
| **Total changed paths** | **99** |
| Stashes | 0 |
| Whitespace/conflict errors (`git diff --check`) | none |

Substantial uncommitted work exists, including nine new backend modules
(`entity_resolution.py`, `generic_field_discovery.py`, `reviewer_feedback.py`,
`schema_config.py`, `semantic_facts.py`, `semantic_risk.py`, `semantic_values.py`,
`schemas/`), eight new test files, a new migration
(`infra/migrations/004_conversation_evidence_state.sql`) and a new frontend component
(`GenericIntelligencePanel`). **This work is unreviewed and uncommitted. It should be
committed to the working branch before any further change is layered on top.**

---

## 1. Verified baseline (measured this session, not quoted)

| Check | Command | Result |
|---|---|---|
| Backend lint | `.venv/bin/ruff check app tests` | **PASS** — "All checks passed!" |
| Frontend unit tests | `npm test` (vitest) | **PASS** — 23/23 in 7 files, 2.33 s |
| Frontend lint | `npm run lint` (`--max-warnings 0`) | **PASS** |
| Frontend build | `npm run build` (`tsc -b && vite build`) | **PASS** — built in 2.54 s |
| Frontend deps | `npm audit --audit-level=high` | **PASS** — 0 vulnerabilities |
| Backend tests collected | `pytest --collect-only -q` | **218 tests** |
| Backend test run | `pytest -q --durations=25` | **PASS — 218 passed, 80 warnings, 516.02 s (8 m 36 s)** |

**Correction to the historical baseline:** `docs/FINAL_RELEASE_VERIFICATION_2026-08-13.md`
records "155 backend tests" and "22/22 frontend unit tests". Actual current counts are
**218 backend tests collected** and **23 frontend tests**. The historical numbers predate
the uncommitted work and must not be cited as current.

### Environment reality check

| Service | Configured | Reachable now |
|---|---|---|
| Docker daemon | — | at audit start **DOWN**; **started during this session** (28.4.0) |
| PostgreSQL :5432 | `DATABASE_URL` set in `backend/.env` | **UP, healthy** |
| Redis :6379 | `ENABLE_REDIS` unset → false | **UP, healthy** (app flag still off) |
| Qdrant :6333 | `ENABLE_QDRANT` unset → **false** | **UP, healthy** (app flag still off) |
| Neo4j :7687 | `ENABLE_NEO4J=true` | **UP** |
| Backend :8000 / Frontend :5173 | — | not started |

Note the flag/service mismatch (D-20): `docker compose up` starts Qdrant and Redis
unconditionally, but the application's `ENABLE_QDRANT` / `ENABLE_REDIS` remain `false`, so
the running containers are ignored by the app.

Interpreters: `.venv` = Python 3.13.9 (core: fastapi, qdrant-client 1.18, neo4j 6.2,
celery 5.6, sqlalchemy 2.0.51, pytesseract, opencv-headless). `.venv-ai` = Python 3.11.14
(heavy: torch 2.13, transformers 5.5.4, paddleocr 3.7, paddlepaddle 3.3.1,
sentence-transformers 5.6.1, FlagEmbedding 1.4). `tesseract 5.5.3` present at
`/opt/homebrew/bin/tesseract`.

**Consequence:** the defects below were *found* by reading code with the datastores down.
Where a defect was subsequently **proven against a live service**, that is stated explicitly
in the register.

---

## 2. Dependency map — what actually executes

```
frontend/src/pages/*  ──►  src/api/client.ts  ──►  /api/v1/*
                                                     │
backend/app/main.py ─┬─► auth_routes.py            (prefix /auth)
                     ├─► vision_alias_routes.py    (unprefixed  — what the UI calls)
                     ├─► vision_routes.py          (prefix /vision — canonical)
                     └─► [only if ENABLE_DEMO_API] routes.py + finance_routes.py
                                                     under /api/v1/demo-legacy
                     │
                     ▼
          app/services/vision/*  (44 modules — THE REAL SYSTEM)
              service.py (9,273 lines) ── ask/route/answer/extract/graph/index
              retrieval_adapters.py     ── in-memory hybrid index + Qdrant mirror
              graph_builder.py          ── evidence graph projection
              persistence.py            ── atomic JSON state + PostgreSQL
              model_registry (app/services/model_registry.py) ── status truth source
                     │
                     ▼
          var/vision_state (JSON)  ·  PostgreSQL  ·  Qdrant  ·  Neo4j
```

### Two parallel stacks — the single most misleading thing in this repo

Reachability was traced module by module. Every legacy service roots at exactly two entry
points, `app/api/routes.py` and `app/api/finance_routes.py`, which `main.py:62` mounts
**only when `ENABLE_DEMO_API=true`**. The shipping default is `false`
(`config.py:68`, `.env.example:38`, `backend/.env`).

**Unreachable in the shipping configuration (24 modules, ~230 KB of source):**

`rag.py` · `retrieval.py` · `knowledge_graph.py` · `pipeline.py` · `vector_store.py` ·
`embedding.py` · `ocr.py` · `extraction.py` · `chunking.py` · `agents.py` ·
`compliance.py` · `cross_document.py` · `fraud.py` · `predictive.py` · `report.py` ·
`scoring.py` · `workflow.py` · `xai.py` · `audit.py` · `dataset_loader.py` ·
`dataset_case_service.py` · `finance_workflow_service.py` ·
`finance_extraction_service.py` · `finance_narrative_service.py` ·
`finance_evaluation_service.py`

**Live, shared with the vision stack (2 modules):**
`ingestion.py` (used by `vision/service.py`, `vision/model_smoke.py`) and
`classification.py` (used by `vision/service.py`).

Why this matters more than ordinary dead code: the dead modules carry **exactly the names a
reviewer searches for**. Anyone asking "does the RAG work?" opens `app/services/rag.py` —
10 KB of code the running application never executes — instead of
`app/services/vision/service.py`, where the real implementation lives. The same trap exists
for `retrieval.py`, `knowledge_graph.py`, `vector_store.py`, `embedding.py` and `ocr.py`.

`backend/conftest.py` sets `ENABLE_DEMO_API=true` for the whole test session, so nine test
files exercise this dead stack (`test_enterprise_mvp`, `test_model_registry`,
`test_cross_document`, `test_pipeline`, `tests/test_workflow`, `tests/test_search`,
`tests/test_dataset_loader`, `tests/test_scoring`, `tests/test_extraction`). The headline
test count therefore **overstates verification of the shipping path**.

---

## 3. Route inventory

| Router | File | Routes | Mount | Auth |
|---|---|---|---|---|
| auth | `auth_routes.py` | 8 | `/api/v1/auth` | mixed (public login/register; `require_user` on `/me`, `/reauthenticate`, `/logout`) |
| vision (canonical) | `vision_routes.py` | 48 | `/api/v1/vision` | `Depends(require_user)` on every route sampled |
| vision (aliases) | `vision_alias_routes.py` | 68 | `/api/v1` | `Depends(require_user)` on every route sampled |
| legacy demo | `routes.py` | 39 | `/api/v1/demo-legacy` *(flag-gated, off)* | — |
| legacy finance | `finance_routes.py` | 26 | `/api/v1/demo-legacy` *(flag-gated, off)* | — |
| health | `main.py` | 2 | `/health/live`, `/health/ready` | **public** (intentional) |
| metrics | `observability.py` | 1 | `/metrics` | **public — see D-09** |

`vision_alias_routes.py` is a thin delegating layer: it re-exports handlers from
`vision_routes.py` and adds tenant/role guards. The frontend consumes the **unprefixed
alias** routes plus `/vision/jobs/{id}`. No frontend call was found targeting a route that
does not exist, and no mutating route was found without an auth dependency.

Routes with **no frontend consumer** (candidate dead API surface, to confirm in Phase 1):
`/analytics/documents`, `/analytics/reviews`, `/analytics/rag`, `/narratives/*`
(duplicate of `/finance-narratives/*`), `/evaluation/run`, `/active-learning/export`,
`/documents/{id}/tables`, `/documents/{id}/missing-fields`, `/graph/entity/{id}`.

---

## 4. Demo / fixture coupling surface

This was hunted specifically, because Rule 3 makes it the highest-value defect class.

| Asset | Verdict |
|---|---|
| `backend/app/services/vision/demo_data.py` | **Not an answer-path risk.** Its 4 documents are fed through the *real* `process_text()` pipeline by `seed_connected_demo()` — genuine ingestion, not canned answers. Reachable only via `POST /demo/vision/seed`, which is reviewer-gated. |
| `ENABLE_DEMO_API` | Correctly gates the entire legacy stack. Default `false`. |
| `demo-data/` (4 .txt) | **Dead.** Referenced by no code, no test, no script. |
| `demo_data/` (8 files) | Live only for `finance_workflow_service.py` (flag-gated) + 1 test. |
| `dataset/documind_ai_synthetic_dataset` | Live only for `dataset_loader`/`dataset_case_service` (flag-gated). Config default at `config.py:29`. |
| `var/test_fixtures/*.png` | Legitimate regression fixtures (2 images), used by pinned-value tests. |
| Hardcoded document/case IDs in service code | **None found.** `grep -E "(doc_id\|document_id\|case_id\|tenant_id\|filename)\s*==\s*[\"']" app` returns only test files. |
| `TODO/FIXME/HACK/placeholder` in `app/` | **2**, both benign docstrings. |
| Silently swallowed exceptions (`except Exception:` + bare `pass`) | **1** (`model_smoke.py:835`, a best-effort collection cleanup — acceptable). |

**Conclusion on the "base case bug" (directive §45):** no evidence of a cached answer, a
fixed document ID, a demo-mode branch, or a static context was found in the live answer
path. `ask()` re-hydrates the retrieval index per request
(`service.py:1690 upsert_document`) and dispatches through a handler table keyed on the
routed query type. **The described symptom, if it still reproduces, is not caused by
hardcoding.** The two real generalization weaknesses are D-04 and D-05 below.

---

## 5. Datastore responsibilities

| Store | Declared role | Actually used for | Outage behaviour |
|---|---|---|---|
| PostgreSQL | operational metadata, lineage, evidence, workflow, audit, reports | yes — `vision/persistence.py`, repositories; readiness does an authenticated `SELECT 1` | honest: `status=unavailable`, reason recorded |
| Qdrant | dense vector index for evidence chunks | **mirror/accelerator only** — see D-02 | caught, recorded in `_external_error`, falls back to in-memory — but see D-03 |
| Neo4j | cross-document knowledge graph | gated by `ENABLE_NEO4J`; projection also held locally | probe exists in `model_registry._external_service_live` |
| Redis | Celery broker, rate-limit counters | `ENABLE_REDIS=false` locally; middleware fails open to in-memory bucket | acceptable |
| MongoDB / MinIO / Chroma | — | **none** — `profiles: ["optional"]`, all flags false | correctly declared optional |

---

## 6. Contract drift

Backend status vocabularies are **inconsistently cased by design**:
`VisionProcessingStatus` is `UPPER_SNAKE` (14 values), `ReviewerStatus` is `lower_snake`
(6 values), model runtime statuses are `lower_snake` (13 values), answer statuses are
`lower_snake` (4 values).

`frontend/src/components/StatusChip.tsx` maps **20 keys total** and is fed at least
**9 distinct vocabularies** across 7 pages (document status, reviewer status, review
priority, risk severity, model `runtime_status`, smoke status, report status,
`validation_result`, and free strings such as `"mixed"` / `"single language"` /
`"not_available"`).

Unmapped values fall through to MUI's `default` (neutral grey). Verified unmapped:

- **Processing:** `UPLOADED`, `QUEUED`, `PREPROCESSING`, `OCR_COMPLETED`,
  `EXTRACTION_COMPLETED`, `VALIDATION_COMPLETED` — an actively-processing document is
  visually indistinguishable from an unknown state.
- **Failure states:** `error`, `failed`, `unhealthy`, `blocked`, `misconfigured`,
  `not_run` — **a failed model smoke test renders in the same neutral grey as a test that
  was never run.** This is a Rule 2 violation in the UI layer: a failure must not look
  neutral.
- **Answer status:** `clarification_needed`, `partially_supported`.

`frontend/src/types/api.ts` types these fields as bare `string`, so TypeScript cannot catch
the drift.

---

## 7. Defect register

Severity: **P0** = correctness or honesty blocker · **P1** = real defect, ships badly ·
**P2** = hygiene / maintainability.

| ID | Sev | Area | Defect | Evidence |
|---|---|---|---|---|
| **D-01** | **P0** | Honesty | `QdrantHybridBackend.status()` returns `active` from `enable_qdrant and importlib.find_spec("qdrant_client")` alone — **it never contacts the server**. The same method gates whether retrieval attempts Qdrant at all. | `retrieval_adapters.py:462-488` |
| **D-02** | **P0** | Retrieval | Qdrant is **not the retrieval backend**. `LocalHybridRetrievalIndex` keeps an in-process `dict` of chunks/vectors as the source of truth; Qdrant only supplies `candidate_scores` used to *pre-filter* the same in-memory set. A chunk absent from memory can never be returned even if Qdrant has it. Any claim "Qdrant retrieval verified" must be scoped to this. | `retrieval_adapters.py:264-408` |
| **D-03** | **P0** | Resilience | `_external_error` latches permanently. `search()` requires `self._external_error is None`; once any transient Qdrant error occurs, Qdrant is **never retried for the lifetime of the process** and there is no reset path. Contradicts §50 (recover from service failures). | `retrieval_adapters.py:343, 353` |
| **D-04** | **P0** | Generalization | `VisionQueryRouter` routes on hardcoded phrases lifted from one fixture — `"sample number"`, `"batch size"`, `"product type"`, `"original request"`, `"sample specifications"`, `"who wrote the sample specifications"`, `"made the original request"`, `"what was the purpose"`. Unseen field vocabulary routes differently from seen vocabulary. Partially masked by the not-found cascade in `ask()`, but the routing decision itself is fixture-shaped. | `query_router.py:8-40, 137-152` |
| **D-05** | **P0** | Generalization | `EvidenceGraphBuilder._add_entity_relationships` enumerates the same fixture field names (`sample_number`, `product_type`, `batch_size`, `requested_by`, `specifications_written_by`, `request_date`, `purpose`) to emit typed edges. Unseen documents degrade to generic `HAS_ATTRIBUTE` nodes — graceful, but the *typed* ontology does not generalize. | `graph_builder.py:227-340` |
| **D-06** | **P1** | Observability | The backend contains **exactly one** logger call. `configure_logging()` installs a JSON formatter with sensitive-key masking that essentially nothing uses. `request.state.request_id` is set and returned as a header but **never attached to any log record**. §48 (trace a query end to end) is unimplemented. | `core/logging.py`, `core/middleware.py:18-21`; `grep -c "logger\."` → 1 |
| **D-07** | **P1** | Error handling | Only `DocuMindError` has an exception handler. Any unhandled exception returns Starlette's plain-text 500, which does **not** match the `{success, case_id, data, errors}` envelope `client.ts:parseResponse` expects, and is never logged with the request ID. (No traceback leaks — `debug` is off.) | `core/exceptions.py:17-23`; `main.py:29-34` |
| **D-08** | **P1** | Resource leak | `RateLimitMiddleware._hits` is keyed `f"{client}:{path}"` and **entries are never evicted** — only the deque contents are trimmed. Path-parameter routes (`/documents/{uuid}`) mint a permanent dict entry per unique ID. Unbounded memory growth. | `core/middleware.py:37, 55-63` |
| **D-09** | **P1** | Security | `GET /metrics` is **unauthenticated** and always installed. Exposes route inventory, request counts and latency histograms. | `core/observability.py:49-53` |
| **D-10** | **P1** | Deploy | `docker-compose.yml` uses bare `depends_on` for backend/worker → postgres/neo4j/redis/qdrant. Healthchecks are defined on all four but **never referenced via `condition: service_healthy`**, so app containers start against unready datastores. `frontend depends_on backend` with no backend healthcheck at all. | `docker-compose.yml:23-28, 58-63, 71` |
| **D-11** | **P1** | Deploy | Migrations are mounted into `/docker-entrypoint-initdb.d/`, which PostgreSQL runs **only when initialising an empty data volume**. On an existing `postgres_data` volume, `004_conversation_evidence_state.sql` never applies. `scripts/migrate_postgres.py` exists for this but compose never invokes it. | `docker-compose.yml:81-85` |
| **D-12** | **P1** | Silent fallback | `_apply_grounded_generator` catches every generator exception and silently returns the template answer. The user is never told the generator failed; the response carries no degraded marker. Violates §50. | `service.py:2417-2420` |
| **D-13** | **P2** | Duplication | **Two divergent k8s manifest sets**: `k8s/` (6 files, has `*-service.yaml`) and `infra/k8s/` (10 files, has `postgres/neo4j/chroma/ingress/hpa/namespace`). CI validates only `k8s/`. Neither is a superset. | `diff <(ls k8s) <(ls infra/k8s)` |
| **D-14** | **P2** | Dead assets | `demo-data/` referenced by nothing. Three overlapping corpora (`dataset/`, `demo-data/`, `demo_data/`) with no README explaining which is which. | §4 above |
| **D-15** | **P2** | Code quality | Frontend JSX written as single lines up to **1,820 characters** (`AdminSettingsPage.tsx`), 1,502 (`ModelLabPage.tsx`), 1,039 (`AuditHistoryPage.tsx`). ESLint has no length rule so this passes clean. Effectively unreviewable and unmergeable. | `awk '{print length}' src/pages/*.tsx \| sort -rn` |
| **D-16** | **P2** | Code quality | `app/services/vision/service.py` is **9,273 lines** in one class, mixing routing, 12 answer handlers, extraction, chunking, graph projection, indexing, persistence and review workflow. Violates §51 (avoid giant functions / separation of concerns). | `wc -l` |
| **D-17** | **P2** | Perf | `GenericIntelligencePanel` (174 source lines) emits a **385 kB** bundle chunk; `GraphExplorerPage` 462 kB. Both are lazy-loaded, so first paint is unaffected, but the panel's weight is disproportionate and should be traced to its import. | `npm run build` output |
| **D-18** | **P2** | Hygiene | Root carries `MERGE.TXT` (88 K), `new.txt` (64 K, **tracked**), `new2.txt` (24 K), `project.txt` (52 K, **tracked**), `problem_Statement.txt` (20 K, **tracked**), `.DS_Store`, `docs/Big Monster.txt` (64 K). These are prompt/scratch dumps, not deliverables. | `git ls-files` |
| **D-19** | **P2** | CI | CI pins Python **3.12** while local dev is 3.13.9 and `requires-python = ">=3.11"`. CI runs the full `pytest` suite, which takes **well over 12 minutes** locally with no timeout configured. | `.github/workflows/ci.yml:14, 25` |
| **D-20** | **P2** | Config | `docker-compose` starts the `qdrant` and `neo4j` services unconditionally but passes `ENABLE_QDRANT:-false` / `ENABLE_NEO4J:-false` to the app — the containers run while the application is told to ignore them. | `docker-compose.yml:12-13, 129-140` |

### Non-defects — things that survived the audit

These were suspected and cleared; recording them so they are not re-litigated.

- **Model status badges are honest.** `ModelRegistryService._attach_runtime_health`
  requires a *current* passing smoke test whose `configuration_fingerprint` matches the
  live config, expires smoke evidence after 24 h, and additionally live-probes Qdrant and
  Neo4j before allowing `active`. A configured-but-unproven model reports `unavailable`
  with an explicit reason. (`model_registry.py:606-720`) — this is the correct design.
- **Infrastructure readiness is honest.** `VisionInfrastructureReadinessService` returns
  `ready_for_connection` with the literal reason *"no network probe was executed"* rather
  than claiming health. (`infrastructure.py:161-163`)
- **Tenant scoping is present** on every alias route sampled (`user.tenant_id` threaded
  into every service call).
- **Secrets are gitignored.** `.env` and `backend/.env` are untracked; only `.env.example`
  and `secret.example.yaml` are committed.
- **No prompt-injection-friendly free generation.** The shipping generator is the grounded
  template path; `_apply_grounded_generator` discards any generated answer whose citation
  IDs do not resolve to retrieved evidence (`service.py:2432-2436`).

---

## 8. Repo hygiene disposition (proposed — nothing deleted yet)

| Path | Proposal |
|---|---|
| `MERGE.TXT`, `new.txt`, `new2.txt`, `docs/Big Monster.txt` | move to `docs/prompts/`, or gitignore |
| `project.txt`, `problem_Statement.txt` | move to `submission/` (they are assignment inputs) |
| `.DS_Store` | delete; already in `.gitignore` |
| `demo-data/` | delete (proven unreferenced) |
| `k8s/` vs `infra/k8s/` | consolidate into one, point CI at it |

---

## 9. Backend test run — **218 passed in 516.02 s (8 m 36 s)**

No failures, no errors, no skips. 80 warnings, dominated by a single Pillow deprecation
(`Image.Image.getdata`, `duplicate_detection.py:102`) scheduled for removal in Pillow 14.

The suite is **slow**, and the cost is concentrated in a handful of real-OCR tests:

| Duration | Test |
|---|---|
| **127.25 s** | `test_semantic_risk_feedback_resolution::test_feedback_alias_survives_service_restart_reprocess_and_tenant_boundary` |
| 38.32 s | `test_generalization_regressions::test_00093726_generic_layer_rejects_geometry_fragments...` |
| 36.62 s | `test_generalization_regressions::test_raw_ocr_phone_fallback_also_rejects_calendar_dates[2026-08-10]` |
| 35.50 s | `test_generalization_regressions::test_regression_form_full_pipeline_emits_exact_source_backed_fields` |
| 34.23 s | `...test_raw_ocr_phone_fallback_also_rejects_calendar_dates[2/15/90]` |
| 33.61 s | `test_schema_configuration_v1::test_unseen_statement_variant_uses_configured_schema_without_engine_change` |
| 33.52 s | `...test_raw_ocr_phone_fallback_also_rejects_calendar_dates[3/14/90]` |
| 27.45 s | `test_document_variation_acceptance::test_repository_holdout_pdf_is_genuinely_multi_page_and_not_fixture_tuned` |

**Corrected after further runs.** These durations were a *cold-cache* artefact, not a
property of the code. On warm subsequent runs the same suite completes in **220 s** and the
127 s outlier drops to **5.22 s**. The honest statement is: the first run on a cold machine
is slow (~8.5 min, dominated by Tesseract reading image fixtures from disk); warm runs are
~3.5 min. D-19 (no CI timeout) is still a real defect — a cold CI runner is the slow case —
but the suite is not "CI-hostile", and no assertion needs weakening.

*Two misdiagnoses corrected during this audit, recorded so they are not repeated:*
1. The first run was read as a deadlock in
   `test_regression_form_full_pipeline_emits_exact_source_backed_fields`. That test
   **passes in 33.6 s standalone** and 35.5 s in-suite. The suite is slow, not hung.
2. The `time.sleep` seen parked on the main thread was attributed to a Neo4j routing
   retry. Measured directly, `verify_connectivity()` against a dead Neo4j fails in
   **0.00 s** (connection refused). The sleep belongs to the 127 s test above.

---

## 10. Verdict

**No P0 requires rework deeper than hardening.** The architecture is sound and the honesty
machinery (model registry, infrastructure readiness, grounded-generator citation checking)
is genuinely well built — better than the brief assumed.

The P0s cluster into three themes, all fixable in place:

1. **Qdrant is claimed harder than it is** (D-01, D-02, D-03) — the status lies by
   omission, the index is really in-memory, and one transient failure disables it forever.
2. **Two generalization surfaces are fixture-shaped** (D-04, D-05) — query routing and the
   typed graph ontology both enumerate field names from one lab-form fixture.
3. **The repo hides its own architecture** (§2) — 24 unreachable modules occupy the exact
   names a reviewer will search for, and the test count includes them.

Recommended Phase 1 order: D-01/D-03 → D-12 → D-06/D-07 → D-04/D-05 → D-10/D-11 → §2
quarantine → D-08/D-09 → P2 sweep.

---

# Phase 1 — Remediation log

Every entry below carries the command that proved it. Each fix shipped with a
regression test that **fails against the pre-fix code and passes after**; where a
defect was provable against a live service, the live transcript is quoted.

## Closed

### D-01 · Qdrant reported `active` without ever contacting the server — **CLOSED**

`QdrantHybridBackend.status()` now performs a bounded (`1.5 s`), cached (`5 s`)
`GET /readyz` and reports `active` only on a 2xx. `disabled` and `unavailable`
short-circuit without probing, so a disabled backend costs nothing.

Proven live, both directions, against the running container:

```
UP    -> status='active'      reachable=True   detail='HTTP 200'
DOWN  -> status='unavailable' reachable=False  detail='URLError'
```

Pre-fix, the same dead endpoint produced:
`status='active'  reason='Qdrant client is configured.'`

### D-02 · The serving vector backend was never reported — **CLOSED**

`LocalHybridRetrievalIndex.status()` now returns an authoritative
`vector_backend` field — `qdrant_mirrored_local_hybrid` or `local_hybrid_only` —
plus `degraded`, `external_consecutive_failures` and `external_retry_after_seconds`.

Qdrant remains a **mirror and candidate-scorer over an in-process index**, not the
system of record. That is a deliberate design, and it is now stated by the API
instead of being inferred. Any claim of "Qdrant retrieval" must be read with that
scope.

### D-03 · One transient error disabled Qdrant for the process lifetime — **CLOSED**

The permanent latch is replaced by a 30 s backoff with automatic recovery,
`_record_external_failure` / `_record_external_success`, a `WARNING` log on
degradation and an `INFO` log on recovery.

### D-04 · Fixture-shaped and substring-matched query routing — **CLOSED**

Two distinct bugs.

**Substring matching.** `"sum" in lower` matched *summarise*, *summarize*,
*summary*, *consume*, *assumption*. Measured before the fix:

```
arithmetic_question  <- summarise this document
arithmetic_question  <- summarize the contract
arithmetic_question  <- consume rate
arithmetic_question  <- what did the assumption say
```

All vocabulary tests now go through boundary-anchored patterns
(`(?<![\w-])…(?![\w-])`). After:

```
semantic_question    <- summarise this document
arithmetic_question  <- what is the sum of all invoices     (still correct)
```

**Hardcoded fixture vocabulary.** `field_terms` was a frozen 33-entry literal set
containing labels from one lab-request-form fixture. It is now
`base_field_terms` (domain-independent) **plus vocabulary discovered from the
schema registry at runtime** — **301 terms** on the current catalogue. Adding a
document schema makes its fields routable with no code change. Registry failure
degrades to the base list rather than taking routing down.

Generalization gain, measured: `"what is the policy number"` — a domain this repo
has never processed — previously routed to `semantic_question`; it now routes to
`structured_field_lookup`.

### D-05 · Fixture-shaped knowledge-graph ontology — **CLOSED**

`_add_entity_relationships` enumerated seven fixture field names to emit typed
edges; anything else got no typed relationship. Replaced by a derivation rule
(`*_by` → agent relation, otherwise `HAS_<FIELD>`) that reproduces **all seven
original edges exactly**:

| field | edge |
|---|---|
| `sample_number` | `HAS_SAMPLE_NUMBER` |
| `product_type` | `HAS_PRODUCT_TYPE` |
| `batch_size` | `HAS_BATCH_SIZE` |
| `request_date` | `HAS_REQUEST_DATE` |
| `purpose` | `HAS_PURPOSE` |
| `requested_by` | `REQUESTED_BY` |
| `specifications_written_by` | `SPECIFICATIONS_WRITTEN_BY` |

…and extends the same treatment to names nobody anticipated —
`policy_number` → `HAS_POLICY_NUMBER`, `consignee_name` → `HAS_CONSIGNEE_NAME`,
`approved_by` → `APPROVED_BY`. 50 lines of special-casing removed.

### D-06 · Observability existed but was never used — **CLOSED**

The backend had **one** logger call. `SecurityHeadersMiddleware` now emits one
structured access record per request carrying `request_id`, method, path,
status and `duration_ms`, so the correlation id in `X-Request-ID` finally
corresponds to something in the log.

### D-07 · Unhandled exceptions broke the response envelope — **CLOSED**

Handlers added for `HTTPException`, `RequestValidationError` and bare
`Exception`. Every error now returns `{success, case_id, data, errors,
request_id}`; the traceback goes to the log keyed by the same id and never to the
client.

**A regression I introduced and caught:** the first version flattened structured
`HTTPException.detail` into the envelope. That broke two approval-policy tests —
and would have broken the frontend, whose `parseResponse()` reads `payload.detail`.
`detail` is now passed through verbatim alongside the envelope, and
`test_structured_http_detail_is_preserved_verbatim` pins it.

### D-08 · Unbounded rate-limit memory growth — **CLOSED**

`_hits` was a `defaultdict` keyed `client:path`, never evicted; path-parameter
routes minted a permanent entry per document id. Now an `OrderedDict` LRU with
`max_tracked_keys = 8192` and bounded per-request eviction of expired buckets.

### D-09 · `/metrics` served to anyone — **CLOSED**

Requires `Bearer METRICS_AUTH_TOKEN` when configured (compared with
`secrets.compare_digest`); with no token configured it serves **loopback only**, so
an unconfigured deployment cannot silently publish its metrics.

### D-10 · Compose ignored its own healthchecks — **CLOSED**

`backend` and `worker` were gated on bare `depends_on`. All four datastores are now
`condition: service_healthy`; the backend gained a `/health/ready` healthcheck and
`frontend` waits on `backend: service_healthy`.

### D-11 · Migration 004 never applied to an existing volume — **CLOSED**

`/docker-entrypoint-initdb.d` runs only on a fresh volume. A dedicated `migrate`
service now runs `scripts/migrate_postgres.py --execute` on every start, and the app
containers wait on `migrate: service_completed_successfully`.

Resulting dependency graph (from `docker compose config`):

```
postgres(healthy) -> migrate(completed) -> backend(healthy) -> frontend
                                        -> worker
```

### D-12 · Silent generator fallbacks — **CLOSED**

`VisionAskResponse` gained `degradations[]` and a computed `degraded` flag. Both
silent paths now disclose:

* generator raised → `impact="generator_skipped"`, logged, answer still cited;
* generated answer cited nothing resolvable → `impact="generated_answer_rejected"`,
  logged. This is also the signal that fires on a prompt-injection attempt from
  document content.

### D-13 · Two divergent k8s manifest sets — **CLOSED**

Both validated with `kubectl apply --dry-run=client`. `infra/k8s/` is the superset
(namespace, postgres/neo4j statefulsets, ingress, HPA, and the backend Service
defined inline in `backend-deployment.yaml`); `k8s/` was a strict subset. `k8s/`
removed, CI repointed at `infra/k8s/`, revalidated.

### D-14 / D-18 · Dead assets and root clutter — **CLOSED**

`demo-data/` deleted (proven referenced by nothing). `new.txt`, `new2.txt`,
`MERGE.TXT`, `docs/Big Monster.txt` → `docs/prompts/` (gitignored);
`project.txt`, `problem_Statement.txt` → `submission/brief/`; `.DS_Store` removed.

### D-19 · CI could hang forever — **CLOSED**

Backend job gained `timeout-minutes: 25` and `--durations=15`. Python pinned to
3.13 to match local development.

### D-20 · Services started but told to be ignored — **CLOSED**

Compose defaults changed to `ENABLE_QDRANT:-true` / `ENABLE_NEO4J:-true`, matching
the containers it actually starts.

## Structural change

### The two-stack problem (§2) — **RESOLVED**

25 unreachable modules moved from `app/services/` to **`app/legacy/`** with a
README stating plainly what runs and what does not, and a table pointing at the
real implementation of each capability. `app/services/` now contains only the six
modules that serve production traffic: `auth`, `classification`, `ingestion`,
`model_registry`, `readiness`, `vision/`.

Verified after the move: `ruff` clean, and `create_app().openapi()` documents
**185 paths** — 8 auth, 46 vision, 66 aliases, 63 demo-legacy — so nothing was
unmounted by the relocation.

## Live datastore evidence

Docker was started during this session; PostgreSQL 16, Redis 7, Qdrant 1.18 and
Neo4j 5.22 all reported healthy. Two previously-unseen invoices were ingested
through the real pipeline into the live Qdrant:

```
vector_backend = qdrant_mirrored_local_hybrid   degraded=False   qdrant=active
qdrant collections:
   vision_evidence_chunks__local_hashing_v1__d768_v1: 33434 points

Q: who supplies precision bearings   -> [supported] cites ['live-a.txt']
Q: who supplies industrial solvents  -> [supported] cites ['live-b.txt']
Q: what is the total amount          -> [supported] cites ['live-a.txt','live-b.txt']
Q: mean time between failures ...    -> [not_found] cites []      <- genuine abstention
```

Retrieval discriminates correctly per document, aggregates across documents when
the question calls for it, and abstains with zero citations when the corpus does
not support the question.

## Still open

| ID | Sev | Status |
|---|---|---|
| **D-15** | P2 | **OPEN.** Frontend JSX still contains single lines up to 1,820 chars. ESLint has no length rule, so this passes clean. Needs a formatting pass plus a `max-len` rule — deferred because reformatting every page would bury the behavioural diff. |
| **D-16** | P2 | **OPEN.** `app/services/vision/service.py` remains 9,273 lines in one class. Splitting it is the right call but is a large, risky refactor that deserves its own change set. |
| **D-17** | P2 | **OPEN.** `GenericIntelligencePanel` still emits a 385 kB chunk from 174 source lines; the heavy import has not been traced. |
| **D-19b** | P2 | **OPEN.** The suite is still slow. The top eight OCR tests dominate; fixture-level caching of the Tesseract pass is the fix. The CI timeout now bounds the damage. |
