# DocuMind remediation and generalization handoff

Last updated: 2026-08-12 10:55 IST  
Branch/HEAD at snapshot: `main` / `e6151ee`  
Repository: `/Users/kumar/Downloads/DocuMind AI`

This is the durable resume point for both supplied directives:

- the full DocuMind Vision AI remediation/activation directive; and
- the document-generalization/open-set extraction directive.

It records what was actually implemented and verified during the time-boxed
session. It is not a claim that every item in either directive has reached the
final release gate. Runtime files under `var/` are evidence, not source control
substitutes.

## 1. Protect this work before doing anything else

The worktree is intentionally dirty. Do not reset it, clean it, or perform a
blanket stash. In particular, these files predated this remediation work and
belong to the user:

- `new.txt` — modified;
- `new2.txt` — untracked; and
- `docs/Big Monster.txt` — untracked.

The two new source test fixtures are the only files intended to escape the
otherwise ignored `backend/var/` tree and should be included when this work is
committed:

- `backend/var/test_fixtures/00093726.png` (120,650 bytes); and
- `backend/var/test_fixtures/87093315_87093318.png` (87,156 bytes).

Before editing, inspect the exact state:

```bash
cd "/Users/kumar/Downloads/DocuMind AI"
git status --short --untracked-files=all
git diff --check
git diff -- new.txt
```

Do not add runtime OCR, model, upload, review, conversation, or index artifacts
from `backend/var/` or `var/`. Do not print or commit `.env`, `backend/.env`,
credentials, tokens, or database connection strings.

## 2. Implemented repository changes

The changes below exist in the uncommitted worktree.

### OCR, layout, classification, and extraction

- Added adaptive Tesseract consensus over original/processed/grayscale/enhanced
  page variants, plausibility scoring, selected-pass metadata, and per-word OCR
  variant/pass provenance.
- Tightened OCR line reconstruction so tall or adjacent word boxes do not merge
  distinct form rows.
- Added geometry-driven table discovery with scale-relative gaps, repeated
  column alignment, multiword cells, true cell boxes, Markdown output, and a
  prose false-positive guard. Existing pipe/tab table parsing remains intact.
- Removed filename semantics from document classification and added content-led
  open-set behavior plus `sample_form`, `product_specification`, `email`, and
  `generic_form` families.
- Extended the schema registry with schema version/family/hints/module metadata,
  richer field validation metadata, aliases, `request_date`, and `purpose`.
- Fixed global date/phone incompatibility: calendar dates such as `3/14/90`,
  `2/15/90`, ISO dates, and date-shaped OCR strings cannot validate as phone
  numbers. Two-digit years normalize deterministically.
- Improved candidate resolution using delimiter, label/value geometry,
  compatibility, evidence conflicts, and fuzzy OCR-label matching. Added form
  value normalization such as `100m Filter` to `100 mm Filter` and `Ibs.` to
  `lbs.` without hard-coding either regression filename.

### Generic/open-set intelligence

- Added a reusable semantic value registry for DATE, MONEY, PERCENTAGE, PHONE,
  EMAIL, ADDRESS, PERSON, ORGANIZATION, IDENTIFIER, QUANTITY, BANK_ACCOUNT,
  CURRENCY, TEXT, and BOOLEAN-like values, with context-sensitive validation.
- Added generic key/value discovery for colon, delimiter, whitespace-gap, and
  label-above-value layouts. It preserves raw/canonical labels, raw/normalized
  values, page, label/value/evidence boxes, word IDs, extraction strategy,
  confidence, type, and review state.
- Added curated aliases, fuzzy aliases with a configurable threshold, and
  open-set normalized labels instead of forcing weak matches.
- Added schema modes A (registered), B (registered plus unexpected fields), and
  C (generic/open set), plus a schema profile returned to clients.
- Kept generic fields and typed entities additive to registered schema fields.
  The new information is retained in document records, source-backed chunks,
  evidence endpoints, graph projection, and the React viewer.
- Added `VerifiedVisionFact` records and tenant-scoped feedback alias reuse when
  reviewers accept or correct source-backed fields.
- Added a React generic-intelligence panel and overlays for generic, typed,
  verified, and low-confidence evidence.

### Retrieval, Ask Documents, graph, persistence, and runtime truthfulness

- Added generic fields, typed entities, and verified facts to retrieval chunks;
  raw OCR remains available as a lower-priority evidence layer.
- Added conversational `Why?`, evidence, source/page, and location follow-ups
  that preserve selected document scope and reuse prior citations.
- Added generic `Attribute` graph nodes and source-backed `HAS_ATTRIBUTE` edges,
  including tenant/document/evidence provenance.
- Added evidence aliases/routes and persistence for the new semantic fields.
- Added PostgreSQL migration
  `infra/migrations/004_conversation_evidence_state.sql`; it stores the expanded
  conversation/evidence state and is included in Compose initialization.
- Made model-smoke health expire and configuration fingerprints invalidate stale
  results, including live service rechecks for Qdrant and Neo4j.
- Added environment/Compose wiring for explicit Qdrant and Neo4j activation.

### Security and dependency work

- Upgraded/overrode `nanoid` to 3.3.18 and refreshed the frontend lockfile.
- Made the CI Trivy HIGH/CRITICAL filesystem scan blocking (`exit-code: "1"`).
- No Trivy ignore file or undocumented suppression was added.

## 3. Exact verification evidence at handoff

The following results were observed after the current source changes unless
explicitly bounded below. The optional-model/Core-AI smoke artifacts were
created earlier in this same session, before the final service-layer edits, so
they must be refreshed at the release gate.

| Area | Result | Evidence/boundary |
| --- | --- | --- |
| Backend full suite | **155 passed**, 49 warnings, 134.42 s | Final rerun at 10:54 IST, including the protected conversation-evidence state. |
| Generalization-focused backend group | **38 passed** | Semantic types, generic discovery, OCR/layout, both image regressions, open-set contracts. |
| Migration/graph/regression group | **16 passed** | Migration 004 and graph/regression contracts. |
| RAG/persistence/graph group | **13 passed** | Conversation state, PostgreSQL contracts, graph behavior. |
| Updated contract tests | **17 passed** | OCR provenance, completeness accounting, model-smoke freshness, follow-up citations. This overlaps the full suite. |
| Ruff | **Full check passed** | `ruff check app tests ../scripts` completed at 10:51 IST. |
| Frontend unit tests | **22/22 passed** across 7 files | Includes the new generic-intelligence and evidence-overlay tests. |
| Frontend lint | Passed | Current frontend changes. |
| Frontend production build | Passed | Current frontend changes. |
| npm audit | **0 vulnerabilities** at `--audit-level=high` | After the `nanoid` override/lock refresh. |
| `git diff --check` | Passed at 10:55 IST | Includes the final documentation and persistence edits. |
| PostgreSQL | Authenticated, migration 004 applied, **37 tables** | Migration path and persistence tests passed. |
| API health | `/health/live` returned `{"status":"ok"}` | Local Uvicorn on `127.0.0.1:8000`. |
| API readiness | `ready`; PostgreSQL, Redis, Qdrant, Neo4j enabled with no errors | This process used transient runtime flags; see section 5. |
| Core AI verifier | **13 passed, 0 failed, 0 blocked** | `var/ai-core/verification-2026-08-12.json`, generated 10:21 IST before final service edits; both service probes were reachable. Refresh before release. |
| Optional model smokes | Passed: BGE-M3, BGE reranker v2-m3, PP-OCRv5 mobile, PP-StructureV3/PP-DocLayoutV3, Qdrant, Neo4j | Durable entries are in `var/vision_state/model_smoke/results.json`; they were generated at 10:21–10:28 IST before final service edits and are session evidence, not permanent deployment activation. |

The prior pre-change baseline was 127 backend tests and 18 frontend tests. Those
older counts in `docs/FINAL_CORE_INTELLIGENCE_V4_REPORT.md` are historical, not
the current acceptance result.

## 4. Regression acceptance results

The stable fixtures are no longer read from an ignored uploads directory.

For `87093315_87093318.png`, the live full pipeline was also run under a renamed
filename. It classified the content as `business_form`, selected registered
schema mode A, produced source boxes, emitted no PHONE entity, and extracted the
following exact eight values:

```text
sample_number: 1194-90
date: 3/14/90                 -> 1990-03-14
product_type: 100 mm Filter
batch_size: 47.5 lbs.
requested_by: J.H. Bell
request_date: 2/15/90         -> 1990-02-15
purpose: Cigarette Modification B-451
specifications_written_by: C.W. Lassiter
```

`00093726.png` also passed its current extraction/provenance regression. The
adaptive OCR test deliberately accepts different raw OCR variants while still
requiring source-backed corrected semantic output.

## 5. Services and deployment/model status

At 10:44 IST these Compose services were running on localhost:

| Service | Observed version/image | Ports | State |
| --- | --- | --- | --- |
| PostgreSQL | 16.14 / `postgres:16-alpine` | 5432 | running; migrated |
| Redis | 7.4.10 / `redis:7-alpine` | 6379 | running |
| Qdrant | 1.18.0 | 6333/6334 | running; health and integrated retrieval smoke passed |
| Neo4j | Docker `neo4j:5.22-community` | 7474/7687 | running; live semantic Ask/path smoke passed |

The local Uvicorn process was started separately and was healthy at handoff.
The Vite frontend also responded at `http://127.0.0.1:5174`. Processes started
in this interactive session may stop when the session closes.

Important truthfulness boundary: service availability, cached checkpoints, and
successful isolated smoke inference are not the same as permanent deployment
activation. The checked-in defaults remain opt-in. The local `.env` files
persistently enable Neo4j, while the verified live backend was started with
additional transient PostgreSQL/Redis/Qdrant/Neo4j settings. Recheck the actual
process and `/api/v1/admin/models` after every restart.

Latest smoke state records these real passes:

- BAAI BGE-M3 and exact BGE reranker v2-m3;
- PP-OCRv5 mobile and PP-StructureV3/PP-DocLayoutV3;
- Qdrant storage behavior and Neo4j connectivity/path behavior;
- Tesseract, deterministic schema extraction, local layout, visual cues,
  forensics, anomaly model, local hashing/vector/graph, and grounded templates.

The following remain correctly blocked/unavailable rather than falsely active:

- LayoutLMv3 project classification/extraction checkpoint and labels;
- Donut project checkpoint;
- YOLO project detector checkpoint/labels;
- DINOv2/SegFormer/tamper project assets;
- DeBERTa CUAD project integration assets;
- project-trained XGBoost risk model;
- PaddleOCR-VL deployment path;
- Qwen 3.5 9B checkpoint/runtime; and
- external provider generation (no approved provider configuration/credential).

Disk is critically constrained: `df -h .` showed about 3.5 GiB free and 99%
use at 10:44 IST. This is a genuine blocker for a multi-gigabyte Qwen download
or additional large checkpoints. Do not delete repository, Docker, model-cache,
or user data to recover space without explicit target review.

## 6. Genuine remaining work, in resume order

### P0 — close the release safely

1. Re-run `git status` and `git diff --check` after all concurrent edits are
   stable; the full Ruff and frontend gates already pass on the current source.
2. Review the entire diff, separating the three protected user files from the
   implementation. Commit only after that review. No commit was created during
   this time-box.
3. Run the current live browser/Playwright upload-to-answer journey. The 2026-08-10
   Playwright evidence predates this open-set/UI/persistence change set.
4. Run a real Trivy HIGH/CRITICAL filesystem scan in CI or on a machine with
   Trivy installed. Trivy was not installed locally; only the blocking workflow
   configuration was validated.
5. Recheck API/admin model statuses after a clean process restart. Do not rely
   on historical smoke records or the process that may disappear with this
   session.
6. Reconcile stale documentation: several older reports say Docker Neo4j is
   stopped and Neo4j Desktop is the live target. At this snapshot Docker Neo4j
   5.22 is running and used by the current local Compose path.

### P1 — finish the generalization directive beyond the current regressions

1. Add a genuinely diverse unseen-document acceptance corpus: at minimum a new
   invoice, purchase order, email/letter, sparse form, visually complex table,
   multi-page document, low-contrast/rotated scan, and an unknown-domain form.
   Current tests are strong but still centered on synthetic layouts and the two
   supplied regression images.
2. Add property-based/generative tests for label/value permutations, date/phone/
   identifier ambiguity, Unicode and punctuation aliases, row alignment, and
   filename/content invariance. The current parametrized tests are not a full
   property-based suite.
3. Move or mirror the schema registry into validated, versioned YAML/JSON config
   if strict compliance with the directive's external structured-registry
   requirement is required. The current registry is extensible but Python-based.
4. Add the optional embedding-similarity stage to canonical label mapping when
   BGE is active; current label mapping is curated plus fuzzy/contextual.
5. Validate OCR consensus on deskew/rotation, blur, noise, compression, mixed
   fonts, handwritten content, multilingual text, multi-column pages, and
   multi-page PDFs. Add ROI/crop retries where measured evidence shows benefit.
6. Validate geometry table extraction on real scans with merged cells, sparse
   values, multiline cells, ruled/unruled tables, repeated page headers, and
   variable column counts. Preserve abstention where alignment is ambiguous.
7. Add an end-to-end unknown-document Ask test that intentionally has no schema
   fields and proves raw OCR/layout/generic chunks remain searchable with
   citations and grounded abstention.
8. Extend the general risk engine to consume typed generic facts explicitly.
   Current generic facts are retained/indexed/graphed, but no new generic-risk
   rule integration was demonstrated in this session.
9. Validate tenant-scoped entity resolution and deduplication across multiple
   unseen documents, including aliases and temporal conflicts; keep ambiguous
   relationships proposed rather than verified.
10. Exercise reviewer corrections through PostgreSQL persistence, restart,
    reprocess, feedback reuse, export, and evaluation. Unit coverage exists for
    tenant-scoped feedback alias reuse, but a restart-level live acceptance is
    still needed.
11. Reprocess a bounded existing corpus with the current pipeline, preserving
    archive/history, then prove vector replacement and Neo4j projection are
    idempotent and contain no stale facts.

### P2 — finish conditional/full-platform items from the remediation directive

1. Decide which optional heavy adapters are deployment requirements. Supply
   project-trained checkpoints and label maps for LayoutLMv3, Donut, YOLO,
   DINOv2/SegFormer/tamper, CUAD, and XGBoost before claiming them active.
2. Recover sufficient disk, then obtain and verify the selected Qwen checkpoint
   if local Qwen is mandatory. Until initialization, real inference, schema
   validation, and no-fallback proof pass, keep it blocked. The grounded local
   template/FLAN path remains the safe fallback.
3. Consolidate normal backend and AI dependencies into a reproducible deployment
   image/profile. The normal `.venv` and heavy `.venv-ai` are currently separate;
   isolated smoke success does not guarantee the default Uvicorn interpreter can
   load every heavy adapter.
4. Perform a clean integrated BGE-M3 -> versioned Qdrant -> exact reranker ->
   cited Ask run after restart, including tenant filters, replacement, deletion,
   dimension mismatch protection, and lexical fallback.
5. Perform two clean Neo4j rebuilds after bounded corpus reprocessing and compare
   node/edge identities. Re-run source-backed GraphRAG for known and generic
   `HAS_ATTRIBUTE` facts and tenant isolation.
6. Regress the existing finance, report, analytics, fraud, risk, review, auth,
   authorization, audit, upload failure, and processing-trace workflows against
   this change set. No new defect was demonstrated there, but the full directive
   requires live evidence beyond unit-suite success.
7. Specifically confirm finance calculations use verified `Decimal` values,
   narratives cite facts rather than inventing them, report snapshots retain
   processing/data version, analytics excludes legacy/impossible durations, and
   approval/escalation remains backend-protected.
8. Run secret scanning and inspect the built frontend bundle for credential-like
   strings. Confirm no `VITE_*` variable contains database, Neo4j, model-provider,
   or API credentials.
9. Refresh screenshots and the final release report only after all current live
   acceptance checks pass. The required final response must retain the 16-section
   order specified in the primary directive.

## 7. Exact commands to resume

### Inspect and recover the local infrastructure

```bash
cd "/Users/kumar/Downloads/DocuMind AI"
df -h .
docker compose ps
docker compose up -d postgres redis qdrant neo4j
docker compose ps
```

Do not run `docker compose down -v`; the volumes contain persistent state.

### Inspect/apply PostgreSQL migrations

```bash
cd "/Users/kumar/Downloads/DocuMind AI"
set -a
source ./.env
set +a
export DATABASE_URL="postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:${POSTGRES_PORT:-5432}/${POSTGRES_DB}"
./.venv/bin/python scripts/migrate_postgres.py
./.venv/bin/python scripts/migrate_postgres.py --execute
./.venv/bin/python scripts/migrate_postgres.py
```

The first and third commands are read-only inspections. `--execute` is
checksum-tracked and was idempotent in this session.

### Start the local application

Terminal 1:

```bash
cd "/Users/kumar/Downloads/DocuMind AI"
set -a
source ./.env
set +a
export DATABASE_URL="postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@127.0.0.1:${POSTGRES_PORT:-5432}/${POSTGRES_DB}"
export REDIS_URL="redis://127.0.0.1:${REDIS_PORT:-6379}/0"
export ENABLE_REDIS=true
export ENABLE_QDRANT=true
export QDRANT_URL="http://127.0.0.1:${QDRANT_HTTP_PORT:-6333}"
export ENABLE_NEO4J=true
export NEO4J_URI="bolt://127.0.0.1:${NEO4J_BOLT_PORT:-7687}"
cd backend
../.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd "/Users/kumar/Downloads/DocuMind AI/frontend"
npm run dev -- --host 127.0.0.1
```

Then check:

```bash
curl --fail http://127.0.0.1:8000/health/live
curl --fail http://127.0.0.1:8000/health/ready
```

This command deliberately derives the host PostgreSQL URL from the root Compose
environment because the current ignored `backend/.env` contains a different
local database identity. Reconcile that file manually before simplifying the
command; do not overwrite it blindly. To enable a heavy adapter, set its
backend-only variables before starting Uvicorn. Keep all secret values in an
ignored local environment or secrets manager. Never place them in these docs or
the frontend.

### Run all source gates

```bash
cd "/Users/kumar/Downloads/DocuMind AI/backend"
../.venv/bin/python -m pytest -q
../.venv/bin/ruff check app tests ../scripts

cd "/Users/kumar/Downloads/DocuMind AI/frontend"
npm ci
npm test -- --run
npm run lint
npm run build
npm audit --audit-level=high

cd "/Users/kumar/Downloads/DocuMind AI"
git diff --check
```

### Re-run the core model/runtime verifier without downloads

```bash
cd "/Users/kumar/Downloads/DocuMind AI"
./.venv-ai/bin/python scripts/verify_ai_core.py \
  --run-smoke \
  --smoke-timeout 300 \
  --output var/ai-core/verification-resume.json \
  --strict
```

Read the JSON `verification.passed`, per-model `status`, initialization,
inference, output-schema, and service results. A zero exit under the wrong set of
feature flags is not evidence that an optional adapter is active. Downloads are
off by default; do not enable them while disk remains critically low.

To refresh the already-cached optional targets without downloads, run each
adapter explicitly in `.venv-ai` so an omitted feature flag cannot look like a
successful model smoke:

```bash
cd "/Users/kumar/Downloads/DocuMind AI"
set -a
source ./.env
set +a
export ENABLE_HEAVY_MODELS=true
export ENABLE_BGE_M3=true
export ENABLE_PADDLEOCR=true
export ENABLE_PADDLEOCR_STRUCTURE=true
export PADDLEOCR_DEVICE=cpu
export PADDLE_PDX_MODEL_SOURCE=huggingface
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true
export ENABLE_QDRANT=true
export QDRANT_URL="http://127.0.0.1:${QDRANT_HTTP_PORT:-6333}"
export BGE_ALLOW_DOWNLOAD=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
./.venv-ai/bin/python scripts/smoke_test_models.py \
  --model BAAI/bge-m3 --strict --compact
./.venv-ai/bin/python scripts/smoke_test_models.py \
  --model BAAI/bge-reranker-v2-m3 --strict --compact
./.venv-ai/bin/python scripts/smoke_test_models.py \
  --model Qdrant --strict --compact
./.venv-ai/bin/python scripts/smoke_test_models.py \
  --model PP-OCRv5_mobile --strict --compact
./.venv-ai/bin/python scripts/smoke_test_models.py \
  --model 'PP-StructureV3 / PP-DocLayoutV3' --strict --compact
```

### Reprocess safely

Dry-run first and replace the example tenant/document with an inspected target:

```bash
cd "/Users/kumar/Downloads/DocuMind AI"
./.venv/bin/python scripts/reprocess_documents.py \
  --organisation example \
  --pipeline-version current \
  --document-id vision-doc-example
```

Only after the source and archive target are confirmed:

```bash
./.venv/bin/python scripts/reprocess_documents.py \
  --organisation example \
  --pipeline-version current \
  --document-id vision-doc-example \
  --execute
```

## 8. Release declaration boundary

The current implementation and automated suite are in substantially better
shape and the full backend/frontend gates pass. However, do not issue the
directive's final “completed” declaration until the P0 live release checks are
closed and each blocked optional model is either (a) supplied, integrated, and
smoke-passed or (b) explicitly accepted as a truthful deployment limitation.
The P1 items are the remaining evidence needed to support the stronger claim
that the system generalizes beyond the supplied documents rather than merely
passing their regressions.
