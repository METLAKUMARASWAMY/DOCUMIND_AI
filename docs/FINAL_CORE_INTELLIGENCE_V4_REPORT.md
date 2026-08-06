# DocuMind Core Intelligence v4 — Final Engineering Report

Date: 2026-08-06  
Reference document: `vision-doc-1097f9461f` / `00093726.png`  
Implementation commit: `ea8b494`

## 1. Repository and environment audit

- macOS 26.5.2, Apple Silicon arm64, 16 GiB RAM.
- Node 22.19, npm 10.9, application Python 3.13.9, isolated AI Python 3.11.14.
- Docker 28.4; Qdrant 1.18 and PostgreSQL available. The supplied Neo4j Desktop binary is 2026.07.0.
- A pre-change checkpoint was created as `8b917f2`.

## 2. Root causes found

- Single-pass OCR and loose aliases promoted dates/titles incorrectly and missed form fields.
- Completeness counters had multiple incompatible meanings across API/UI/reporting.
- Ask routing lacked durable clarification state and could cascade into unrelated evidence.
- Graph projections used weak display semantics and incomplete source metadata.
- Analytics mixed legacy, incomplete and post-processing review timestamps.
- Model status confused requested configuration with verified runtime health.
- Qdrant used one collection name across incompatible embedding dimensions.
- Relative/runtime stores and external-service credentials created stale-health risks.

## 3. Files modified

The final commit changes 79 files: extraction/schema/model services, RAG/generator/retrieval/graph services, persistence and readiness, API contracts, frontend workspaces, migrations, operational scripts, documentation and regression tests. Use `git show --stat ea8b494` for the authoritative list.

## 4. Migrations added

`infra/migrations/003_core_intelligence_v4.sql` adds OCR provenance, candidate resolutions, reliable processing timings, conversations and retrieval traces. `scripts/migrate_postgres.py` is dry-run by default, checksum-tracked, advisory-locked and idempotent.

The configured native PostgreSQL 17.10 database has 37 tables; migrations 001/002/003 are tracked, all required tables exist, and the second execution returned `already_applied` for all three.

## 5. Packages installed

The isolated AI environment contains Torch 2.13.0, Transformers 4.57.6, Sentence Transformers 5.6.1, FlagEmbedding 1.4.0, PaddlePaddle 3.3.1, PaddleOCR 3.7.0, PaddleX 3.7.2, qdrant-client 1.19.0 and neo4j driver 6.2.0.

## 6. Services started or connected

- Qdrant 1.18: live filtered replace/query/delete passed.
- PostgreSQL 17.10 (Postgres.app): authenticated `SELECT 1`, migrations and v4 persistence mirror passed.
- Neo4j Desktop `22201bd7-b52f-473a-a809-0c14666764e2`: endpoint and version verified; authentication remains blocked pending its matching credential.
- The redundant Docker Neo4j container was stopped without deleting its volume.

## 7. Checkpoints configured

- `BAAI/bge-m3`: cached and verified on CPU.
- `google/flan-t5-small`: cached and verified as the compatible grounded local generator.
- BGE reranker and PP-Structure assets are not marked active because their bounded downloads did not complete.

## 8. Extraction before/after

The target is now `business_form`, not an unknown/title-only record. Six required fields are selected once each; the date correction is traceable from raw `september 21, 11/6` to `September 21, 1976` by ROI re-OCR and date validation. Every field has page/source text/bounding box/model/confidence/review status.

## 9. RAG before/after

Ask now has tenant/user/conversation state, five safety clarification options, a 12-source retrieval trace, exact field routing, grounded abstention, source citations and scope preservation. A bare “yes” cannot become an independent semantic query.

## 10. Graph before/after

The semantic graph contains 16 unique nodes and 21 unique edges for the reference record, with readable entity/relationship labels, source field/page/text/evidence/confidence/status, tenant/document scope and idempotent cleanup. The requester path is evidence-linked rather than invented.

## 11. Processing-time correction

Twenty current-pipeline samples are included. Mean is 4224.074 ms, median 2646.778 ms and p95 10999.747 ms; queue mean is 0.959 ms. Legacy/incomplete/negative/over-one-hour samples are excluded and review activity after the first terminal processing event cannot inflate processing time.

## 12. Active model registry

Verified active paths include local parser, Tesseract 5, local layout, PP-OCRv5 mobile, deterministic schema extractor v3, OpenCV visual cues, ELA/noise screening, Isolation Forest, local hashing, BGE-M3, local vector store, Qdrant, local graph projection, grounded template generator and FLAN-T5-small.

## 13. Blocked model registry and attempts

- BGE reranker v2-m3 and reranker-base: Xet downloads stalled beyond bounded five-minute attempts; checkpoint incomplete; local ranking fallback remains.
- PP-StructureV3: dependency extra installed and multiple assets downloaded, then bounded initialization stopped during remaining formula/layout assets; deterministic geometry layout remains.
- Other advanced adapters remain disabled without project checkpoints.
- Neo4j Desktop is unavailable only for the missing matching credential; local semantic GraphRAG remains active.

## 14. Automated tests

- Backend: 127 passed, 44 known deprecation warnings, 108.92 seconds.
- Ruff: passed.
- Frontend: 18 tests passed; 3192-module production build passed; lint passed.
- `git diff --check`: passed.

## 15. Manual acceptance

Live API acceptance passed upload/processing, the exact six-field viewer, all seven counters and idempotent report reuse. A screenshot is stored at `docs/screenshots/core-v4-00093726-fields.png`. The final browser test reached Ask Documents but its duplicate-filename option selector timed out; the selector is fixed but was not rerun before the session cutoff. This is recorded as pending rather than reported as a pass.

## 16. Sample extracted JSON

```json
{
  "document_type": "business_form",
  "fields": {
    "date": "September 21, 1976",
    "sample_number": "6030",
    "product_type": "85 mm Filter",
    "batch_size": "50 lbs.",
    "requested_by": "Dr. A. W. Spears",
    "specifications_written_by": "W. E. Routh"
  },
  "counts": {"required_expected": 6, "required_matched": 6, "required_missing": 0, "optional_extracted": 0, "total_extracted": 6, "duplicates_rejected": 0, "disputed": 0}
}
```

## 17. Sample RAG response

Question: `What is the sample number?`  
Answer: `The sample number shown is 6030.`  
Status: `supported`, review required because extraction is unverified. Citation: `00093726.png`, page 1, `Sample No. 6030`, box `[274,48,423,62]`, evidence `vis-ev-acb647e84af0`.

## 18. Sample graph path

`00093726.png → REQUESTED BY → Dr. A. W. Spears`  
Relationship status: proposed; page 1; evidence `vis-ev-85de3cb6feea`; source field `requested_by`; confidence 0.7916.

## 19. Remaining genuine limitations

See `docs/PENDING_TASKS_2026-08-06.md`. The concrete items are the Neo4j Desktop credential, one final Playwright rerun, incomplete reranker/PP-Structure assets and project-specific advanced checkpoints. Local JSON remains the recoverable document source of truth while PostgreSQL mirrors v4 operational intelligence records.

## 20. PASS/FAIL/BLOCKED checklist

- PASS: exact extraction, provenance, seven counters, clarification state, 12-source retrieval, citations, abstention, semantic local GraphRAG, analytics, report identity, reprocessing idempotency, Qdrant, PostgreSQL, model truthfulness, automated tests.
- BLOCKED: Neo4j Desktop live write/query until its credential is supplied; optional incomplete model assets.
- PENDING: rerun the corrected live Playwright journey.
- FAIL: none in the committed automated core.
