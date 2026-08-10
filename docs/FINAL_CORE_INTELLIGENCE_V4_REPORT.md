# DocuMind Core Intelligence v4 — Final Engineering Report

Implementation date: 2026-08-06

Final pending-task closure: 2026-08-10

Reference document: `vision-doc-1097f9461f` / `00093726.png`

Core implementation commit: `d8d9a2b`

## 1. Repository and environment audit

- macOS 26.5.2, Apple Silicon arm64, 16 GiB RAM.
- Node 22.19, npm 10.9, application Python 3.13.9, isolated AI Python 3.11.14.
- Docker 28.4; Qdrant 1.18 and PostgreSQL 17.10 available. The exact supplied Neo4j Desktop instance is running Neo4j Enterprise 2026.07.0.
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

The core implementation commit changes 80 files: extraction/schema/model services, RAG/generator/retrieval/graph services, persistence and readiness, API contracts, frontend workspaces, migrations, operational scripts, documentation and regression tests. Use `git show --stat d8d9a2b` for the authoritative core list. The 2026-08-10 closure adds only acceptance-test corrections, refreshed evidence, and truthful runtime documentation.

## 4. Migrations added

`infra/migrations/003_core_intelligence_v4.sql` adds OCR provenance, candidate resolutions, reliable processing timings, conversations and retrieval traces. `scripts/migrate_postgres.py` is dry-run by default, checksum-tracked, advisory-locked and idempotent.

The configured native PostgreSQL 17.10 database has 37 tables; migrations 001/002/003 are tracked, all required tables exist, and the second execution returned `already_applied` for all three.

## 5. Packages installed

The isolated AI environment contains Torch 2.13.0, Transformers 4.57.6, Sentence Transformers 5.6.1, FlagEmbedding 1.4.0, PaddlePaddle 3.3.1, PaddleOCR 3.7.0, PaddleX 3.7.2, qdrant-client 1.19.0 and neo4j driver 6.2.0.

## 6. Services started or connected

- Qdrant 1.18: live filtered replace/query/delete passed.
- PostgreSQL 17.10 (Postgres.app): authenticated `SELECT 1`, migrations and v4 persistence mirror passed.
- Neo4j Desktop `22201bd7-b52f-473a-a809-0c14666764e2`: the configured credential authenticates against Neo4j Enterprise 2026.07.0; all databases are online and the default `neo4j` database is connected.
- The redundant Docker Neo4j 5.22 container is stopped without deleting its volume. Neo4j Desktop manages the live process on port 7687.

## 7. Checkpoints configured

- `BAAI/bge-m3`: cached and verified on CPU.
- `google/flan-t5-small`: cached and verified as the compatible grounded local generator.
- Exact `BAAI/bge-reranker-v2-m3`: cached and verified by real offline CPU reranking.
- `PP-StructureV3 / PP-DocLayoutV3`: all required components are cached and verified by real offline CPU layout inference.

## 8. Extraction before/after

The target is now `business_form`, not an unknown/title-only record. Six required fields are selected once each; the date correction is traceable from raw `september 21, 11/6` to `September 21, 1976` by ROI re-OCR and date validation. Every field has page/source text/bounding box/model/confidence/review status.

## 9. RAG before/after

Ask now has tenant/user/conversation state, five safety clarification options, a 12-source retrieval trace, exact field routing, grounded abstention, source citations and scope preservation. A bare “yes” cannot become an independent semantic query.

## 10. Graph before/after

The semantic graph contains 16 unique nodes and 21 unique edges for the reference record, with readable entity/relationship labels, source field/page/text/evidence/confidence/status, tenant/document scope and idempotent cleanup. The requester path is evidence-linked rather than invented. The live Desktop database contains 203/203 unique tenant entity IDs and 272/272 unique relationship IDs after two 22-document rebuilds; the second rebuild changed no document.

## 11. Processing-time correction

Twenty-two current-pipeline samples are included. Mean is 4961.886 ms, median 2874.443 ms and p95 13004.021 ms; queue mean is 2.384 ms. Structured extraction coverage is 0.7273. Legacy/incomplete/negative/over-one-hour samples are excluded and review activity after the first terminal processing event cannot inflate processing time.

## 12. Verified model runtime

Measured smoke passes include local parser, Tesseract 5, local layout, PP-OCRv5 mobile, PP-StructureV3, deterministic schema extractor v3, OpenCV visual cues, ELA/noise screening, Isolation Forest, local hashing, BGE-M3, exact BGE reranking, local vector store, Qdrant, local graph projection, Neo4j Desktop, grounded template generation and FLAN-T5-small. Heavy adapters remain explicit deployment flags; a passed smoke is not described as currently loaded when its flag is disabled.

## 13. Model completion and conditional boundaries

- Exact BGE reranker v2-m3: the resumed 2,271,071,852-byte checkpoint completed and strict offline initialization/inference passed in 9.209 seconds. The smaller fallback was superseded and intentionally not downloaded.
- PP-StructureV3: the missing 735,468,877-byte FormulaNet component completed and full strict offline layout inference passed in 18.225 seconds.
- Other advanced adapters remain correctly disabled because no project-trained checkpoint was supplied; no generic public weight is substituted for a project model.
- The external provider remains unconfigured because no provider credential was supplied. The verified local grounded generator remains available.
- Exact commands and machine-readable artifact names are in `docs/MODEL_ASSET_VERIFICATION_2026-08-10.md`.

## 14. Automated tests

- Backend: 127 passed, zero skipped/deselected, 44 known deprecation warnings, 99.72 seconds.
- Ruff: passed.
- Frontend: 18 tests passed; 3192-module production build passed; lint passed.
- Live Playwright: 1 exact end-to-end journey passed in 9.5 seconds with zero captured browser errors.
- `git diff --check`: passed.

## 15. Manual acceptance

Live Playwright acceptance passed in 9.5 seconds against the final Neo4j Desktop target. It covered upload/terminal processing, the exact six-field viewer, all seven counters, idempotent report reuse, a cited sample-number answer, safety clarification with cited document-quality evidence, the evidence-linked requester GraphRAG path, analytics, and Admin Models. No browser console or page errors were captured. Five refreshed screenshots are stored under `docs/screenshots/core-v4-*.png`.

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

No actionable item from the 2026-08-06 hand-off remains; see `docs/PENDING_TASKS_2026-08-06.md` for the closure evidence. Project-trained advanced checkpoints and an external provider are conditional future deployment inputs, not incomplete current work. Local JSON remains the recoverable document source of truth while PostgreSQL mirrors v4 operational intelligence records.

## 20. PASS/FAIL/BLOCKED checklist

- PASS: exact extraction, provenance, seven counters, clarification state, 12-source retrieval, citations, abstention, semantic local and live Neo4j GraphRAG, analytics, report identity, reprocessing idempotency, Qdrant, PostgreSQL, exact reranking, PP-Structure, live Playwright, model truthfulness, and automated tests.
- BLOCKED: none.
- PENDING: none from the recorded hand-off.
- FAIL: none.
