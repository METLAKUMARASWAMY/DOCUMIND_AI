# Pending tasks at session cutoff

Cutoff: 2026-08-06 12:50 IST

The core implementation is saved and the automated quality gates pass. The items below are genuine external or bounded-runtime follow-ups; none is hidden behind an `active` flag.

## 1. Authenticate the user's Neo4j Desktop DBMS

- Target: `neo4j://127.0.0.1:7687`
- Instance: `22201bd7-b52f-473a-a809-0c14666764e2`
- Server binary: `2026.07.0`
- Current result: endpoint/version verified, authentication rejected because the repository's old Docker credential is not the Desktop credential.
- Safe next action: set the matching `NEO4J_USER` and `NEO4J_PASSWORD` (or `NEO4J_AUTH`) in the ignored `backend/.env`, then run:

```bash
ENABLE_NEO4J=true .venv-ai/bin/python scripts/smoke_test_models.py --model Neo4j --strict --compact
.venv/bin/python scripts/rebuild_graph.py --organisation documind-demo --execute
```

Do not reset the Desktop password or delete either Neo4j volume merely to make a smoke test green.

## 2. Rerun the final live Playwright journey

- The live API upload, processing, six-field viewer, seven counters, and idempotent report generation completed.
- The first browser run exposed and fixed the report enum in the test.
- The second run reached Ask Documents but timed out because duplicate filenames made an exact accessible-name selector too strict.
- The selector is now fixed to select the first option containing the filename, but the 12:50 cutoff prevented another full browser run.

```bash
# Start backend on 8000 and frontend on 5174, then provide a bootstrapped admin:
cd frontend
DOCUMIND_E2E_EMAIL='...' DOCUMIND_E2E_PASSWORD='...' \
  npx playwright test e2e/core-intelligence-v4.spec.ts --reporter=list
```

## 3. Optional model assets that did not complete

- `BAAI/bge-reranker-v2-m3`: exact ~2.2 GB Xet download stalled beyond the bounded window; partial cache is not active. Local lexical/dense ranking remains the fallback.
- `BAAI/bge-reranker-base`: compatible fallback download also stalled; not active.
- `PP-StructureV3 / PP-DocLayoutV3`: OCR extra was installed and several submodels downloaded, but formula/layout assets did not finish inside the bounded window; not active. PP-OCRv5 itself passed real inference.
- LayoutLMv3, Donut, DINOv2, YOLO, DeBERTa/CUAD, SegFormer and XGBoost remain conditional on project-trained or explicitly configured checkpoints.
- External LLM remains disabled because no provider credentials were supplied; the verified local `google/flan-t5-small` grounded generator is active.

## 4. Operational follow-up

- Rotate the local Neo4j/Docker credential if it has been exposed outside the ignored environment file.
- Review and commit future model cache downloads only as deployment metadata; never commit model weights, `.env`, `var/`, uploads, or credentials.
- The PostgreSQL database is connected, migrated, checksum-tracked, and already mirrors v4 completeness/timing/candidate/conversation/retrieval records. Local JSON remains the recoverable document source of truth by design.

## Completed verification baseline

- Backend: 127 tests passed; Ruff passed; diff check clean.
- Frontend: 18 tests passed; production build and lint passed.
- PostgreSQL 17.10: authenticated, 37 tables, three migrations checksum-tracked, second execution idempotent.
- Exact fixture: six selected fields, six required matched, zero required missing, zero duplicate persisted fields.
- Qdrant: live scoped replacement/query/filter/delete passed; BGE-M3 dimension-safe collection passed.
- Local GraphRAG: 16 unique nodes, 21 unique edges, evidence-linked `REQUESTED_BY` path passed.
