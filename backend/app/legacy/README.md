# `app/legacy` — flag-gated research-case stack

**Nothing in this package runs in the shipping configuration.**

## Why this package exists

These 25 modules were previously in `app/services/`, sitting beside the modules
that actually serve production traffic. That was actively misleading: they carry
exactly the names a reviewer searches for — `rag.py`, `retrieval.py`,
`knowledge_graph.py`, `vector_store.py`, `embedding.py`, `ocr.py` — so anyone
asking *"does the RAG work?"* would open `app/services/rag.py`, read code the
running application never executes, and draw a conclusion about the wrong system.

Moving them here removes the trap without deleting working code or its tests.

## Where the real system is

| If you are looking for | Read |
|---|---|
| RAG / question answering | `app/services/vision/service.py` → `ask()` |
| Retrieval, embeddings, vector index | `app/services/vision/retrieval_adapters.py` |
| Knowledge graph construction | `app/services/vision/graph_builder.py` |
| OCR | `app/services/vision/ocr_adapters.py`, `ocr_lines.py` |
| Field extraction | `app/services/vision/deterministic_extraction.py`, `generic_field_discovery.py` |
| Model status / health | `app/services/model_registry.py`, `app/services/vision/model_smoke.py` |
| Persistence | `app/services/vision/persistence.py` |

## Reachability

```
app/main.py
  └── if settings.enable_demo_api:            # ENABLE_DEMO_API, default false
        ├── app/api/routes.py          ──►  /api/v1/demo-legacy/*
        └── app/api/finance_routes.py  ──►  /api/v1/demo-legacy/finance/*
                                              └──► app/legacy/*
```

With `ENABLE_DEMO_API=false` (the default in `.env.example` and `backend/.env`)
neither router is mounted and no module in this package is imported at runtime.

## Modules

Document pipeline: `pipeline.py` · `ocr.py` · `extraction.py` · `chunking.py` ·
`classification` *(stayed in `app/services/` — still used by the vision stack)*

Retrieval / RAG: `rag.py` · `retrieval.py` · `embedding.py` · `vector_store.py` ·
`knowledge_graph.py`

Analysis: `agents.py` · `compliance.py` · `cross_document.py` · `fraud.py` ·
`predictive.py` · `scoring.py` · `xai.py`

Workflow / output: `workflow.py` · `report.py` · `audit.py`

Research dataset: `dataset_loader.py` · `dataset_case_service.py`

Finance case study: `finance_workflow_service.py` · `finance_extraction_service.py` ·
`finance_narrative_service.py` · `finance_evaluation_service.py`

## Two modules deliberately stayed behind

`app/services/ingestion.py` and `app/services/classification.py` are imported by
`app/services/vision/service.py` and `app/services/vision/model_smoke.py`. They are
live production code and must not be moved here.

## Tests

`backend/conftest.py` sets `ENABLE_DEMO_API=true` for the whole test session, so
these modules are still covered by:

`app/tests/test_enterprise_mvp.py` · `app/tests/test_pipeline.py` ·
`app/tests/test_cross_document.py` · `app/tests/test_model_registry.py` ·
`tests/test_workflow.py` · `tests/test_search.py` · `tests/test_dataset_loader.py` ·
`tests/test_scoring.py` · `tests/test_extraction.py`

**When reading the backend test count, subtract these.** They verify the
research-case stack, not the shipping path.

## Corpora used only from here

`demo_data/` and `dataset/documind_ai_synthetic_dataset` are consumed exclusively
by this package. `demo-data/` (hyphen) is referenced by nothing and is a deletion
candidate.
