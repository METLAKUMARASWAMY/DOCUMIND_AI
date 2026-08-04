# Core Intelligence v3 Acceptance Report

Generated: 2026-08-04

## 1. Repository audit summary

The failure chain was traced through the classifier, schema registry, word-only OCR,
deterministic extractor, evidence chunks, Ask routing, graph projection, language
routing, artefact listing, status UI, and model smoke gates. The backlog-to-file
mapping was:

| Backlog | Primary implementation |
| --- | --- |
| Classification and taxonomy | `document_classification.py`, `vision.py` |
| Schema hierarchy and validators | `schema_registry.py` |
| OCR geometry and extraction | `ocr_lines.py`, `deterministic_extraction.py` |
| Pipeline, Ask, chunks, status | `service.py`, `ProcessingStatusPage.tsx`, `AskDocumentsPage.tsx` |
| Graph and analytics | `graph_builder.py`, `analytics.py`, frontend graph/analytics pages |
| Legacy and duplicate cleanup | `legacy_audit.py`, three scripts in `scripts/` |
| Runtime truthfulness | `model_smoke.py`, admin/model UI, dependency groups |

## 2. Files changed

Backend changes cover API role guards, models, persistence, classification, OCR
lines, schema selection, deterministic extraction, pipeline routing, graph,
analytics, language routing, model smoke gates, and approval policy. Frontend
changes cover Ask scope, processing semantics, analytics, graph evidence, reports,
legacy warnings, model status, and fraud-role visibility. Tests, installation
guidance, dependency groups, cleanup tools, and one MongoDB migration were added.

## 3. Database migrations

`backend/migrations/003_core_intelligence_v3.mongodb.js` adds tenant-scoped OCR-line
reading-order and identity indexes, the narrative identity/version unique index,
and a pipeline-policy audit index. The migration is ready but was not applied to
MongoDB because this local runtime has MongoDB disabled. JSON-state legacy flags
and duplicate archives were applied separately and safely; no evidence was deleted.

## 4. New schemas

The exact taxonomy now includes invoice, purchase order, contract, receipt, bank
document, bank confirmation, fax transmission, business form, legal form,
insurance document, financial report, general report, correspondence, letter,
statement, and unknown. Selection follows exact schema, parent schema, generic
key-value form, then generic document metadata. Fax transmission has all 17
requested fields. Generic key-value and generic metadata schemas remain available
when a specialized schema cannot be selected.

## 5. Classification rules

`hybrid-document-classifier-v3` evaluates explicit override, strong title terms,
layout, OCR features, statistical hints, and confidence fallback. Strong fax,
invoice, purchase-order, bank-confirmation, agreement/contract, and report titles
carry explicit weights. The real fax was classified as `fax_transmission` at 0.94
from the `FAX TRANSMISSION` title, with `business_form` as a 0.72 alternative.

## 6. Extraction aliases

The v3 extractor resolves OCR-tolerant aliases for fax, phone, date, recipient,
sender, company, client/reference number, title, time, subject, pages, message,
email, address, amount, account, signature, and document-specific labels. It also
handles observed corruptions such as `ATE`, `SLIENTNO`, and `MESSAGETO`, while
keeping raw OCR values and never silently correcting source text.

## 7. Query-routing flow

Scope validation -> greeting/help -> structured field and alias lookup -> OCR
label-value lookup -> OCR line search -> local lexical/hybrid chunks -> graph lookup
-> clarification -> not found. Current, selected, approved, and accessible scopes
are explicit; the service never broadens a query silently. Every supported factual
answer is returned with document, page, source text, and evidence geometry.

## 8. Model runtime table

| Runtime | Result | Evidence |
| --- | --- | --- |
| Real deterministic fax extraction v3 | PASS | 8.776 ms, 165 words, 15 lines, 11 labels, 11 fields |
| Local model smoke stack | PASS | 11 passed, 0 failed |
| Advanced model/service smoke gates | BLOCKED | 16 blocked, inference `not_run`, schema `not_evaluated` |
| LayoutLMv3 training activation | BLOCKED | 60 source records, 0 valid document records, no field labels |

The 16 blocked gates are PP-OCRv5, PP-Structure/DocLayout, LayoutLMv3,
PaddleOCR-VL/advanced document adapters, Donut, DINOv2, YOLO, DeBERTa-CUAD,
SegFormer, XGBoost, BGE-M3, BGE reranker, Qdrant, Neo4j, Qwen, and the external
LLM provider. Each remains disabled with its exact package, checkpoint, service,
credential, or label blocker and a preparation command.

## 9. Test commands

```bash
cd backend && .venv/bin/pytest -q
cd backend && .venv/bin/ruff check app ../scripts/audit_legacy_documents.py ../scripts/reprocess_documents.py ../scripts/archive_duplicate_artifacts.py
cd frontend && npm run lint
cd frontend && npm test -- --reporter=dot
cd frontend && npm run build
backend/.venv/bin/python scripts/smoke_test_models.py
backend/.venv/bin/python scripts/prepare_layoutlmv3_dataset.py --audit-only
```

## 10. Test results

- Backend: PASS, 93 passed; 43 deprecation warnings; 0 failures.
- Frontend: PASS, 4 files and 15 tests; lint and production build passed.
- Browser: PASS at 1440x1000 and 390x844; no overflow or browser errors.
- Model smoke: PASS truthfulness gate; 27 total, 11 passed, 16 blocked, 0 failed.
- Live API: PASS readiness, authenticated document retrieval, reprocess, analytics,
  and citation-backed Ask against `vision-doc-e563d27d31`.

## 11. Sample extracted JSON

```json
{
  "document_type": "fax_transmission",
  "selected_schema": "fax_transmission",
  "extraction_outcome": "LOW_CONFIDENCE_FIELDS",
  "fields_extracted": 11,
  "recipient_fax": {
    "raw_value": "910/535-7707",
    "normalized_value": "910-535-7707",
    "validation_status": "valid",
    "page": 1,
    "bounding_box": {"x1": 103, "y1": 350, "x2": 279, "y2": 381}
  }
}
```

The low-confidence outcome is intentional: fields exist and are source-backed,
but OCR noise remains in values such as the recipient name and message, so human
review is required rather than overstating accuracy.

## 12. Sample citation-backed answer

```json
{
  "answer": "The fax number shown is 910/535-7707.",
  "answer_status": "supported",
  "retrieval_path": "structured_field_lookup",
  "page_number": 1,
  "source_text": "FAX NUMBER: 910/535-7707",
  "bounding_boxes": [{"x1": 103, "y1": 350, "x2": 279, "y2": 381}]
}
```

A separate test removes structured fields and returns the same grounded answer via
`ocr_label_value_fallback`.

## 13. Sample graph output

The real fax produces 22 nodes and 30 edges. Its `HAS_FAX` edge links an
Organisation to a ContactMethod with source field `recipient_fax`, page 1,
confidence 0.7252, proposed/unverified status, the source text above, and the same
bounding box. Document, person, organisation, contact, date, client-number, and
evidence-linked relationship data are present.

## 14. Legacy-data cleanup results

For `tenant-documind-demo`, 13 legacy documents were safely flagged: 13 outdated
pipeline, 8 reprocess required, 6 stale language review reasons, and 1 legacy
approval re-verification. The real fax was reprocessed in place. One duplicate
narrative and one duplicate report were archived while their canonical active
records were retained. No source document or audit evidence was deleted.

## 15. Remaining genuine blockers

Advanced model activation remains blocked until compatible packages/checkpoints,
service health, credentials, or labelled datasets exist. In particular,
LayoutLMv3 is `BLOCKED_BY_LABELS`; Qdrant and Neo4j are disabled because their
services are not configured/running; BGE and Qwen lack permitted local checkpoints;
and no external LLM credential is configured. The deterministic OCR-backed local
pipeline remains fully operational without them. Platform-specific preparation is
documented in `docs/AI_INSTALLATION.md`.

## 16. Final acceptance checklist

- PASS - Fax transmission documents are classified correctly.
- PASS - Exact or generic schema is selected for readable forms.
- PASS - Readable label-value documents produce structured fields.
- PASS - Fax number is extracted with bounding-box evidence.
- PASS - Phone number is extracted with bounding-box evidence.
- PASS - Date is extracted with bounding-box evidence.
- PASS - Completed extractor runs store runtime provenance.
- PASS - Ask searches fields, aliases, OCR pairs, OCR lines, chunks, and graph.
- PASS - The fax question returns a supported citation-backed answer.
- PASS - Wrong scope does not hallucinate or broaden silently.
- PASS - Graph entity labels are readable.
- PASS - Graph edges contain source evidence.
- PASS - English OCR noise does not trigger mixed-language routing.
- PASS - Reprocessing removes obsolete routing reasons and replaces chunks.
- PASS - Legacy approvals without evidence are flagged.
- PASS - Generate-or-reuse exposes one canonical narrative card.
- PASS - Pipeline completion and review decision are displayed separately.
- PASS - Analytics exposes structured extraction coverage.
- PASS - Confirm Fraud is role-restricted and evidence-gated.
- PASS - Blocked models display `not_evaluated`, not schema invalid.
- PASS - No model is active without a passing smoke test.
- PASS - No fields are fabricated.
- PASS - No factual answers are emitted without evidence.
- PASS - All backend and frontend tests pass.

Final gate: 24 PASS, 0 FAIL, 0 BLOCKED. Advanced optional model activations remain
separately and truthfully BLOCKED as listed above.
