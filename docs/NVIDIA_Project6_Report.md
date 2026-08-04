# NVIDIA Project 6 Report

## Problem Statement

NVIDIA Project 6 asks for a finance document-extraction and narrative-generation workflow with human review. The system must process finance documents, extract structured values, generate grounded narratives, store evidence in a VectorDB, route AI output to reviewers, measure quality, and provide demo-ready artifacts.

## Objectives

- Upload invoices, purchase orders, contracts, and vendor bank-change documents.
- Extract structured finance JSON with confidence, source evidence, missing fields, and risk flags.
- Store raw chunks, extractions, narratives, risk flags, reviewer decisions, and evaluation artifacts in a VectorDB layer.
- Generate finance-focused narratives grounded only in extracted fields and source chunks.
- Support human review decisions: approve, reject, request correction, and escalate.
- Calculate extraction, hallucination, latency, human-review, retrieval, safety, and cost metrics.
- Provide demo cases and project evidence for final presentation.

## Existing DocuMind AI Overview

DocuMind AI already provided FastAPI, React, upload, OCR/text parsing, document classification, extraction, local retrieval, RAG, fraud, compliance, workflow, audit, reports, and deployment documentation. Project 6 is implemented as a dedicated finance workflow while reusing the existing backend and frontend architecture.

## NVIDIA Project 6 Requirement Overview

The target project is a production-style GenAI finance workflow. It requires document ingestion, field extraction, narrative generation, human review, feedback/evaluation, guardrails, monitoring readiness, and a stakeholder demo.

## Final System Architecture

- Frontend: React analyst console with Finance, Human Review, Evaluation, Demo Cases, and VectorDB views.
- Backend: optional FastAPI finance demo router under `/api/v1/demo-legacy/finance` when `ENABLE_DEMO_API=true`; verified finance narratives also exist in the default Vision API.
- Services: finance extraction, narrative generation, workflow coordination, evaluation, and vector storage.
- VectorDB: local Chroma-compatible vector boundary using deterministic local hashing embeddings.
- Storage: in-memory runtime stores plus uploaded file storage under the existing upload folder.
- Demo data: finance documents and ground-truth JSON under `demo_data/`.

## Mapping DocuMind AI to NVIDIA Project 6

| NVIDIA Project 6 Requirement | DocuMind AI Implementation | Evidence / File / Module | Status | Remarks |
| --- | --- | --- | --- | --- |
| Finance document extraction | Dedicated finance extraction pipeline for invoice, PO, contract, bank-change request | `backend/app/services/finance_extraction_service.py` | Complete | Outputs structured JSON, missing fields, risk flags, confidence, evidence |
| Structured field extraction | Pydantic schemas validate extraction payloads | `backend/app/models/schemas.py` | Complete | Includes `ExtractedFinanceFields`, `SourceReference`, `RiskFlag` |
| Finance narrative generation | Grounded narrative service with narrative type selector | `backend/app/services/finance_narrative_service.py`, `frontend/src/pages/FinanceNarrativesPage.tsx` | Complete | Supports executive, payment, risk, compliance, exception, reviewer notes |
| Human review workflow | Queue, approve, reject, request correction, escalate, history, approval guardrails | `backend/app/services/finance_workflow_service.py`, `/finance/review/*` | Complete | Stores original AI output and human-corrected extraction; high-risk approval requires verification evidence |
| VectorDB storage | Chroma-compatible vector service stores chunks, extractions, narratives, risk flags, reviews, evaluation | `backend/app/services/vector_store.py` | Complete | Uses local hashing embeddings by default |
| RAG-based evidence retrieval | Finance vector search retrieves evidence and prior decisions | `/api/v1/demo-legacy/finance/vector/search`, `FinanceVectorSearch` | Complete | Existing DocuMind RAG remains available for case-level queries |
| Prompt templates | Narrative templates are deterministic and finance-focused | `finance_narrative_service.py` | Complete | No unsupported finance approval language |
| Guardrails | Prompt injection, unsupported claims, sensitive bank masking, auto type correction, unsafe approval checks | `finance_extraction_service.py`, `finance_narrative_service.py`, `finance_workflow_service.py`, `frontend/src/pages/ReviewWorkspacePage.tsx` | Complete | Bank-change requests always require review and cannot be casually approved |
| Prompt-injection protection | Detects instruction-like text inside uploaded documents | `prompt_injection_terms` in `finance_extraction_service.py` | Complete | Flagged as high-risk guardrail event |
| Hallucination detection | Unsupported narrative claims measured against fields and source text | `FinanceNarrativeService.unsupported_claims` | Complete | Evaluation reports hallucination rate |
| Feedback loop | Reviewer corrections and decisions are stored and vector-indexed | `HumanReview`, `human_review_decision` vector records | Complete | Supports acceptance categories and correction count |
| Evaluation metrics | Extraction, hallucination, latency, human review, retrieval, safety, cost | `backend/app/services/finance_evaluation_service.py` | Complete | Frontend dashboard included |
| Demo cases | Invoice, PO, contract, bank-change risk with ground truth | `demo_data/invoice/`, `demo_data/purchase_order/`, `demo_data/contract/`, `demo_data/bank_change_risk/`, `demo_data/ground_truth/` | Complete | Demo API runs full workflow |
| Monitoring | Metrics API and dashboard expose quality/latency/review metrics | `/api/v1/demo-legacy/finance/metrics`, Finance Evaluation page | Complete | Production monitoring adapters can be added later |
| Failure-case catalogue | Demo and tests cover bank-change risk, missing fields, prompt-injection detection path, unsupported claims | `backend/app/tests/test_finance_project6.py` | Complete | Can be expanded with larger UAT sets |
| Deployment readiness | Existing Docker, Kubernetes, CI, and readiness docs remain compatible | `README.md`, `docs/MLOPS_PIPELINE.md`, `docs/SECURITY_ARCHITECTURE.md` | Complete | Finance APIs run inside existing FastAPI app |
| Stakeholder demo readiness | Demo and workflow pages run the end-to-end workflow | `frontend/src/pages/UploadWizardPage.tsx`, `frontend/src/pages/ReviewQueuePage.tsx`, `frontend/src/pages/FinanceNarrativesPage.tsx` | Complete | Load demo, extract, narrate, review, evaluate, search VectorDB |

## Frontend Changes

- Finance Upload Page
- Document Type Selector
- Finance Document List
- Document Detail Page
- Extracted Fields Viewer
- Editable Extracted Fields Form
- Finance Narrative Generation Page
- Finance Narrative Viewer
- Source Evidence Panel
- Risk Flags Panel
- Human Review Dashboard
- Pending Review Queue
- Completed Reviews Page
- Review Decision Form
- Evaluation Metrics Dashboard
- Demo Cases Page
- VectorDB Search Page

## Backend Changes

- Added finance router with upload, extraction, narrative, review, evaluation, vector, and demo endpoints.
- Added finance schemas to the central Pydantic schema module.
- Added modular finance services for extraction, narrative generation, evaluation, workflow orchestration, and vector storage.
- Added classifier support for bank-change request documents.
- Added auto-detection override when bank-change risk content is uploaded under the wrong document type.
- Added approval guardrail requiring independent verification notes before high-risk or bank-change approval.
- Added backend tests for Project 6 workflow behavior.

## VectorDB Design

Each finance vector record stores:

- `document_id`
- `document_type`
- `file_name`
- `uploaded_by`
- `upload_timestamp`
- `page_number`
- `chunk_id`
- `extraction_version`
- `review_status`
- `risk_level`
- `source_type`
- `created_at`
- `text`
- `metadata`

Stored source types include raw chunks, extracted JSON, finance narrative, risk flag, human review queue, human review decision, final approved narrative, and evaluation result.

## Finance Extraction Design

The extraction service uses deterministic finance rules and produces JSON for:

- Invoice values
- Purchase order values
- Contract clauses
- Bank-change risk fields

It includes confidence scores, source references, missing fields, risk flags, masked bank accounts, and original extraction before human correction.

## Finance Narrative Generation Design

The narrative generator supports:

- Executive Summary
- Payment Summary
- Risk Summary
- Compliance Summary
- Exception Summary
- Reviewer Notes Draft

Narratives are grounded in extracted fields and source text. Unsupported amounts, dates, IDs, or bank details are flagged.

## Human Review Workflow

Statuses:

- Uploaded
- Extracted
- Narrative Generated
- Pending Review
- Approved
- Needs Correction
- Rejected
- Escalated

Reviewers can edit extracted fields, approve narratives, reject outputs, request correction, escalate risky documents, and add comments. Decisions are stored and vector-indexed.

For bank-change and high-risk finance documents, approval is guarded. The UI defaults reviewers toward escalation/correction, and the backend blocks approval unless reviewer comments document independent trusted verification. This prevents a bank-change request from being approved merely because a reviewer selected `Approve`.

Manual upload now defaults to auto-detection. If bank-change language is detected while the user selected another document type, the workflow overrides the document to `bank_change_request`, stores the selected/detected type metadata, and shows a frontend warning.

## Evaluation Metrics

Implemented metrics:

- Field-level exact match accuracy
- Normalized field accuracy
- Precision, recall, F1
- Line-item extraction accuracy
- Amount mismatch rate
- Missing field rate
- Hallucination rate
- Missing citation count
- Hallucinated amount/date/bank-detail counts
- Groundedness score
- Upload, extraction, embedding, VectorDB, narrative, review, and end-to-end latency
- Human review acceptance rate
- Correction count
- Retrieval top-k hit rate
- Unsafe response rate
- Prompt-injection detection/failure rate
- Cost per 100 requests placeholder for local mode

## Demo Cases

- Invoice Demo: extracts invoice payment details and generates payment narrative.
- Purchase Order Demo: extracts procurement values and approval status.
- Contract Demo: extracts clauses, obligations, missing/risky clauses.
- Bank-Change Risk Demo: detects account change, masks bank details, escalates for human verification.
- Manual Bank-Change Safety Demo: uploading `bank_change_risk_demo.txt` as `Invoice` is auto-corrected to `Bank Change Request`, produces a bank-change payment-hold narrative, blocks unsafe approval, and supports verified approval only with independent verification comments.

## Failure-Case Testing

Current tests cover:

- Invoice extraction and masking
- Narrative grounding
- VectorDB retrieval
- Human review approval
- Bank-change escalation and wrong-type upload override
- Unsafe bank-change approval blocking
- Demo case availability
- Evaluation metrics

## Prompt-Injection Testing

The finance extraction service checks uploaded documents for instruction-like attacks such as ignoring prior instructions, bypassing review, revealing prompts, or approving payment automatically. These are flagged as high-risk guardrail events.

## Limitations

- The default VectorDB is local and Chroma-compatible; production Chroma persistence can be enabled as a future adapter.
- Extraction is deterministic and rule-based for demo reliability; production can add OCR/LayoutLM/LLM extraction adapters.
- Cost metrics are zero in local deterministic mode.
- Metadata persistence is runtime-local unless connected to a database.

## Future Improvements

- Add persistent ChromaDB collections.
- Add database tables for finance documents, reviews, and evaluations.
- Add OCR layout coordinates and page thumbnails.
- Add larger finance evaluation datasets.
- Add LLM provider adapters for narrative generation with strict guardrails.
- Add reviewer identity/RBAC enforcement on finance endpoints.

## Conclusion

DocuMind AI now satisfies NVIDIA Project 6 as a complete finance-focused GenAI workflow. It can upload finance documents, extract structured fields, store evidence in a VectorDB layer, generate grounded narratives, route output through human review, calculate evaluation metrics, run demo cases, and support final presentation.
