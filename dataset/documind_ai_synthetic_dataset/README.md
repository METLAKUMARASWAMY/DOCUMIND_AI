# DocuMind AI Synthetic Enterprise Document Dataset

This is a synthetic dataset for the project:

**DocuMind AI – Intelligent Enterprise Document Intelligence Platform**

It is designed for local development, demos, testing, and model/pipeline prototyping.

## Contents

- `documents/`: synthetic enterprise document bundles grouped by case.
- `labels/cases.csv`: case-level risk, fraud, compliance, and workflow labels.
- `labels/documents.csv`: document inventory and document type labels.
- `labels/entities.csv`: extracted entities such as vendors, amounts, PO numbers, invoice numbers, clauses.
- `labels/evidence_chunks.csv`: citation/evidence chunks for RAG.
- `labels/compliance_findings.csv`: missing clauses and compliance issues.
- `labels/fraud_findings.csv`: fraud and payment anomaly labels.
- `labels/cross_document_checks.csv`: invoice-to-PO and invoice-to-contract verification labels.
- `labels/workflow_decisions.csv`: approval, hold, escalation, or request-change recommendations.
- `labels/agent_decisions.csv`: multi-agent output labels.
- `labels/knowledge_graph_edges.csv`: graph relationship seed data.
- `labels/train_document_classification.csv`: text samples for document classification.
- `labels/ner_records.jsonl`: entity extraction examples.
- `labels/rag_questions.csv`: retrieval QA evaluation set.

## Suggested Use

1. Upload files from one case folder into your app.
2. Parse and chunk the documents.
3. Store chunks in ChromaDB/FAISS/pgvector.
4. Run compliance and fraud rules.
5. Compare your app output against the CSV labels.

## Important

All data is synthetic and safe for academic/project use. It is not real financial, legal, banking, insurance, or healthcare data.



cd "/Users/kumar/Downloads/DocuMind AI/backend"
source .venv/bin/activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000



cd "/Users/kumar/Downloads/DocuMind AI/frontend"
npm run dev -- --host 127.0.0.1 --port 5174