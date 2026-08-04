# Project Report

## Abstract
DocuMind AI is an enterprise document intelligence platform that analyzes multi-document business cases, grounds outputs in evidence, detects compliance and fraud risks, and routes workflow decisions.

## Introduction
Manual review of contracts, invoices, purchase orders, policies, and payment emails is slow and error-prone. Enterprises require traceability, explainability, and safe approval gates.

## Problem Statement
Existing OCR demos extract text but do not connect findings, evidence, risk, graph, workflow, and reports into one auditable decision system.

## Literature/Research Gap
Traditional document AI focuses on classification or OCR. The gap is end-to-end reasoning across documents with evidence citations and workflow automation.

## Proposed System
The proposed system ingests documents, authenticates reviewers, extracts entities, chunks evidence, retrieves citations, detects compliance and fraud issues, verifies cross-document consistency, builds graph relationships, runs agents, and generates reports.

## Modules
Login/signup/auth, ingestion, classification, extraction, retrieval/RAG, compliance, fraud, cross-document checks, graph, agents, workflow, audit, reports, frontend dashboard, deployment.

## Implementation
The backend uses FastAPI and Pydantic. The frontend uses React, TypeScript, MUI, Recharts, custom CSS, and lucide icons. The synthetic dataset is the main source of truth.

## Results
The demo case `case-df5ea07cd6` shows missing indemnity, manual override without approval, vendor bank account change, urgent wire wording, expected payout of USD 42,500.00, risk score 100/100, and workflow action `hold_payment`. The dashboard, fraud page, workflow view, agent center, and report use the same backend score object.

## Screenshots Placeholders
- Dashboard
- Login and Signup
- Search
- Search payout query
- Compliance
- Fraud
- Graph
- Workflow
- Reports
- Agent Center

## Conclusion
DocuMind AI demonstrates evidence-grounded enterprise document intelligence suitable for academic submission and further production extension.

## Future Scope
Integrate persistent PostgreSQL, ChromaDB/FAISS, Neo4j, advanced OCR, RVL-CDIP classifier training, production identity provider RBAC, and human evaluation datasets.
