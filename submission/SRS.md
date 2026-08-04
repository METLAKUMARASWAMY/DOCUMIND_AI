# Software Requirements Specification

## Introduction
DocuMind AI assists enterprise reviewers in analyzing contracts, invoices, purchase orders, policy documents, emails, and bank evidence.

## Objectives
- Reduce manual document review time.
- Ground all answers in document evidence.
- Detect compliance and fraud risks.
- Route approvals safely.
- Preserve auditability for enterprise review.

## Functional Requirements
- Upload TXT, PDF, DOCX, and image documents with validation.
- Load demo cases from the synthetic dataset.
- Classify documents and extract entities.
- Create evidence chunks and retrieve relevant citations.
- Detect missing clauses and policy mismatches.
- Detect duplicate invoice, amount mismatch, bank change, urgent wording, manual override, vendor anomaly, and fake document signals.
- Run cross-document checks.
- Generate graph nodes and relationships.
- Generate workflow recommendations and audit events.
- Export reports as Markdown, JSON, and PDF.
- Authenticate users with local login/signup/logout for the demo.
- Persist local auth token in the browser and show the active role badge.
- Protect report export and workflow decision APIs.
- Answer payout/payment queries with direct amount, payment status, blockers, citations, and recommended action.

## Non-Functional Requirements
- Consistent API responses.
- Traceable evidence citations.
- Secure file validation and checksums.
- Configurable dataset paths and secrets.
- Local demo operation without paid APIs.

## User Roles
Admin, Compliance Officer, Finance Reviewer, Legal Reviewer, Risk Analyst, and Auditor.

## Demo Credentials
Local users are created through signup. Privileged demo roles are bootstrapped with `scripts/bootstrap_auth_users.py`.

## System Constraints
The local MVP uses deterministic services and fallback retrieval. ChromaDB, Neo4j, OCR engines, and RVL-CDIP are optional upgrade paths.
