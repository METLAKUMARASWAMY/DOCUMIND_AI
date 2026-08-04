# Testing Strategy

## Backend

- Unit tests for classification, extraction, compliance, fraud, cross-document checks, retrieval, RAG, workflow, and reporting.
- API tests for upload, analyze, query, metrics, decisions, and reports.
- Contract tests for response schemas.
- Golden datasets for invoices, contracts, POs, statements, policies, and emails.

## Frontend

- Component tests for dashboard, upload, search, risk, graph, and workflow views.
- Accessibility checks for keyboard navigation and contrast.
- Visual regression screenshots for desktop and mobile.
- API mocking for offline UI states.

## AI Quality

- Retrieval evaluation: recall@k, nDCG, citation precision.
- Extraction evaluation: exact match and normalized match.
- Compliance evaluation: recall for critical missing clauses.
- Fraud evaluation: precision, recall, cost-weighted loss, reviewer acceptance.
- Explanation evaluation: evidence presence, source accuracy, page accuracy, confidence calibration.

## Security and Reliability

- SAST and dependency scanning in CI.
- Fuzzing for parsers and file upload boundaries.
- Load tests for concurrent upload and query.
- Disaster recovery tests for PostgreSQL, vector store, graph store, and object storage.

