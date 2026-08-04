# API Design

Default app base path: `/api/v1`

This document describes the optional legacy research-case API. It is mounted at
`/api/v1/demo-legacy` only when `ENABLE_DEMO_API=true`. The default Vision app
uses authenticated routes such as `/api/v1/documents`, `/api/v1/reviews`,
`/api/v1/ask`, `/api/v1/reports`, and `/api/v1/admin/models`.

## Health

`GET /health` under `/api/v1/demo-legacy`, or `GET /health/live` and
`GET /health/ready` in the default app.

Returns service status and version.

## Capabilities

`GET /capabilities`

Returns enabled ingestion, AI, risk, and enterprise modules.

## Text Analysis

`POST /documents/analyze`

```json
{
  "filename": "invoice-1001.txt",
  "text": "Invoice INV-1001 Vendor: Acme Amount Due USD 12,500",
  "metadata": {
    "business_unit": "Finance"
  }
}
```

Returns document records, findings, signals, agent decisions, workflow, report URL, and graph projection.

## Demo Case

`POST /demo/run`

Runs the seeded four-document enterprise case from `demo-data/` and returns a full `AnalysisResponse`.

## File Upload

`POST /documents/upload`

Multipart fields:

- `files`: one or more files.
- `metadata`: JSON string.

## Retrieval QA

`POST /query`

```json
{
  "case_id": "case-abc123",
  "question": "Which exceptions require review?",
  "top_k": 5
}
```

Returns grounded answer, citations, confidence, and reasoning.

## Dashboard Metrics

`GET /dashboard/metrics`

Returns processed document count, open cases, average compliance score, high-risk cases, fraud alerts, and SLA breach probability.

## Cases

`GET /cases`

Returns analyzed case summaries held by the local runtime.

`GET /cases/{case_id}`

Returns the full analysis response for one case.

`GET /cases/{case_id}/evidence`

Returns the deduplicated evidence bundle collected from layout analysis, extracted fields, compliance findings, fraud signals, cross-document checks, and agent decisions.

`GET /cases/{case_id}/audit`

Returns timestamped backend audit events for upload, analysis, report export, and workflow decisions.

## Compliance And Fraud

`GET /compliance/findings?case_id={case_id}`

Returns compliance findings for a case, or all analyzed cases when `case_id` is omitted.

`GET /fraud/findings?case_id={case_id}`

Returns fraud signals for a case, or all analyzed cases when `case_id` is omitted.

## Agent Runs

`GET /agents/runs/{case_id}`

Returns multi-agent decisions, confidence, evidence, and reasoning chains for one case.

## Workflow Decision

`POST /workflow/decisions`

```json
{
  "case_id": "case-abc123",
  "approver": "jane.smith",
  "decision": "approved",
  "comment": "Approved with vendor confirmation attached."
}
```

## Knowledge Graph

`GET /graph/cases/{case_id}`

Returns the local graph projection for a case, including tenant, case, document, entity, finding, match, conflict, and evidence relationships.

`GET /graph/entities/{entity_id}?case_id={case_id}`

Returns an entity neighborhood. `case_id` is optional and scopes the lookup when provided.

`GET /graph/risk-paths?min_risk_score=50`

Returns high-risk case-to-finding paths for graph-assisted review.

`POST /graph/project/{case_id}`

Rebuilds the local graph projection from the current case analysis.

## Readiness

The later delivery phases expose local readiness snapshots:

- `GET /mlops/readiness`
- `GET /security/readiness`
- `GET /deployment/readiness`
- `GET /testing/readiness`
- `GET /research/readiness`
- `GET /portfolio/readiness`

## Report

`GET /reports/{case_id}?format=md|json|pdf`

Returns a generated markdown report, JSON report payload, or lightweight PDF export.
