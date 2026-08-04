# API Documentation

All page APIs return:
```json
{ "success": true, "case_id": "case-df5ea07cd6", "data": {}, "errors": [] }
```

## Endpoints
- `GET /health/live` and `GET /health/ready` - health status.
- `POST /api/v1/auth/register` or `/api/v1/auth/signup` - create a local viewer account with `name`, `email`, `password`, and organisation.
- `POST /api/v1/auth/login` - returns bearer token, expiry, and user role.
- `GET /api/v1/auth/me` - returns the current authenticated user from the bearer token.
- `POST /api/v1/auth/logout` - revokes the current local token.
- `POST /api/v1/documents/upload` - upload TXT/PDF/DOCX/images. Requires `Authorization: Bearer <token>`.
- `GET /api/v1/documents` - list processed Vision documents.
- `POST /api/v1/ask` - citation-grounded Vision question answering.
- `GET /api/v1/reviews` - human review queue.
- `POST /api/v1/reports/generate` - generate report records.
- `GET /api/v1/reports/{report_id}?format=md|json|pdf` - report export. Requires `Authorization: Bearer <token>`.
- Optional legacy dataset endpoints are under `/api/v1/demo-legacy`, including `/cases`, `/demo/run`, `/query`, `/reports/{case_id}`, and `/finance/*`, when `ENABLE_DEMO_API=true`.

## Local Auth
Use signup for viewer accounts. Use `python scripts/bootstrap_auth_users.py --role ADMIN --email admin@documind.local` for local admin/reviewer demos. Supported roles are `ADMIN`, `PROCESSOR`, `REVIEWER`, `AUDITOR`, and `VIEWER`; legacy role strings are mapped to these canonical roles.

## Sample Search Response
```json
{
  "success": true,
  "case_id": "case-df5ea07cd6",
  "data": {
  "answer": "Exceptions requiring review:\n1. Missing indemnity requirement - High severity",
  "citations": [
    {
      "source_document": "contract.txt",
      "page_number": 1,
      "chunk_id": "ev-case-df5ea07cd6-contract-001",
      "confidence": 0.9
    }
  ]
  },
  "errors": []
}
```

## Sample Payout Search Response
Question: `What is the expected payout?`

```json
{
  "success": true,
  "case_id": "case-df5ea07cd6",
  "data": {
    "answer": "Direct Answer:\n- Invoice amount: USD 42,500.00\n- PO amount: USD 42,500.00\n- Amount match status: Passed\n- Payment terms: Net 30\n- Due date: 2026-02-15\n- Current payment status: Hold payment / approval blocked",
    "citations": [
      { "source_document": "invoice.txt", "page_number": 1, "chunk_id": "chk-case-df5ea07cd6-invoice-001", "confidence": 0.52 },
      { "source_document": "purchase_order.txt", "page_number": 1, "chunk_id": "chk-case-df5ea07cd6-purchase_order-001", "confidence": 0.52 }
    ]
  },
  "errors": []
}
```

## Sample Dashboard Scores
```json
{
  "success": true,
  "case_id": "case-df5ea07cd6",
  "data": {
    "scores": {
      "compliance_score": 82,
      "fraud_probability_score": 70,
      "risk_score": 100,
      "risk_severity": "critical",
      "approval_blocked": true
    },
    "workflow": {
      "recommended_action": "hold_payment",
      "approval_blocked": true
    }
  },
  "errors": []
}
```
