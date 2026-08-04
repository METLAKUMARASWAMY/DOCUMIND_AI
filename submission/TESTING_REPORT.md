# Testing Report

## Backend Tests
Command:
```bash
cd backend
source .venv/bin/activate
pytest
```

Actual result on final validation: 26 passed.

## Test Cases
- Dataset loader loads cases, documents, evidence, findings, workflow, agents, graph, RAG questions, and NER records. Pass.
- Compliance score calculation follows severity penalties. Pass.
- Fraud probability and risk score follow formula. Pass.
- Search for "What exceptions require review?" returns evidence and does not return "no evidence found". Pass.
- Workflow defaults high-risk demo to hold_payment and blocks approve. Pass.
- Markdown, JSON, and PDF reports are generated. Pass.
- Auth login returns a bearer token and user role. Pass.
- Auth `/me` returns the active reviewer from the token. Pass.
- Signup creates a local reviewer account. Pass.
- Payout query returns direct amount, payment status, recommended action, and citations. Pass.
- Multipart upload runs full analysis. Pass.
- Vendor/customer extraction does not mislabel customer as vendor. Pass.
- Negated indemnity language creates a missing-clause finding and no extracted indemnity clause. Pass.
- Backend search returns structured exception answers with document, page, confidence, and chunk IDs. Pass.

## Frontend Checks
Command:
```bash
cd frontend
npm install
npm run build
```

Actual result on final validation: production build successful.

## Quality Checks
Command:
```bash
cd backend
ruff check app tests
```

Actual result: all checks passed.
