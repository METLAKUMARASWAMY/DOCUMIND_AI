from fastapi.testclient import TestClient

from app.main import create_app
from app.services.pipeline import DocumentIntelligencePipeline
from app.services.rag import RAGEngine

client = TestClient(create_app())


def test_exceptions_query_returns_grounded_evidence():
    pipeline = DocumentIntelligencePipeline()
    analysis = pipeline.process_demo_case()

    response = RAGEngine(pipeline.retrieval).answer("What exceptions require review?", analysis.case_id)

    assert response.citations
    assert "could not find" not in response.answer.lower()
    assert "Missing indemnity requirement" in response.answer
    assert any("indemnity" in citation.evidence_text.lower() for citation in response.citations)
    assert all(citation.source_document and citation.page_number and citation.chunk_id for citation in response.citations)


def test_api_search_returns_professional_structured_answer():
    demo = client.post("/api/v1/demo-legacy/demo/run", json={}).json()
    case_id = demo["case_id"]

    response = client.get(f"/api/v1/demo-legacy/cases/{case_id}/search", params={"q": "What exceptions require review?"})
    assert response.status_code == 200
    payload = response.json()["data"]

    assert "Exceptions requiring review:" in payload["answer"]
    assert "Missing indemnity requirement" in payload["answer"]
    assert "Manual override lacks approval evidence" in payload["answer"]
    assert "Bank account change request" in payload["answer"]
    assert payload["citations"]
    assert all(citation["source_document"] and citation["page_number"] and citation["chunk_id"] for citation in payload["citations"])


def test_payout_query_returns_direct_amount_status_and_citations():
    demo = client.post("/api/v1/demo-legacy/demo/run", json={"case_id": "case-df5ea07cd6"}).json()
    case_id = demo["case_id"]

    response = client.get(f"/api/v1/demo-legacy/cases/{case_id}/search", params={"q": "What is the expected payout?"})
    assert response.status_code == 200
    payload = response.json()["data"]

    assert "Expected payout:" in payload["answer"]
    assert "Invoice amount: USD 42,500.00" in payload["answer"]
    assert "Purchase Order amount: USD 42,500.00" in payload["answer"]
    assert "Match status: Passed" in payload["answer"]
    assert "Current payment status: Hold Payment" in payload["answer"]
    assert "Approval gate: Blocked" in payload["answer"]
    assert "vendor bank account change, urgent wire request, manual override without approval" in payload["answer"]
    assert "missing indemnity requirement" in payload["answer"]
    assert "Recommended Action:" in payload["answer"]
    documents = {citation["source_document"] for citation in payload["citations"]}
    assert {"invoice.txt", "purchase_order.txt", "email.txt", "bank_statement.txt", "policy.txt"}.issubset(documents)
