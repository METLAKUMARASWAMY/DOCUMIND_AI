from fastapi.testclient import TestClient

from app.legacy.pipeline import DocumentIntelligencePipeline
from app.legacy.rag import RAGEngine
from app.main import create_app
from app.models.schemas import TextAnalysisRequest
from app.tests.auth_helpers import auth_headers

client = TestClient(create_app())


def _auth_headers() -> dict[str, str]:
    return auth_headers(client)


def test_demo_case_surfaces_required_enterprise_findings():
    service = DocumentIntelligencePipeline()
    response = service.process_demo_case()

    compliance_titles = {finding.title for finding in response.compliance_findings}
    fraud_titles = {signal.title for signal in response.fraud_signals}
    check_statuses = {check.title: check.status for check in response.cross_document_checks}

    assert "Missing indemnity requirement" in compliance_titles
    assert any(finding.approval_blocking for finding in response.compliance_findings)
    assert "Vendor bank account change request" in fraud_titles
    assert "Manual override without approval" in fraud_titles
    assert check_statuses["Invoice amount matches po amount"] == "Passed"
    assert response.workflow.recommended_action.value in {"hold_payment", "request_changes"}
    assert response.workflow.approval_blocked is True


def test_rag_retrieves_generated_findings_and_citations():
    service = DocumentIntelligencePipeline()
    response = service.process_demo_case()
    answer = RAGEngine(service.retrieval).answer(
        "What exceptions require review?", response.case_id
    )

    assert answer.citations
    assert any("indemnity" in citation.evidence_text.lower() for citation in answer.citations)
    assert all(citation.chunk_id for citation in answer.citations)
    assert all(citation.source_document for citation in answer.citations)
    assert all(citation.confidence > 0 for citation in answer.citations)


def test_single_document_fraud_is_low_confidence_when_verification_missing():
    service = DocumentIntelligencePipeline()
    response = service.process_text(
        TextAnalysisRequest(
            filename="invoice-only.txt",
            text="Invoice INV-7000\nVendor: Example Co\nAmount Due USD 100.00\nPayment terms Net 30",
        )
    )

    assert response.cross_document_checks == []
    assert response.fraud_confidence == "low"
    assert response.fraud_probability < 0.1


def test_report_exports_are_backend_generated():
    case = client.post("/api/v1/demo-legacy/demo/run").json()
    case_id = case["case_id"]

    headers = _auth_headers()

    json_report = client.get(
        f"/api/v1/demo-legacy/reports/{case_id}", params={"format": "json"}, headers=headers
    )
    assert json_report.status_code == 200
    assert json_report.json()["case_id"] == case_id
    assert json_report.json()["success"] is True
    assert json_report.json()["data"]["compliance_report"]

    pdf_report = client.get(
        f"/api/v1/demo-legacy/reports/{case_id}", params={"format": "pdf"}, headers=headers
    )
    assert pdf_report.status_code == 200
    assert pdf_report.headers["content-type"] == "application/pdf"
    assert pdf_report.content.startswith(b"%PDF")


def test_multipart_document_upload_runs_full_analysis():
    response = client.post(
        "/api/v1/demo-legacy/documents/upload",
        files=[
            (
                "files",
                (
                    "invoice.txt",
                    b"Invoice INV-88\nVendor: Upload Co\nAmount Due USD 200.00\nPO Reference: PO-88",
                    "text/plain",
                ),
            ),
            (
                "files",
                (
                    "purchase_order.txt",
                    b"Purchase Order PO-88\nVendor: Upload Co\nAmount USD 200.00",
                    "text/plain",
                ),
            ),
        ],
        data={"metadata": '{"source":"pytest"}'},
    )

    assert response.status_code == 200
    payload = response.json()["data"]
    assert len(payload["documents"]) == 2
    assert payload["evidence_chunks"]
    assert payload["cross_document_checks"]
