from fastapi.testclient import TestClient

from app.main import create_app
from app.tests.auth_helpers import auth_headers

client = TestClient(create_app())


def _auth_headers() -> dict[str, str]:
    return auth_headers(client)


def test_markdown_json_and_pdf_reports_are_generated():
    demo = client.post("/api/v1/demo-legacy/demo/run", json={})
    assert demo.status_code == 200
    case_id = demo.json()["case_id"]
    headers = _auth_headers()

    markdown = client.get(f"/api/v1/demo-legacy/reports/{case_id}", params={"format": "md"}, headers=headers)
    assert markdown.status_code == 200
    assert "DocuMind AI Report" in markdown.text
    assert "Cross-Document Checks" in markdown.text
    assert "risk is critical at 100/100" in markdown.text
    assert "risk is critical at 10/100" not in markdown.text
    assert "Recommended workflow action is Hold Payment." in markdown.text

    json_report = client.get(f"/api/v1/demo-legacy/reports/{case_id}", params={"format": "json"}, headers=headers)
    assert json_report.status_code == 200
    assert json_report.json()["case_id"] == case_id
    assert json_report.json()["success"] is True
    assert json_report.json()["data"]["scores"]["risk_score"] == demo.json()["data"]["risk_score"]
    assert "risk is critical at 100/100" in json_report.json()["data"]["executive_summary"]

    pdf_report = client.get(f"/api/v1/demo-legacy/reports/{case_id}", params={"format": "pdf"}, headers=headers)
    assert pdf_report.status_code == 200
    assert pdf_report.headers["content-type"] == "application/pdf"
    assert pdf_report.content.startswith(b"%PDF")
