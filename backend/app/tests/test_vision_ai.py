from fastapi.testclient import TestClient

from app.main import create_app
from app.tests.auth_helpers import auth_headers

client = TestClient(create_app())


def _token() -> str:
    return auth_headers(client)["Authorization"].split(" ", 1)[1]


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_token()}"}


def test_vision_text_pipeline_preserves_boxes_tables_risk_and_review():
    response = client.post(
        "/api/v1/vision/documents/analyze",
        headers=_headers(),
        json={
            "filename": "vision-invoice.txt",
            "text": (
                "Invoice INV-1042\n"
                "Vendor: Vendor A\n"
                "Invoice Date: 2026-03-01\n"
                "Due Date: 2026-03-20\n"
                "Bank Account: ACCT-99999999\n"
                "Subtotal: INR 400000.00\n"
                "Tax: INR 25000.00\n"
                "Discount: INR 0.00\n"
                "Total Amount: INR 430000.00\n"
                "item | qty | price | tax | total\n"
                "service | 1 | 400000 | 25000 | 430000\n"
                "Signature: Finance Manager\n"
            ),
            "metadata": {
                "approved_bank_identifier": "ACCT-11112222",
                "required_signature": True,
            },
        },
    )

    assert response.status_code == 200
    document = response.json()["data"]

    assert document["document_type"] == "invoice"
    assert document["ocr_words"]
    assert document["ocr_words"][0]["bounding_box"]["x2"] > document["ocr_words"][0]["bounding_box"]["x1"]
    assert document["tables"]
    assert any(item["detection_class"] == "signature" for item in document["visual_detections"])
    assert any(item["title"] == "Amount Arithmetic Mismatch" for item in document["risk_indicators"])
    assert any(item["title"] == "Bank Detail Mismatch" for item in document["risk_indicators"])
    assert document["status"] == "REVIEW_REQUIRED"
    assert document["review_task"]
    assert "PaddleOCR-VL-1.6" in {
        item["model_id"] for item in document["model_routing"]["fallback_models"]
    }


def test_vision_upload_endpoint_processes_files_with_background_task():
    response = client.post(
        "/api/v1/vision/documents/upload",
        headers=_headers(),
        data={"metadata": '{"required_signature": true}'},
        files={
            "files": (
                "uploaded-vision-invoice.txt",
                b"Invoice INV-7777\nVendor: Upload Vendor\nTotal Amount: USD 50.00\n",
                "text/plain",
            )
        },
    )

    assert response.status_code == 200
    document_id = response.json()["data"]["document_ids"][0]
    assert response.json()["data"]["dispatch"]["mode"] == "fastapi_background_task"
    detail = client.get(f"/api/v1/vision/documents/{document_id}", headers=_headers())

    assert detail.status_code == 200
    assert detail.json()["data"]["filename"] == "uploaded-vision-invoice.txt"
    assert detail.json()["data"]["ocr_words"]


def test_vision_evaluation_requires_labels_and_calculates_when_supplied():
    empty = client.post("/api/v1/vision/evaluation/run", json={})
    assert empty.status_code == 200
    assert empty.json()["data"]["status"] == "not_calculated"
    assert empty.json()["data"]["fake_metrics_reported"] is False

    calculated = client.post(
        "/api/v1/vision/evaluation/run",
        json={
            "references": [
                {
                    "document_type": "invoice",
                    "fields": {"invoice_number": "INV-1", "total_amount": "USD 10.00"},
                    "answer": "USD 10.00",
                },
                {
                    "document_type": "contract",
                    "fields": {"governing_law": "New York"},
                    "answer": "New York",
                },
            ],
            "predictions": [
                {
                    "document_type": "invoice",
                    "fields": {"invoice_number": "INV-1", "total_amount": "USD 11.00"},
                    "answer": "USD 11.00",
                },
                {
                    "document_type": "contract",
                    "fields": {"governing_law": "New York"},
                    "answer": "New York",
                },
            ],
        },
    )

    assert calculated.status_code == 200
    metrics = calculated.json()["data"]
    assert metrics["status"] == "calculated"
    assert metrics["classification"]["accuracy"] == 1.0
    assert metrics["extraction"]["field_precision"] < 1.0
    assert metrics["fake_metrics_reported"] is False


def test_vision_analyze_idempotency_key_returns_existing_document():
    headers = {**_headers(), "Idempotency-Key": "vision-test-idempotent-001"}
    payload = {
        "filename": "idempotent-invoice.txt",
        "text": "Invoice INV-IDEMP\nVendor: Stable Vendor\nTotal Amount: USD 25.00\nSignature: Reviewer",
        "metadata": {},
    }

    first = client.post("/api/v1/vision/documents/analyze", headers=headers, json=payload)
    second = client.post("/api/v1/vision/documents/analyze", headers=headers, json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["data"]["document_id"] == second.json()["data"]["document_id"]


def test_vision_review_ask_report_and_model_comparison_flow():
    headers = _headers()
    created = client.post(
        "/api/v1/vision/documents/analyze",
        headers=headers,
        json={
            "filename": "vision-review-invoice.txt",
            "text": (
                "Invoice INV-2048\n"
                "Vendor: Contoso Finance\n"
                "Invoice Date: 2026-04-01\n"
                "Due Date: 2026-04-15\n"
                "Bank Account: ACCT-33334444\n"
                "Total Amount: USD 1000.00\n"
                "Signature: Jane Reviewer"
            ),
            "metadata": {"approved_bank_identifier": "ACCT-55556666"},
        },
    )
    assert created.status_code == 200
    document = created.json()["data"]
    document_id = document["document_id"]

    fields = client.get(f"/api/v1/vision/documents/{document_id}/fields", headers=headers)
    assert fields.status_code == 200
    invoice_field = next(
        field for field in fields.json()["data"]["fields"] if field["name"] == "invoice_number"
    )

    reviews = client.get("/api/v1/vision/reviews", headers=headers)
    assert reviews.status_code == 200
    matching = [
        review for review in reviews.json()["data"]["reviews"] if review["document_id"] == document_id
    ]
    assert matching
    review_id = matching[0]["review_id"]

    patched = client.patch(
        f"/api/v1/vision/reviews/{review_id}/fields/{invoice_field['field_id']}",
        headers=headers,
        json={
            "reviewed_value": "INV-2048-CORRECTED",
            "reviewer_status": "edited",
            "comment": "Corrected from reviewer workspace.",
        },
    )
    assert patched.status_code == 200
    assert patched.json()["data"]["reviewed_value"] == "INV-2048-CORRECTED"

    approved = client.post(
        f"/api/v1/vision/reviews/{review_id}/approve",
        headers=headers,
        json={"decision": "APPROVED", "comment": "Verified against source document."},
    )
    assert approved.status_code == 200
    assert approved.json()["data"]["status"] == "APPROVED"

    answer = client.post(
        "/api/v1/vision/ask",
        headers=headers,
        json={"question": "What is the invoice number?", "document_id": document_id},
    )
    assert answer.status_code == 200
    assert answer.json()["data"]["answer_status"] == "supported"
    assert answer.json()["data"]["citations"]

    comparison = client.get(
        f"/api/v1/vision/models/comparison/{document_id}",
        headers=headers,
    )
    assert comparison.status_code == 200
    assert comparison.json()["data"]["field_output"]

    report = client.get(
        f"/api/v1/vision/reports/{document_id}",
        params={"format": "json"},
        headers=headers,
    )
    assert report.status_code == 200
    assert report.json()["data"]["report_type"] == "document_intelligence_report"

    exported = client.get("/api/v1/vision/active-learning/export", headers=_headers())
    assert exported.status_code == 200
    assert exported.json()["data"]["total_corrections"] >= 1
