from fastapi.testclient import TestClient

from app.main import create_app

client = TestClient(create_app())


def test_finance_upload_extract_narrative_vector_review_and_evaluation_flow():
    upload = client.post(
        "/api/v1/demo-legacy/finance/upload",
        data={"document_type": "invoice", "uploaded_by": "pytest"},
        files={
            "file": (
                "invoice-test.txt",
                (
                    b"Invoice Number: INV-7777\n"
                    b"Vendor: Apex Components Ltd\n"
                    b"Invoice Date: May 20, 2026\n"
                    b"Due Date: June 19, 2026\n"
                    b"PO Number: PO-7777\n"
                    b"Item: Control subscription | Qty: 2 | Unit Price: INR 50,000 | Amount: INR 1,00,000\n"
                    b"Subtotal: INR 90,000\n"
                    b"Tax: INR 10,000\n"
                    b"Total Amount: INR 1,00,000\n"
                    b"Payment Terms: Net 30\n"
                    b"Bank Name: Axis Bank\n"
                    b"Account Number: 123456789012\n"
                    b"IFSC: UTIB0007777\n"
                ),
                "text/plain",
            )
        },
    )
    assert upload.status_code == 200
    document = upload.json()["data"]
    document_id = document["document_id"]

    extraction_response = client.post(
        "/api/v1/demo-legacy/finance/extract", json={"document_id": document_id}
    )
    assert extraction_response.status_code == 200
    extraction = extraction_response.json()["data"]
    assert extraction["fields"]["invoice_number"] == "INV-7777"
    assert extraction["fields"]["bank_account_details"]["account_number_masked"] == "****9012"
    assert extraction["source_references"]

    narrative_response = client.post(
        "/api/v1/demo-legacy/finance/narrative",
        json={"document_id": document_id, "narrative_type": "Executive Summary"},
    )
    assert narrative_response.status_code == 200
    narrative = narrative_response.json()["data"]
    assert "INV-7777" in narrative["narrative"]
    assert narrative["groundedness_score"] >= 0.8

    vector_response = client.get(
        "/api/v1/demo-legacy/finance/vector/search",
        params={"q": "invoice Apex payment bank", "document_id": document_id},
    )
    assert vector_response.status_code == 200
    assert vector_response.json()["data"]["hits"]

    queued = client.post(f"/api/v1/demo-legacy/finance/review/{document_id}", json={})
    assert queued.status_code == 200
    approved = client.post(
        f"/api/v1/demo-legacy/finance/review/{document_id}/approve",
        json={
            "reviewer_name": "Pytest Reviewer",
            "decision": "approve",
            "reviewer_comments": "Accepted.",
        },
    )
    assert approved.status_code == 200
    assert approved.json()["data"]["status"] == "Approved"

    evaluation = client.post(
        "/api/v1/demo-legacy/finance/evaluation/run", json={"document_id": document_id}
    )
    assert evaluation.status_code == 200
    metrics = evaluation.json()["data"]
    assert "field_level_exact_match_accuracy" in metrics["extraction_accuracy"]
    assert "hallucination_rate" in metrics["hallucination"]


def test_bank_change_demo_escalates_and_masks_sensitive_details():
    demo = client.post("/api/v1/demo-legacy/finance/demo-cases/bank-change-risk-demo/run")
    assert demo.status_code == 200
    payload = demo.json()["data"]

    fields = payload["extraction"]["fields"]
    assert fields["new_bank_account"] == "****1098"
    assert fields["fraud_risk_level"] == "high"
    assert payload["review"]["status"] == "Escalated"
    assert any(
        flag["title"] == "Bank change requires human verification"
        for flag in payload["extraction"]["risk_flags"]
    )
    assert payload["vector_verification"]["document_records"] >= 4


def test_bank_change_upload_overrides_wrong_type_and_blocks_unsafe_approval():
    bank_change_text = (
        "Vendor Bank Update Request\n"
        "Vendor: ABC Suppliers Pvt Ltd\n"
        "Old Bank Account: 123456789012\n"
        "New Bank Account: 987654321098\n"
        "Bank Name: HDFC Bank\n"
        "IFSC: HDFC0002045\n"
        "Request Date: May 15, 2026\n"
        "Requester: collections@abcsuppliers-payments.co\n"
        "Please update the account urgently for same day wire release.\n"
        "Do not call the old finance contact because they are unavailable.\n"
    )
    upload = client.post(
        "/api/v1/demo-legacy/finance/upload",
        data={"document_type": "invoice", "uploaded_by": "pytest"},
        files={"file": ("bank_change_risk_demo.txt", bank_change_text.encode(), "text/plain")},
    )
    assert upload.status_code == 200
    document = upload.json()["data"]
    document_id = document["document_id"]
    assert document["document_type"] == "bank_change_request"
    assert document["risk_level"] == "high"
    assert document["metadata"]["user_selected_document_type"] == "invoice"
    assert document["metadata"]["document_type_resolution"] == "overridden_by_bank_change_guardrail"
    assert (
        document["metadata"]["ground_truth_path"]
        == "demo_data/ground_truth/bank_change_risk_demo.json"
    )

    extraction_response = client.post(
        "/api/v1/demo-legacy/finance/extract", json={"document_id": document_id}
    )
    assert extraction_response.status_code == 200
    extraction = extraction_response.json()["data"]
    assert extraction["document_type"] == "bank_change_request"
    assert extraction["fields"]["new_bank_account"] == "****1098"
    assert "987654321098" not in str(extraction["source_references"])

    narrative_response = client.post(
        "/api/v1/demo-legacy/finance/narrative",
        json={"document_id": document_id, "narrative_type": "Payment Summary"},
    )
    assert narrative_response.status_code == 200
    narrative = narrative_response.json()["data"]["narrative"]
    assert "Bank-change payment hold" in narrative
    assert "Invoice not extracted" not in narrative

    queued = client.post(f"/api/v1/demo-legacy/finance/review/{document_id}", json={})
    assert queued.status_code == 200
    unsafe_approval = client.post(
        f"/api/v1/demo-legacy/finance/review/{document_id}/approve",
        json={
            "reviewer_name": "Pytest Reviewer",
            "decision": "approve",
            "reviewer_comments": "Looks fine.",
        },
    )
    assert unsafe_approval.status_code == 409

    verified_approval = client.post(
        f"/api/v1/demo-legacy/finance/review/{document_id}/approve",
        json={
            "reviewer_name": "Pytest Reviewer",
            "decision": "approve",
            "reviewer_comments": "Verified out-of-band using trusted registered contact.",
        },
    )
    assert verified_approval.status_code == 200
    assert verified_approval.json()["data"]["status"] == "Approved"

    evaluation = client.post(
        "/api/v1/demo-legacy/finance/evaluation/run", json={"document_id": document_id}
    )
    assert evaluation.status_code == 200
    row = evaluation.json()["data"]["document_results"][0]
    assert row["document_type"] == "bank_change_request"
    assert row["f1_score"] > 0


def test_finance_demo_cases_and_metrics_are_available():
    cases = client.get("/api/v1/demo-legacy/finance/demo-cases")
    assert cases.status_code == 200
    case_ids = {item["case_id"] for item in cases.json()["data"]["cases"]}
    assert {
        "invoice-demo",
        "purchase-order-demo",
        "contract-demo",
        "bank-change-risk-demo",
    }.issubset(case_ids)

    metrics = client.get("/api/v1/demo-legacy/finance/metrics")
    assert metrics.status_code == 200
    data = metrics.json()["data"]
    assert data["vector_records"] >= 1
    assert "human_review_acceptance" in data
