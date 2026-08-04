from fastapi.testclient import TestClient

from app.main import create_app
from app.tests.auth_helpers import auth_headers

client = TestClient(create_app())


def _auth_headers() -> dict[str, str]:
    return auth_headers(client)


def test_connected_demo_seed_powers_ask_and_finance_narrative():
    headers = _auth_headers()

    seeded = client.post("/api/v1/demo/vision/seed", headers=headers)
    assert seeded.status_code == 200, seeded.text
    seed_data = seeded.json()["data"]
    primary_document_id = seed_data["primary_document_id"]

    documents = client.get("/api/v1/documents", headers=headers)
    assert documents.status_code == 200
    seeded_ids = set(seed_data["document_ids"])
    listed_ids = {item["document_id"] for item in documents.json()["data"]["documents"]}
    assert seeded_ids.issubset(listed_ids)

    answer = client.post(
        "/api/v1/ask",
        headers=headers,
        json={
            "question": "What is the total amount?",
            "document_id": primary_document_id,
        },
    )
    assert answer.status_code == 200, answer.text
    answer_data = answer.json()["data"]
    assert answer_data["answer_status"] == "supported"
    assert "472,000.00" in answer_data["answer"]
    assert answer_data["citations"]

    narrative = client.post(
        "/api/v1/finance-narratives",
        headers=headers,
        json={
            "narrative_type": "monthly_invoice_summary",
            "vendor_name": "Balu Enterprises",
            "document_ids": [primary_document_id],
        },
    )
    assert narrative.status_code == 200, narrative.text
    assert "472000.00" in narrative.json()["data"]["narrative"].replace(",", "")
