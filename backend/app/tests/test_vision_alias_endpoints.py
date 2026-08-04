from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import UserRole
from app.tests.auth_helpers import auth_headers

client = TestClient(create_app())


def _headers(role: UserRole = UserRole.reviewer) -> dict[str, str]:
    return auth_headers(client, role)


def test_top_level_vision_aliases_questions_analytics_graph_and_admin():
    headers = _headers()
    created = client.post(
        "/api/v1/vision/documents/analyze",
        headers=headers,
        json={
            "filename": "alias-invoice.txt",
            "text": (
                "Invoice INV-ALIAS\n"
                "Vendor: Alias Systems\n"
                "Invoice Date: 2026-05-01\n"
                "Due Date: 2026-05-20\n"
                "Bank Account: ACCT-20262026\n"
                "Total Amount: USD 1225.00\n"
                "Signature: Alias Approver"
            ),
            "metadata": {"approved_bank_identifier": "ACCT-11110000"},
        },
    )
    assert created.status_code == 200
    document_id = created.json()["data"]["document_id"]

    fields = client.get(f"/api/v1/documents/{document_id}/fields", headers=headers)
    assert fields.status_code == 200
    assert any(field["name"] == "total_amount" for field in fields.json()["data"]["fields"])

    answer = client.post(
        "/api/v1/ask",
        headers=headers,
        json={"question": "What is the total amount?", "document_id": document_id},
    )
    assert answer.status_code == 200
    answer_data = answer.json()["data"]
    question_id = answer_data["question_id"]
    assert answer_data["answer_status"] == "supported"

    question = client.get(f"/api/v1/questions/{question_id}", headers=headers)
    assert question.status_code == 200
    assert question.json()["data"]["answer"]["answer_status"] == "supported"

    feedback = client.post(
        f"/api/v1/questions/{question_id}/feedback",
        headers=headers,
        json={"rating": "helpful", "comment": "Citation opened the correct source."},
    )
    assert feedback.status_code == 200
    assert feedback.json()["data"]["question_id"] == question_id

    graph = client.get(f"/api/v1/graph/document/{document_id}", headers=headers)
    assert graph.status_code == 200
    assert graph.json()["data"]["nodes"]

    entity = client.get("/api/v1/graph/entity/alias systems", headers=headers)
    assert entity.status_code == 200
    assert entity.json()["data"]["matches"]

    for endpoint in [
        "/api/v1/analytics/overview",
        "/api/v1/analytics/vendors",
        "/api/v1/analytics/risk",
        "/api/v1/analytics/review",
        "/api/v1/analytics/models",
    ]:
        response = client.get(endpoint, headers=headers)
        assert response.status_code == 200
        assert response.json()["success"] is True

    admin_headers = _headers(UserRole.admin)
    models = client.get("/api/v1/admin/models", headers=admin_headers)
    assert models.status_code == 200
    assert models.json()["data"]["registry"]["project_name"] == "DOCUMIND VISION AI"

    model_patch = client.patch(
        "/api/v1/admin/models/LayoutLMv3-base",
        headers=admin_headers,
        json={"enabled": False, "notes": "Keep disabled until adapter smoke tests pass."},
    )
    assert model_patch.status_code == 200
    assert model_patch.json()["data"]["status"] == "recorded_for_deployment_review"

    rules = client.get("/api/v1/admin/rules", headers=admin_headers)
    assert rules.status_code == 200
    assert "amount_arithmetic" in rules.json()["data"]["validation_rules"]

    rule_patch = client.patch(
        "/api/v1/admin/rules/amount_arithmetic",
        headers=admin_headers,
        json={"enabled": True, "threshold": 0.01, "notes": "Default invoice tolerance."},
    )
    assert rule_patch.status_code == 200
    assert rule_patch.json()["data"]["status"] == "recorded_for_local_runtime"

    audit = client.get("/api/v1/admin/audit", headers=admin_headers)
    assert audit.status_code == 200
    assert audit.json()["data"]["total"] >= 1
