from fastapi.testclient import TestClient

from app.main import create_app
from app.services.pipeline import DocumentIntelligencePipeline
from app.tests.auth_helpers import auth_headers

client = TestClient(create_app())


def _auth_headers() -> dict[str, str]:
    return auth_headers(client)


def test_high_risk_demo_defaults_to_hold_payment_and_blocks_approval():
    pipeline = DocumentIntelligencePipeline()
    analysis = pipeline.process_demo_case()

    assert analysis.risk_score >= 75
    assert analysis.workflow.recommended_action.value == "hold_payment"
    assert analysis.workflow.approval_blocked is True
    assert "approve" not in [action.value for action in analysis.workflow.allowed_actions]


def test_approve_decision_is_blocked_for_high_risk_case():
    demo = client.post("/api/v1/demo-legacy/demo/run", json={}).json()
    case_id = demo["case_id"]

    response = client.post(
        f"/api/v1/demo-legacy/cases/{case_id}/workflow/decision",
        json={"approver": "Reviewer", "decision": "approve", "comment": "Attempt approval."},
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["status"] == "blocked"
    assert payload["blocked_reasons"]
