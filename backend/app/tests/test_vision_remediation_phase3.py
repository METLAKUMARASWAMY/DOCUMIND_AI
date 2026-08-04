from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import UserRole
from app.tests.auth_helpers import auth_credentials, auth_headers

client = TestClient(create_app())


def _create_review(
    *,
    headers: dict[str, str],
    filename: str,
    text: str,
    document_type: str,
) -> tuple[dict, str]:
    response = client.post(
        "/api/v1/vision/documents/analyze",
        headers=headers,
        json={
            "filename": filename,
            "text": text,
            "metadata": {"document_type_hint": document_type},
        },
    )
    assert response.status_code == 200
    document = response.json()["data"]
    assert document["review_task"]
    return document, document["review_task"]["review_id"]


def test_no_evidence_approval_is_blocked_and_manual_override_is_audited():
    headers = auth_headers(client, UserRole.reviewer)
    unique = uuid4().hex[:10]
    document, review_id = _create_review(
        headers=headers,
        filename=f"unreadable-invoice-{unique}.txt",
        text=f"Unreadable source {unique} with no recognised labels or values.",
        document_type="invoice",
    )
    assert document["extracted_fields"] == []
    assert document["field_completeness"]["unresolved_required_count"] == 5

    direct = client.post(
        f"/api/v1/reviews/{review_id}/approve",
        headers=headers,
        json={"decision": "APPROVED", "comment": "Approve after quick review."},
    )
    assert direct.status_code == 422
    policy = direct.json()["detail"]
    assert policy["allowed"] is False
    assert any(item["code"] == "no_structured_fields" for item in policy["blockers"])

    detail = client.get(f"/api/v1/reviews/{review_id}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["data"]["approval_policy"]["manual_override_available"] is True

    override = client.post(
        f"/api/v1/reviews/{review_id}/approve",
        headers=headers,
        json={
            "decision": "APPROVED",
            "comment": "Completed a full manual check against the original source.",
            "approval_reason": "Readable values were independently verified by the reviewer.",
            "manual_verification_completed": True,
            "confirmation_acknowledged": True,
            "confirmed_fields": {
                "vendor_name": "Manual Vendor",
                "invoice_number": "MANUAL-1001",
                "invoice_date": "2026-08-04",
                "total_amount": "100.00",
                "currency": "USD",
            },
        },
    )
    assert override.status_code == 200
    assert override.json()["data"]["status"] == "APPROVED"

    updated = client.get(
        f"/api/v1/documents/{document['document_id']}",
        headers=headers,
    ).json()["data"]
    manual_fields = [
        item for item in updated["extracted_fields"] if item["extractor"] == "manual-review"
    ]
    assert len(manual_fields) == 5
    assert updated["field_completeness"]["unresolved_required_count"] == 0
    override_events = [
        item
        for item in updated["audit_events"]
        if item["event_type"] == "review_decision_recorded"
    ]
    assert override_events[-1]["metadata"]["approval_mode"] == "manual_override"
    assert override_events[-1]["metadata"]["approval_reason"]
    assert "no_structured_fields" in override_events[-1]["metadata"][
        "overridden_policy_conditions"
    ]


def test_approval_requires_comment_and_selected_mandatory_evidence():
    headers = auth_headers(client, UserRole.reviewer)
    document, review_id = _create_review(
        headers=headers,
        filename="policy-complete-invoice.txt",
        text=(
            "Invoice INV-POLICY-1\n"
            "Vendor: Policy Vendor\n"
            "Invoice Date: 2026-08-04\n"
            "Total Amount: USD 50.00\n"
        ),
        document_type="invoice",
    )
    response = client.post(
        f"/api/v1/reviews/{review_id}/approve",
        headers=headers,
        json={"decision": "APPROVED", "comment": ""},
    )
    assert response.status_code == 422
    codes = {item["code"] for item in response.json()["detail"]["blockers"]}
    assert "reviewer_comment_required" in codes
    assert "mandatory_evidence_not_reviewed" in codes
    assert document["status"] == "REVIEW_REQUIRED"


def test_confirm_fraud_requires_admin_reauthentication_evidence_and_rationale():
    reviewer_headers = auth_headers(client, UserRole.reviewer)
    document, review_id = _create_review(
        headers=reviewer_headers,
        filename="fraud-safeguard-invoice.txt",
        text=(
            "Invoice INV-FRAUD-7\n"
            "Vendor: Evidence Vendor\n"
            "Invoice Date: 2026-08-04\n"
            "Total Amount: USD 700.00\n"
        ),
        document_type="invoice",
    )
    reviewer_attempt = client.post(
        f"/api/v1/reviews/{review_id}/confirm-fraud",
        headers=reviewer_headers,
        json={"decision": "CONFIRMED_FRAUD", "comment": "Reviewer attempt."},
    )
    assert reviewer_attempt.status_code == 403

    admin_headers = auth_headers(client, UserRole.admin)
    _, admin_password = auth_credentials(UserRole.admin)
    reauthentication = client.post(
        "/api/v1/auth/reauthenticate",
        headers=admin_headers,
        json={"password": admin_password},
    )
    assert reauthentication.status_code == 200
    token = reauthentication.json()["data"]["reauthentication_token"]

    missing_evidence = client.post(
        f"/api/v1/reviews/{review_id}/confirm-fraud",
        headers=admin_headers,
        json={
            "decision": "CONFIRMED_FRAUD",
            "comment": "Human investigation completed.",
            "decision_rationale": "Multiple independently reviewed records conflict materially.",
            "automated_indicators_acknowledged": True,
            "confirmation_acknowledged": True,
            "reauthentication_token": token,
        },
    )
    assert missing_evidence.status_code == 422

    evidence_id = document["extracted_fields"][0]["evidence"]["evidence_id"]
    confirmed = client.post(
        f"/api/v1/reviews/{review_id}/confirm-fraud",
        headers=admin_headers,
        json={
            "decision": "CONFIRMED_FRAUD",
            "comment": "Human investigation completed and source evidence retained.",
            "decision_rationale": (
                "The selected source evidence was independently checked and supports "
                "the recorded decision."
            ),
            "evidence_ids": [evidence_id],
            "automated_indicators_acknowledged": True,
            "confirmation_acknowledged": True,
            "reauthentication_token": token,
        },
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["data"]["status"] == "CONFIRMED_FRAUD"

    updated = client.get(
        f"/api/v1/documents/{document['document_id']}",
        headers=admin_headers,
    ).json()["data"]
    event = next(
        item
        for item in reversed(updated["audit_events"])
        if item["event_type"] == "confirmed_fraud_by_reviewer"
    )
    assert event["actor"].startswith("pytest-admin-")
    assert event["metadata"]["evidence_ids"] == [evidence_id]
    assert event["metadata"]["decision_rationale"]
    assert event["metadata"]["reauthenticated_at"] is not None
