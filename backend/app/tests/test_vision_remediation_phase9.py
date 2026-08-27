from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import SignupRequest, UserRole
from app.services.auth import auth_service
from app.tests.auth_helpers import auth_credentials, auth_headers

client = TestClient(create_app())


def _approved_invoice(headers: dict[str, str], prefix: str) -> dict:
    created = client.post(
        "/api/v1/vision/documents/analyze",
        headers=headers,
        json={
            "filename": f"{prefix}.txt",
            "text": (
                f"Invoice INV-{prefix}\n"
                f"Vendor: Acceptance {prefix}\n"
                "Invoice Date: 2026-08-04\n"
                "Subtotal: USD 100.00\n"
                "Tax: USD 18.00\n"
                "Total Amount: USD 118.00\n"
                "Currency: USD\n"
                "Signature: Acceptance Reviewer\n"
            ),
            "metadata": {},
        },
    )
    assert created.status_code == 200, created.text
    document = created.json()["data"]
    approved = client.post(
        f"/api/v1/reviews/{document['review_task']['review_id']}/approve",
        headers=headers,
        json={
            "decision": "APPROVED",
            "comment": "Acceptance evidence verified against the source.",
            "evidence_ids": [
                field["evidence"]["evidence_id"]
                for field in document["extracted_fields"]
                if field["required"]
            ],
        },
    )
    assert approved.status_code == 200, approved.text
    return document


def test_required_phase9_api_contract_is_present_in_openapi():
    paths = create_app().openapi()["paths"]
    required = {
        "/api/v1/admin/models/{model_id}/health": "get",
        "/api/v1/admin/models/{model_id}/smoke-test": "post",
        "/api/v1/admin/models/{model_id}/requested-state": "patch",
        "/api/v1/analytics/documents": "get",
        "/api/v1/analytics/reviews": "get",
        "/api/v1/analytics/rag": "get",
        "/api/v1/narratives/generate": "post",
        "/api/v1/narratives": "get",
        "/api/v1/narratives/{narrative_id}": "get",
    }
    for path, method in required.items():
        assert method in paths[path]


def test_analytics_health_and_narrative_aliases_execute_real_service_paths():
    reviewer = auth_headers(client, UserRole.reviewer)
    admin = auth_headers(client, UserRole.admin)
    unique = uuid4().hex[:8]
    document = _approved_invoice(reviewer, f"phase9-{unique}")

    health = client.get(
        "/api/v1/admin/models/BAAI%2Fbge-m3/health",
        headers=admin,
    )
    assert health.status_code == 200, health.text
    assert health.json()["data"]["model_id"] == "BAAI/bge-m3"
    assert health.json()["data"]["status"] != "active"

    for endpoint in (
        "overview",
        "documents",
        "risk",
        "reviews",
        "models",
        "rag",
    ):
        response = client.get(f"/api/v1/analytics/{endpoint}", headers=reviewer)
        assert response.status_code == 200, response.text
        assert response.json()["success"] is True

    generated = client.post(
        "/api/v1/narratives/generate",
        headers=reviewer,
        json={
            "narrative_type": "monthly_invoice_summary",
            "document_ids": [document["document_id"]],
        },
    )
    assert generated.status_code == 200, generated.text
    narrative = generated.json()["data"]
    assert narrative["all_values_verified"] is True
    assert narrative["citations"]

    listing = client.get("/api/v1/narratives", headers=reviewer)
    assert listing.status_code == 200, listing.text
    assert narrative["narrative_id"] in {
        item["narrative_id"] for item in listing.json()["data"]["narratives"]
    }
    fetched = client.get(
        f"/api/v1/narratives/{narrative['narrative_id']}",
        headers=reviewer,
    )
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["data"]["identity_hash"] == narrative["identity_hash"]


def test_admin_mutations_and_report_download_are_tenant_scoped_and_audited():
    admin = auth_headers(client, UserRole.admin)
    admin_email, _ = auth_credentials(UserRole.admin)
    unique = uuid4().hex[:8]
    auditor_email = f"phase9-auditor-{unique}@documind.test"
    auditor_password = f"Phase9-{uuid4().hex}-Aa1!"
    auth_service.create_bootstrap_user(
        SignupRequest(
            name="Phase 9 Isolated Auditor",
            email=auditor_email,
            password=auditor_password,
            role=UserRole.auditor,
            organisation_name=f"Phase 9 Isolated Tenant {unique}",
        )
    )
    auditor_login = client.post(
        "/api/v1/auth/login",
        json={"email": auditor_email, "password": auditor_password},
    )
    assert auditor_login.status_code == 200, auditor_login.text
    auditor = {"Authorization": f"Bearer {auditor_login.json()['data']['access_token']}"}

    requested = client.patch(
        "/api/v1/admin/models/BAAI%2Fbge-m3/requested-state",
        headers=admin,
        json={"requested": False, "notes": f"phase9-{unique}"},
    )
    assert requested.status_code == 200, requested.text
    malformed = client.patch(
        "/api/v1/admin/models/BAAI%2Fbge-m3/requested-state",
        headers=admin,
        json={"notes": "missing requested state"},
    )
    assert malformed.status_code == 422

    smoke = client.post(
        "/api/v1/admin/models/deterministic-schema-extractor-v2/smoke-test",
        headers=admin,
    )
    assert smoke.status_code == 200, smoke.text
    assert smoke.json()["data"]["status"] == "passed"

    rule = client.patch(
        f"/api/v1/admin/rules/phase9-{unique}",
        headers=admin,
        json={"enabled": True, "threshold": 0.25, "notes": "Acceptance test"},
    )
    assert rule.status_code == 200, rule.text

    document = _approved_invoice(admin, f"phase9-audit-{unique}")
    report_response = client.post(
        "/api/v1/reports/generate",
        headers=admin,
        json={
            "report_type": "risk_screening_report",
            "document_id": document["document_id"],
        },
    )
    assert report_response.status_code == 200, report_response.text
    report = report_response.json()["data"]
    downloaded = client.get(
        f"/api/v1/reports/{report['report_id']}/download",
        headers=admin,
    )
    assert downloaded.status_code == 200, downloaded.text
    assert downloaded.content.startswith(b"%PDF-1.4")

    audit = client.get("/api/v1/admin/audit", headers=admin)
    assert audit.status_code == 200, audit.text
    events = audit.json()["data"]["events"]
    event_types = {item["event_type"] for item in events}
    assert {
        "model_state_change",
        "model_smoke_test",
        "rule_state_change",
        "report_downloaded",
    }.issubset(event_types)
    assert any(
        item["actor"] == admin_email
        and item["event_type"] == "rule_state_change"
        and item["metadata"].get("rule_id") == f"phase9-{unique}"
        for item in events
    )

    other_tenant_audit = client.get("/api/v1/admin/audit", headers=auditor)
    assert other_tenant_audit.status_code == 200, other_tenant_audit.text
    assert all(item["actor"] != admin_email for item in other_tenant_audit.json()["data"]["events"])
