"""Ingestion must enforce the role restriction the console advertises.

The navigation has always hidden Upload from VIEWER and AUDITOR, but
``POST /documents/upload`` accepted any authenticated caller, so the
restriction existed only in the UI. These tests pin the enforcement to the
API, where it is actually load-bearing.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import UserRole
from app.tests.auth_helpers import auth_headers

client = TestClient(create_app())

UPLOAD_PATHS = ("/api/v1/documents/upload", "/api/v1/vision/documents/upload")

WRITE_ROLES = (
    UserRole.admin,
    UserRole.processor,
    UserRole.reviewer,
    UserRole.senior_reviewer,
    UserRole.compliance_officer,
)
READ_ONLY_ROLES = (UserRole.viewer, UserRole.auditor)


def _upload(headers: dict[str, str], path: str):
    return client.post(
        path,
        headers=headers,
        data={"metadata": "{}"},
        files={
            "files": (
                "authorization-probe.txt",
                b"Invoice INV-4242\nVendor: Probe\nTotal Amount: USD 10.00\n",
                "text/plain",
            )
        },
    )


def test_read_only_roles_cannot_upload():
    for role in READ_ONLY_ROLES:
        headers = auth_headers(client, role)
        for path in UPLOAD_PATHS:
            response = _upload(headers, path)
            assert response.status_code == 403, (
                f"{role.value} uploaded via {path}: {response.status_code}"
            )
            assert "role" in response.json()["detail"].lower()


def test_processing_roles_can_still_upload():
    for role in WRITE_ROLES:
        headers = auth_headers(client, role)
        response = _upload(headers, UPLOAD_PATHS[0])
        assert response.status_code == 200, f"{role.value} blocked: {response.text[:200]}"
        assert response.json()["data"]["document_ids"]


def test_analyze_enforces_the_same_boundary():
    payload = {"filename": "probe.txt", "text": "Invoice INV-1\nTotal Amount: USD 1.00\n"}
    denied = client.post(
        "/api/v1/vision/documents/analyze",
        headers=auth_headers(client, UserRole.viewer),
        json=payload,
    )
    assert denied.status_code == 403

    allowed = client.post(
        "/api/v1/vision/documents/analyze",
        headers=auth_headers(client, UserRole.reviewer),
        json=payload,
    )
    assert allowed.status_code == 200


def test_upload_still_requires_authentication_at_all():
    for path in UPLOAD_PATHS:
        assert _upload({}, path).status_code == 401


def test_public_registration_cannot_self_assign_a_writing_role():
    """A signup body carrying role=ADMIN must not grant one."""
    import secrets

    email = f"selfassign-{secrets.token_hex(4)}@documind.test"
    password = f"Probe-{secrets.token_urlsafe(12)}-Aa1!"
    created = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Self Assign",
            "email": email,
            "password": password,
            "role": "ADMIN",
            "organisation_name": "Probe Org",
        },
    )
    assert created.status_code == 200, created.text
    assert created.json()["data"]["user"]["role"] == "VIEWER"

    login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    token = login.json()["data"]["access_token"]
    blocked = _upload({"Authorization": f"Bearer {token}"}, UPLOAD_PATHS[0])
    assert blocked.status_code == 403
