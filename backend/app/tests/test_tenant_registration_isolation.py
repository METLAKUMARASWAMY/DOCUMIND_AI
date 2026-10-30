"""Public self-registration must never place a stranger inside an existing tenant.

Found by end-to-end probing against the live stack. ``tenant_id`` was
``"tenant-" + slug(organisation_name)`` for every signup, and
``organisation_name`` is a free-text field on the public, unauthenticated
``/auth/register`` endpoint. So registering with a customer's organisation name
put the registrant inside that customer's tenant. The probe read the victim
document's full text, listed their documents, and got RAG answers drawn from
them -- with a VIEWER account created from nothing but the sign-up form.

Server-side provisioning (``create_bootstrap_user``) still resolves a named
tenant, because that is how an operator deliberately seeds one.
"""

from __future__ import annotations

import secrets

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import SignupRequest, UserRole
from app.services.auth import auth_service

client = TestClient(create_app())


def _register(organisation: str) -> dict:
    email = f"reg-{secrets.token_hex(6)}@documind.test"
    password = f"Reg-{secrets.token_urlsafe(14)}-A1!"
    response = client.post(
        "/api/v1/auth/register",
        json={
            "name": "Registrant",
            "email": email,
            "password": password,
            "organisation_name": organisation,
        },
    )
    assert response.status_code == 200, response.text
    return {"user": response.json()["data"]["user"], "email": email, "password": password}


def test_two_public_signups_naming_the_same_org_are_separated():
    organisation = f"Shared Name {secrets.token_hex(3)}"
    first = _register(organisation)
    second = _register(organisation)
    assert first["user"]["tenant_id"] != second["user"]["tenant_id"], (
        "public registration put two unrelated registrants in one tenant"
    )


def test_a_stranger_cannot_join_a_provisioned_tenant_by_naming_it():
    organisation = f"Provisioned Org {secrets.token_hex(3)}"
    owner_email = f"owner-{secrets.token_hex(5)}@documind.test"
    auth_service.create_bootstrap_user(
        SignupRequest(
            name="Tenant Owner",
            email=owner_email,
            password=f"Owner-{secrets.token_urlsafe(14)}-A1!",
            role=UserRole.reviewer,
            organisation_name=organisation,
        )
    )
    owner_tenant = auth_service._users[owner_email].user.tenant_id

    stranger = _register(organisation)
    assert stranger["user"]["tenant_id"] != owner_tenant, (
        "a self-registered stranger landed inside a provisioned tenant"
    )


def test_a_stranger_cannot_read_another_tenants_document():
    """The full breach path, end to end through the API."""
    organisation = f"Victim Org {secrets.token_hex(3)}"
    owner_email = f"victim-{secrets.token_hex(5)}@documind.test"
    owner_password = f"Victim-{secrets.token_urlsafe(14)}-A1!"
    auth_service.create_bootstrap_user(
        SignupRequest(
            name="Victim Reviewer",
            email=owner_email,
            password=owner_password,
            role=UserRole.reviewer,
            organisation_name=organisation,
        )
    )
    owner_token = client.post(
        "/api/v1/auth/login", json={"email": owner_email, "password": owner_password}
    ).json()["data"]["access_token"]

    uploaded = client.post(
        "/api/v1/documents/upload",
        headers={"Authorization": f"Bearer {owner_token}"},
        data={"metadata": "{}"},
        files={
            "files": (
                "victim-contract.txt",
                b"Service Provider: Confidential Counterparty Ltd\nTotal Amount: USD 99.00\n",
                "text/plain",
            )
        },
    )
    assert uploaded.status_code == 200, uploaded.text
    document_id = uploaded.json()["data"]["document_ids"][0]

    stranger = _register(organisation)
    stranger_token = client.post(
        "/api/v1/auth/login",
        json={"email": stranger["email"], "password": stranger["password"]},
    ).json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {stranger_token}"}

    read = client.get(f"/api/v1/documents/{document_id}", headers=headers)
    assert read.status_code in (403, 404), f"stranger read the document: {read.status_code}"

    listing = client.get("/api/v1/documents", headers=headers)
    visible = {item["document_id"] for item in listing.json()["data"].get("documents", [])}
    assert document_id not in visible, "stranger's listing exposed another tenant's document"

    answer = client.post(
        "/api/v1/ask",
        headers=headers,
        json={"question": "who is the service provider"},
    )
    body = answer.json().get("data") or {}
    assert "confidential counterparty" not in (body.get("answer") or "").lower(), (
        "stranger's RAG answer leaked another tenant's content"
    )


def test_server_side_provisioning_still_targets_a_named_tenant():
    """Operators must still be able to add users to one workspace."""
    organisation = f"Ops Org {secrets.token_hex(3)}"
    tenants = set()
    for index in range(2):
        email = f"ops-{index}-{secrets.token_hex(5)}@documind.test"
        auth_service.create_bootstrap_user(
            SignupRequest(
                name=f"Ops {index}",
                email=email,
                password=f"Ops-{secrets.token_urlsafe(14)}-A1!",
                role=UserRole.reviewer,
                organisation_name=organisation,
            )
        )
        tenants.add(auth_service._users[email].user.tenant_id)
    assert len(tenants) == 1, "bootstrap provisioning stopped sharing a named tenant"
