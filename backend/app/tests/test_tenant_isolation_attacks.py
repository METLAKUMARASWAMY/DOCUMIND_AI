"""Adversarial tenant-isolation tests (directive §6, §33, §41).

The audit found tenant scoping *present* on every route sampled, but present is
not the same as proven. These tests build two real tenants, put a document in
each, and then actively attack the boundary from the other side.

Frontend route protection is not security; the assertions here all run against
the API with a valid token for the *wrong* tenant.
"""

from __future__ import annotations

import secrets

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import SignupRequest, UserRole
from app.services.auth import auth_service

RUN = secrets.token_hex(6)
PASSWORD = f"Tenant-{secrets.token_urlsafe(18)}-Aa1!"


class Tenant:
    def __init__(self, client: TestClient, label: str) -> None:
        self.label = label
        self.email = f"pytest-iso-{label}-{RUN}@documind.test"
        auth_service.create_bootstrap_user(
            SignupRequest(
                name=f"Isolation {label}",
                email=self.email,
                password=PASSWORD,
                role=UserRole.admin,
                organisation_name=f"Isolation Org {label} {RUN}",
            )
        )
        response = client.post(
            "/api/v1/auth/login", json={"email": self.email, "password": PASSWORD}
        )
        assert response.status_code == 200, response.text
        data = response.json()["data"]
        self.token = data["access_token"]
        self.tenant_id = data["user"]["tenant_id"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
        self.document_id: str | None = None
        self.review_id: str | None = None

    def ingest(self, client: TestClient, text: str, filename: str) -> str:
        response = client.post(
            "/api/v1/vision/documents/analyze",
            headers=self.headers,
            json={"filename": filename, "text": text, "metadata": {}},
        )
        assert response.status_code == 200, response.text
        payload = response.json()["data"]
        self.document_id = payload["document_id"]
        review = payload.get("review_task") or {}
        self.review_id = review.get("review_id")
        return self.document_id


@pytest.fixture(scope="module")
def app_client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


@pytest.fixture(scope="module")
def tenants(app_client: TestClient) -> tuple[Tenant, Tenant]:
    alpha = Tenant(app_client, "alpha")
    beta = Tenant(app_client, "beta")
    alpha.ingest(
        app_client,
        "Invoice INV-ALPHA-001\nVendor: Alpha Confidential Supplies\n"
        "Total Amount: INR 111,111.00\nBank Account: ACCT-ALPHA-9999\n",
        "alpha-secret.txt",
    )
    beta.ingest(
        app_client,
        "Invoice INV-BETA-002\nVendor: Beta Separate Traders\n"
        "Total Amount: INR 222,222.00\nBank Account: ACCT-BETA-8888\n",
        "beta-secret.txt",
    )
    assert alpha.tenant_id != beta.tenant_id, "the two tenants must be distinct"
    return alpha, beta


# ------------------------------------------------------------- baseline sanity


def test_each_tenant_sees_only_its_own_documents(
    app_client: TestClient, tenants: tuple[Tenant, Tenant]
) -> None:
    alpha, beta = tenants

    alpha_docs = app_client.get("/api/v1/documents", headers=alpha.headers).json()["data"]
    beta_docs = app_client.get("/api/v1/documents", headers=beta.headers).json()["data"]

    alpha_names = {item["filename"] for item in alpha_docs["documents"]}
    beta_names = {item["filename"] for item in beta_docs["documents"]}

    assert "alpha-secret.txt" in alpha_names
    assert "beta-secret.txt" not in alpha_names, "alpha can see beta's document"
    assert "beta-secret.txt" in beta_names
    assert "alpha-secret.txt" not in beta_names, "beta can see alpha's document"


# ------------------------------------------------------------- direct ID access


@pytest.mark.parametrize(
    "path_template",
    [
        "/api/v1/documents/{doc}",
        "/api/v1/documents/{doc}/fields",
        "/api/v1/documents/{doc}/ocr",
        "/api/v1/documents/{doc}/layout",
        "/api/v1/documents/{doc}/tables",
        "/api/v1/documents/{doc}/risk",
        "/api/v1/documents/{doc}/warnings",
        "/api/v1/documents/{doc}/related",
        "/api/v1/documents/{doc}/status",
        "/api/v1/documents/{doc}/source",
        "/api/v1/documents/{doc}/pages",
        "/api/v1/graph/document/{doc}",
        "/api/v1/models/comparison/{doc}",
    ],
)
def test_guessing_another_tenants_document_id_is_refused(
    app_client: TestClient, tenants: tuple[Tenant, Tenant], path_template: str
) -> None:
    """Knowing the id must not be enough. This is the core attack."""

    alpha, beta = tenants
    path = path_template.format(doc=alpha.document_id)

    response = app_client.get(path, headers=beta.headers)

    assert response.status_code in {403, 404}, (
        f"{path} leaked cross-tenant with status {response.status_code}: {response.text[:200]}"
    )
    assert "Alpha Confidential" not in response.text
    assert "ACCT-ALPHA-9999" not in response.text


def test_cross_tenant_delete_is_refused(
    app_client: TestClient, tenants: tuple[Tenant, Tenant]
) -> None:
    alpha, beta = tenants

    response = app_client.delete(f"/api/v1/documents/{alpha.document_id}", headers=beta.headers)
    assert response.status_code in {403, 404}

    # ...and alpha's document must still be there.
    still_there = app_client.get(f"/api/v1/documents/{alpha.document_id}", headers=alpha.headers)
    assert still_there.status_code == 200


def test_cross_tenant_reprocess_is_refused(
    app_client: TestClient, tenants: tuple[Tenant, Tenant]
) -> None:
    alpha, beta = tenants
    response = app_client.post(
        f"/api/v1/documents/{alpha.document_id}/reprocess", headers=beta.headers
    )
    assert response.status_code in {403, 404}


# ------------------------------------------------------------- retrieval / RAG


def test_ask_cannot_retrieve_another_tenants_evidence(
    app_client: TestClient, tenants: tuple[Tenant, Tenant]
) -> None:
    """The RAG path is the subtlest leak: retrieval must be tenant-filtered."""

    alpha, beta = tenants

    response = app_client.post(
        "/api/v1/ask",
        headers=beta.headers,
        json={"question": "who is Alpha Confidential Supplies", "top_k": 10},
    )

    assert response.status_code == 200, response.text
    payload = response.json()["data"]
    assert "Alpha Confidential" not in str(payload), "alpha evidence leaked into beta's answer"
    assert "ACCT-ALPHA-9999" not in str(payload)
    for citation in payload.get("citations", []):
        assert citation["document_id"] != alpha.document_id


def test_ask_scoped_to_another_tenants_document_id_does_not_leak(
    app_client: TestClient, tenants: tuple[Tenant, Tenant]
) -> None:
    """Explicitly naming the victim's document must not widen the scope."""

    alpha, beta = tenants

    response = app_client.post(
        "/api/v1/ask",
        headers=beta.headers,
        json={
            "question": "what is the total amount",
            "document_ids": [alpha.document_id],
            "top_k": 10,
        },
    )

    assert response.status_code in {200, 403, 404}
    if response.status_code == 200:
        payload = response.json()["data"]
        assert "111,111" not in str(payload), "alpha's amount leaked via document_ids scoping"
        for citation in payload.get("citations", []):
            assert citation["document_id"] != alpha.document_id


def test_graph_query_is_tenant_scoped(
    app_client: TestClient, tenants: tuple[Tenant, Tenant]
) -> None:
    alpha, beta = tenants

    response = app_client.post(
        "/api/v1/graph/query",
        headers=beta.headers,
        json={"question": "which vendor issued this", "document_ids": [alpha.document_id]},
    )

    assert response.status_code in {200, 400, 403, 404}
    assert "Alpha Confidential" not in response.text


# ------------------------------------------------------------- review workflow


def test_cross_tenant_review_actions_are_refused(
    app_client: TestClient, tenants: tuple[Tenant, Tenant]
) -> None:
    alpha, beta = tenants
    if not alpha.review_id:
        pytest.skip("alpha's document did not produce a review task")

    for action in ("approve", "reject", "escalate"):
        response = app_client.post(
            f"/api/v1/reviews/{alpha.review_id}/{action}",
            headers=beta.headers,
            json={"comment": "cross-tenant attempt", "evidence_ids": []},
        )
        assert response.status_code in {400, 403, 404, 422}, (
            f"{action} succeeded across tenants with {response.status_code}"
        )


def test_review_queue_is_tenant_scoped(
    app_client: TestClient, tenants: tuple[Tenant, Tenant]
) -> None:
    _alpha, beta = tenants

    beta_queue = app_client.get("/api/v1/reviews", headers=beta.headers)
    assert beta_queue.status_code == 200
    assert "alpha-secret.txt" not in beta_queue.text
    assert "Alpha Confidential" not in beta_queue.text


# ------------------------------------------------------------- unauthenticated


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/v1/documents"),
        ("GET", "/api/v1/reviews"),
        ("GET", "/api/v1/analytics/overview"),
        ("GET", "/api/v1/admin/models"),
        ("GET", "/api/v1/admin/audit"),
        ("POST", "/api/v1/ask"),
        ("POST", "/api/v1/vision/documents/analyze"),
        ("POST", "/api/v1/reports/generate"),
        ("POST", "/api/v1/graph/query"),
    ],
)
def test_unauthenticated_access_is_refused(app_client: TestClient, method: str, path: str) -> None:
    """Calling the URL directly, with no token at all."""

    response = app_client.request(method, path, json={})
    assert response.status_code in {401, 403}, (
        f"{method} {path} answered {response.status_code} without a token"
    )


def test_a_forged_token_is_refused(app_client: TestClient) -> None:
    forged = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhdHRhY2tlciJ9.not-a-signature"
    response = app_client.get("/api/v1/documents", headers={"Authorization": f"Bearer {forged}"})
    assert response.status_code in {401, 403}


def test_a_token_is_not_transferable_between_tenants(
    app_client: TestClient, tenants: tuple[Tenant, Tenant]
) -> None:
    """A valid token must carry its own tenant, not one supplied by the caller."""

    alpha, beta = tenants

    response = app_client.get(
        "/api/v1/documents",
        headers={**beta.headers, "X-Tenant-ID": alpha.tenant_id},
    )

    assert response.status_code == 200
    assert "alpha-secret.txt" not in response.text, "a header overrode the token's tenant"
