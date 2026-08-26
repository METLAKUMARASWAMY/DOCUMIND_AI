from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import SignupRequest, UserRole
from app.services.auth import auth_service
from app.tests.auth_helpers import auth_headers

client = TestClient(create_app())


def test_verified_finance_narrative_and_typed_report_flow():
    headers = auth_headers(client, UserRole.reviewer)
    unique = uuid4().hex[:10]
    vendor_name = f"Narrative Industries {unique}"
    created = client.post(
        "/api/v1/vision/documents/analyze",
        headers=headers,
        json={
            "filename": f"verified-finance-invoice-{unique}.txt",
            "text": (
                f"Invoice INV-NARRATIVE-{unique}\n"
                f"Vendor: {vendor_name}\n"
                "Invoice Date: 2026-07-05\n"
                "Subtotal: USD 100.00\n"
                "Tax: USD 18.00\n"
                "Total Amount: USD 118.00\n"
                "Currency: USD\n"
                "Signature: Finance Reviewer\n"
            ),
            "metadata": {},
        },
    )
    assert created.status_code == 200
    document = created.json()["data"]
    review_id = document["review_task"]["review_id"]

    approved = client.post(
        f"/api/v1/reviews/{review_id}/approve",
        headers=headers,
        json={
            "decision": "APPROVED",
            "comment": "Verified all mandatory evidence against the source.",
            "evidence_ids": [
                field["evidence"]["evidence_id"]
                for field in document["extracted_fields"]
                if field["required"]
            ],
        },
    )
    assert approved.status_code == 200, approved.text

    narrative_response = client.post(
        "/api/v1/finance-narratives",
        headers=headers,
        json={
            "narrative_type": "monthly_invoice_summary",
            "vendor_name": vendor_name,
            "date_from": "2026-07-01",
            "date_to": "2026-07-31",
            "document_ids": [document["document_id"]],
        },
    )
    assert narrative_response.status_code == 200, narrative_response.text
    narrative = narrative_response.json()["data"]
    assert "USD 118.00" in narrative["narrative"]
    assert narrative["all_values_verified"] is True
    assert narrative["identity_hash"]
    assert narrative["source_data_version"]
    assert narrative["version"] == 1
    assert all(item["source_document_ids"] for item in narrative["calculations"])
    assert any(
        item["calculation_function"].startswith("sum(Decimal") for item in narrative["calculations"]
    )

    repeated_narrative = client.post(
        "/api/v1/finance-narratives",
        headers=headers,
        json={
            "narrative_type": "monthly_invoice_summary",
            "vendor_name": vendor_name,
            "date_from": "2026-07-01",
            "date_to": "2026-07-31",
            "document_ids": [document["document_id"]],
        },
    ).json()["data"]
    assert repeated_narrative["narrative_id"] == narrative["narrative_id"]
    listed_narratives = client.get(
        "/api/v1/finance-narratives",
        headers=headers,
    ).json()["data"]["narratives"]
    assert (
        sum(item["identity_hash"] == narrative["identity_hash"] for item in listed_narratives) == 1
    )

    versioned_narrative = client.post(
        "/api/v1/finance-narratives",
        headers=headers,
        json={
            "narrative_type": "monthly_invoice_summary",
            "vendor_name": vendor_name,
            "date_from": "2026-07-01",
            "date_to": "2026-07-31",
            "document_ids": [document["document_id"]],
            "generate_new_version": True,
        },
    ).json()["data"]
    assert versioned_narrative["version"] == 2
    assert versioned_narrative["supersedes"] == narrative["narrative_id"]
    assert versioned_narrative["identity_hash"] == narrative["identity_hash"]

    report_response = client.post(
        "/api/v1/reports/generate",
        headers=headers,
        json={
            "report_type": "finance_narrative_report",
            "narrative_id": narrative["narrative_id"],
        },
    )
    assert report_response.status_code == 200, report_response.text
    report = report_response.json()["data"]
    assert report["report_id"].startswith("report-")
    assert report["content"]["narrative_id"] == narrative["narrative_id"]
    assert report["version"] == 1
    assert report["identity_hash"]
    assert report["source_data_version"]
    assert report["content"]["source_document_ids"] == [document["document_id"]]

    repeated_report = client.post(
        "/api/v1/reports/generate",
        headers=headers,
        json={
            "report_type": "finance_narrative_report",
            "narrative_id": narrative["narrative_id"],
        },
    ).json()["data"]
    assert repeated_report["report_id"] == report["report_id"]

    versioned_report = client.post(
        "/api/v1/reports/generate",
        headers=headers,
        json={
            "report_type": "finance_narrative_report",
            "narrative_id": narrative["narrative_id"],
            "generate_new_version": True,
        },
    ).json()["data"]
    assert versioned_report["version"] == 2
    assert versioned_report["supersedes"] == report["report_id"]

    concurrent_payload = {
        "report_type": "risk_screening_report",
        "document_id": document["document_id"],
    }
    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(
            executor.map(
                lambda _: client.post(
                    "/api/v1/reports/generate",
                    headers=headers,
                    json=concurrent_payload,
                ),
                range(4),
            )
        )
    assert all(item.status_code == 200 for item in responses)
    assert len({item.json()["data"]["report_id"] for item in responses}) == 1

    fetched = client.get(f"/api/v1/reports/{report['report_id']}", headers=headers)
    assert fetched.status_code == 200
    assert fetched.json()["data"]["report_id"] == report["report_id"]

    downloaded = client.get(
        f"/api/v1/reports/{report['report_id']}/download",
        headers=headers,
    )
    assert downloaded.status_code == 200
    assert downloaded.content.startswith(b"%PDF-1.4")

    archived = client.post(
        f"/api/v1/reports/{versioned_report['report_id']}/archive",
        headers=headers,
    )
    assert archived.status_code == 200
    assert archived.json()["data"]["status"] == "archived"

    intelligence_report = client.post(
        "/api/v1/reports/generate",
        headers=headers,
        json={
            "report_type": "document_intelligence_report",
            "document_id": document["document_id"],
        },
    ).json()["data"]
    payload = intelligence_report["content"]["documents"][0]
    assert payload["ocr_layout"]["ocr_word_count"] > 0
    assert "page_artifacts" in payload["ocr_layout"]
    assert {"nodes", "edges"} <= set(payload["graph"])
    assert "structured_extraction_status" in payload["limitations"]


def test_document_access_is_isolated_between_tenants():
    suffix = uuid4().hex[:10]
    password = f"Tenant-{suffix}-Aa1!"
    credentials = []
    for organisation in ("Tenant Isolation Alpha", "Tenant Isolation Beta"):
        email = f"{organisation.split()[-1].lower()}-{suffix}@documind.test"
        auth_service.create_bootstrap_user(
            SignupRequest(
                name=organisation,
                email=email,
                password=password,
                role=UserRole.reviewer,
                organisation_name=f"{organisation} {suffix}",
            )
        )
        login = client.post("/api/v1/auth/login", json={"email": email, "password": password})
        assert login.status_code == 200
        credentials.append({"Authorization": f"Bearer {login.json()['data']['access_token']}"})

    created = client.post(
        "/api/v1/vision/documents/analyze",
        headers=credentials[0],
        json={
            "filename": "tenant-private.txt",
            "text": "Invoice INV-PRIVATE\nVendor: Private Tenant\nTotal Amount: USD 9.00",
            "metadata": {},
        },
    )
    assert created.status_code == 200
    document_id = created.json()["data"]["document_id"]

    assert client.get(f"/api/v1/documents/{document_id}", headers=credentials[0]).status_code == 200
    assert client.get(f"/api/v1/documents/{document_id}", headers=credentials[1]).status_code == 404
    other_listing = client.get("/api/v1/documents", headers=credentials[1])
    assert document_id not in {
        item["document_id"] for item in other_listing.json()["data"]["documents"]
    }
