from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.tests.auth_helpers import auth_headers

client = TestClient(create_app())


def _headers() -> dict[str, str]:
    return auth_headers(client)


def _analyse(filename: str, text: str, document_type: str | None = None) -> dict:
    metadata = {"document_type_hint": document_type} if document_type else {}
    response = client.post(
        "/api/v1/vision/documents/analyze",
        headers=_headers(),
        json={"filename": filename, "text": text, "metadata": metadata},
    )
    assert response.status_code == 200
    return response.json()["data"]


def test_supported_form_and_invoice_extract_real_boxed_schema_fields():
    form = _analyse(
        "organisation-registration.txt",
        (
            "Organisation Registration Form\n"
            "Organisation Name: Northwind Research\n"
            "Contact Name: Priya Shah\n"
            "FAX NO.: 212-403-2211\n"
            "Email: priya@example.org\n"
        ),
        "form",
    )
    form_fields = {item["name"]: item for item in form["extracted_fields"]}
    assert {"organisation_name", "contact_name"} <= set(form_fields)
    assert form_fields["organisation_name"]["value"] == "Northwind Research"
    assert form_fields["contact_name"]["evidence"]["bounding_box"] is not None
    assert form_fields["contact_name"]["ocr_confidence"] is not None
    assert form["confidence_summary"]["field_count"] == len(form_fields)
    assert form["confidence_summary"]["mean_field_confidence"] is not None

    invoice = _analyse(
        "schema-invoice.txt",
        (
            "Invoice INV-SCHEMA-42\n"
            "Vendor: Fabrikam Services\n"
            "Invoice Date: 2026-08-01\n"
            "Total Amount: USD 825.50\n"
        ),
        "invoice",
    )
    invoice_fields = {item["name"]: item for item in invoice["extracted_fields"]}
    assert {"invoice_number", "invoice_date", "total_amount"} <= set(invoice_fields)
    assert invoice_fields["invoice_number"]["value"] == "INV-SCHEMA-42"
    assert invoice_fields["total_amount"]["normalized_value"] == "825.50"
    assert all(item["extractor"] == "deterministic-fallback" for item in invoice_fields.values())


def test_unknown_document_uses_unavailable_confidence_and_schema_message_contract():
    document = _analyse(
        "free-note.txt",
        "A short free-form note with no financial or administrative labels.",
    )
    assert document["document_type"] == "unknown"
    assert document["extracted_fields"] == []
    assert document["confidence_summary"]["mean_field_confidence"] is None
    assert document["confidence_summary"]["confidence_available"] is False
    assert document["field_completeness"]["schema_available"] is False
    assert document["field_completeness"]["unresolved_required_count"] is None


def test_document_evidence_endpoints_expose_persisted_contracts():
    document = _analyse(
        "endpoint-form.txt",
        "Registration Form\nOrganisation: Endpoint Labs\nContact: Asha Rao\n",
        "form",
    )
    document_id = document["document_id"]
    headers = _headers()

    ocr = client.get(f"/api/v1/documents/{document_id}/ocr", headers=headers)
    layout = client.get(f"/api/v1/documents/{document_id}/layout", headers=headers)
    fields = client.get(f"/api/v1/documents/{document_id}/fields", headers=headers)
    missing = client.get(
        f"/api/v1/documents/{document_id}/missing-fields",
        headers=headers,
    )
    visual = client.get(
        f"/api/v1/documents/{document_id}/visual-elements",
        headers=headers,
    )
    warnings = client.get(
        f"/api/v1/documents/{document_id}/warnings",
        headers=headers,
    )

    assert ocr.status_code == layout.status_code == fields.status_code == 200
    assert missing.status_code == visual.status_code == warnings.status_code == 200
    assert ocr.json()["data"]["word_count"] > 0
    assert layout.json()["data"]["region_count"] > 0
    assert fields.json()["data"]["confidence"]["field_count"] > 0
    assert missing.json()["data"]["schema_available"] is True
    assert isinstance(visual.json()["data"]["visual_elements"], list)
    assert isinstance(warnings.json()["data"]["warnings"], list)


def test_reprocess_reuses_document_fields_chunks_and_review_identity():
    document = _analyse(
        "stable-reprocess-invoice.txt",
        (
            "Invoice INV-STABLE-9\n"
            "Vendor: Stable Industries\n"
            "Invoice Date: 2026-08-02\n"
            "Total Amount: USD 90.00\n"
        ),
        "invoice",
    )
    document_id = document["document_id"]
    initial_field_ids = sorted(item["field_id"] for item in document["extracted_fields"])
    initial_chunk_ids = sorted(item["chunk_id"] for item in document["evidence_chunks"])
    initial_review_id = document["review_task"]["review_id"]

    response = client.post(
        f"/api/v1/documents/{document_id}/reprocess",
        headers=_headers(),
    )
    assert response.status_code == 200
    assert response.json()["data"]["document_ids"] == [document_id]

    detail = client.get(f"/api/v1/documents/{document_id}", headers=_headers())
    assert detail.status_code == 200
    updated = detail.json()["data"]
    assert sorted(item["field_id"] for item in updated["extracted_fields"]) == initial_field_ids
    assert sorted(item["chunk_id"] for item in updated["evidence_chunks"]) == initial_chunk_ids
    assert updated["review_task"]["review_id"] == initial_review_id
    assert updated["metadata"]["reprocess_count"] == 1

    first_reprocess_field_ids = [item["field_id"] for item in updated["extracted_fields"]]
    first_reprocess_chunk_ids = [item["chunk_id"] for item in updated["evidence_chunks"]]
    first_reprocess_node_ids = [item["id"] for item in updated["graph"]["nodes"]]
    first_reprocess_edge_ids = [item["id"] for item in updated["graph"]["edges"]]
    second_response = client.post(
        f"/api/v1/documents/{document_id}/reprocess",
        headers=_headers(),
    )
    assert second_response.status_code == 200
    second = client.get(
        f"/api/v1/documents/{document_id}", headers=_headers()
    ).json()["data"]

    assert sorted(item["field_id"] for item in second["extracted_fields"]) == sorted(
        first_reprocess_field_ids
    )
    assert sorted(item["chunk_id"] for item in second["evidence_chunks"]) == sorted(
        first_reprocess_chunk_ids
    )
    assert sorted(item["id"] for item in second["graph"]["nodes"]) == sorted(
        first_reprocess_node_ids
    )
    assert sorted(item["id"] for item in second["graph"]["edges"]) == sorted(
        first_reprocess_edge_ids
    )
    assert len(first_reprocess_field_ids) == len(set(first_reprocess_field_ids))
    assert len(first_reprocess_chunk_ids) == len(set(first_reprocess_chunk_ids))
    assert len(first_reprocess_node_ids) == len(set(first_reprocess_node_ids))
    assert len(first_reprocess_edge_ids) == len(set(first_reprocess_edge_ids))
    assert second["review_task"]["review_id"] == initial_review_id
    assert second["metadata"]["reprocess_count"] == 2

    listed = client.get("/api/v1/documents", headers=_headers()).json()["data"]["documents"]
    assert sum(item["document_id"] == document_id for item in listed) == 1


def test_page_image_modes_are_distinct_reversible_same_size_artifacts():
    document = _analyse(
        "viewer-modes-invoice.txt",
        (
            "Invoice INV-VIEWER-1\n"
            "Vendor: Viewer Modes\n"
            "Invoice Date: 2026-08-04\n"
            "Total Amount: USD 25.00\n"
        ),
        "invoice",
    )
    document_id = document["document_id"]
    artifacts: dict[str, bytes] = {}
    dimensions: set[tuple[int, int]] = set()
    for variant in ("original", "enhanced", "grayscale", "binary"):
        response = client.get(
            f"/api/v1/documents/{document_id}/pages/1/image",
            params={"variant": variant},
            headers=_headers(),
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/png")
        artifacts[variant] = response.content
        image = Image.open(BytesIO(response.content))
        dimensions.add(image.size)
        assert image.getbbox() is not None

    assert len(dimensions) == 1
    assert artifacts["original"] != artifacts["binary"]
    assert artifacts["enhanced"] != artifacts["binary"]

    if any(item["heatmap_path"] for item in document["tamper_analysis"]):
        forensic = client.get(
            f"/api/v1/documents/{document_id}/pages/1/image",
            params={"variant": "forensic"},
            headers=_headers(),
        )
        assert forensic.status_code == 200
        assert Image.open(BytesIO(forensic.content)).getbbox() is not None
