from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import UserRole
from app.services.vision.document_classification import hybrid_document_classifier
from app.services.vision.legacy_audit import legacy_document_flags
from app.services.vision.service import vision_ai_service
from app.tests.auth_helpers import auth_credentials, auth_headers

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
    assert response.status_code == 200, response.text
    return response.json()["data"]


def _fax_document(suffix: str) -> dict:
    return _analyse(
        f"fax-transmission-{suffix}.txt",
        (
            "FAX TRANSMISSION\n"
            "DATE: February 25, 1998\n"
            "CLIENT NO: L8587.002\n"
            "MESSAGE TO: Ronald S. Milstein\n"
            "COMPANY: Lorillard Tobacco Company\n"
            "FAX NUMBER: 910/535-7707\n"
            "PHONE: 910/335-7718\n"
            "FROM: Andy Zausner\n"
            "PHONE: 202/828-2259\n"
            "PAGES: 12\n"
            "MESSAGE: Please call to discuss the attached committee notice.\n"
        ),
    )


@pytest.mark.parametrize(
    ("title", "expected_type"),
    [
        ("FACSIMILE", "fax_transmission"),
        ("FAX COVER SHEET", "fax_transmission"),
        ("BILL", "invoice"),
        ("PO NUMBER", "purchase_order"),
        ("ACCOUNT CONFIRMATION", "bank_confirmation"),
        ("AGREEMENT", "contract"),
    ],
)
def test_strong_title_rules_override_generic_classification(title, expected_type):
    result = hybrid_document_classifier.classify(title, "generic-document.txt")
    assert result.predicted_type.value == expected_type
    assert result.signals[0].signal_type == "title_keyword"


def test_fax_classification_extraction_runtime_chunks_and_graph():
    document = _fax_document("core")
    assert document["document_type"] == "fax_transmission"
    assert document["metadata"]["classification"]["classifier"] == ("hybrid-document-classifier-v3")
    assert document["metadata"]["classification"]["signals"][0]["type"] == ("title_keyword")
    assert document["metadata"]["schema_selection"]["selected_schema"] == ("fax_transmission")
    assert document["metadata"]["pipeline_status"] == "COMPLETED"
    assert document["metadata"]["decision_status"] == "REVIEW_REQUIRED"

    fields = {item["name"]: item for item in document["extracted_fields"]}
    assert {
        "date",
        "recipient_name",
        "recipient_company",
        "recipient_fax",
        "recipient_phone",
    } <= set(fields)
    assert fields["recipient_fax"]["value"] == "910/535-7707"
    assert fields["recipient_fax"]["normalized_value"] == "910-535-7707"
    assert fields["recipient_fax"]["evidence"]["bounding_box"] is not None
    assert fields["recipient_fax"]["validation_status"] == "valid"
    assert document["ocr_lines"]

    extraction_run = next(
        item
        for item in document["model_runs"]
        if item["stage"] == "extraction" and item["status"] == "completed"
    )
    assert extraction_run["processing_duration_ms"] is not None
    assert extraction_run["runtime_ms"] == extraction_run["processing_duration_ms"]
    assert extraction_run["model_run_id"] == extraction_run["run_id"]
    assert extraction_run["model_id"] == "deterministic-schema-extractor-v3"
    assert extraction_run["input_schema"] == "ocr-lines-v1"
    assert extraction_run["output_schema"] == "extracted-fields-v1"
    assert extraction_run["result_status"] in {
        "SUCCESS_WITH_FIELDS",
        "LOW_CONFIDENCE_FIELDS",
    }

    source_types = {item["source_type"] for item in document["evidence_chunks"]}
    assert {"field", "ocr_line", "key_value", "page_text"} <= source_types
    node_types = {item["type"] for item in document["graph"]["nodes"]}
    assert {"Document", "Person", "Organisation", "ContactMethod"} <= node_types
    assert any(
        edge["type"] == "HAS_FAX"
        and edge["evidence_text"]
        and edge["source_field"] == "recipient_fax"
        for edge in document["graph"]["edges"]
    )


def test_unknown_label_value_form_uses_generic_schema():
    document = _analyse(
        "unknown-business-note.txt",
        "Reference Number: GEN-42\nContact Number: 212-555-0198\n",
    )
    assert document["document_type"] == "unknown"
    assert document["metadata"]["schema_selection"]["selected_schema"] == ("generic_key_value_form")
    assert document["metadata"]["schema_selection"]["fallback_used"] is True
    assert document["extracted_fields"]


def test_fax_question_uses_source_backed_field_or_ocr_fallback():
    document = _fax_document("ask")
    response = client.post(
        "/api/v1/ai/ask",
        headers=_headers(),
        json={
            "question": "What is the fax number?",
            "document_id": document["document_id"],
            "scope": "current_document",
        },
    )
    assert response.status_code == 200, response.text
    answer = response.json()["data"]
    assert answer["answer_status"] == "supported"
    assert "910/535-7707" in answer["answer"]
    assert answer["retrieval_path"] in {
        "structured_field_lookup",
        "ocr_label_value_fallback",
    }
    assert answer["citations"][0]["page_number"] == 1
    assert answer["citations"][0]["bounding_boxes"]


def test_fax_question_uses_ocr_when_structured_fields_are_absent():
    document = _fax_document("ocr-only-ask")
    vision_ai_service.documents[document["document_id"]].extracted_fields = []

    response = client.post(
        "/api/v1/ai/ask",
        headers=_headers(),
        json={
            "question": "What is the fax number?",
            "document_id": document["document_id"],
            "scope": "current_document",
        },
    )
    assert response.status_code == 200, response.text
    answer = response.json()["data"]
    assert answer["answer_status"] == "supported"
    assert answer["retrieval_path"] == "ocr_label_value_fallback"
    assert "910/535-7707" in answer["answer"]
    assert answer["citations"][0]["bounding_boxes"]


def test_wrong_document_scope_never_broadens_silently():
    _fax_document("outside-scope")
    report = _analyse(
        "scope-report.txt",
        "GENERAL REPORT\nThis report contains no contact details.\n",
        "general_report",
    )
    response = client.post(
        "/api/v1/ai/ask",
        headers=_headers(),
        json={
            "question": "What is the fax number?",
            "document_id": report["document_id"],
            "scope": "current_document",
        },
    )
    assert response.status_code == 200, response.text
    answer = response.json()["data"]
    assert answer["answer_status"] in {"not_found", "clarification_needed"}
    assert answer["citations"] == []
    assert "910/535-7707" not in answer["answer"]
    assert answer["scope"]["documents"][0]["document_id"] == report["document_id"]
    assert answer["broader_scope_available"] is True


@pytest.mark.parametrize("scope", ["current_document", "selected_documents"])
def test_explicit_scope_requires_its_document_selection(scope):
    response = client.post(
        "/api/v1/ai/ask",
        headers=_headers(),
        json={"question": "What is the fax number?", "scope": scope},
    )
    assert response.status_code == 422


def test_approved_no_field_legacy_record_is_flagged_for_reverification():
    assert "LEGACY_APPROVAL_REVIEW_REQUIRED" in legacy_document_flags(
        {
            "status": "APPROVED",
            "metadata": {},
            "extracted_fields": [],
            "ocr_words": [{"text": "visible"}],
            "audit_events": [],
            "review_task": None,
        }
    )


def test_only_privileged_review_roles_can_confirm_fraud():
    document = _fax_document("fraud-roles")
    review_id = document["review_task"]["review_id"]
    evidence_id = document["extracted_fields"][0]["evidence"]["evidence_id"]

    reviewer_headers = auth_headers(client, UserRole.reviewer)
    reviewer_attempt = client.post(
        f"/api/v1/reviews/{review_id}/confirm-fraud",
        headers=reviewer_headers,
        json={"decision": "CONFIRMED_FRAUD", "comment": "Reviewer attempt."},
    )
    assert reviewer_attempt.status_code == 403

    compliance_headers = auth_headers(client, UserRole.compliance_officer)
    _, password = auth_credentials(UserRole.compliance_officer)
    reauthentication = client.post(
        "/api/v1/auth/reauthenticate",
        headers=compliance_headers,
        json={"password": password},
    )
    assert reauthentication.status_code == 200, reauthentication.text
    token = reauthentication.json()["data"]["reauthentication_token"]

    confirmed = client.post(
        f"/api/v1/reviews/{review_id}/confirm-fraud",
        headers=compliance_headers,
        json={
            "decision": "CONFIRMED_FRAUD",
            "comment": "Source evidence independently reviewed and retained.",
            "decision_rationale": (
                "The reviewed source evidence supports the recorded fraud decision."
            ),
            "evidence_ids": [evidence_id],
            "automated_indicators_acknowledged": True,
            "confirmation_acknowledged": True,
            "reauthentication_token": token,
        },
    )
    assert confirmed.status_code == 200, confirmed.text
    assert confirmed.json()["data"]["status"] == "CONFIRMED_FRAUD"
