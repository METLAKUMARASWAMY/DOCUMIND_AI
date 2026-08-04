from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import create_app
from app.tests.auth_helpers import auth_headers

client = TestClient(create_app())


def _headers() -> dict[str, str]:
    return auth_headers(client)


def _analyse(filename: str, text: str, metadata: dict | None = None) -> dict:
    response = client.post(
        "/api/v1/vision/documents/analyze",
        headers=_headers(),
        json={"filename": filename, "text": text, "metadata": metadata or {}},
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]


def test_clean_english_text_with_symbols_does_not_trigger_mixed_language_routing():
    unique = uuid4().hex[:8]
    document = _analyse(
        f"english-language-{unique}.txt",
        (
            "Organisation Registration Form\n"
            f"Organisation Name: English Evidence Services {unique}\n"
            "Contact Name: Priya Shah\n"
            "Address: Forty two Market Street, London\n"
            "Notes: The registration details are complete and written in English.\n"
            "Amount: INR 472000.00 | Symbols: ₹ © • # @\n"
        ),
        {"document_type_hint": "form"},
    )

    analysis = document["language_analysis"]
    assert analysis["valid_character_count"] >= 40
    assert analysis["mixed_language_routing"] is False
    assert analysis["decision_reason"] == "single_valid_language_group"
    assert "mixed_languages_detected" not in document["model_routing"]["reasons"]
    assert all(
        warning["title"] != "Multiple-language uncertainty"
        for warning in document["processing_warnings"]
    )


def test_genuine_mixed_language_text_routes_with_stored_language_evidence():
    unique = uuid4().hex[:8]
    document = _analyse(
        f"mixed-language-{unique}.txt",
        (
            "Organisation Registration Form\n"
            f"Organisation Name: Bilingual Evidence Services {unique}\n"
            "The registration details and contact information are written in English.\n"
            "यह पंजीकरण विवरण हिंदी भाषा में भी स्पष्ट रूप से लिखा गया है।\n"
            "संपर्क और पते की जानकारी सत्यापन के लिए उपलब्ध है।\n"
        ),
        {"document_type_hint": "form"},
    )

    analysis = document["language_analysis"]
    assert analysis["mixed_language_routing"] is True
    assert analysis["decision_reason"] == "mixed_language_thresholds_met"
    assert analysis["secondary_language_characters"] >= 20
    assert analysis["secondary_language_ratio"] >= 0.20
    assert analysis["confidence"] >= 0.80
    assert {item["script"] for item in analysis["detected_languages"]} >= {
        "Latin",
        "Devanagari",
    }
    assert "mixed_languages_detected" in document["model_routing"]["reasons"]
    warning = next(
        item
        for item in document["processing_warnings"]
        if item["title"] == "Multiple-language uncertainty"
    )
    assert warning["category"] == "processing_warning"
    assert warning["rule_id"] == "processing_warning.mixed_languages_detected"


def test_processing_warnings_and_business_risks_have_distinct_counted_contracts():
    unique = uuid4().hex[:8]
    document = _analyse(
        f"risk-contract-{unique}.txt",
        (
            f"Invoice INV-{unique}\n"
            "Vendor: Risk Evidence Ltd\n"
            "Invoice Date: 2026-08-01\n"
            "Bank Account: ACCT-99999999\n"
            "Subtotal: USD 100.00\n"
            "Tax: USD 10.00\n"
            "Total Amount: USD 125.00\n"
        ),
        {
            "document_type_hint": "invoice",
            "approved_bank_identifier": "ACCT-11112222",
        },
    )
    document_id = document["document_id"]

    risk_response = client.get(f"/api/v1/documents/{document_id}/risk", headers=_headers())
    warning_response = client.get(
        f"/api/v1/documents/{document_id}/warnings",
        headers=_headers(),
    )
    assert risk_response.status_code == warning_response.status_code == 200
    risks = risk_response.json()["data"]["risk_indicators"]
    warnings = warning_response.json()["data"]["warnings"]
    bank = next(item for item in risks if item["title"] == "Bank Detail Mismatch")

    assert bank["category"] == "business_risk"
    assert bank["evidence"]
    assert bank["model_or_rule"]
    assert all(item["category"] == "processing_warning" for item in warnings)
    assert risk_response.json()["data"]["total"] == len(risks)
    assert warning_response.json()["data"]["total"] == len(warnings)


def test_review_queue_returns_readable_metadata_filters_and_pagination():
    unique = uuid4().hex[:8]
    filename = f"queue-readable-{unique}.txt"
    document = _analyse(
        filename,
        (
            "Organisation Registration Form\n"
            f"Organisation Name: Queue Evidence {unique}\n"
            "Contact Name: Mina Rao\n"
            "The registration details are written clearly in English for review.\n"
            "यह पंजीकरण विवरण हिंदी भाषा में भी सत्यापन के लिए उपलब्ध है।\n"
            "संपर्क जानकारी दूसरी भाषा में स्पष्ट रूप से लिखी गई है।\n"
        ),
        {"document_type_hint": "form"},
    )

    response = client.get(
        "/api/v1/reviews",
        headers=_headers(),
        params={
            "search": unique,
            "document_type": "form",
            "reason": "mixed_languages_detected",
            "status": "all",
            "page": 1,
            "page_size": 1,
        },
    )
    assert response.status_code == 200, response.text
    queue = response.json()["data"]
    assert queue["total"] == 1
    assert queue["page"] == 1
    assert queue["page_size"] == 1
    review = queue["reviews"][0]
    assert review["document_id"] == document["document_id"]
    assert review["filename"] == filename
    assert review["document_type"] == "form"
    assert review["page_count"] == 1
    assert "Multiple-language uncertainty" in review["reason_labels"]
    assert review["thumbnail_url"].endswith("variant=original")


def test_dashboard_library_queue_risk_and_report_totals_are_consistent():
    headers = _headers()
    library = client.get("/api/v1/documents", headers=headers).json()["data"]
    queue = client.get(
        "/api/v1/reviews",
        headers=headers,
        params={"status": "all", "page_size": 100},
    ).json()["data"]
    reports = client.get("/api/v1/reports", headers=headers).json()["data"]
    overview = client.get("/api/v1/analytics/overview", headers=headers).json()["data"]
    risk = client.get("/api/v1/analytics/risk", headers=headers).json()["data"]

    assert overview["documents"] == library["total"]
    assert overview["document_count"] == library["total"]
    assert sum(overview["status_counts"].values()) == library["total"]
    assert overview["review_tasks"] == queue["unfiltered_total"]
    assert overview["report_count"] == reports["total"]
    assert sum(risk["by_category"].values()) == risk["total_indicators"]
    assert overview["sample_sizes"]["documents"] == library["total"]

    empty_window = client.get(
        "/api/v1/analytics/overview",
        headers=headers,
        params={"date_from": "2099-01-01", "date_to": "2099-12-31"},
    ).json()["data"]
    assert empty_window["documents"] == 0
    assert empty_window["mean_field_confidence"] is None
