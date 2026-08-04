from fastapi.testclient import TestClient

from app.main import create_app
from app.models.schemas import TextAnalysisRequest
from app.services.pipeline import DocumentIntelligencePipeline

client = TestClient(create_app())


def test_model_registry_reports_local_runtime_and_disabled_heavy_models():
    response = client.get("/api/v1/demo-legacy/models/registry")

    assert response.status_code == 200
    payload = response.json()["data"]
    backends = {item["model_id"]: item for item in payload["backends"]}

    assert payload["model_profile"] == "LITE"
    assert backends["local-layout-adapter-v1"]["status"] == "active"
    assert backends["PP-OCRv5_mobile"]["status"] in {"disabled", "unavailable"}
    assert backends["LayoutLMv3-base"]["status"] in {"disabled", "unavailable"}
    assert "local-layout-adapter-v1" in payload["active_local_pipeline"]


def test_routing_preview_requests_fallback_without_fake_activation():
    response = client.post(
        "/api/v1/demo-legacy/models/routing-preview",
        json={
            "ocr_mean_confidence": 0.42,
            "reading_order_confidence": 0.55,
            "has_complex_table": True,
            "model_outputs_conflict": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()["data"]
    fallback_ids = {item["model_id"] for item in payload["fallback_models"]}

    assert payload["fallback_required"] is True
    assert "PaddleOCR-VL-1.6" in fallback_ids
    assert "Donut" in fallback_ids
    assert payload["unavailable_requested_models"]
    assert all(item["status"] != "active" for item in payload["unavailable_requested_models"])


def test_processed_document_contains_model_routing_metadata():
    service = DocumentIntelligencePipeline()
    response = service.process_text(
        TextAnalysisRequest(
            filename="complex-table-invoice.txt",
            text=(
                "Invoice INV-7777\n"
                "Vendor: Northstar Systems\n"
                "item | qty | price | tax | total | status | notes\n"
                "scanner output contains Café Mumbai amount INR 12,500.00"
            ),
        )
    )

    routing = response.documents[0].metadata["model_routing"]

    assert routing["model_profile"] == "LITE"
    assert routing["source"]["layout_engine"] == "local-layout-adapter"
    assert routing["fallback_required"] is True
    assert "mixed_languages_detected" in routing["reasons"]
    assert "PaddleOCR-VL-1.6" in {item["model_id"] for item in routing["fallback_models"]}
