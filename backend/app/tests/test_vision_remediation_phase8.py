from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from app.core.config import settings
from app.main import create_app
from app.models.schemas import UserRole
from app.services.model_registry import ModelRegistryService
from app.services.vision import runtime_health
from app.services.vision.extraction_adapters import LayoutLMv3ExtractionAdapter
from app.services.vision.model_smoke import ModelSmokeService
from app.services.vision.runtime_health import ModelSmokeStore
from app.tests.auth_helpers import auth_headers

client = TestClient(create_app())


def test_active_registry_state_requires_matching_smoke_inference(tmp_path: Path, monkeypatch):
    store = ModelSmokeStore(tmp_path / "smoke")
    monkeypatch.setattr(runtime_health, "model_smoke_store", store)
    registry = ModelRegistryService()

    before = {item["model_id"]: item for item in registry.backends()}
    assert before["local-layout-adapter-v1"]["status"] == "unavailable"
    assert before["local-layout-adapter-v1"]["health_status"] == "unknown"

    result = ModelSmokeService(registry=registry, store=store).run(
        "local-layout-adapter-v1"
    )
    assert result["status"] == "passed"
    assert result["initialization"]["status"] == "passed"
    assert result["inference"]["status"] == "passed"
    assert result["output_schema"]["valid"] is True
    assert result["duration_ms"] > 0

    after = {item["model_id"]: item for item in registry.backends()}
    assert after["local-layout-adapter-v1"]["status"] == "active"
    assert after["local-layout-adapter-v1"]["health_status"] == "healthy"
    assert after["local-layout-adapter-v1"]["last_smoke_test"]


def test_generic_layoutlm_checkpoint_is_not_a_field_extractor(
    tmp_path: Path,
    monkeypatch,
):
    checkpoint = tmp_path / "generic-layoutlm"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"id2label": {"0": "LABEL_0", "1": "LABEL_1"}}),
        encoding="utf-8",
    )
    (checkpoint / "model.safetensors").write_bytes(b"not-loaded-by-status-check")
    monkeypatch.setattr(settings, "enable_heavy_models", True)
    monkeypatch.setattr(settings, "enable_layoutlmv3", True)
    monkeypatch.setattr(settings, "layoutlmv3_checkpoint", str(checkpoint))

    status = LayoutLMv3ExtractionAdapter().status()

    assert status.status == "unavailable"
    assert "generic pretrained head is not an extractor" in status.reason


def test_admin_smoke_and_requested_state_endpoints_are_truthful_and_authorized():
    admin = auth_headers(client, UserRole.admin)
    viewer = auth_headers(client, UserRole.viewer)

    forbidden = client.post(
        "/api/v1/admin/models/deterministic-schema-extractor-v2/smoke-test",
        headers=viewer,
    )
    assert forbidden.status_code == 403

    smoke = client.post(
        "/api/v1/admin/models/deterministic-schema-extractor-v2/smoke-test",
        headers=admin,
    )
    assert smoke.status_code == 200, smoke.text
    result = smoke.json()["data"]
    assert result["status"] == "passed"
    assert result["output_schema"]["name"] == "extracted-fields-v1"

    requested = client.patch(
        "/api/v1/admin/models/BAAI%2Fbge-m3/requested-state",
        headers=admin,
        json={"requested": True, "notes": "Prepare local checkpoint first."},
    )
    assert requested.status_code == 200, requested.text
    assert requested.json()["data"]["requested_enabled"] is True

    models = client.get("/api/v1/admin/models", headers=admin)
    registered = {
        item["model_id"]: item
        for item in models.json()["data"]["registry"]["backends"]
    }
    assert registered["BAAI/bge-m3"]["requested"] is True
    assert registered["BAAI/bge-m3"]["status"] != "active"
    assert registered["BAAI/bge-m3"]["requirements"]["missing_checkpoint"] is True


def test_model_lab_returns_expected_fields_and_null_unavailable_outputs():
    headers = auth_headers(client, UserRole.reviewer)
    unique = uuid4().hex[:8]
    analyzed = client.post(
        "/api/v1/vision/documents/analyze",
        headers=headers,
        json={
            "filename": f"phase8-invoice-{unique}.txt",
            "text": (
                "INVOICE\n"
                "Vendor: Runtime Evidence Ltd\n"
                f"Invoice Number: INV-{unique}\n"
                "Invoice Date: 2026-08-04\n"
                "Total Amount: INR 1250.00\n"
                "Currency: INR\n"
            ),
            "metadata": {"document_type_hint": "invoice"},
        },
    )
    assert analyzed.status_code == 200, analyzed.text
    document_id = analyzed.json()["data"]["document_id"]

    comparison = client.get(
        f"/api/v1/models/comparison/{document_id}",
        headers=headers,
    )
    assert comparison.status_code == 200, comparison.text
    payload = comparison.json()["data"]
    rows = {item["field_name"]: item for item in payload["field_output"]}

    assert payload["comparison_mode"] == "observed_runtime_outputs"
    assert rows["invoice_number"]["primary_result"] == f"INV-{unique}"
    assert rows["invoice_number"]["evidence"]["page_number"] == 1
    assert rows["vendor_address"]["primary_result"] is None
    assert rows["vendor_address"]["primary_confidence"] is None
    assert rows["vendor_address"]["layoutlmv3_result"] is None
    assert rows["vendor_address"]["donut_result"] is None
    assert payload["availability"]
