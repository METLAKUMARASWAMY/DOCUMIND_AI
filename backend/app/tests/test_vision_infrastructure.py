import json

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.vision.infrastructure import (
    InfrastructureUnavailableError,
    VisionInfrastructureReadinessService,
    vision_infrastructure_readiness_service,
)

client = TestClient(create_app())


def test_vision_infrastructure_readiness_defaults_to_local_runtime_without_secret_leakage():
    response = client.get("/api/v1/vision/infrastructure/readiness")

    assert response.status_code == 200
    payload = response.json()["data"]
    services = {item["backend_id"]: item for item in payload["services"]}

    assert payload["status"] == "local_runtime_only"
    assert services["mongodb"]["status"] == "disabled"
    assert services["redis"]["status"] == "disabled"
    assert services["minio"]["status"] == "disabled"
    assert services["qdrant"]["status"] == "disabled"
    assert services["neo4j"]["status"] == "disabled"
    assert services["celery_worker"]["status"] == "disabled"

    dumped = json.dumps(payload).lower()
    assert "documind_dev_password" not in dumped
    assert "minio_secret_key" not in dumped


def test_infrastructure_service_can_report_enabled_driver_readiness(monkeypatch):
    monkeypatch.setattr("app.services.vision.infrastructure.settings.enable_mongodb", True)
    service = VisionInfrastructureReadinessService(package_checker=lambda package: package == "motor")

    status = service.service_status("mongodb")
    payload = service.readiness()

    assert status["status"] == "ready_for_connection"
    assert status["installed"] is True
    assert payload["status"] == "external_backends_ready_for_connection"


def test_infrastructure_service_blocks_enabled_backend_with_missing_driver(monkeypatch):
    monkeypatch.setattr("app.services.vision.infrastructure.settings.enable_qdrant", True)
    service = VisionInfrastructureReadinessService(package_checker=lambda package: package != "qdrant_client")

    status = service.service_status("qdrant")

    assert status["status"] == "unavailable"
    assert "qdrant_client" in status["reason"]


def test_require_ready_raises_for_disabled_backend():
    with pytest.raises(InfrastructureUnavailableError):
        vision_infrastructure_readiness_service.require_ready("mongodb")
