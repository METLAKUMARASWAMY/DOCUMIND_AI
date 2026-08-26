from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from app.services.vision import runtime_health
from app.services.vision.checkpoint_utils import checkpoint_assets_complete
from app.services.vision.generator_adapters import (
    GroundedGeneratorError,
    normalize_grounded_text,
)
from app.services.vision.retrieval_adapters import QdrantHybridBackend
from app.services.vision.runtime_health import ModelSmokeStore, model_configuration_fingerprint


def test_checkpoint_requires_real_weight_assets(tmp_path) -> None:
    checkpoint = tmp_path / "interrupted-model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model.safetensors").touch()

    assert checkpoint_assets_complete(checkpoint) is False

    (checkpoint / "model.safetensors").write_bytes(b"weights")
    assert checkpoint_assets_complete(checkpoint) is True


def test_checkpoint_requires_every_indexed_shard(tmp_path) -> None:
    checkpoint = tmp_path / "sharded-model"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"first")

    assert checkpoint_assets_complete(checkpoint) is False

    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"second")
    assert checkpoint_assets_complete(checkpoint) is True


def test_qdrant_collection_is_embedding_and_dimension_versioned() -> None:
    bge = QdrantHybridBackend(
        vector_namespace="BAAI/bge-m3",
        expected_vector_size=1024,
    )
    local = QdrantHybridBackend(
        vector_namespace="local-hashing-v1",
        expected_vector_size=768,
    )

    assert bge.collection_name == "vision_evidence_chunks__baai_bge_m3__d1024_v1"
    assert local.collection_name == "vision_evidence_chunks__local_hashing_v1__d768_v1"
    assert bge.collection_name != local.collection_name


def test_plain_generator_normalizer_rejects_unsupported_claims() -> None:
    evidence = [
        {
            "id": "evidence-1",
            "text": "The invoice total is INR 1250.00.",
            "review_status": "accepted",
        }
    ]

    supported = normalize_grounded_text("INR 1250.00", evidence)
    assert supported["citation_ids"] == ["evidence-1"]

    with pytest.raises(GroundedGeneratorError, match="not covered"):
        normalize_grounded_text("INR 9999.00", evidence)


def test_neo4j_deployment_changes_make_historical_smoke_stale(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = {
        "model_id": "Neo4j",
        "adapter_id": "Neo4j",
        "stage": "graph",
        "provider": "Neo4j",
        "enabled": True,
        "installed": True,
        "configured_status": "active",
        "runtime": {},
        "package_version": "6.2.0",
    }
    monkeypatch.setattr(runtime_health.settings, "neo4j_uri", "neo4j://first:7687")
    monkeypatch.setattr(runtime_health.settings, "neo4j_database", "neo4j")
    monkeypatch.setattr(runtime_health.settings, "neo4j_instance_id", "instance-first")
    monkeypatch.setattr(runtime_health.settings, "neo4j_user", "neo4j")
    monkeypatch.setattr(runtime_health.settings, "neo4j_password", "credential-first")
    first_fingerprint = model_configuration_fingerprint(backend)
    store = ModelSmokeStore(root=tmp_path)
    store.record(
        {
            "model_id": "Neo4j",
            "configuration_fingerprint": first_fingerprint,
            "status": "passed",
            "completed_at": datetime.now(UTC).isoformat(),
        }
    )
    monkeypatch.setattr(runtime_health, "model_smoke_store", store)

    from app.services.model_registry import ModelRegistryService

    monkeypatch.setattr(ModelRegistryService, "_external_service_live", lambda *_: True)
    current = ModelRegistryService()._attach_runtime_health(backend)
    assert current["status"] == "active"
    assert current["health_status"] == "healthy"
    assert "deployment_identity" not in current
    assert "credential-first" not in json.dumps(current)

    monkeypatch.setattr(runtime_health.settings, "neo4j_uri", "neo4j://second:7687")
    changed_uri = ModelRegistryService()._attach_runtime_health(backend)
    assert changed_uri["status"] == "unavailable"
    assert changed_uri["smoke_test_status"] == "not_run"
    assert changed_uri["stale_smoke_test"]["status"] == "passed"

    monkeypatch.setattr(runtime_health.settings, "neo4j_uri", "neo4j://first:7687")
    monkeypatch.setattr(runtime_health.settings, "neo4j_password", "credential-second")
    changed_credential = ModelRegistryService()._attach_runtime_health(backend)
    assert changed_credential["status"] == "unavailable"
    assert changed_credential["configuration_fingerprint"] != first_fingerprint

    monkeypatch.setattr(runtime_health.settings, "neo4j_password", "credential-first")
    monkeypatch.setattr(runtime_health.settings, "neo4j_database", "analytics")
    assert model_configuration_fingerprint(backend) != first_fingerprint
    monkeypatch.setattr(runtime_health.settings, "neo4j_database", "neo4j")
    monkeypatch.setattr(runtime_health.settings, "neo4j_instance_id", "instance-second")
    assert model_configuration_fingerprint(backend) != first_fingerprint


def test_expired_historical_smoke_is_stale_and_cannot_report_active(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = {
        "model_id": "Neo4j",
        "adapter_id": "Neo4j",
        "stage": "graph",
        "provider": "Neo4j",
        "enabled": True,
        "installed": True,
        "configured_status": "active",
        "runtime": {},
        "package_version": "6.2.0",
    }
    monkeypatch.setattr(runtime_health.settings, "neo4j_uri", "neo4j://first:7687")
    monkeypatch.setattr(runtime_health.settings, "neo4j_database", "neo4j")
    monkeypatch.setattr(runtime_health.settings, "neo4j_instance_id", "instance-first")
    monkeypatch.setattr(runtime_health.settings, "neo4j_user", "neo4j")
    monkeypatch.setattr(runtime_health.settings, "neo4j_password", "credential-first")
    store = ModelSmokeStore(root=tmp_path)
    store.record(
        {
            "model_id": "Neo4j",
            "configuration_fingerprint": model_configuration_fingerprint(backend),
            "status": "passed",
            "completed_at": (datetime.now(UTC) - timedelta(hours=25)).isoformat(),
        }
    )
    monkeypatch.setattr(runtime_health, "model_smoke_store", store)

    from app.services.model_registry import ModelRegistryService

    result = ModelRegistryService()._attach_runtime_health(backend)

    assert result["status"] == "unavailable"
    assert result["health_status"] == "unknown"
    assert result["smoke_test_status"] == "not_run"
    assert result["smoke_test"] is None
    assert result["last_smoke_test"] is None
    assert result["stale_smoke_test"]["status"] == "passed"
    assert result["stale_smoke_test"]["reason"] == "expired"


def test_fresh_smoke_cannot_hide_a_stopped_external_runtime(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = {
        "model_id": "Neo4j",
        "adapter_id": "Neo4j",
        "stage": "graph",
        "provider": "Neo4j",
        "enabled": True,
        "installed": True,
        "configured_status": "active",
        "runtime": {},
    }
    store = ModelSmokeStore(root=tmp_path)
    store.record(
        {
            "model_id": "Neo4j",
            "configuration_fingerprint": model_configuration_fingerprint(backend),
            "status": "passed",
            "completed_at": datetime.now(UTC).isoformat(),
        }
    )
    monkeypatch.setattr(runtime_health, "model_smoke_store", store)

    from app.services.model_registry import ModelRegistryService

    monkeypatch.setattr(ModelRegistryService, "_external_service_live", lambda *_: False)
    result = ModelRegistryService()._attach_runtime_health(backend)

    assert result["status"] == "unavailable"
    assert result["health_status"] == "unhealthy"
    assert result["smoke_test_status"] == "passed"
    assert result["last_error"] == "Live runtime probe failed."


def test_qdrant_endpoint_change_invalidates_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = {
        "model_id": "Qdrant",
        "adapter_id": "qdrant-hybrid-retrieval",
        "stage": "vector_store",
        "enabled": True,
        "installed": True,
        "configured_status": "active",
        "runtime": {},
    }
    monkeypatch.setattr(runtime_health.settings, "qdrant_url", "http://first:6333")
    first = model_configuration_fingerprint(backend)
    monkeypatch.setattr(runtime_health.settings, "qdrant_url", "http://second:6333")

    assert model_configuration_fingerprint(backend) != first
