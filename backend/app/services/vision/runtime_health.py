from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from app.core.config import settings


class ModelSmokeStore:
    """Small, atomic store for the latest verified result per model configuration."""

    def __init__(self, root: Path | None = None) -> None:
        configured = root or Path(settings.model_smoke_state_dir)
        self.root = (
            configured
            if configured.is_absolute()
            else Path(__file__).resolve().parents[4] / configured
        )
        self.path = self.root / "results.json"
        self._lock = threading.RLock()

    def all(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            payload = self._read()
            return {
                str(model_id): dict(result)
                for model_id, result in dict(payload.get("results") or {}).items()
                if isinstance(result, dict)
            }

    def latest(self, model_id: str) -> dict[str, Any] | None:
        return self.all().get(model_id)

    def record(self, result: dict[str, Any]) -> dict[str, Any]:
        model_id = str(result.get("model_id") or "").strip()
        if not model_id:
            raise ValueError("Smoke-test result requires model_id")
        with self._lock:
            payload = self._read()
            results = dict(payload.get("results") or {})
            results[model_id] = dict(result)
            self._write({"schema_version": 1, "results": results})
        return result

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema_version": 1, "results": {}}
        return value if isinstance(value, dict) else {"schema_version": 1, "results": {}}

    def _write(self, payload: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temporary.replace(self.path)


def model_configuration_fingerprint(backend: dict[str, Any]) -> str:
    material = {
        "model_id": backend.get("model_id"),
        "adapter_id": backend.get("adapter_id"),
        "stage": backend.get("stage"),
        "provider": backend.get("provider"),
        "enabled": backend.get("enabled"),
        "installed": backend.get("installed"),
        "configured_status": backend.get("configured_status", backend.get("status")),
        "runtime": backend.get("runtime") or {},
        "package_version": backend.get("package_version"),
    }
    deployment_identity = _deployment_identity(backend)
    if deployment_identity:
        # This material is never returned or stored. Only the outer SHA-256 value is
        # exposed as the configuration fingerprint.
        material["deployment_identity"] = deployment_identity
    serialized = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _deployment_identity(backend: dict[str, Any]) -> dict[str, Any] | None:
    model_id = str(backend.get("model_id") or "")
    adapter_id = str(backend.get("adapter_id") or "")
    if model_id == "Neo4j" or adapter_id == "Neo4j":
        return {
            "endpoint": _safe_endpoint_identity(settings.neo4j_uri),
            "database": settings.neo4j_database,
            "instance_id": settings.neo4j_instance_id,
            "credentials": _neo4j_credential_identity(),
        }
    if model_id == "Qdrant" or adapter_id == "qdrant-hybrid-retrieval":
        return {"endpoint": _safe_endpoint_identity(settings.qdrant_url)}
    return None


def _safe_endpoint_identity(value: str) -> dict[str, Any]:
    """Build endpoint identity without retaining userinfo, query values, or fragments."""

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return {
            "configured": bool(value),
            "invalid_value_digest": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
    return {
        "scheme": parsed.scheme.lower(),
        "hostname": hostname.lower(),
        "port": port,
        "path": parsed.path,
    }


def _neo4j_credential_identity() -> dict[str, Any]:
    user = settings.neo4j_user or ""
    password = settings.neo4j_password or ""
    fallback_auth = ""
    if not user and not password:
        fallback_auth = settings.neo4j_auth or ""
    present = bool(user or password or fallback_auth)
    if not present:
        return {"present": False, "complete": False, "digest": None}
    credential_material = json.dumps(
        {"user": user, "password": password, "fallback_auth": fallback_auth},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "present": True,
        "complete": bool(user and password) or bool(fallback_auth),
        "digest": hashlib.sha256(credential_material.encode("utf-8")).hexdigest(),
    }


def safe_model_slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", model_id).strip("-") or "model"


model_smoke_store = ModelSmokeStore()
