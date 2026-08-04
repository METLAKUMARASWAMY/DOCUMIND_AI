from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path
from typing import Any

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
    serialized = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def safe_model_slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", model_id).strip("-") or "model"


model_smoke_store = ModelSmokeStore()
