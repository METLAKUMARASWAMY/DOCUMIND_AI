from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointValidation:
    ready: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


def validate_transformer_checkpoint(
    checkpoint: Path | None,
    *,
    expected_tasks: set[str],
    require_domain_labels: bool = False,
) -> CheckpointValidation:
    if checkpoint is None:
        return CheckpointValidation(False, "No checkpoint path is configured.")
    if not checkpoint.is_dir():
        return CheckpointValidation(False, f"Checkpoint directory does not exist: {checkpoint}")
    config_path = checkpoint / "config.json"
    weights = next(
        (
            candidate
            for candidate in (
                checkpoint / "model.safetensors",
                checkpoint / "pytorch_model.bin",
            )
            if candidate.is_file()
        ),
        None,
    )
    if not config_path.is_file() or weights is None:
        return CheckpointValidation(
            False,
            "Checkpoint must contain config.json and local model weights.",
            {"config_present": config_path.is_file(), "weights_present": weights is not None},
        )
    config = _read_json(config_path)
    if config is None:
        return CheckpointValidation(False, "Checkpoint config.json is invalid JSON.")
    if require_domain_labels and not _has_domain_labels(config):
        return CheckpointValidation(
            False,
            "Checkpoint has no project field labels; a generic pretrained head is not an extractor.",
            {"labels": _labels(config)},
        )
    manifest_path, manifest = _training_manifest(checkpoint)
    if manifest is None:
        return CheckpointValidation(
            False,
            "A DocuMind run_manifest.json with labelled validation/test provenance is required.",
            {"manifest": str(manifest_path) if manifest_path else None},
        )
    task = str(dict(manifest.get("config") or {}).get("task", ""))
    result = dict(manifest.get("result") or {})
    validation = dict(manifest.get("dataset_validation") or {})
    test_metrics = result.get("test")
    ready = (
        task in expected_tasks
        and result.get("status") == "trained"
        and isinstance(test_metrics, dict)
        and bool(test_metrics)
        and validation.get("leakage_detected") is False
    )
    if not ready:
        return CheckpointValidation(
            False,
            "Training manifest does not prove the expected task, held-out evaluation, and leakage gate.",
            {
                "manifest": str(manifest_path),
                "task": task or None,
                "expected_tasks": sorted(expected_tasks),
                "trained": result.get("status") == "trained",
                "test_metrics_present": isinstance(test_metrics, dict) and bool(test_metrics),
                "leakage_detected": validation.get("leakage_detected"),
            },
        )
    return CheckpointValidation(
        True,
        "Project checkpoint and labelled held-out evaluation manifest are present.",
        {
            "manifest": str(manifest_path),
            "task": task,
            "weights": str(weights),
            "test_metric_names": sorted(test_metrics),
            "labels": _labels(config),
        },
    )


def validate_project_file_checkpoint(
    checkpoint: Path | None,
    *,
    expected_tasks: set[str],
) -> CheckpointValidation:
    if checkpoint is None:
        return CheckpointValidation(False, "No checkpoint path is configured.")
    if not checkpoint.is_file():
        return CheckpointValidation(False, f"Checkpoint file does not exist: {checkpoint}")
    manifest_path, manifest = _training_manifest(checkpoint.parent)
    if manifest is None:
        return CheckpointValidation(
            False,
            "A DocuMind run_manifest.json is required beside the trained project checkpoint.",
        )
    task = str(dict(manifest.get("config") or {}).get("task", ""))
    result = dict(manifest.get("result") or {})
    validation = dict(manifest.get("dataset_validation") or {})
    test_metrics = result.get("test")
    ready = (
        task in expected_tasks
        and result.get("status") == "trained"
        and isinstance(test_metrics, dict)
        and bool(test_metrics)
        and validation.get("leakage_detected") is False
    )
    if not ready:
        return CheckpointValidation(
            False,
            "Checkpoint manifest does not prove project training and held-out evaluation.",
            {"manifest": str(manifest_path), "task": task or None},
        )
    return CheckpointValidation(
        True,
        "Trained project checkpoint and held-out evaluation manifest are present.",
        {"manifest": str(manifest_path), "task": task},
    )


def _training_manifest(checkpoint: Path) -> tuple[Path | None, dict[str, Any] | None]:
    for root in (checkpoint, *list(checkpoint.parents)[:4]):
        candidate = root / "run_manifest.json"
        value = _read_json(candidate)
        if value is not None:
            return candidate, value
    return None, None


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _labels(config: dict[str, Any]) -> list[str]:
    mapping = config.get("id2label") or {}
    if not isinstance(mapping, dict):
        return []
    return sorted({str(value) for value in mapping.values()})


def _has_domain_labels(config: dict[str, Any]) -> bool:
    labels = _labels(config)
    meaningful = [
        value
        for value in labels
        if value.upper() not in {"O", "OTHER"} and not value.upper().startswith("LABEL_")
    ]
    return bool(meaningful)
