from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrainingConfig:
    task: str
    base_model: str
    train_manifest: Path
    validation_manifest: Path
    test_manifest: Path
    output_dir: Path
    dataset_name: str
    dataset_version: str
    seed: int = 42
    epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 5
    mixed_precision: bool = True
    weighted_loss: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    mlflow_enabled: bool = True
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "documind-vision-ai"
    model_version: str = "0.1.0"
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path) -> TrainingConfig:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("Training configuration must be a YAML object")
        expanded = _expand(payload)
        for key in ("train_manifest", "validation_manifest", "test_manifest", "output_dir"):
            if key in expanded:
                expanded[key] = Path(expanded[key]).expanduser()
        return cls(**expanded)


def parse_training_cli(description: str) -> tuple[TrainingConfig, bool]:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args()
    config = TrainingConfig.from_file(arguments.config)
    if arguments.no_mlflow:
        config.mlflow_enabled = False
    return config, arguments.validate_only


def validate_training_inputs(
    config: TrainingConfig,
    *,
    required_record_keys: set[str] | None = None,
) -> dict[str, Any]:
    required_record_keys = required_record_keys or set()
    manifests = {
        "train": config.train_manifest,
        "validation": config.validation_manifest,
        "test": config.test_manifest,
    }
    records = {name: read_manifest(path) for name, path in manifests.items()}
    document_sets = {
        name: {str(record["document_id"]) for record in values}
        for name, values in records.items()
    }
    leakage = {
        "train_validation": document_sets["train"] & document_sets["validation"],
        "train_test": document_sets["train"] & document_sets["test"],
        "validation_test": document_sets["validation"] & document_sets["test"],
    }
    leaked = {key: sorted(value) for key, value in leakage.items() if value}
    if leaked:
        raise ValueError(f"Document-level data leakage detected: {leaked}")
    missing = []
    for split_name, split_records in records.items():
        for index, record in enumerate(split_records):
            absent = sorted(key for key in required_record_keys if record.get(key) is None)
            if absent:
                missing.append({"split": split_name, "record": index, "keys": absent})
    if missing:
        raise ValueError(f"Required training fields are missing: {missing[:10]}")
    validation = {
        "task": config.task,
        "dataset_name": config.dataset_name,
        "dataset_version": config.dataset_version,
        "seed": config.seed,
        "split_unit": "complete_document",
        "records": {name: len(values) for name, values in records.items()},
        "documents": {name: len(values) for name, values in document_sets.items()},
        "manifest_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in manifests.items()
        },
        "leakage_detected": False,
    }
    return validation


def read_manifest(path: Path) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("document_id"):
            raise ValueError(f"Invalid manifest record at {path}:{line_number}")
        records.append(value)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass


def class_weights(labels: list[int], class_count: int) -> list[float]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=class_count)
    total = int(counts.sum())
    return [
        float(total / (class_count * count)) if count else 0.0
        for count in counts
    ]


def write_run_manifest(
    config: TrainingConfig,
    validation: dict[str, Any],
    result: dict[str, Any],
) -> Path:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "dataset_validation": validation,
        "result": result,
        "environment": package_versions(
            ["torch", "transformers", "datasets", "ultralytics", "xgboost", "scikit-learn", "mlflow"]
        ),
        "metrics_source": "validation_or_test_predictions_only",
    }
    path = config.output_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return path


def package_versions(names: list[str]) -> dict[str, str | None]:
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


class ExperimentTracker(AbstractContextManager["ExperimentTracker"]):
    def __init__(self, config: TrainingConfig, validation: dict[str, Any]) -> None:
        self.config = config
        self.validation = validation
        self._mlflow: Any | None = None
        self._run: Any | None = None

    def __enter__(self) -> ExperimentTracker:
        if not self.config.mlflow_enabled:
            return self
        try:
            import mlflow
        except ImportError as exc:
            raise RuntimeError(
                "MLflow tracking is enabled but mlflow is not installed; install ai-standard.txt "
                "or pass --no-mlflow."
            ) from exc
        self._mlflow = mlflow
        if self.config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.mlflow_experiment)
        self._run = mlflow.start_run(run_name=f"{self.config.task}-{self.config.dataset_version}")
        mlflow.log_params(
            {
                "task": self.config.task,
                "base_model": self.config.base_model,
                "dataset_name": self.config.dataset_name,
                "dataset_version": self.config.dataset_version,
                "seed": self.config.seed,
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "weight_decay": self.config.weight_decay,
                "mixed_precision": self.config.mixed_precision,
                "weighted_loss": self.config.weighted_loss,
                "model_version": self.config.model_version,
            }
        )
        for split, digest in self.validation["manifest_sha256"].items():
            mlflow.set_tag(f"dataset.{split}.sha256", digest)
        mlflow.set_tag("dataset.split_unit", "complete_document")
        return self

    def log_metrics(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if not self._mlflow:
            return
        numeric = _flatten_numeric(metrics)
        if numeric:
            self._mlflow.log_metrics(numeric, step=step)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if self._mlflow and path.exists():
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._mlflow and self._run:
            status = "FAILED" if exc_type else "FINISHED"
            self._mlflow.end_run(status=status)
        return False


def transformer_training_arguments(config: TrainingConfig) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for this training task") from exc
    use_cuda = torch.cuda.is_available()
    return {
        "output_dir": str(config.output_dir),
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 20,
        "load_best_model_at_end": True,
        "metric_for_best_model": config.metric_for_best_model,
        "greater_is_better": config.greater_is_better,
        "save_total_limit": 2,
        "fp16": bool(config.mixed_precision and use_cuda),
        "bf16": False,
        "seed": config.seed,
        "data_seed": config.seed,
        "report_to": [],
        "remove_unused_columns": False,
    }


def finish_validation_only(
    config: TrainingConfig,
    validation: dict[str, Any],
) -> None:
    result = {
        "status": "validated_only",
        "training_started": False,
        "metrics_reported": False,
    }
    path = write_run_manifest(config, validation, result)
    print(json.dumps({**result, "run_manifest": str(path.resolve())}, indent=2))


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _flatten_numeric(payload: dict[str, Any], prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    for key, value in payload.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, bool):
            output[name] = float(value)
        elif isinstance(value, (int, float)) and np.isfinite(value):
            output[name] = float(value)
        elif isinstance(value, dict):
            output.update(_flatten_numeric(value, name))
    return output
