from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ml.training.common import (
    ExperimentTracker,
    finish_validation_only,
    parse_training_cli,
    seed_everything,
    validate_training_inputs,
    write_run_manifest,
)


def train_yolo26(config: Any) -> dict[str, Any]:
    validation = validate_training_inputs(config)
    seed_everything(config.seed)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install ultralytics from ai-standard.txt before YOLO26 training") from exc
    data_yaml = Path(str(config.parameters.get("data_yaml", ""))).expanduser()
    if not data_yaml.is_file():
        raise FileNotFoundError("parameters.data_yaml must reference a prepared YOLO dataset YAML")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with ExperimentTracker(config, validation) as tracker:
        model = YOLO(config.base_model)
        result = model.train(
            data=str(data_yaml),
            epochs=config.epochs,
            batch=config.batch_size,
            imgsz=int(config.parameters.get("image_size", 1280)),
            patience=config.early_stopping_patience,
            optimizer=str(config.parameters.get("optimizer", "auto")),
            lr0=config.learning_rate,
            weight_decay=config.weight_decay,
            amp=config.mixed_precision,
            seed=config.seed,
            deterministic=True,
            project=str(config.output_dir),
            name="training",
            exist_ok=True,
            plots=True,
        )
        validation_result = model.val(
            data=str(data_yaml),
            split="val",
            plots=True,
        )
        test_result = model.val(
            data=str(data_yaml),
            split="test",
            plots=True,
        )
        best = config.output_dir / "training" / "weights" / "best.pt"
        metrics = {
            "status": "trained",
            "checkpoint": str(best.resolve()),
            "validation": _metrics(validation_result),
            "test": _metrics(test_result),
            "training": _metrics(result),
            "validation_based_model_selection": True,
        }
        manifest = write_run_manifest(config, validation, metrics)
        tracker.log_metrics({"validation": metrics["validation"], "test": metrics["test"]})
        tracker.log_artifact(manifest, "run")
        for artifact in (config.output_dir / "training").glob("*.png"):
            tracker.log_artifact(artifact, "evaluation")
    return metrics


def _metrics(result: Any) -> dict[str, float]:
    source = getattr(result, "results_dict", {}) or {}
    return {
        str(key): float(value)
        for key, value in source.items()
        if isinstance(value, (int, float))
    }


def main() -> None:
    config, validate_only = parse_training_cli("Fine-tune YOLO26 on document visual elements")
    validation = validate_training_inputs(config)
    if validate_only:
        finish_validation_only(config, validation)
        return
    print(json.dumps(train_yolo26(config), indent=2, default=str))


if __name__ == "__main__":
    main()
