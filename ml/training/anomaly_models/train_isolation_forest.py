from __future__ import annotations

import json
from typing import Any

import joblib

from ml.training.anomaly_models.common import binary_metrics, tabular_split, validation_threshold
from ml.training.common import (
    ExperimentTracker,
    finish_validation_only,
    parse_training_cli,
    seed_everything,
    validate_training_inputs,
    write_run_manifest,
)


def train_isolation_forest(config: Any) -> dict[str, Any]:
    from sklearn.ensemble import IsolationForest

    validation = validate_training_inputs(config, required_record_keys={"features"})
    seed_everything(config.seed)
    train_x, _, feature_names, _ = tabular_split(config.train_manifest)
    validation_x, validation_y, _, _ = tabular_split(config.validation_manifest, feature_names)
    test_x, test_y, _, _ = tabular_split(config.test_manifest, feature_names)
    model = IsolationForest(
        n_estimators=int(config.parameters.get("n_estimators", 300)),
        contamination=config.parameters.get("contamination", "auto"),
        max_samples=config.parameters.get("max_samples", "auto"),
        random_state=config.seed,
        n_jobs=int(config.parameters.get("n_jobs", -1)),
    )
    with ExperimentTracker(config, validation) as tracker:
        model.fit(train_x)
        validation_scores = -model.decision_function(validation_x)
        threshold = validation_threshold(
            validation_y,
            validation_scores,
            float(config.parameters.get("review_quantile", 0.95)),
        )
        test_scores = -model.decision_function(test_x)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = config.output_dir / "isolation_forest.joblib"
        joblib.dump(
            {"model": model, "features": feature_names, "threshold": threshold},
            checkpoint,
        )
        result = {
            "status": "trained",
            "checkpoint": str(checkpoint.resolve()),
            "features": feature_names,
            "threshold_selected_on_validation": threshold,
            "validation": binary_metrics(validation_y, validation_scores, threshold),
            "test": binary_metrics(test_y, test_scores, threshold),
            "validation_based_model_selection": True,
        }
        manifest = write_run_manifest(config, validation, result)
        tracker.log_metrics({"validation": result["validation"], "test": result["test"]})
        tracker.log_artifact(checkpoint, "checkpoint")
        tracker.log_artifact(manifest, "run")
    return result


def main() -> None:
    config, validate_only = parse_training_cli("Train an Isolation Forest document-risk model")
    validation = validate_training_inputs(config, required_record_keys={"features"})
    if validate_only:
        finish_validation_only(config, validation)
        return
    print(json.dumps(train_isolation_forest(config), indent=2, default=str))


if __name__ == "__main__":
    main()
