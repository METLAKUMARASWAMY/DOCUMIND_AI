from __future__ import annotations

import json
from typing import Any

import numpy as np

from ml.training.anomaly_models.common import binary_metrics, tabular_split, validation_threshold
from ml.training.common import (
    ExperimentTracker,
    finish_validation_only,
    parse_training_cli,
    seed_everything,
    validate_training_inputs,
    write_run_manifest,
)


def train_xgboost(config: Any) -> dict[str, Any]:
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("Install xgboost from ai-standard.txt before supervised training") from exc
    validation = validate_training_inputs(config, required_record_keys={"features", "label"})
    seed_everything(config.seed)
    train_x, train_y, feature_names, _ = tabular_split(config.train_manifest)
    validation_x, validation_y, _, validation_ids = tabular_split(config.validation_manifest, feature_names)
    test_x, test_y, _, test_ids = tabular_split(config.test_manifest, feature_names)
    if train_y is None or validation_y is None or test_y is None:
        raise ValueError("XGBoost requires binary labels in every split")
    positive = int(train_y.sum())
    negative = int(train_y.size - positive)
    scale_positive = negative / max(positive, 1)
    model = xgb.XGBClassifier(
        n_estimators=int(config.parameters.get("n_estimators", 800)),
        max_depth=int(config.parameters.get("max_depth", 5)),
        learning_rate=config.learning_rate,
        subsample=float(config.parameters.get("subsample", 0.85)),
        colsample_bytree=float(config.parameters.get("colsample_bytree", 0.85)),
        reg_lambda=float(config.parameters.get("reg_lambda", 1.0)),
        reg_alpha=float(config.parameters.get("reg_alpha", 0.0)),
        scale_pos_weight=scale_positive if config.weighted_loss else 1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        early_stopping_rounds=config.early_stopping_patience,
        random_state=config.seed,
        n_jobs=int(config.parameters.get("n_jobs", -1)),
    )
    with ExperimentTracker(config, validation) as tracker:
        model.fit(
            train_x,
            train_y,
            eval_set=[(validation_x, validation_y)],
            verbose=False,
        )
        validation_scores = model.predict_proba(validation_x)[:, 1]
        threshold = validation_threshold(validation_y, validation_scores, 0.95)
        test_scores = model.predict_proba(test_x)[:, 1]
        config.output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = config.output_dir / "xgboost-risk.json"
        model.save_model(checkpoint)
        explanations = _shap_explanations(
            model,
            test_x,
            feature_names,
            test_ids,
            int(config.parameters.get("shap_samples", 100)),
        )
        shap_path = config.output_dir / "shap_explanations.json"
        shap_path.write_text(json.dumps(explanations, indent=2), encoding="utf-8")
        result = {
            "status": "trained",
            "checkpoint": str(checkpoint.resolve()),
            "features": feature_names,
            "scale_pos_weight": scale_positive,
            "best_iteration": getattr(model, "best_iteration", None),
            "threshold_selected_on_validation": threshold,
            "validation": binary_metrics(validation_y, validation_scores, threshold),
            "test": binary_metrics(test_y, test_scores, threshold),
            "shap_explanations": str(shap_path.resolve()),
            "validation_document_ids": validation_ids,
            "validation_based_model_selection": True,
        }
        manifest = write_run_manifest(config, validation, result)
        tracker.log_metrics({"validation": result["validation"], "test": result["test"]})
        tracker.log_artifact(checkpoint, "checkpoint")
        tracker.log_artifact(shap_path, "explainability")
        tracker.log_artifact(manifest, "run")
    return result


def _shap_explanations(
    model: Any,
    matrix: np.ndarray,
    feature_names: list[str],
    document_ids: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install shap from ai-standard.txt to produce required explanations") from exc
    selected = matrix[: max(1, min(limit, len(matrix)))]
    values = np.asarray(shap.TreeExplainer(model).shap_values(selected))
    return [
        {
            "document_id": document_ids[index],
            "contributions": {
                feature: round(float(values[index, feature_index]), 8)
                for feature_index, feature in enumerate(feature_names)
            },
        }
        for index in range(len(selected))
    ]


def main() -> None:
    config, validate_only = parse_training_cli("Train the supervised XGBoost document-risk model")
    validation = validate_training_inputs(config, required_record_keys={"features", "label"})
    if validate_only:
        finish_validation_only(config, validation)
        return
    print(json.dumps(train_xgboost(config), indent=2, default=str))


if __name__ == "__main__":
    main()
