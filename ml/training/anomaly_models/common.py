from __future__ import annotations

from typing import Any

import numpy as np

from ml.training.common import read_manifest


def tabular_split(path: Any, feature_names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray | None, list[str], list[str]]:
    records = read_manifest(path)
    feature_names = feature_names or sorted(
        {
            str(key)
            for record in records
            for key in (record.get("features") or {}).keys()
        }
    )
    if not feature_names:
        raise ValueError("Each anomaly manifest record must contain a numeric features object")
    matrix = []
    labels = []
    has_labels = True
    document_ids = []
    for record in records:
        features = record.get("features") or {}
        try:
            matrix.append([float(features.get(name, 0.0)) for name in feature_names])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric features in {record['document_id']}") from exc
        document_ids.append(str(record["document_id"]))
        if "label" not in record:
            has_labels = False
        labels.append(int(bool(record.get("label", False))))
    return (
        np.asarray(matrix, dtype=np.float64),
        np.asarray(labels, dtype=np.int64) if has_labels else None,
        feature_names,
        document_ids,
    )


def binary_metrics(labels: np.ndarray | None, scores: np.ndarray, threshold: float) -> dict[str, float | int | str]:
    if labels is None:
        return {
            "status": "not_calculated",
            "reason": "Labelled validation/test records are required for risk metrics.",
        }
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support

    predictions = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    false_positive = int(((predictions == 1) & (labels == 0)).sum())
    true_negative = int(((predictions == 0) & (labels == 0)).sum())
    return {
        "sample_count": int(labels.size),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "auprc": round(float(average_precision_score(labels, scores)), 4),
        "false_positive_rate": round(false_positive / max(false_positive + true_negative, 1), 4),
        "threshold": round(float(threshold), 8),
    }


def validation_threshold(labels: np.ndarray | None, scores: np.ndarray, default_quantile: float) -> float:
    if labels is None or len(set(labels.tolist())) < 2:
        return float(np.quantile(scores, default_quantile))
    candidates = np.unique(scores)
    best = (float("-inf"), float(candidates[0]))
    for threshold in candidates:
        metrics = binary_metrics(labels, scores, float(threshold))
        f1 = float(metrics.get("f1", 0.0))
        if f1 > best[0]:
            best = (f1, float(threshold))
    return best[1]
