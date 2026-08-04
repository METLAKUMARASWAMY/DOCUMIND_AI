from __future__ import annotations

import importlib.util
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

from app.core.config import settings
from app.models.vision import AnomalyAnalysisResult, VisionDocumentRecord
from app.services.vision.adapters import AdapterStatus, VisionAdapter

FEATURE_NAMES = (
    "amount_z_score_by_vendor",
    "amount_to_vendor_average_ratio",
    "time_since_previous_invoice_days",
    "invoice_frequency_30d",
    "account_change_frequency",
    "validation_failure_count",
    "ocr_uncertainty",
    "document_similarity",
    "submission_hour_utc",
    "linked_vendor_account_count",
)


class IsolationForestRiskAdapter(VisionAdapter):
    adapter_id = "isolation-forest-risk"
    stage = "unsupervised_anomaly"
    model_version = "sklearn-runtime-fit-v1"

    def status(self) -> AdapterStatus:
        installed = importlib.util.find_spec("sklearn") is not None
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if installed else "unavailable",
            reason=(
                "Scikit-learn Isolation Forest is available."
                if installed
                else "Install scikit-learn to enable unsupervised anomaly detection."
            ),
            enabled=True,
            installed=installed,
            model_version=self.model_version,
        )

    def analyze(
        self,
        document: VisionDocumentRecord,
        corpus: list[VisionDocumentRecord],
    ) -> AnomalyAnalysisResult:
        self.ensure_ready()
        eligible = [item for item in corpus if item.tenant_id == document.tenant_id]
        features_by_document = {
            item.document_id: self.features(item, eligible) for item in eligible
        }
        current_features = features_by_document.get(document.document_id) or self.features(
            document,
            eligible,
        )
        sample_count = len(features_by_document)
        if sample_count < settings.anomaly_min_samples:
            return AnomalyAnalysisResult(
                layer="unsupervised",
                status="insufficient_data",
                threshold=settings.anomaly_review_percentile,
                sample_count=sample_count,
                features=current_features,
                explanations=[
                    {
                        "reason": "minimum_sample_gate",
                        "required": settings.anomaly_min_samples,
                        "available": sample_count,
                    }
                ],
                model_name=self.adapter_id,
                model_version=self.model_version,
            )

        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import RobustScaler

        ordered_ids = list(features_by_document)
        matrix = np.asarray(
            [
                [features_by_document[item_id].get(name) for name in FEATURE_NAMES]
                for item_id in ordered_ids
            ],
            dtype=float,
        )
        imputed = matrix.copy()
        for column_index in range(imputed.shape[1]):
            column = imputed[:, column_index]
            finite = column[np.isfinite(column)]
            fill_value = float(np.median(finite)) if finite.size else 0.0
            column[~np.isfinite(column)] = fill_value
            imputed[:, column_index] = column
        scaled = RobustScaler().fit_transform(imputed)
        model = IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=1,
        )
        model.fit(scaled)
        raw_scores = -model.decision_function(scaled)
        current_index = ordered_ids.index(document.document_id)
        current_raw = float(raw_scores[current_index])
        percentile = float(np.count_nonzero(raw_scores <= current_raw) / len(raw_scores))
        deviations = self._deviations(scaled[current_index], FEATURE_NAMES)
        return AnomalyAnalysisResult(
            layer="unsupervised",
            status="completed",
            anomaly_score=round(percentile, 4),
            calibrated=False,
            threshold=settings.anomaly_review_percentile,
            sample_count=sample_count,
            features=current_features,
            explanations=deviations,
            model_name=self.adapter_id,
            model_version=self.model_version,
        )

    def features(
        self,
        document: VisionDocumentRecord,
        corpus: list[VisionDocumentRecord],
    ) -> dict[str, float | None]:
        fields = self._field_map(document)
        vendor = self._normalize(fields.get("vendor_name"))
        amount = self._money(fields.get("total_amount"))
        same_vendor = [
            item
            for item in corpus
            if item.document_id != document.document_id
            and self._normalize(self._field_map(item).get("vendor_name")) == vendor
        ]
        vendor_amounts = [
            value
            for item in same_vendor
            if (value := self._money(self._field_map(item).get("total_amount"))) is not None
        ]
        vendor_average = mean(vendor_amounts) if vendor_amounts else None
        vendor_std = pstdev(vendor_amounts) if len(vendor_amounts) >= 2 else None
        amount_z_score = (
            (amount - vendor_average) / vendor_std
            if amount is not None and vendor_average is not None and vendor_std not in {None, 0.0}
            else None
        )
        amount_ratio = (
            amount / vendor_average
            if amount is not None and vendor_average not in {None, 0.0}
            else None
        )
        current_date = self._date(fields.get("invoice_date")) or self._timestamp(document.created_at)
        prior_dates = sorted(
            date
            for item in same_vendor
            if (date := self._date(self._field_map(item).get("invoice_date")) or self._timestamp(item.created_at))
            < current_date
        )
        days_since = (current_date - prior_dates[-1]).total_seconds() / 86400 if prior_dates else None
        frequency = sum(0 <= (current_date - date).days <= 30 for date in prior_dates)
        account_values = {
            self._normalize(value)
            for item in same_vendor
            if (value := self._field_map(item).get("bank_account_number"))
        }
        current_account = fields.get("bank_account_number")
        account_change_frequency = (
            float(bool(account_values and current_account and self._normalize(current_account) not in account_values))
            if current_account
            else None
        )
        validation_failures = sum(
            risk.category in {"validation", "cross_document", "format_validation", "visual_validation"}
            for risk in document.risk_indicators
        )
        ocr_uncertainty = (
            1.0 - mean(word.confidence for word in document.ocr_words)
            if document.ocr_words
            else None
        )
        similarities = [
            candidate.similarity
            for candidate in document.duplicate_candidates
            if candidate.status != "no_duplicate"
        ]
        timestamp = self._timestamp(document.created_at)
        linked_values = {
            value
            for item in corpus
            for name in ("vendor_name", "bank_account_number")
            if (value := self._normalize(self._field_map(item).get(name)))
            and (
                value == vendor
                or (current_account and value == self._normalize(current_account))
            )
        }
        return {
            "amount_z_score_by_vendor": self._finite(amount_z_score),
            "amount_to_vendor_average_ratio": self._finite(amount_ratio),
            "time_since_previous_invoice_days": self._finite(days_since),
            "invoice_frequency_30d": float(frequency),
            "account_change_frequency": account_change_frequency,
            "validation_failure_count": float(validation_failures),
            "ocr_uncertainty": self._finite(ocr_uncertainty),
            "document_similarity": max(similarities) if similarities else 0.0,
            "submission_hour_utc": float(timestamp.hour),
            "linked_vendor_account_count": float(len(linked_values)),
        }

    def _deviations(
        self,
        row: np.ndarray,
        names: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        ranked = sorted(
            ((name, float(abs(value)), float(value)) for name, value in zip(names, row, strict=True)),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            {"feature": name, "robust_scaled_deviation": round(signed, 4)}
            for name, _, signed in ranked[:5]
        ]

    def _field_map(self, document: VisionDocumentRecord) -> dict[str, str]:
        values: dict[str, str] = {}
        for field in document.extracted_fields:
            values.setdefault(field.name, field.reviewed_value or field.value)
        return values

    def _money(self, value: str | None) -> float | None:
        if not value:
            return None
        import re

        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
        return float(match.group(0).replace(",", "")) if match else None

    def _date(self, value: str | None) -> datetime | None:
        if not value:
            return None
        cleaned = value.strip().rstrip(".")
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y"):
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
        return None

    def _timestamp(self, value: str) -> datetime:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)

    def _normalize(self, value: str | None) -> str:
        return " ".join((value or "").lower().split())

    def _finite(self, value: float | None) -> float | None:
        return value if value is not None and math.isfinite(value) else None


class XGBoostRiskAdapter(VisionAdapter):
    adapter_id = "xgboost-supervised-risk"
    stage = "supervised_anomaly"
    model_version = "project-checkpoint"

    def __init__(self) -> None:
        self._model: Any | None = None

    def status(self) -> AdapterStatus:
        installed = importlib.util.find_spec("xgboost") is not None
        checkpoint = (
            Path(settings.xgboost_risk_checkpoint).expanduser()
            if settings.xgboost_risk_checkpoint
            else None
        )
        ready = bool(checkpoint and checkpoint.is_file())
        enabled = settings.enable_heavy_models and bool(checkpoint)
        active = enabled and installed and ready
        return AdapterStatus(
            adapter_id=self.adapter_id,
            stage=self.stage,
            status="active" if active else ("disabled" if not enabled else "unavailable"),
            reason=(
                "Trained XGBoost checkpoint is ready."
                if active
                else "Provide XGBOOST_RISK_CHECKPOINT after training on labelled risk data."
            ),
            enabled=enabled,
            installed=installed,
            model_version=self.model_version,
        )

    def analyze(self, features: dict[str, float | None]) -> AnomalyAnalysisResult:
        self.ensure_ready()
        import xgboost as xgb

        model = self._load()
        values = [
            float(value) if (value := features.get(name)) is not None else np.nan
            for name in FEATURE_NAMES
        ]
        matrix = xgb.DMatrix([values], feature_names=list(FEATURE_NAMES), missing=np.nan)
        probability = float(model.predict(matrix)[0])
        contributions = model.predict(matrix, pred_contribs=True)[0]
        explanations = sorted(
            (
                {"feature": name, "shap_value": round(float(value), 6)}
                for name, value in zip(FEATURE_NAMES, contributions[:-1], strict=True)
            ),
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )[:5]
        return AnomalyAnalysisResult(
            layer="supervised",
            status="completed",
            supervised_probability=max(0.0, min(1.0, probability)),
            calibrated=False,
            features=features,
            explanations=explanations,
            model_name=self.adapter_id,
            model_version=self.model_version,
        )

    def _load(self) -> Any:
        if self._model is None:
            import xgboost as xgb

            self._model = xgb.Booster()
            self._model.load_model(str(Path(settings.xgboost_risk_checkpoint).expanduser()))
        return self._model
