"""Confidence scoring and the thresholds that route work to review.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.models.vision import (
    OCRWord,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionRiskIndicator,
    VisionRiskSeverity,
)
from app.services.vision.ocr_lines import ocr_line_reconstructor


class ConfidenceMixin:
    def _load_confidence_policy(self) -> dict[str, Any]:
        fallback = {
            "field_confidence_weights": {
                "ocr_confidence": 0.18,
                "layoutlmv3_probability": 0.18,
                "layout_agreement": 0.12,
                "donut_agreement": 0.10,
                "visual_parser_agreement": 0.08,
                "field_format_validity": 0.10,
                "arithmetic_consistency": 0.10,
                "cross_document_consistency": 0.08,
                "retrieval_evidence_quality": 0.06,
            },
            "review_thresholds": {
                "auto_approval_min_confidence": 0.90,
                "medium_review_min_confidence": 0.70,
                "mandatory_review_below_confidence": 0.70,
            },
            "calibration": {"calibrated": False},
        }
        path = Path(__file__).resolve().parents[3] / "config" / "vision_confidence.yml"
        if not path.exists():
            return fallback
        try:
            import yaml

            loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return {**fallback, **loaded}
        except Exception:
            return fallback

    def _field_confidence(
        self,
        value: str,
        words: list[OCRWord],
        box: VisionBoundingBox | None,
    ) -> tuple[float, dict[str, float]]:
        weights = self.confidence_policy["field_confidence_weights"]
        matched_words = self._words_for_text(value, words)
        components = {
            "ocr_confidence": round(
                sum(word.confidence for word in matched_words) / len(matched_words),
                4,
            )
            if matched_words
            else 0.0,
            "layout_agreement": 1.0 if box and matched_words else 0.0,
            "field_format_validity": 1.0 if self._format_valid(value) else 0.0,
        }
        return self._score_confidence_components(components, weights), components

    def _enrich_field_confidence(
        self,
        document: VisionDocumentRecord,
        risks: list[VisionRiskIndicator],
    ) -> None:
        weights = self.confidence_policy["field_confidence_weights"]
        risk_evidence_ids = {evidence.evidence_id for risk in risks for evidence in risk.evidence}
        donut = document.model_routing.get("donut_comparison") or {}
        donut_output = json.dumps(donut.get("output") or {}, sort_keys=True).lower()
        donut_disagreements = {
            str(item.get("field_name"))
            for item in donut.get("disagreements", [])
            if isinstance(item, dict)
        }
        advanced = document.model_routing.get("advanced_parsing") or {}
        advanced_text = json.dumps(advanced.get("blocks") or {}, sort_keys=True).lower()
        field_map = self._field_map(document)
        arithmetic_available = all(
            field_map.get(name) for name in ("subtotal", "tax", "total_amount")
        )
        arithmetic_failed = any(risk.title == "Amount Arithmetic Mismatch" for risk in risks)
        cross_metadata_present = any(
            key in document.metadata
            for key in (
                "approved_bank_identifier",
                "approved_bank_account",
                "po_vendor",
                "purchase_order_vendor",
                "po_amount",
                "purchase_order_amount",
                "contract_payment_terms",
                "contract_effective_date",
                "contract_expiry_date",
            )
        )
        for field in document.extracted_fields:
            components = dict(field.confidence_components)
            if field.model_name.lower().startswith("layoutlm"):
                if field.confidence is not None:
                    components.setdefault("layoutlmv3_probability", field.confidence)
            normalized_value = self._normalize(field.reviewed_value or field.value)
            if donut_output and field.name.lower() in donut_output:
                components["donut_agreement"] = 0.0 if field.name in donut_disagreements else 1.0
            if advanced_text and normalized_value:
                components["visual_parser_agreement"] = float(
                    normalized_value in self._normalize(advanced_text)
                )
            if arithmetic_available and field.name in {
                "subtotal",
                "tax",
                "discount",
                "total_amount",
            }:
                components["arithmetic_consistency"] = 0.0 if arithmetic_failed else 1.0
            if cross_metadata_present and field.name in {
                "vendor_name",
                "purchase_order_number",
                "bank_account_number",
                "IFSC_or_bank_code",
                "total_amount",
                "currency",
                "payment_terms",
                "invoice_date",
            }:
                components["cross_document_consistency"] = float(
                    field.evidence.evidence_id not in risk_evidence_ids
                )
            if document.evidence_chunks:
                components["retrieval_evidence_quality"] = float(
                    any(
                        normalized_value and normalized_value in self._normalize(chunk.text)
                        for chunk in document.evidence_chunks
                    )
                )
            field.confidence_components = components
            field.confidence = self._score_confidence_components(components, weights)
            field.combined_confidence = field.confidence
            field.ocr_confidence = components.get("ocr_confidence")
            field.extraction_confidence = components.get(
                "layoutlmv3_probability",
                components.get("extraction_confidence", field.extraction_confidence),
            )
            field.validation_confidence = components.get("field_format_validity")
            agreement_values = [
                components.get(name)
                for name in (
                    "layout_agreement",
                    "donut_agreement",
                    "visual_parser_agreement",
                    "cross_document_consistency",
                )
                if components.get(name) is not None
            ]
            field.agreement_confidence = (
                round(sum(agreement_values) / len(agreement_values), 4)
                if agreement_values
                else None
            )
            field.calibrated_confidence = field.confidence if field.calibrated else None
            field.evidence.model_confidence = field.confidence
            field.evidence.metadata["confidence_components"] = components

    def _score_confidence_components(
        self,
        components: dict[str, float],
        weights: dict[str, Any],
    ) -> float:
        active_weights = {
            key: float(weights[key])
            for key in components
            if key in weights and components[key] is not None
        }
        denominator = sum(active_weights.values())
        score = (
            sum(components[key] * weight for key, weight in active_weights.items()) / denominator
            if denominator
            else 0.0
        )
        return round(max(0.0, min(1.0, score)), 4)

    def _mean_word_confidence(self, document: VisionDocumentRecord) -> float | None:
        if not document.ocr_words:
            return None
        return round(
            sum(word.confidence for word in document.ocr_words) / len(document.ocr_words), 3
        )

    def _text_from_ocr(self, words: list[OCRWord]) -> str:
        lines = ocr_line_reconstructor.reconstruct(words)
        pages: dict[int, list[str]] = {}
        for line in lines:
            pages.setdefault(line.page, []).append(line.text)
        return "\n\n".join("\n".join(pages[page_number]) for page_number in sorted(pages))

    def _mean_field_confidence(self, document: VisionDocumentRecord) -> float | None:
        if not document.extracted_fields:
            return None
        values = [
            field.confidence for field in document.extracted_fields if field.confidence is not None
        ]
        return round(sum(values) / len(values), 3) if values else None

    def _mean_available_field_confidence(
        self,
        fields: Any,
    ) -> float | None:
        values = [field.confidence for field in fields if field.confidence is not None]
        return round(sum(values) / len(values), 4) if values else None

    def _layout_confidence(self, document: VisionDocumentRecord) -> float | None:
        if not document.layout_regions:
            return None
        return round(
            sum(region.confidence for region in document.layout_regions)
            / len(document.layout_regions),
            3,
        )

    def _review_threshold(
        self,
        name: str,
        document: VisionDocumentRecord | None = None,
    ) -> float:
        value = self.confidence_policy["review_thresholds"][name]
        if document:
            document_override = self.confidence_policy.get("document_type_overrides", {}).get(
                document.document_type.value,
                {},
            )
            organisation_override = self.confidence_policy.get("organisation_overrides", {}).get(
                document.tenant_id,
                {},
            )
            value = document_override.get(name, value)
            value = organisation_override.get(name, value)
        return float(value)

    def _mandatory_review_threshold(
        self,
        document: VisionDocumentRecord | None = None,
    ) -> float:
        return self._review_threshold("mandatory_review_below_confidence", document)

    def _review_priority(self, document: VisionDocumentRecord) -> VisionRiskSeverity:
        severities = [
            item.severity for item in [*document.risk_indicators, *document.processing_warnings]
        ]
        if VisionRiskSeverity.critical in severities:
            return VisionRiskSeverity.critical
        if VisionRiskSeverity.high in severities:
            return VisionRiskSeverity.high
        if VisionRiskSeverity.medium in severities:
            return VisionRiskSeverity.medium
        return VisionRiskSeverity.low
