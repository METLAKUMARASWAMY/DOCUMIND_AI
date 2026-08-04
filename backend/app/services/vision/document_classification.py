from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.models.vision import LayoutRegion, OCRLineRecord, VisionDocumentType

PARENT_CATEGORY: dict[VisionDocumentType, str] = {
    VisionDocumentType.invoice: "transaction_document",
    VisionDocumentType.purchase_order: "transaction_document",
    VisionDocumentType.receipt: "transaction_document",
    VisionDocumentType.bank_document: "transaction_document",
    VisionDocumentType.bank_confirmation: "transaction_document",
    VisionDocumentType.statement: "transaction_document",
    VisionDocumentType.contract: "contract_document",
    VisionDocumentType.legal_form: "contract_document",
    VisionDocumentType.financial_report: "report_document",
    VisionDocumentType.general_report: "report_document",
    VisionDocumentType.report: "report_document",
    VisionDocumentType.fax_transmission: "structured_form",
    VisionDocumentType.business_form: "structured_form",
    VisionDocumentType.form: "structured_form",
    VisionDocumentType.insurance_document: "structured_form",
    VisionDocumentType.correspondence: "correspondence",
    VisionDocumentType.letter: "correspondence",
    VisionDocumentType.unknown: "unknown",
}


@dataclass(frozen=True)
class ClassificationSignal:
    document_type: VisionDocumentType
    signal_type: str
    value: str
    weight: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": self.signal_type,
            "value": self.value,
            "weight": self.weight,
            "candidate_type": self.document_type.value,
        }


@dataclass(frozen=True)
class HybridClassificationResult:
    predicted_type: VisionDocumentType
    parent_category: str
    confidence: float
    signals: tuple[ClassificationSignal, ...]
    alternatives: tuple[dict[str, Any], ...]
    classifier: str = "hybrid-document-classifier-v3"
    user_override: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicted_type": self.predicted_type.value,
            "parent_category": self.parent_category,
            "confidence": self.confidence,
            "classifier": self.classifier,
            "signals": [signal.as_dict() for signal in self.signals],
            "alternatives": list(self.alternatives),
            "user_override": self.user_override,
        }


class HybridDocumentClassifier:
    adapter_id = "hybrid-document-classifier"
    model_version = "v3"

    _title_rules: tuple[tuple[VisionDocumentType, re.Pattern[str], float], ...] = (
        (
            VisionDocumentType.fax_transmission,
            re.compile(r"\b(?:fax\s+transmission|facsimile|fax\s+cover\s+sheet)\b", re.I),
            0.94,
        ),
        (
            VisionDocumentType.purchase_order,
            re.compile(r"\b(?:purchase\s+order|po\s+(?:number|no\.?|#))\b", re.I),
            0.93,
        ),
        (
            VisionDocumentType.bank_confirmation,
            re.compile(r"\b(?:bank|account)\s+confirmation\b", re.I),
            0.92,
        ),
        (
            VisionDocumentType.invoice,
            re.compile(r"\b(?:tax\s+invoice|invoice|commercial\s+invoice|bill)\b", re.I),
            0.92,
        ),
        (
            VisionDocumentType.contract,
            re.compile(r"\b(?:agreement|contract)\b", re.I),
            0.89,
        ),
        (
            VisionDocumentType.receipt,
            re.compile(r"\b(?:payment\s+receipt|cash\s+receipt|sales\s+receipt|receipt)\b", re.I),
            0.87,
        ),
        (
            VisionDocumentType.financial_report,
            re.compile(r"\b(?:financial\s+report|annual\s+report|financial\s+statements?)\b", re.I),
            0.88,
        ),
        (
            VisionDocumentType.general_report,
            re.compile(r"\b(?:inspection|incident|status|audit)\s+report\b", re.I),
            0.84,
        ),
        (
            VisionDocumentType.insurance_document,
            re.compile(r"\b(?:insurance\s+(?:policy|certificate|claim)|policy\s+number)\b", re.I),
            0.86,
        ),
        (
            VisionDocumentType.statement,
            re.compile(r"\b(?:bank|account|billing)\s+statement\b", re.I),
            0.86,
        ),
        (
            VisionDocumentType.letter,
            re.compile(r"\b(?:dear\s+(?:sir|madam)|formal\s+letter)\b", re.I),
            0.78,
        ),
    )

    def classify(
        self,
        text: str,
        filename: str,
        *,
        lines: list[OCRLineRecord] | None = None,
        layout_regions: list[LayoutRegion] | None = None,
        statistical_type: VisionDocumentType | None = None,
        statistical_confidence: float | None = None,
    ) -> HybridClassificationResult:
        signals: list[ClassificationSignal] = []
        title_text = self._title_text(text, lines or [], layout_regions or [])
        for document_type, pattern, weight in self._title_rules:
            if match := pattern.search(title_text):
                signals.append(
                    ClassificationSignal(
                        document_type,
                        "title_keyword",
                        match.group(0).upper(),
                        weight,
                    )
                )

        normalized = f"{filename}\n{text}".lower()
        feature_rules = (
            (VisionDocumentType.fax_transmission, r"\b(?:fax|facsimile)\b.*\b(?:to|from|phone|pages?)\b", 0.72),
            (VisionDocumentType.invoice, r"\b(?:invoice\s+(?:number|date)|amount\s+due|grand\s+total)\b", 0.66),
            (VisionDocumentType.purchase_order, r"\b(?:po\s+(?:number|date)|ship\s+to|buyer)\b", 0.65),
            (VisionDocumentType.receipt, r"\b(?:cash|change|tendered|thank\s+you)\b", 0.56),
            (VisionDocumentType.bank_confirmation, r"\b(?:account\s+holder|beneficiary|swift|routing)\b", 0.61),
            (VisionDocumentType.contract, r"\b(?:governing\s+law|termination|hereinafter|parties)\b", 0.65),
            (VisionDocumentType.financial_report, r"\b(?:balance\s+sheet|income\s+statement|cash\s+flow)\b", 0.64),
            (VisionDocumentType.correspondence, r"\b(?:dear|sincerely|regards)\b", 0.55),
        )
        for document_type, pattern, weight in feature_rules:
            if match := re.search(pattern, normalized, re.I | re.S):
                signals.append(
                    ClassificationSignal(
                        document_type,
                        "ocr_text_feature",
                        re.sub(r"\s+", " ", match.group(0))[:80],
                        weight,
                    )
                )

        key_value_count = sum(
            bool(line.metadata.get("key_value_candidate")) for line in (lines or [])
        )
        if key_value_count >= 3:
            signals.append(
                ClassificationSignal(
                    VisionDocumentType.business_form,
                    "layout_characteristic",
                    f"{key_value_count} label-value lines",
                    min(0.72, 0.54 + key_value_count * 0.025),
                )
            )
        elif re.search(r"\b(?:application|registration|request|intake)\s+form\b", normalized):
            signals.append(
                ClassificationSignal(
                    VisionDocumentType.business_form,
                    "ocr_text_feature",
                    "form terminology",
                    0.68,
                )
            )

        if statistical_type and statistical_type != VisionDocumentType.unknown:
            confidence = max(0.0, min(float(statistical_confidence or 0.5), 0.89))
            signals.append(
                ClassificationSignal(
                    statistical_type,
                    "statistical_classifier",
                    statistical_type.value,
                    confidence,
                )
            )

        if not signals:
            return HybridClassificationResult(
                predicted_type=VisionDocumentType.unknown,
                parent_category="unknown",
                confidence=0.35,
                signals=(
                    ClassificationSignal(
                        VisionDocumentType.unknown,
                        "confidence_fallback",
                        "No reliable title, layout, OCR, or model signal",
                        0.35,
                    ),
                ),
                alternatives=(),
            )

        scores: dict[VisionDocumentType, float] = {}
        for signal in signals:
            scores[signal.document_type] = max(scores.get(signal.document_type, 0.0), signal.weight)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0].value))
        predicted, confidence = ranked[0]
        selected_signals = tuple(
            sorted(
                (signal for signal in signals if signal.document_type == predicted),
                key=lambda signal: -signal.weight,
            )
        )
        alternatives = tuple(
            {"document_type": item.value, "confidence": score}
            for item, score in ranked[1:4]
        )
        return HybridClassificationResult(
            predicted_type=predicted,
            parent_category=PARENT_CATEGORY.get(predicted, "unknown"),
            confidence=round(confidence, 4),
            signals=selected_signals,
            alternatives=alternatives,
        )

    def _title_text(
        self,
        text: str,
        lines: list[OCRLineRecord],
        layout_regions: list[LayoutRegion],
    ) -> str:
        title_regions = [
            region.text
            for region in layout_regions
            if "title" in region.region_type.lower() and region.text.strip()
        ]
        top_lines = [
            line.text
            for line in sorted(lines, key=lambda item: (item.page, item.bounding_box.y1))
            if line.page == 1
        ][:12]
        plain_lines = [line for line in text.splitlines() if line.strip()][:12]
        return "\n".join([*title_regions, *top_lines, *plain_lines])[:3000]


hybrid_document_classifier = HybridDocumentClassifier()
