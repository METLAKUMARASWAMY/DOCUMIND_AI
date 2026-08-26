from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from app.models.vision import LayoutRegion, OCRLineRecord, VisionDocumentType
from app.services.vision.schema_registry import vision_schema_registry

PARENT_CATEGORY: dict[VisionDocumentType, str] = {
    VisionDocumentType.invoice: "transaction_document",
    VisionDocumentType.purchase_order: "transaction_document",
    VisionDocumentType.receipt: "transaction_document",
    VisionDocumentType.bank_document: "transaction_document",
    VisionDocumentType.bank_confirmation: "transaction_document",
    VisionDocumentType.statement: "transaction_document",
    VisionDocumentType.product_specification: "structured_form",
    VisionDocumentType.sample_form: "structured_form",
    VisionDocumentType.contract: "contract_document",
    VisionDocumentType.legal_form: "contract_document",
    VisionDocumentType.financial_report: "report_document",
    VisionDocumentType.general_report: "report_document",
    VisionDocumentType.report: "report_document",
    VisionDocumentType.fax_transmission: "structured_form",
    VisionDocumentType.business_form: "structured_form",
    VisionDocumentType.form: "structured_form",
    VisionDocumentType.generic_form: "structured_form",
    VisionDocumentType.insurance_document: "structured_form",
    VisionDocumentType.correspondence: "correspondence",
    VisionDocumentType.letter: "correspondence",
    VisionDocumentType.email: "correspondence",
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
    schema_mode: str = "C"

    def as_dict(self) -> dict[str, Any]:
        return {
            "predicted_type": self.predicted_type.value,
            "parent_category": self.parent_category,
            "confidence": self.confidence,
            "classifier": self.classifier,
            "signals": [signal.as_dict() for signal in self.signals],
            "alternatives": list(self.alternatives),
            "candidate_types": [
                {"type": self.predicted_type.value, "confidence": self.confidence},
                *[
                    {
                        "type": item.get("document_type", "unknown"),
                        "confidence": item.get("confidence", 0.0),
                    }
                    for item in self.alternatives
                ],
            ],
            "reasons": [signal.value for signal in self.signals],
            "schema_mode": self.schema_mode,
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
            VisionDocumentType.product_specification,
            re.compile(r"\b(?:product|technical|material)\s+specifications?\b", re.I),
            0.90,
        ),
        (
            VisionDocumentType.sample_form,
            re.compile(r"\b(?:sample\s+(?:request|requisition|submission)\s+form)\b", re.I),
            0.91,
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
        first_title_line = next(
            (line.strip() for line in title_text.splitlines() if line.strip()),
            "",
        )
        for document_type, pattern, weight in self._title_rules:
            if match := pattern.search(title_text):
                title_weight = (
                    min(0.99, weight + 0.04)
                    if first_title_line and pattern.search(first_title_line)
                    else weight
                )
                signals.append(
                    ClassificationSignal(
                        document_type,
                        "title_keyword",
                        match.group(0).upper(),
                        title_weight,
                    )
                )

        # A filename is source metadata, not semantic evidence. Keeping this argument
        # preserves the public adapter contract while deliberately excluding it from
        # classification so renaming an upload cannot change its meaning.
        del filename
        normalized = text.lower()
        normalized_words = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
        for schema in vision_schema_registry.classification_schemas():
            if schema.document_type is None:
                continue
            if schema.document_type == VisionDocumentType.email:
                matched_hints = tuple(
                    hint
                    for hint in schema.classification_hints
                    if re.search(
                        rf"(?mi)^\s*{re.escape(hint.rstrip(':').strip())}\s*:",
                        text,
                    )
                )
            else:
                matched_hints = tuple(
                    hint
                    for hint in schema.classification_hints
                    if self._contains_phrase(normalized_words, hint)
                )
            distinctive = tuple(
                hint
                for hint in matched_hints
                if len(re.sub(r"[^a-z0-9]+", " ", hint.lower()).split()) >= 2
            )
            if not matched_hints or (not distinctive and len(matched_hints) < 2):
                continue
            signals.append(
                ClassificationSignal(
                    schema.document_type,
                    "schema_registry_hint",
                    f"{schema.name}: {', '.join(matched_hints[:4])}",
                    min(
                        0.84,
                        0.62 + len(matched_hints) * 0.05 + bool(distinctive) * 0.04,
                    ),
                )
            )
        feature_rules = (
            (
                VisionDocumentType.fax_transmission,
                r"\b(?:fax|facsimile)\b.*\b(?:to|from|phone|pages?)\b",
                0.72,
            ),
            (
                VisionDocumentType.invoice,
                r"\b(?:invoice\s+(?:number|date)|amount\s+due|grand\s+total)\b",
                0.66,
            ),
            (
                VisionDocumentType.purchase_order,
                r"\b(?:po\s+(?:number|date)|ship\s+to|buyer)\b",
                0.65,
            ),
            (VisionDocumentType.receipt, r"\b(?:cash|change|tendered|thank\s+you)\b", 0.56),
            (
                VisionDocumentType.bank_confirmation,
                r"\b(?:account\s+holder|beneficiary|swift|routing)\b",
                0.61,
            ),
            (
                VisionDocumentType.contract,
                r"\b(?:governing\s+law|termination|hereinafter|parties)\b",
                0.65,
            ),
            (
                VisionDocumentType.financial_report,
                r"\b(?:balance\s+sheet|income\s+statement|cash\s+flow)\b",
                0.64,
            ),
            (VisionDocumentType.correspondence, r"\b(?:dear|sincerely|regards)\b", 0.55),
            (VisionDocumentType.email, r"(?m)^\s*(?:from|to|sent|subject)\s*:\s*.+$", 0.58),
            (
                VisionDocumentType.product_specification,
                r"\b(?:dimensions?|tolerance|material|specification)\b",
                0.57,
            ),
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
        sample_aliases = {
            "sample_number": ("sample number", "sample no", "sample id"),
            "batch_size": ("batch size", "lot size"),
            "requested_by": ("original request made by", "requested by"),
            "specifications_written_by": (
                "sample specifications written by",
                "specifications written by",
            ),
            "product_type": ("product type", "sample type", "type of cigarette"),
        }
        semantic_lines = [line.text for line in (lines or [])] or text.splitlines()
        sample_labels = {
            label
            for label, aliases in sample_aliases.items()
            if self._contains_semantic_label(semantic_lines, aliases)
        }
        if len(sample_labels) >= 2:
            signals.append(
                ClassificationSignal(
                    VisionDocumentType.business_form,
                    "semantic_label_cluster",
                    f"sample/request form labels: {', '.join(sorted(sample_labels))}",
                    min(0.94, 0.76 + len(sample_labels) * 0.035),
                )
            )
        elif key_value_count >= 3 or re.search(
            r"\b(?:application|registration|request|intake)\s+form\b", normalized
        ):
            signals.append(
                ClassificationSignal(
                    VisionDocumentType.generic_form,
                    "layout_characteristic",
                    (
                        f"{key_value_count} label-value lines"
                        if key_value_count >= 3
                        else "generic form terminology"
                    ),
                    min(0.74, 0.57 + key_value_count * 0.025),
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
                schema_mode="C",
            )

        scores: dict[VisionDocumentType, float] = {}
        for signal in signals:
            scores[signal.document_type] = max(scores.get(signal.document_type, 0.0), signal.weight)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0].value))
        predicted, confidence = ranked[0]
        runner_up_confidence = ranked[1][1] if len(ranked) > 1 else 0.0
        predicted_signals = tuple(signal for signal in signals if signal.document_type == predicted)
        weak_schema_only = confidence < 0.75 and all(
            signal.signal_type == "schema_registry_hint" for signal in predicted_signals
        )
        ambiguous_candidates = (
            confidence < 0.80
            and confidence - runner_up_confidence < 0.08
            and not (
                ranked[1][0] == VisionDocumentType.generic_form
                and any(
                    signal.signal_type in {"schema_registry_hint", "title_keyword"}
                    for signal in predicted_signals
                )
            )
        )
        if weak_schema_only or ambiguous_candidates:
            reason = (
                f"Only weak schema hints supported {predicted.value} ({confidence:.2f})"
                if weak_schema_only
                else (
                    f"Top candidates {ranked[0][0].value} ({confidence:.2f}) and "
                    f"{ranked[1][0].value} ({runner_up_confidence:.2f}) were too close"
                )
            )
            return HybridClassificationResult(
                predicted_type=VisionDocumentType.unknown,
                parent_category="unknown",
                confidence=round(max(0.35, confidence - runner_up_confidence), 4),
                signals=(
                    ClassificationSignal(
                        VisionDocumentType.unknown,
                        "ambiguous_candidate_abstention",
                        reason,
                        round(max(0.35, confidence - runner_up_confidence), 4),
                    ),
                ),
                alternatives=tuple(
                    {"document_type": item.value, "confidence": score} for item, score in ranked[:4]
                ),
                schema_mode="C",
            )
        selected_signals = tuple(
            sorted(
                (signal for signal in signals if signal.document_type == predicted),
                key=lambda signal: -signal.weight,
            )
        )
        alternatives = tuple(
            {"document_type": item.value, "confidence": score} for item, score in ranked[1:4]
        )
        return HybridClassificationResult(
            predicted_type=predicted,
            parent_category=PARENT_CATEGORY.get(predicted, "unknown"),
            confidence=round(confidence, 4),
            signals=selected_signals,
            alternatives=alternatives,
            schema_mode=(
                "C"
                if predicted == VisionDocumentType.unknown
                else "B"
                if predicted == VisionDocumentType.generic_form
                else "A"
            ),
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

    def _contains_semantic_label(
        self,
        lines: list[str],
        aliases: tuple[str, ...],
    ) -> bool:
        for line in lines:
            normalized_line = re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()
            tokens = normalized_line.split()
            for alias in aliases:
                normalized_alias = re.sub(r"[^a-z0-9]+", " ", alias.lower()).strip()
                if normalized_alias in normalized_line:
                    return True
                alias_tokens = normalized_alias.split()
                if not alias_tokens or len(tokens) < len(alias_tokens):
                    continue
                for start in range(min(4, len(tokens) - len(alias_tokens) + 1)):
                    observed = " ".join(tokens[start : start + len(alias_tokens)])
                    if SequenceMatcher(None, observed, normalized_alias).ratio() >= 0.82:
                        return True
        return False

    def _contains_phrase(self, normalized_text: str, phrase: str) -> bool:
        normalized_phrase = re.sub(r"[^a-z0-9]+", " ", phrase.lower()).strip()
        if not normalized_phrase:
            return False
        return (
            re.search(rf"(?:^|\s){re.escape(normalized_phrase)}(?:\s|$)", normalized_text)
            is not None
        )


hybrid_document_classifier = HybridDocumentClassifier()
