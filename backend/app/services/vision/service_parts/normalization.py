"""Value normalisation, geometry helpers and sensitive-value masking.

Part of VisionAIService, split out of service.py by responsibility (D-16).
Method bodies are unchanged; only their home module moved.
"""

from __future__ import annotations

import re
from datetime import datetime

from app.models.vision import (
    ExtractedVisionField,
    OCRWord,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionReviewTask,
)


class NormalizationMixin:
    def _field_map(self, document: VisionDocumentRecord) -> dict[str, str]:
        result = {}
        for field in document.extracted_fields:
            result.setdefault(field.name, field.reviewed_value or field.value)
        return result

    def _review_belongs_to_tenant(self, review: VisionReviewTask, tenant_id: str) -> bool:
        try:
            return self.get_document(review.document_id).tenant_id == tenant_id
        except ValueError:
            return False

    def _field_confidence_by_name(self, document: VisionDocumentRecord, name: str) -> float:
        field = next((item for item in document.extracted_fields if item.name == name), None)
        return field.confidence if field and field.confidence is not None else 0.0

    def _indicator_category(self, category: str) -> str:
        if category == "processing_warning" or category in {
            "processing",
            "ocr_quality",
            "model_routing",
        }:
            return "processing_warning"
        if category in {"validation", "format_validation", "visual_validation"}:
            return "data_quality"
        if category in {"tamper_screening", "visual_risk"}:
            return "visual_risk"
        return "business_risk"

    def _rule_id(self, category: str, title: str) -> str:
        prefix = (
            "cross_document" if category == "cross_document" else self._indicator_category(category)
        )
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        return f"{prefix}.{slug}"

    def _box_for_text(self, text: str, words: list[OCRWord]) -> VisionBoundingBox | None:
        matched = self._words_for_text(text, words)
        return self._union_box([word.bounding_box for word in matched])

    def _words_for_text(self, text: str, words: list[OCRWord]) -> list[OCRWord]:
        targets = [
            normalized
            for token in re.findall(r"\S+", text or "")
            if (normalized := self._normalize_token(token))
        ]
        if not targets:
            return []
        by_page: dict[int, list[OCRWord]] = {}
        for word in sorted(words, key=lambda item: item.reading_order):
            by_page.setdefault(word.page, []).append(word)

        best: list[OCRWord] = []
        for page_words in by_page.values():
            normalized_words = [self._normalize_token(word.text) for word in page_words]
            for target_start in range(len(targets)):
                for word_start, normalized_word in enumerate(normalized_words):
                    if normalized_word != targets[target_start]:
                        continue
                    matched: list[OCRWord] = []
                    target_index = target_start
                    word_index = word_start
                    while (
                        target_index < len(targets)
                        and word_index < len(page_words)
                        and len(matched) < 24
                        and normalized_words[word_index] == targets[target_index]
                    ):
                        matched.append(page_words[word_index])
                        target_index += 1
                        word_index += 1
                    if len(matched) > len(best):
                        best = matched
                    if target_start == 0 and target_index == len(targets):
                        return matched
        minimum = 1 if len(targets) == 1 else min(2, len(targets))
        return best if len(best) >= minimum else []

    def _union_box(self, boxes: list[VisionBoundingBox]) -> VisionBoundingBox | None:
        if not boxes:
            return None
        return VisionBoundingBox(
            x1=min(box.x1 for box in boxes),
            y1=min(box.y1 for box in boxes),
            x2=max(box.x2 for box in boxes),
            y2=max(box.y2 for box in boxes),
        )

    def _dedupe_fields(self, fields: list[ExtractedVisionField]) -> list[ExtractedVisionField]:
        seen = set()
        result = []
        for field in fields:
            key = (field.name, field.normalized_value)
            if key in seen:
                continue
            seen.add(key)
            result.append(field)
        return result

    def _format_valid(self, value: str) -> bool:
        value = value.strip()
        if not value:
            return False
        if re.search(r"\d", value):
            return len(value) >= 2
        return any(character.isalpha() for character in value)

    def _money(self, value: str | None) -> float | None:
        if not value:
            return None
        match = re.search(r"-?\d[\d,]*(?:\.\d+)?", value)
        if not match:
            return None
        try:
            return float(match.group(0).replace(",", ""))
        except ValueError:
            return None

    def _date(self, value: str | None) -> datetime | None:
        if not value:
            return None
        cleaned = value.strip().rstrip(".")
        formats = (
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%d-%m-%Y",
            "%m-%d-%Y",
            "%b %d, %Y",
            "%B %d, %Y",
        )
        for fmt in formats:
            try:
                return datetime.strptime(cleaned, fmt)
            except ValueError:
                continue
        return None

    def _clean_value(self, value: str) -> str:
        value = re.sub(r"\s+", " ", value).strip(" .,:;-")
        value = re.split(
            r"\b(?:invoice date|due date|payment terms|subtotal|tax|discount|total|amount due|signature)\b",
            value,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" .,:;-")
        return value

    def _normalize(self, value: str) -> str:
        return re.sub(r"\s+", " ", str(value).strip().lower()).strip(" .,:;-")

    def _normalize_token(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _mask_sensitive(self, field_name: str, value: str) -> str:
        if "bank" in field_name or "account" in field_name or "IFSC" in field_name:
            return self._mask_account(value)
        return value

    def _mask_account(self, value: str) -> str:
        raw = re.sub(r"\s+", "", str(value))
        if len(raw) <= 4:
            return "****"
        return f"{'*' * max(4, len(raw) - 4)}{raw[-4:]}"
