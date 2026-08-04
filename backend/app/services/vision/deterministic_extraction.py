from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation

from app.core.security import utc_now_iso
from app.models.vision import (
    ExtractedVisionField,
    OCRWord,
    ReviewerStatus,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionEvidence,
)
from app.services.vision.schema_registry import (
    DocumentSchema,
    SchemaFieldSpec,
    vision_schema_registry,
)


@dataclass(frozen=True)
class OCRLine:
    page: int
    line_id: str
    words: tuple[OCRWord, ...]
    text: str
    box: VisionBoundingBox


@dataclass(frozen=True)
class FieldCandidate:
    spec: SchemaFieldSpec
    value: str
    source_text: str
    page: int
    box: VisionBoundingBox | None
    words: tuple[OCRWord, ...]
    strategy: str
    extraction_confidence: float


class DeterministicSchemaExtractor:
    adapter_id = "deterministic-schema-extractor"
    model_version = "v2"

    def extract(self, document: VisionDocumentRecord) -> list[ExtractedVisionField]:
        schema = vision_schema_registry.get(document.document_type)
        if schema is None:
            return []
        lines = self._lines(document.ocr_words)
        candidates: list[FieldCandidate] = []
        for spec in schema.fields:
            candidates.extend(self._field_candidates(spec, schema, lines, document))
        retained: dict[str, FieldCandidate] = {}
        for candidate in candidates:
            current = retained.get(candidate.spec.name)
            if current is None or self._candidate_score(candidate) > self._candidate_score(current):
                retained[candidate.spec.name] = candidate
        return [
            self._to_field(document, candidate)
            for candidate in retained.values()
            if candidate.value.strip()
        ]

    def _field_candidates(
        self,
        spec: SchemaFieldSpec,
        schema: DocumentSchema,
        lines: list[OCRLine],
        document: VisionDocumentRecord,
    ) -> list[FieldCandidate]:
        candidates: list[FieldCandidate] = []
        for index, line in enumerate(lines):
            for alias in sorted(spec.aliases, key=len, reverse=True):
                association = self._label_value(line.text, alias)
                if association is None:
                    continue
                raw_value, strategy = association
                value = self._clean_candidate(raw_value, schema)
                if value:
                    value_words = self._words_for_value(value, line.words)
                    candidates.append(
                        FieldCandidate(
                            spec=spec,
                            value=value,
                            source_text=line.text,
                            page=line.page,
                            box=self._union(word.bounding_box for word in value_words)
                            or line.box,
                            words=value_words or line.words,
                            strategy=strategy,
                            extraction_confidence=(
                                0.92 if strategy == "same_line_label_value" else 0.84
                            ),
                        )
                    )
                    break
                below = self._nearest_below(line, lines[index + 1 :], schema)
                if below:
                    candidates.append(
                        FieldCandidate(
                            spec=spec,
                            value=self._clean_candidate(below.text, schema),
                            source_text=f"{line.text}\n{below.text}",
                            page=below.page,
                            box=below.box,
                            words=below.words,
                            strategy="nearest_lower_value",
                            extraction_confidence=0.76,
                        )
                    )
                    break

        candidates.extend(self._table_candidates(spec, document))
        candidates.extend(self._special_candidates(spec, document, lines))
        if spec.name == "form_title" and not candidates:
            title = next(
                (
                    region
                    for region in document.layout_regions
                    if region.region_type == "document title" and region.text.strip()
                ),
                None,
            )
            if title:
                candidates.append(
                    FieldCandidate(
                        spec=spec,
                        value=title.text.strip(),
                        source_text=title.text.strip(),
                        page=title.page_number,
                        box=title.bounding_box,
                        words=(),
                        strategy="measured_title_region",
                        extraction_confidence=0.82,
                    )
                )
        return [item for item in candidates if self._valid_candidate(item, schema)]

    def _label_value(self, text: str, alias: str) -> tuple[str, str] | None:
        alias_pattern = r"\s+".join(re.escape(token) for token in alias.split())
        prefix = rf"(?i)^\s*{alias_pattern}(?:\s+(?:no\.?|number))?"
        delimited = re.compile(
            prefix + r"\s*(?:[:#=\-]|\bis\b)\s*(?P<value>.*?)\s*$"
        )
        match = delimited.search(text)
        if match:
            return match.group("value"), "same_line_label_value"
        right_side = re.compile(prefix + r"\s+(?P<value>\S.+?)\s*$").search(text)
        if right_side:
            return right_side.group("value"), "nearest_right_value"
        label_only = re.compile(prefix + r"\s*[:#=\-]?\s*$").search(text)
        return ("", "nearest_lower_value") if label_only else None

    def _table_candidates(
        self,
        spec: SchemaFieldSpec,
        document: VisionDocumentRecord,
    ) -> list[FieldCandidate]:
        aliases = {self._normalize(alias) for alias in spec.aliases}
        candidates: list[FieldCandidate] = []
        for table in document.tables:
            cells = {(cell.row_index, cell.column_index): cell for cell in table.cells}
            for row_index, row in enumerate(table.rows):
                for column_index, label in enumerate(row[:-1]):
                    normalized_label = self._normalize(label)
                    if normalized_label not in aliases:
                        continue
                    value = str(row[column_index + 1]).strip()
                    value_cell = cells.get((row_index, column_index + 1))
                    box = value_cell.bounding_box if value_cell else table.bounding_box
                    words = self._words_in_box(
                        document.ocr_words,
                        page=table.page_number,
                        box=box,
                    )
                    candidates.append(
                        FieldCandidate(
                            spec=spec,
                            value=value,
                            source_text=f"{label} | {value}",
                            page=table.page_number,
                            box=box,
                            words=words,
                            strategy="table_cell_relationship",
                            extraction_confidence=0.88,
                        )
                    )
        return candidates

    def _special_candidates(
        self,
        spec: SchemaFieldSpec,
        document: VisionDocumentRecord,
        lines: list[OCRLine],
    ) -> list[FieldCandidate]:
        patterns: dict[str, tuple[str, float]] = {
            "invoice_number": (r"(?im)^\s*(?:invoice|inv)\s+(?!date\b)([A-Z0-9][A-Z0-9./-]{2,39})\s*$", 0.88),
            "purchase_order_number": (r"(?im)^\s*(?:purchase\s+order|po)\s+([A-Z0-9][A-Z0-9./-]{2,39})\s*$", 0.88),
            "email": (r"(?i)\b([^\s@]+@[^\s@]+\.[^\s@]+)\b", 0.86),
            "currency": (r"(?i)(?:\b(USD|INR|EUR|GBP|AUD|CAD|JPY)\b|([$€£₹]))", 0.82),
        }
        configured = patterns.get(spec.name)
        if configured is None:
            return []
        pattern, confidence = configured
        results: list[FieldCandidate] = []
        for match in re.finditer(pattern, document.parsed_text):
            groups = [value for value in match.groups() if value]
            if not groups:
                continue
            value = groups[0]
            line = next((item for item in lines if value.lower() in item.text.lower()), None)
            value_words = self._words_for_value(value, line.words) if line else ()
            results.append(
                FieldCandidate(
                    spec=spec,
                    value=value,
                    source_text=line.text if line else match.group(0),
                    page=line.page if line else 1,
                    box=self._union(word.bounding_box for word in value_words)
                    if value_words
                    else (line.box if line else None),
                    words=value_words or (line.words if line else ()),
                    strategy="schema_regex_with_source_match",
                    extraction_confidence=confidence,
                )
            )
        return results

    def _nearest_below(
        self,
        label: OCRLine,
        following: list[OCRLine],
        schema: DocumentSchema,
    ) -> OCRLine | None:
        label_height = max(1.0, label.box.y2 - label.box.y1)
        for candidate in following[:4]:
            if candidate.page != label.page:
                break
            if candidate.box.y1 - label.box.y2 > label_height * 4:
                break
            if candidate.box.x2 < label.box.x1 or candidate.box.x1 > label.box.x2 + 500:
                continue
            if self._starts_with_known_label(candidate.text, schema):
                return None
            return candidate
        return None

    def _starts_with_known_label(self, text: str, schema: DocumentSchema) -> bool:
        normalized = self._normalize(text)
        return any(
            normalized.startswith(self._normalize(alias))
            for field in schema.fields
            for alias in field.aliases
        )

    def _clean_candidate(self, value: str, schema: DocumentSchema) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip(" \t|:;,#=-")
        boundaries = sorted(
            {
                alias
                for field in schema.fields
                for alias in field.aliases
                if len(alias) >= 3
            },
            key=len,
            reverse=True,
        )
        if boundaries:
            boundary = re.compile(
                r"\s+(?:" + "|".join(re.escape(item) for item in boundaries) + r")\s*[:#=\-]",
                re.I,
            )
            cleaned = boundary.split(cleaned, maxsplit=1)[0]
        return cleaned.strip(" \t|:;,#=-")

    def _valid_candidate(self, candidate: FieldCandidate, schema: DocumentSchema) -> bool:
        value = candidate.value.strip()
        if not value or len(value) > 600:
            return False
        return vision_schema_registry.value_is_valid(schema.field(candidate.spec.name), value)

    def _candidate_score(self, candidate: FieldCandidate) -> float:
        ocr = self._ocr_confidence(candidate.words)
        validation = float(
            vision_schema_registry.value_is_valid(candidate.spec, candidate.value)
        )
        return self._weighted_confidence(
            ((ocr, 0.45), (candidate.extraction_confidence, 0.40), (validation, 0.15))
        )

    def _to_field(
        self,
        document: VisionDocumentRecord,
        candidate: FieldCandidate,
    ) -> ExtractedVisionField:
        ocr_confidence = self._ocr_confidence(candidate.words)
        validation_confidence = float(
            vision_schema_registry.value_is_valid(candidate.spec, candidate.value)
        )
        agreement_confidence = (
            1.0
            if candidate.strategy in {"same_line_label_value", "table_cell_relationship"}
            else 0.75
        )
        combined = self._weighted_confidence(
            (
                (ocr_confidence, 0.40),
                (candidate.extraction_confidence, 0.35),
                (validation_confidence, 0.15),
                (agreement_confidence, 0.10),
            )
        )
        identity = hashlib.sha256(
            (
                f"{document.document_id}|{candidate.spec.name}|{candidate.page}|"
                f"{self._normalize(candidate.value)}"
            ).encode()
        ).hexdigest()
        evidence = VisionEvidence(
            evidence_id=f"vis-ev-{identity[:12]}",
            document_id=document.document_id,
            page_number=candidate.page,
            source_text=candidate.source_text[:700],
            bounding_box=candidate.box,
            visual_region=f"page_{candidate.page}",
            model_confidence=combined,
            model_name=self.adapter_id,
            model_version=self.model_version,
            processing_timestamp=utc_now_iso(),
            reviewer_status=ReviewerStatus.unverified,
            metadata={
                "association_strategy": candidate.strategy,
                "word_ids": [word.word_id for word in candidate.words],
                "confidence_semantics": "observed_heuristic_components_not_calibrated_accuracy",
            },
        )
        return ExtractedVisionField(
            field_id=f"field-{identity[:10]}",
            name=candidate.spec.name,
            schema_field=candidate.spec.name,
            display_name=candidate.spec.display_name,
            value=candidate.value,
            normalized_value=self._normalise_value(candidate.spec, candidate.value),
            confidence=combined,
            ocr_confidence=ocr_confidence,
            extraction_confidence=candidate.extraction_confidence,
            validation_confidence=validation_confidence,
            agreement_confidence=agreement_confidence,
            combined_confidence=combined,
            evidence=evidence,
            model_name=self.adapter_id,
            model_version=self.model_version,
            target_model="LayoutLMv3",
            confidence_components={
                "ocr_confidence": ocr_confidence,
                "extraction_confidence": candidate.extraction_confidence,
                "layout_agreement": agreement_confidence,
                "field_format_validity": validation_confidence,
            },
            calibrated=False,
            extractor="deterministic-fallback",
            verification_status="unverified",
            required=candidate.spec.required,
        )

    def _normalise_value(self, spec: SchemaFieldSpec, value: str) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if spec.value_kind == "email":
            return cleaned.lower()
        if spec.value_kind == "phone":
            digits = re.sub(r"\D", "", cleaned)
            return f"+{digits}" if cleaned.startswith("+") else digits
        if spec.value_kind == "currency":
            return {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}.get(
                cleaned.upper(),
                cleaned.upper(),
            )
        if spec.value_kind == "money":
            match = re.search(r"-?\d[\d,]*(?:\.\d+)?", cleaned)
            if match:
                try:
                    return str(Decimal(match.group(0).replace(",", "")).quantize(Decimal("0.01")))
                except InvalidOperation:
                    pass
        if spec.value_kind == "date":
            for date_format in (
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%d-%m-%Y",
                "%m-%d-%Y",
                "%b %d, %Y",
                "%B %d, %Y",
            ):
                try:
                    return datetime.strptime(cleaned, date_format).date().isoformat()
                except ValueError:
                    continue
        return self._normalize(cleaned)

    def _lines(self, words: list[OCRWord]) -> list[OCRLine]:
        grouped: dict[tuple[int, str], list[OCRWord]] = defaultdict(list)
        for word in sorted(words, key=lambda item: item.reading_order):
            grouped[(word.page, word.line_id)].append(word)
        lines: list[OCRLine] = []
        for (page, line_id), items in grouped.items():
            ordered = tuple(sorted(items, key=lambda item: item.bounding_box.x1))
            box = self._union(item.bounding_box for item in ordered)
            if box is None:
                continue
            lines.append(
                OCRLine(
                    page=page,
                    line_id=line_id,
                    words=ordered,
                    text=" ".join(item.text for item in ordered).strip(),
                    box=box,
                )
            )
        return sorted(lines, key=lambda item: (item.page, item.box.y1, item.box.x1))

    def _words_for_value(
        self,
        value: str,
        words: tuple[OCRWord, ...],
    ) -> tuple[OCRWord, ...]:
        targets = [
            item
            for token in value.split()
            if (item := self._normalize_token(token))
        ]
        normalized = [self._normalize_token(word.text) for word in words]
        for start in range(len(words)):
            matched: list[OCRWord] = []
            for offset, target in enumerate(targets):
                position = start + offset
                if position >= len(words) or normalized[position] != target:
                    break
                matched.append(words[position])
            if len(matched) == len(targets):
                return tuple(matched)
        target_set = set(targets)
        return tuple(word for word, token in zip(words, normalized, strict=True) if token in target_set)

    def _words_in_box(
        self,
        words: list[OCRWord],
        *,
        page: int,
        box: VisionBoundingBox,
    ) -> tuple[OCRWord, ...]:
        return tuple(
            word
            for word in words
            if word.page == page
            and word.bounding_box.x1 < box.x2
            and word.bounding_box.x2 > box.x1
            and word.bounding_box.y1 < box.y2
            and word.bounding_box.y2 > box.y1
        )

    def _ocr_confidence(self, words: Iterable[OCRWord]) -> float | None:
        values = [word.confidence for word in words]
        return round(sum(values) / len(values), 4) if values else None

    def _weighted_confidence(
        self,
        components: Iterable[tuple[float | None, float]],
    ) -> float:
        observed = [(value, weight) for value, weight in components if value is not None]
        if not observed:
            return 0.0
        return round(
            sum(value * weight for value, weight in observed) / sum(weight for _, weight in observed),
            4,
        )

    def _union(
        self,
        boxes: Iterable[VisionBoundingBox],
    ) -> VisionBoundingBox | None:
        values = list(boxes)
        if not values:
            return None
        return VisionBoundingBox(
            x1=min(item.x1 for item in values),
            y1=min(item.y1 for item in values),
            x2=max(item.x2 for item in values),
            y2=max(item.y2 for item in values),
        )

    def _normalize(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower()).strip(" .,:;#=-")

    def _normalize_token(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())


deterministic_schema_extractor = DeterministicSchemaExtractor()
