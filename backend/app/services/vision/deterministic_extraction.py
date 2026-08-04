from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher

from app.core.security import utc_now_iso
from app.models.vision import (
    ExtractedVisionField,
    OCRLineRecord,
    OCRWord,
    ReviewerStatus,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionEvidence,
)
from app.services.vision.ocr_lines import ocr_line_reconstructor, union_boxes
from app.services.vision.schema_registry import (
    DocumentSchema,
    SchemaFieldSpec,
    vision_schema_registry,
)


@dataclass(frozen=True)
class OCRLine:
    record: OCRLineRecord
    words: tuple[OCRWord, ...]

    @property
    def page(self) -> int:
        return self.record.page

    @property
    def line_id(self) -> str:
        return self.record.line_id

    @property
    def text(self) -> str:
        return self.record.text

    @property
    def box(self) -> VisionBoundingBox:
        return self.record.bounding_box


@dataclass(frozen=True)
class LabelMatch:
    alias: str
    value: str
    label_words: tuple[OCRWord, ...]
    value_words: tuple[OCRWord, ...]
    similarity: float
    strategy: str


@dataclass(frozen=True)
class FieldCandidate:
    spec: SchemaFieldSpec
    value: str
    source_text: str
    page: int
    label_box: VisionBoundingBox | None
    value_box: VisionBoundingBox | None
    evidence_box: VisionBoundingBox
    words: tuple[OCRWord, ...]
    strategy: str
    extraction_confidence: float
    validation_status: str
    validation_reason: str


class DeterministicSchemaExtractor:
    adapter_id = "deterministic-schema-extractor"
    model_version = "v3"

    def extract(self, document: VisionDocumentRecord) -> list[ExtractedVisionField]:
        if not document.ocr_lines:
            document.ocr_lines = ocr_line_reconstructor.reconstruct(document.ocr_words)
        lines = self._lines(document)
        prefer_key_value = any(
            bool(line.record.metadata.get("key_value_candidate")) for line in lines
        )
        selection = vision_schema_registry.resolve(
            document.document_type,
            prefer_key_value=prefer_key_value,
        )
        document.metadata["schema_selection"] = selection.as_dict()

        candidates: list[FieldCandidate] = []
        labels_detected = 0
        for spec in selection.schema.fields:
            observed = self._field_candidates(spec, selection.schema, lines, document)
            labels_detected += sum(
                item.strategy
                not in {
                    "measured_title_region",
                    "top_ocr_line",
                    "source_matched_pattern",
                }
                for item in observed
            )
            candidates.extend(observed)

        retained: dict[str, FieldCandidate] = {}
        for candidate in candidates:
            current = retained.get(candidate.spec.name)
            if current is None or self._candidate_score(candidate) > self._candidate_score(current):
                retained[candidate.spec.name] = candidate

        fields = [
            self._to_field(document, candidate)
            for candidate in retained.values()
            if candidate.value.strip()
        ]
        document.metadata["extraction_diagnostics"] = {
            "ocr_words_inspected": len(document.ocr_words),
            "ocr_lines_inspected": len(lines),
            "labels_detected": labels_detected,
            "fields_extracted": len(fields),
            "extractor": f"{self.adapter_id}-{self.model_version}",
            **selection.as_dict(),
        }
        return fields

    def _field_candidates(
        self,
        spec: SchemaFieldSpec,
        schema: DocumentSchema,
        lines: list[OCRLine],
        document: VisionDocumentRecord,
    ) -> list[FieldCandidate]:
        candidates: list[FieldCandidate] = []
        for index, line in enumerate(lines):
            match = self._match_label(line, spec)
            if match is None:
                continue
            if not self._role_context_allows(spec.name, index, lines):
                continue
            if match.value:
                value = self._clean_candidate(match.value, schema)
                if value:
                    candidates.append(
                        self._candidate(
                            spec,
                            value,
                            line.text,
                            line.page,
                            match.label_words,
                            match.value_words,
                            match.strategy,
                            0.93 if match.similarity >= 0.98 else 0.84,
                        )
                    )
                    continue

            associated = self._nearest_spatial_value(line, lines[index + 1 :], schema)
            if associated:
                value_line, strategy = associated
                value = self._clean_candidate(value_line.text, schema)
                if value:
                    candidates.append(
                        self._candidate(
                            spec,
                            value,
                            f"{line.text}\n{value_line.text}",
                            line.page,
                            match.label_words or line.words,
                            value_line.words,
                            strategy,
                            0.78 if strategy == "nearest_lower_value" else 0.81,
                        )
                    )

            if spec.multiline:
                continued = self._multiline_value(line, lines[index + 1 :], schema)
                if continued:
                    continuation_lines, value = continued
                    value_words = tuple(
                        word for item in continuation_lines for word in item.words
                    )
                    candidates.append(
                        self._candidate(
                            spec,
                            value,
                            "\n".join([line.text, *(item.text for item in continuation_lines)]),
                            line.page,
                            match.label_words or line.words,
                            value_words,
                            "multi_line_continuation",
                            0.72,
                        )
                    )

        candidates.extend(self._table_candidates(spec, document))
        candidates.extend(self._special_candidates(spec, document, lines))
        return [item for item in candidates if self._valid_candidate(item)]

    def _match_label(self, line: OCRLine, spec: SchemaFieldSpec) -> LabelMatch | None:
        text = line.text.strip()
        if not text:
            return None
        left, separator, right = text.partition(":")
        label_text = left if separator else text
        normalized_label = self._normalize_label(label_text)
        aliases = sorted(spec.aliases, key=len, reverse=True)

        best: tuple[str, float, int] | None = None
        for alias in aliases:
            normalized_alias = self._normalize_label(alias)
            if not normalized_alias:
                continue
            prefix = normalized_label[: max(len(normalized_alias) + 3, len(normalized_alias))]
            exact = normalized_label == normalized_alias or normalized_label.startswith(
                f"{normalized_alias} "
            )
            similarity = 1.0 if exact else SequenceMatcher(None, prefix, normalized_alias).ratio()
            threshold = 0.82 if len(normalized_alias) <= 3 else 0.78
            if similarity < threshold:
                continue
            alias_tokens = max(1, len(alias.split()))
            if best is None or similarity > best[1]:
                best = (alias, similarity, alias_tokens)

        if best is None:
            return None
        alias, similarity, alias_tokens = best
        if spec.name == "recipient_name" and self._normalize_label(left) == "message":
            return None
        if (
            not separator
            and spec.name in {"organisation_name", "company_name", "recipient_company"}
            and re.search(r"(?i)\b(?:registration|application|request|intake)\s+form\b", text)
        ):
            return None
        words = line.words
        label_word_count = self._label_word_count(words, alias_tokens, separator)
        label_words = words[:label_word_count]
        value_words = words[label_word_count:]
        value = " ".join(word.text for word in value_words).strip(" :#=-")

        if separator and right.strip():
            value = right.strip()
            value_words = self._words_for_value(value, words) or value_words
        elif not separator:
            alias_pattern = r"\s+".join(re.escape(token) for token in alias.split())
            exact_prefix = re.compile(
                rf"(?i)^\s*{alias_pattern}(?:\s+(?:no\.?|number))?\s*(?:[#=\-]|\bis\b)?\s+(.+?)\s*$"
            )
            direct = exact_prefix.match(text)
            if direct:
                value = direct.group(1).strip()
                value_words = self._words_for_value(value, words) or value_words
            elif similarity < 0.98:
                # Fuzzy OCR labels are accepted only with a visible delimiter.
                return None

        strategy = "same_line_label_value" if separator else "nearest_right_value"
        if not value:
            strategy = "label_only"
        return LabelMatch(
            alias=alias,
            value=value,
            label_words=label_words,
            value_words=value_words,
            similarity=similarity,
            strategy=strategy,
        )

    def _label_word_count(
        self,
        words: tuple[OCRWord, ...],
        alias_tokens: int,
        has_separator: str,
    ) -> int:
        if has_separator:
            for index, word in enumerate(words):
                if ":" in word.text:
                    return index + 1
        return min(len(words), alias_tokens)

    def _nearest_spatial_value(
        self,
        label: OCRLine,
        following: list[OCRLine],
        schema: DocumentSchema,
    ) -> tuple[OCRLine, str] | None:
        height = max(1.0, label.box.y2 - label.box.y1)
        candidates: list[tuple[float, OCRLine, str]] = []
        for candidate in following[:8]:
            if candidate.page != label.page:
                break
            vertical_gap = candidate.box.y1 - label.box.y2
            if vertical_gap > height * 5:
                break
            if self._starts_with_known_label(candidate.text, schema):
                continue
            overlap = max(
                0.0,
                min(label.box.y2, candidate.box.y2)
                - max(label.box.y1, candidate.box.y1),
            )
            if candidate.box.x1 >= label.box.x2 and overlap > 0:
                distance = candidate.box.x1 - label.box.x2
                candidates.append((distance, candidate, "nearest_right_value"))
                continue
            horizontal_overlap = max(
                0.0,
                min(label.box.x2, candidate.box.x2)
                - max(label.box.x1, candidate.box.x1),
            )
            if vertical_gap >= -2 and (
                horizontal_overlap > 0 or abs(candidate.box.x1 - label.box.x1) <= 500
            ):
                distance = max(0.0, vertical_gap) + abs(candidate.box.x1 - label.box.x1) * 0.05
                candidates.append((distance, candidate, "nearest_lower_value"))
        if not candidates:
            return None
        _, candidate, strategy = min(candidates, key=lambda item: item[0])
        return candidate, strategy

    def _multiline_value(
        self,
        label: OCRLine,
        following: list[OCRLine],
        schema: DocumentSchema,
    ) -> tuple[list[OCRLine], str] | None:
        retained: list[OCRLine] = []
        for candidate in following[:4]:
            if candidate.page != label.page or self._starts_with_known_label(candidate.text, schema):
                break
            if candidate.box.y1 - (retained[-1].box.y2 if retained else label.box.y2) > 80:
                break
            retained.append(candidate)
        value = " ".join(item.text for item in retained).strip()
        return (retained, value) if value else None

    def _table_candidates(
        self,
        spec: SchemaFieldSpec,
        document: VisionDocumentRecord,
    ) -> list[FieldCandidate]:
        aliases = {self._normalize_label(alias) for alias in spec.aliases}
        candidates: list[FieldCandidate] = []
        for table in document.tables:
            cells = {(cell.row_index, cell.column_index): cell for cell in table.cells}
            for row_index, row in enumerate(table.rows):
                for column_index, label in enumerate(row[:-1]):
                    if self._normalize_label(label) not in aliases:
                        continue
                    value = str(row[column_index + 1]).strip()
                    label_cell = cells.get((row_index, column_index))
                    value_cell = cells.get((row_index, column_index + 1))
                    label_box = label_cell.bounding_box if label_cell else table.bounding_box
                    value_box = value_cell.bounding_box if value_cell else table.bounding_box
                    words = self._words_in_box(
                        document.ocr_words,
                        page=table.page_number,
                        box=value_box,
                    )
                    candidates.append(
                        self._candidate_from_boxes(
                            spec,
                            value,
                            f"{label} | {value}",
                            table.page_number,
                            label_box,
                            value_box,
                            words,
                            "table_cell_relationship",
                            0.90,
                        )
                    )
        return candidates

    def _special_candidates(
        self,
        spec: SchemaFieldSpec,
        document: VisionDocumentRecord,
        lines: list[OCRLine],
    ) -> list[FieldCandidate]:
        if spec.name in {"document_title", "form_title", "title"}:
            strong_title = next(
                (
                    line
                    for line in lines
                    if line.page == 1
                    and re.search(
                        r"(?i)\b(?:fax\s+transmission|facsimile|fax\s+cover\s+sheet|"
                        r"tax\s+invoice|invoice|purchase\s+order|bank\s+confirmation|"
                        r"agreement|contract|receipt|financial\s+report)\b",
                        line.text,
                    )
                ),
                None,
            )
            if strong_title:
                return [
                    self._candidate(
                        spec,
                        strong_title.text,
                        strong_title.text,
                        strong_title.page,
                        strong_title.words,
                        strong_title.words,
                        "strong_title_ocr_line",
                        0.92,
                    )
                ]
            title = next(
                (
                    region
                    for region in document.layout_regions
                    if "title" in region.region_type.lower() and region.text.strip()
                ),
                None,
            )
            if title and len(title.text.strip()) <= 180 and (
                title.bounding_box.y2 - title.bounding_box.y1
            ) <= 120:
                if spec.name == "title" and not self._reliable_generic_title(
                    title.text,
                    document,
                ):
                    return []
                return [
                    self._candidate_from_boxes(
                        spec,
                        title.text.strip(),
                        title.text.strip(),
                        title.page_number,
                        title.bounding_box,
                        title.bounding_box,
                        self._words_in_box(
                            document.ocr_words,
                            page=title.page_number,
                            box=title.bounding_box,
                        ),
                        "measured_title_region",
                        0.86,
                    )
                ]
            top = next(
                (
                    line
                    for line in lines
                    if line.page == 1
                    and 2 <= len(line.text.strip()) <= 160
                    and any(character.isalpha() for character in line.text)
                ),
                None,
            )
            if top:
                if spec.name == "title" and not self._reliable_generic_title(
                    top.text,
                    document,
                ):
                    return []
                return [
                    self._candidate(
                        spec,
                        top.text,
                        top.text,
                        top.page,
                        top.words,
                        top.words,
                        "top_ocr_line",
                        0.78,
                    )
                ]

        patterns: dict[str, str] = {
            "invoice_number": r"(?i)\b(?:invoice|inv)\s+(?!date\b)([A-Z0-9][A-Z0-9./-]{2,39})\b",
            "purchase_order_number": r"(?i)\b(?:purchase\s+order|po)\s+([A-Z0-9][A-Z0-9./-]{2,39})\b",
            "email": r"(?i)\b([^\s@]+@[^\s@]+\.[^\s@]+)\b",
            "currency": r"(?i)(?:\b(USD|INR|EUR|GBP|AUD|CAD|JPY)\b|([$€£₹]))",
            "visible_emails": r"(?i)\b([^\s@]+@[^\s@]+\.[^\s@]+)\b",
            "visible_phone_numbers": r"(?<![A-Za-z0-9])((?:\+?\d[\d()./ -]{6,}\d))(?![A-Za-z0-9])",
            "visible_fax_numbers": r"(?i)\bfax(?:\s+(?:no\.?|number))?\s*[:#-]?\s*((?:\+?\d[\d()./ -]{6,}\d))",
            "visible_dates": r"(?i)\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            "visible_currency_amounts": r"(?i)(\b(?:USD|INR|EUR|GBP|AUD|CAD|JPY)\s*\d[\d,]*(?:\.\d+)?|[$€£₹]\s*\d[\d,]*(?:\.\d+)?)",
            "visible_reference_numbers": r"(?i)\b(?:ref(?:erence)?|client|account)\s*(?:no\.?|number|#)?\s*[:#-]?\s*([A-Z0-9][A-Z0-9._/-]{2,39})",
            "page_count": r"(?i)\bpages?(?:\s*\(including\s+cover\s+sheet\))?\s*[:;#-]?\s*_*(\d{1,4})_*",
            "hard_copy_follow_up": r"(?i)\bhard\s*copy\s+to\s+follow\s*[:;#-]?\s*_*(yes|no|y|n)_*",
        }
        pattern = patterns.get(spec.name)
        if not pattern:
            return []
        results: list[FieldCandidate] = []
        for line in lines:
            match = re.search(pattern, line.text)
            if not match:
                continue
            value = next((group for group in match.groups() if group), match.group(0)).strip()
            value_words = self._words_for_value(value, line.words) or line.words
            results.append(
                self._candidate(
                    spec,
                    value,
                    line.text,
                    line.page,
                    (),
                    value_words,
                    "source_matched_pattern",
                    0.84,
                )
            )
        return results

    def _candidate(
        self,
        spec: SchemaFieldSpec,
        value: str,
        source_text: str,
        page: int,
        label_words: tuple[OCRWord, ...],
        value_words: tuple[OCRWord, ...],
        strategy: str,
        extraction_confidence: float,
    ) -> FieldCandidate:
        label_box = self._union(word.bounding_box for word in label_words)
        value_box = self._union(word.bounding_box for word in value_words)
        evidence_box = self._union(
            box for box in (label_box, value_box) if box is not None
        )
        if evidence_box is None:
            raise ValueError("Source-backed extraction requires a bounding box")
        return self._candidate_from_boxes(
            spec,
            value,
            source_text,
            page,
            label_box,
            value_box,
            value_words or label_words,
            strategy,
            extraction_confidence,
            evidence_box=evidence_box,
        )

    def _candidate_from_boxes(
        self,
        spec: SchemaFieldSpec,
        value: str,
        source_text: str,
        page: int,
        label_box: VisionBoundingBox | None,
        value_box: VisionBoundingBox | None,
        words: tuple[OCRWord, ...],
        strategy: str,
        extraction_confidence: float,
        *,
        evidence_box: VisionBoundingBox | None = None,
    ) -> FieldCandidate:
        resolved_box = evidence_box or self._union(
            box for box in (label_box, value_box) if box is not None
        )
        if resolved_box is None:
            raise ValueError("Source-backed extraction requires a bounding box")
        valid, reason = vision_schema_registry.validate_value(spec, value)
        return FieldCandidate(
            spec=spec,
            value=value,
            source_text=source_text,
            page=page,
            label_box=label_box,
            value_box=value_box,
            evidence_box=resolved_box,
            words=words,
            strategy=strategy,
            extraction_confidence=extraction_confidence,
            validation_status="valid" if valid else "invalid",
            validation_reason=reason,
        )

    def _valid_candidate(self, candidate: FieldCandidate) -> bool:
        return (
            bool(candidate.value.strip())
            and len(candidate.value) <= 800
            and candidate.validation_status == "valid"
        )

    def _candidate_score(self, candidate: FieldCandidate) -> float:
        ocr = self._ocr_confidence(candidate.words)
        validation = 1.0 if candidate.validation_status == "valid" else 0.0
        return self._weighted_confidence(
            ((ocr, 0.45), (candidate.extraction_confidence, 0.40), (validation, 0.15))
        )

    def _to_field(
        self,
        document: VisionDocumentRecord,
        candidate: FieldCandidate,
    ) -> ExtractedVisionField:
        normalized_value, normalization_reason = self._normalise_value(
            candidate.spec,
            candidate.value,
        )
        validation_reason = candidate.validation_reason
        validation_status = candidate.validation_status
        if normalization_reason == "ambiguous_date":
            validation_status = "ambiguous"
            validation_reason = normalization_reason
        ocr_confidence = self._ocr_confidence(candidate.words)
        validation_confidence = 1.0 if validation_status == "valid" else 0.65
        agreement_confidence = (
            1.0
            if candidate.strategy in {"same_line_label_value", "table_cell_relationship"}
            else 0.78
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
            bounding_box=candidate.evidence_box,
            visual_region=f"page_{candidate.page}",
            model_confidence=combined,
            model_name=self.adapter_id,
            model_version=self.model_version,
            processing_timestamp=utc_now_iso(),
            reviewer_status=ReviewerStatus.unverified,
            metadata={
                "association_strategy": candidate.strategy,
                "word_ids": [word.word_id for word in candidate.words],
                "label_box": candidate.label_box.model_dump(mode="json")
                if candidate.label_box
                else None,
                "value_box": candidate.value_box.model_dump(mode="json")
                if candidate.value_box
                else None,
                "evidence_box": candidate.evidence_box.model_dump(mode="json"),
                "validation_status": validation_status,
                "validation_reason": validation_reason,
                "normalization_reason": normalization_reason,
                "confidence_semantics": "observed_heuristic_components_not_calibrated_accuracy",
            },
        )
        return ExtractedVisionField(
            field_id=f"field-{identity[:10]}",
            name=candidate.spec.name,
            schema_field=candidate.spec.name,
            display_name=candidate.spec.display_name,
            value=candidate.value,
            raw_value=candidate.value,
            normalized_value=normalized_value,
            validation_status=validation_status,
            validation_reason=validation_reason,
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

    def _normalise_value(
        self,
        spec: SchemaFieldSpec,
        value: str,
    ) -> tuple[str, str]:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if spec.value_kind in {"email", "email_list"}:
            return cleaned.lower(), "lowercase_email"
        if spec.value_kind in {"phone", "phone_list"}:
            digits = re.sub(r"\D", "", cleaned)
            if cleaned.startswith("+"):
                return f"+{digits}", "explicit_international_prefix"
            if len(digits) == 11 and digits.startswith("1"):
                return f"+{digits}", "explicit_country_code"
            if len(digits) == 10:
                return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}", "country_not_assumed"
            return digits, "digits_only_country_unknown"
        if spec.value_kind == "currency":
            return (
                {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}.get(
                    cleaned.upper(), cleaned.upper()
                ),
                "currency_code_normalized",
            )
        if spec.value_kind in {"money", "money_list"}:
            match = re.search(r"-?\d[\d,]*(?:\.\d+)?", cleaned)
            if match:
                try:
                    amount = Decimal(match.group(0).replace(",", ""))
                    return format(amount.quantize(Decimal("0.01")), "f"), "decimal_amount"
                except InvalidOperation:
                    pass
            return cleaned, "amount_preserved_unparsed"
        if spec.value_kind in {"date", "date_list"}:
            date_cleaned = cleaned.rstrip(" .,:;")
            numeric = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", date_cleaned)
            if numeric and int(numeric.group(1)) <= 12 and int(numeric.group(2)) <= 12:
                return date_cleaned, "ambiguous_date"
            for date_format in (
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%d-%m-%Y",
                "%m-%d-%Y",
                "%b %d, %Y",
                "%B %d, %Y",
                "%b %d %Y",
                "%B %d %Y",
            ):
                try:
                    return datetime.strptime(date_cleaned, date_format).date().isoformat(), "iso_date"
                except ValueError:
                    continue
            return date_cleaned, "date_preserved_unparsed"
        if spec.value_kind == "account":
            digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            return f"sha256:{digest}", "sensitive_identifier_hashed"
        if spec.value_kind in {"reference", "reference_list"}:
            return cleaned, "punctuation_preserved"
        return self._normalize(cleaned), "whitespace_normalized"

    def _lines(self, document: VisionDocumentRecord) -> list[OCRLine]:
        by_id = {word.word_id: word for word in document.ocr_words}
        return [
            OCRLine(
                record=line,
                words=tuple(by_id[word_id] for word_id in line.word_ids if word_id in by_id),
            )
            for line in sorted(document.ocr_lines, key=lambda item: item.reading_order)
            if line.text.strip()
        ]

    def _starts_with_known_label(self, text: str, schema: DocumentSchema) -> bool:
        left = text.partition(":")[0]
        normalized = self._normalize_label(left)
        return any(
            normalized == self._normalize_label(alias)
            or normalized.startswith(f"{self._normalize_label(alias)} ")
            for field in schema.fields
            for alias in field.aliases
            if self._normalize_label(alias)
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

    def _words_for_value(
        self,
        value: str,
        words: tuple[OCRWord, ...],
    ) -> tuple[OCRWord, ...]:
        targets = [
            normalized
            for token in value.split()
            if (normalized := self._normalize_token(token))
        ]
        normalized_words = [self._normalize_token(word.text) for word in words]
        for start in range(len(words)):
            matched: list[OCRWord] = []
            for offset, target in enumerate(targets):
                position = start + offset
                if position >= len(words) or normalized_words[position] != target:
                    break
                matched.append(words[position])
            if targets and len(matched) == len(targets):
                return tuple(matched)
        target_set = set(targets)
        return tuple(
            word
            for word, token in zip(words, normalized_words, strict=True)
            if token in target_set
        )

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
            sum(value * weight for value, weight in observed)
            / sum(weight for _, weight in observed),
            4,
        )

    def _union(
        self,
        boxes: Iterable[VisionBoundingBox],
    ) -> VisionBoundingBox | None:
        values = list(boxes)
        return union_boxes(values) if values else None

    def _normalize(self, value: str) -> str:
        return re.sub(r"\s+", " ", value.strip().lower()).strip(" .,:;#=-")

    def _normalize_label(self, value: str) -> str:
        expanded = value.lower().replace("#", " number ")
        return re.sub(r"[^a-z0-9]+", " ", expanded).strip()

    def _normalize_token(self, value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    def _reliable_generic_title(
        self,
        value: str,
        document: VisionDocumentRecord,
    ) -> bool:
        cleaned = " ".join(value.split()).strip()
        if document.document_type.value != "unknown":
            return True
        return (
            len(cleaned) <= 100
            and (
                cleaned.isupper()
                or bool(
                    re.search(
                        r"(?i)(?:\b(?:report|statement|letter|notice|confirmation|transmission)\b\s*$|"
                        r"\b(?:registration|application|request|intake|business)\s+form\b)",
                        cleaned,
                    )
                )
            )
        )

    def _role_context_allows(
        self,
        field_name: str,
        index: int,
        lines: list[OCRLine],
    ) -> bool:
        if not field_name.startswith(("recipient_", "sender_")):
            return True
        sender_boundary = next(
            (
                position
                for position, line in enumerate(lines)
                if re.match(r"(?i)^\s*(?:from|sender|sent\s+by)\s*[:#=-]", line.text)
            ),
            None,
        )
        if sender_boundary is None:
            return not field_name.startswith("sender_") or field_name == "sender_name"
        if field_name.startswith("recipient_"):
            return index < sender_boundary
        return index >= sender_boundary


deterministic_schema_extractor = DeterministicSchemaExtractor()
