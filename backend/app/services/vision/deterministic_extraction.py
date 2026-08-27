from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.security import utc_now_iso
from app.models.vision import (
    ExtractedVisionField,
    OCRLineRecord,
    OCRWord,
    ReviewerStatus,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionEvidence,
    VisionSchemaProfile,
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
    label_similarity: float = 1.0
    spatial_score: float = 1.0
    schema_compatibility: float = 1.0
    raw_value: str | None = None
    raw_ocr: str | None = None
    corrected_ocr: str | None = None
    correction_method: str | None = None
    correction_confidence: float | None = None
    ocr_candidates: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class RegionOCRCandidate:
    text: str
    confidence: float | None
    variant: str
    psm: int
    bounding_box: VisionBoundingBox
    words: tuple[dict[str, Any], ...]

    def provenance(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "variant": self.variant,
            "psm": self.psm,
            "bounding_box": self.bounding_box.model_dump(mode="json"),
        }


class FieldCandidateResolver:
    """Selects one schema field per evidence span and records every rejection."""

    def resolve(
        self,
        candidates: Iterable[FieldCandidate],
        *,
        score: Any,
        conflicts: Any,
    ) -> tuple[list[FieldCandidate], list[dict[str, Any]]]:
        ordered = sorted(
            candidates,
            key=lambda item: (
                score(item),
                int(item.spec.required),
                item.label_similarity,
                item.spec.name,
            ),
            reverse=True,
        )
        retained: list[FieldCandidate] = []
        retained_names: set[str] = set()
        rejected: list[dict[str, Any]] = []
        # Resolve globally in score order. A field's highest-scoring candidate can
        # conflict with a stronger assignment for another field; in that case the
        # next non-conflicting candidate for the first field must remain eligible.
        # Pre-collapsing to one candidate per name would incorrectly drop that
        # fallback and is especially harmful for similar labels such as period
        # start/end or sender/recipient.
        for candidate in ordered:
            candidate_score = score(candidate)
            if candidate.spec.name in retained_names:
                rejected.append(
                    self._rejection(
                        candidate,
                        "lower_score_same_field",
                        candidate.spec.name,
                        candidate_score,
                    )
                )
                continue
            conflicting = next(
                (item for item in retained if conflicts(candidate, item)),
                None,
            )
            if conflicting is not None:
                rejected.append(
                    self._rejection(
                        candidate,
                        "duplicate_evidence_conflict",
                        conflicting.spec.name,
                        candidate_score,
                    )
                )
                continue
            retained.append(candidate)
            retained_names.add(candidate.spec.name)
        return retained, rejected

    def _rejection(
        self,
        candidate: FieldCandidate,
        reason: str,
        conflicts_with: str,
        score: float,
    ) -> dict[str, Any]:
        return {
            "field": candidate.spec.name,
            "value": candidate.value,
            "page": candidate.page,
            "strategy": candidate.strategy,
            "reason": reason,
            "conflicts_with": conflicts_with,
            "score": round(score, 4),
            "evidence_box": candidate.evidence_box.model_dump(mode="json"),
        }


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
        selection_payload = selection.as_dict()
        document.schema_profile = VisionSchemaProfile(
            mode=str(selection_payload["mode"]),
            schema_status=str(selection_payload["schema_status"]),
            schema_id=selection.selected_schema,
            schema_version=selection.schema.version,
            selection_source=selection.source,
        )

        candidates: list[FieldCandidate] = []
        roi_cache: dict[tuple[int, float, float, float, float], tuple[RegionOCRCandidate, ...]] = {}
        labels_detected = 0
        for spec in selection.schema.fields:
            observed = self._field_candidates(
                spec,
                selection.schema,
                lines,
                document,
                roi_cache,
            )
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

        retained, rejected = FieldCandidateResolver().resolve(
            candidates,
            score=self._candidate_score,
            conflicts=self._evidence_conflicts,
        )

        fields = [
            self._to_field(document, candidate) for candidate in retained if candidate.value.strip()
        ]
        duplicate_rejections = [
            item for item in rejected if item["reason"] == "duplicate_evidence_conflict"
        ]
        document.metadata["extraction_diagnostics"] = {
            "ocr_words_inspected": len(document.ocr_words),
            "ocr_lines_inspected": len(lines),
            "labels_detected": labels_detected,
            "fields_extracted": len(fields),
            "candidates_generated": len(candidates),
            "candidates_rejected": len(rejected),
            "duplicates_rejected": len(duplicate_rejections),
            "disputed_candidates": sum(item.validation_status != "valid" for item in retained),
            "rejected_candidates": rejected,
            "roi_regions_reprocessed": len(roi_cache),
            "roi_ocr_candidates": sum(len(items) for items in roi_cache.values()),
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
        roi_cache: dict[
            tuple[int, float, float, float, float],
            tuple[RegionOCRCandidate, ...],
        ],
    ) -> list[FieldCandidate]:
        candidates: list[FieldCandidate] = []
        for index, line in enumerate(lines):
            match = self._match_label(line, spec)
            if match is None:
                continue
            if not self._role_context_allows(spec.name, index, lines):
                continue
            direct_value_reliable = False
            if match.value:
                raw_value = self._clean_candidate(match.value, schema)
                value = self._coerce_candidate_value(spec, raw_value)
                if value:
                    resolved_value_words = (
                        self._words_for_value(value, line.words) or match.value_words
                    )
                    direct_valid, direct_reason = vision_schema_registry.validate_value(
                        spec,
                        value,
                    )
                    direct_value_reliable = direct_valid and direct_reason not in {
                        "ambiguous_numeric_date",
                        "date_uncertain_needs_review",
                    }
                    candidates.append(
                        self._candidate(
                            spec,
                            value,
                            line.text,
                            line.page,
                            match.label_words,
                            resolved_value_words,
                            match.strategy,
                            0.93 if match.similarity >= 0.98 else 0.84,
                            label_similarity=match.similarity,
                            raw_value=raw_value,
                            raw_ocr=line.text,
                        )
                    )

            roi_candidates = (
                ()
                if direct_value_reliable
                else self._region_ocr_candidates(
                    line,
                    document,
                    roi_cache,
                )
            )
            roi_provenance = tuple(item.provenance() for item in roi_candidates)
            parsed_roi: list[tuple[RegionOCRCandidate, LabelMatch, str]] = []
            for roi_candidate in roi_candidates:
                roi_line = OCRLine(
                    record=line.record.model_copy(update={"text": roi_candidate.text}),
                    words=(),
                )
                roi_match = self._match_label(roi_line, spec)
                if roi_match is None or not roi_match.value:
                    continue
                roi_raw_value = self._clean_candidate(roi_match.value, schema)
                roi_value = self._coerce_candidate_value(spec, roi_raw_value)
                if not roi_value:
                    continue
                parsed_roi.append((roi_candidate, roi_match, roi_value))
            roi_value_counts = Counter(self._normalize(item[2]) for item in parsed_roi)
            strongest_consensus = max(roi_value_counts.values(), default=1)
            for roi_candidate, roi_match, roi_value in parsed_roi:
                consensus = roi_value_counts[self._normalize(roi_value)] / strongest_consensus
                value_box = self._region_value_box(roi_candidate, roi_value)
                evidence_box = self._union(box for box in (line.box, value_box) if box is not None)
                if evidence_box is None:
                    continue
                value_words = (
                    self._words_in_box(
                        document.ocr_words,
                        page=line.page,
                        box=value_box,
                    )
                    if value_box
                    else match.value_words
                )
                correction_confidence = self._weighted_confidence(
                    (
                        (roi_candidate.confidence, 0.60),
                        (roi_match.similarity, 0.20),
                        (consensus, 0.20),
                    )
                )
                candidates.append(
                    self._candidate_from_boxes(
                        spec,
                        roi_value,
                        roi_candidate.text,
                        line.page,
                        line.box,
                        value_box or line.box,
                        value_words or line.words,
                        "roi_reocr_same_line",
                        self._weighted_confidence(
                            (
                                (roi_candidate.confidence, 0.55),
                                (roi_match.similarity, 0.25),
                                (consensus, 0.20),
                            )
                        ),
                        evidence_box=evidence_box,
                        label_similarity=roi_match.similarity,
                        spatial_score=1.0,
                        schema_compatibility=consensus,
                        raw_value=(
                            self._clean_candidate(match.value, schema) if match.value else None
                        ),
                        raw_ocr=line.text,
                        corrected_ocr=f"{spec.display_name}: {roi_value}",
                        correction_method=(
                            "roi_reocr_and_date_validation"
                            if spec.value_kind in {"date", "date_list"}
                            else "roi_reocr_and_type_validation"
                        ),
                        correction_confidence=correction_confidence,
                        ocr_candidates=roi_provenance,
                    )
                )

            associated = (
                self._nearest_spatial_value(line, lines, schema) if not match.value else None
            )
            if associated is not None:
                value_line, strategy = associated
                raw_value = self._clean_candidate(value_line.text, schema)
                value = self._coerce_candidate_value(spec, raw_value)
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
                            label_similarity=match.similarity,
                            spatial_score=(0.92 if strategy == "nearest_right_value" else 0.78),
                            raw_value=raw_value,
                            raw_ocr=f"{line.text}\n{value_line.text}",
                        )
                    )

            if spec.multiline and not match.value:
                continued = self._multiline_value(line, lines[index + 1 :], schema)
                if continued:
                    continuation_lines, value = continued
                    value_words = tuple(word for item in continuation_lines for word in item.words)
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
                            label_similarity=match.similarity,
                            spatial_score=0.70,
                            raw_value=value,
                            raw_ocr="\n".join(
                                [line.text, *(item.text for item in continuation_lines)]
                            ),
                        )
                    )

        candidates.extend(self._table_candidates(spec, document))
        candidates.extend(self._special_candidates(spec, document, lines))
        return [
            item
            for item in candidates
            if self._valid_candidate(item)
            and (not spec.extraction_strategies or item.strategy in spec.extraction_strategies)
        ]

    def _match_label(self, line: OCRLine, spec: SchemaFieldSpec) -> LabelMatch | None:
        text = line.text.strip()
        if not text:
            return None
        normalized_text = self._normalize_label(text)
        aliases = sorted(spec.aliases, key=len, reverse=True)
        separators = list(re.finditer(r"[:;#=,]", text))

        best: tuple[str, float, int, re.Match[str] | None, bool] | None = None
        for alias in aliases:
            normalized_alias = self._normalize_label(alias)
            if not normalized_alias:
                continue
            observed_delimited_label = (
                self._normalize_label(text[: separators[0].start()]) if separators else ""
            )
            exact = (
                normalized_text == normalized_alias
                or observed_delimited_label == normalized_alias
                or (not separators and normalized_text.startswith(f"{normalized_alias} "))
            )
            selected_separator: re.Match[str] | None = None
            similarity = 1.0 if exact else 0.0
            fuzzy = False
            if not exact:
                # OCR frequently damages one character in an otherwise clear label
                # and may omit the delimiter. Compare the leading label-length token
                # window before resorting to delimiter-based matching.
                observed_tokens = normalized_text.split()[: len(normalized_alias.split())]
                observed_prefix = " ".join(observed_tokens)
                prefix_similarity = SequenceMatcher(
                    None,
                    observed_prefix,
                    normalized_alias,
                ).ratio()
                prefix_threshold = 0.78 if len(normalized_alias) >= 8 else 0.84
                if prefix_similarity >= prefix_threshold:
                    similarity = prefix_similarity
                    fuzzy = True
                for separator in separators:
                    observed = self._normalize_label(text[: separator.start()])
                    if not observed:
                        continue
                    observed_prefix = observed[: max(len(normalized_alias) + 3, 1)]
                    observed_similarity = SequenceMatcher(
                        None,
                        observed_prefix,
                        normalized_alias,
                    ).ratio()
                    threshold = 0.72 if len(normalized_alias) <= 4 else 0.78
                    if observed_similarity >= threshold and observed_similarity > similarity:
                        similarity = observed_similarity
                        selected_separator = separator
                        fuzzy = True
            if not exact and not fuzzy:
                continue
            alias_tokens = max(1, len(alias.split()))
            if best is None or similarity > best[1]:
                best = (alias, similarity, alias_tokens, selected_separator, fuzzy)

        if best is None:
            return None
        alias, similarity, alias_tokens, fuzzy_separator, fuzzy = best
        words = line.words
        label_word_count = self._label_word_count(
            words,
            alias_tokens,
            fuzzy_separator.group(0) if fuzzy_separator else "",
        )
        label_words = words[:label_word_count]
        value_words = words[label_word_count:]
        value = ""
        if fuzzy and fuzzy_separator is not None:
            value = text[fuzzy_separator.end() :].strip()
        elif fuzzy:
            value = " ".join(text.split()[alias_tokens:]).strip()
        else:
            alias_tokens_normalized = self._normalize_label(alias).split()
            alias_pattern = r"[\W_]+".join(re.escape(token) for token in alias_tokens_normalized)
            direct = re.match(rf"(?i)^\s*[\W_]*{alias_pattern}\b", text)
            if direct:
                value = re.sub(
                    r"^[\s.:;,#=_\-]+",
                    "",
                    text[direct.end() :],
                ).strip()
            elif similarity < 0.98:
                return None
        if value:
            value_words = self._words_for_value(value, words) or value_words
        if spec.name in {"organisation_name", "company_name", "recipient_company"} and re.search(
            r"(?i)\b(?:registration|application|request|intake)\s+form\b",
            value or text,
        ):
            return None

        strategy = "same_line_label_value" if value else "label_only"
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
        lines: list[OCRLine],
        schema: DocumentSchema,
    ) -> tuple[OCRLine, str] | None:
        height = max(1.0, label.box.y2 - label.box.y1)
        candidates: list[tuple[int, float, OCRLine, str]] = []
        for candidate in lines:
            if candidate.line_id == label.line_id or candidate.page != label.page:
                continue
            vertical_gap = candidate.box.y1 - label.box.y2
            if vertical_gap > height * 5 or candidate.box.y2 < label.box.y1 - height:
                continue
            if self._starts_with_known_label(candidate.text, schema):
                continue
            overlap = max(
                0.0,
                min(label.box.y2, candidate.box.y2) - max(label.box.y1, candidate.box.y1),
            )
            if candidate.box.x1 >= label.box.x2 and overlap > 0:
                distance = candidate.box.x1 - label.box.x2
                candidates.append((1, distance, candidate, "nearest_right_value"))
                continue
            horizontal_overlap = max(
                0.0,
                min(label.box.x2, candidate.box.x2) - max(label.box.x1, candidate.box.x1),
            )
            if vertical_gap >= -2 and (
                horizontal_overlap > 0 or abs(candidate.box.x1 - label.box.x1) <= 500
            ):
                distance = max(0.0, vertical_gap) + abs(candidate.box.x1 - label.box.x1) * 0.05
                candidates.append((4, distance, candidate, "nearest_lower_value"))
        if not candidates:
            return None
        _, _, candidate, strategy = min(candidates, key=lambda item: (item[0], item[1]))
        return candidate, strategy

    def _multiline_value(
        self,
        label: OCRLine,
        following: list[OCRLine],
        schema: DocumentSchema,
    ) -> tuple[list[OCRLine], str] | None:
        retained: list[OCRLine] = []
        for candidate in following[:4]:
            if candidate.page != label.page or self._starts_with_known_label(
                candidate.text, schema
            ):
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
            if (
                title
                and len(title.text.strip()) <= 180
                and (title.bounding_box.y2 - title.bounding_box.y1) <= 120
            ):
                if not self._reliable_title_candidate(
                    title.text,
                    document,
                    confidence=title.confidence,
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
                if not self._reliable_title_candidate(
                    top.text,
                    document,
                    confidence=top.record.mean_confidence,
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
            "request_date": r"(?i)\b(?:request(?:ed)?|requisition)\b[^\n]{0,100}?\bon\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
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
        *,
        label_similarity: float = 1.0,
        spatial_score: float = 1.0,
        schema_compatibility: float = 1.0,
        raw_value: str | None = None,
        raw_ocr: str | None = None,
        corrected_ocr: str | None = None,
        correction_method: str | None = None,
        correction_confidence: float | None = None,
        ocr_candidates: tuple[dict[str, Any], ...] = (),
    ) -> FieldCandidate:
        label_box = self._union(word.bounding_box for word in label_words)
        value_box = self._union(word.bounding_box for word in value_words)
        evidence_box = self._union(box for box in (label_box, value_box) if box is not None)
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
            label_similarity=label_similarity,
            spatial_score=spatial_score,
            schema_compatibility=schema_compatibility,
            raw_value=raw_value,
            raw_ocr=raw_ocr,
            corrected_ocr=corrected_ocr,
            correction_method=correction_method,
            correction_confidence=correction_confidence,
            ocr_candidates=ocr_candidates,
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
        label_similarity: float = 1.0,
        spatial_score: float = 1.0,
        schema_compatibility: float = 1.0,
        raw_value: str | None = None,
        raw_ocr: str | None = None,
        corrected_ocr: str | None = None,
        correction_method: str | None = None,
        correction_confidence: float | None = None,
        ocr_candidates: tuple[dict[str, Any], ...] = (),
    ) -> FieldCandidate:
        resolved_box = evidence_box or self._union(
            box for box in (label_box, value_box) if box is not None
        )
        if resolved_box is None:
            raise ValueError("Source-backed extraction requires a bounding box")
        valid, reason = vision_schema_registry.validate_value(spec, value)
        ambiguous = reason in {
            "ambiguous_numeric_date",
            "date_uncertain_needs_review",
        }
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
            validation_status=(
                "ambiguous" if valid and ambiguous else "valid" if valid else "invalid"
            ),
            validation_reason=reason,
            label_similarity=label_similarity,
            spatial_score=spatial_score,
            schema_compatibility=schema_compatibility,
            raw_value=raw_value,
            raw_ocr=raw_ocr,
            corrected_ocr=corrected_ocr,
            correction_method=correction_method,
            correction_confidence=correction_confidence,
            ocr_candidates=ocr_candidates,
        )

    def _valid_candidate(self, candidate: FieldCandidate) -> bool:
        return (
            bool(candidate.value.strip())
            and len(candidate.value) <= 800
            and candidate.validation_status in {"valid", "ambiguous"}
        )

    def _candidate_score(self, candidate: FieldCandidate) -> float:
        ocr = self._weighted_confidence(
            (
                (self._ocr_confidence(candidate.words), 0.35),
                (candidate.correction_confidence, 0.65),
            )
        )
        validation = 1.0 if candidate.validation_status == "valid" else 0.55
        return self._weighted_confidence(
            (
                (candidate.label_similarity, 0.24),
                (candidate.spatial_score, 0.16),
                (validation, 0.18),
                (ocr, 0.18),
                (candidate.schema_compatibility, 0.12),
                (candidate.extraction_confidence, 0.12),
            )
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
        if normalization_reason in {"ambiguous_date", "date_unresolved"}:
            validation_status = "ambiguous"
            validation_reason = normalization_reason
        ocr_confidence = self._weighted_confidence(
            (
                (self._ocr_confidence(candidate.words), 0.45),
                (candidate.correction_confidence, 0.55),
            )
        )
        validation_confidence = 1.0 if validation_status == "valid" else 0.65
        agreement_confidence = (
            1.0
            if candidate.strategy
            in {
                "same_line_label_value",
                "table_cell_relationship",
                "roi_reocr_same_line",
            }
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
                "raw_ocr": candidate.raw_ocr or candidate.source_text,
                "corrected_ocr": candidate.corrected_ocr,
                "correction_method": candidate.correction_method,
                "correction_confidence": candidate.correction_confidence,
                "candidates": list(candidate.ocr_candidates),
                "candidate_score_components": {
                    "label_match": candidate.label_similarity,
                    "spatial_relationship": candidate.spatial_score,
                    "type_validity": (1.0 if validation_status == "valid" else 0.55),
                    "ocr_confidence": ocr_confidence,
                    "schema_compatibility": candidate.schema_compatibility,
                    "duplicate_penalty": 0.0,
                    "conflict_penalty": 0.0,
                },
                "confidence_semantics": "observed_heuristic_components_not_calibrated_accuracy",
            },
        )
        return ExtractedVisionField(
            field_id=f"field-{identity[:10]}",
            name=candidate.spec.name,
            schema_field=candidate.spec.name,
            display_name=candidate.spec.display_name,
            value=candidate.value,
            raw_value=candidate.raw_value or candidate.value,
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
            target_model=f"{self.adapter_id}-{self.model_version}",
            confidence_components={
                "ocr_confidence": ocr_confidence,
                "extraction_confidence": candidate.extraction_confidence,
                "layout_agreement": agreement_confidence,
                "field_format_validity": validation_confidence,
                "label_match": candidate.label_similarity,
                "spatial_relationship": candidate.spatial_score,
                "schema_compatibility": candidate.schema_compatibility,
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
    ) -> tuple[str | None, str]:
        cleaned = re.sub(r"\s+", " ", value).strip()
        configured_normalizer = spec.normalizers[0] if spec.normalizers else None
        if configured_normalizer == "preserve":
            reason = {
                "name": "person_name_preserved",
                "title": "title_text_preserved",
                "quantity": "quantity_and_unit_preserved",
                "reference": "punctuation_preserved",
                "reference_list": "punctuation_preserved",
            }.get(spec.value_kind, "configured_value_preserved")
            return cleaned, reason
        if configured_normalizer == "casefold_whitespace":
            return self._normalize(cleaned), "whitespace_normalized"
        if configured_normalizer == "lowercase":
            return cleaned.lower(), "lowercase_value"
        if configured_normalizer == "integer":
            return cleaned, "integer_preserved"
        if configured_normalizer == "boolean":
            return cleaned.lower(), "boolean_normalized"
        kind = {
            "phone": "phone",
            "currency_code": "currency",
            "decimal_amount": "money",
            "iso_date": "date",
            "sensitive_hash": "account",
        }.get(configured_normalizer or "", spec.value_kind)
        if kind == "name":
            return cleaned, "person_name_preserved"
        if kind == "title":
            return cleaned, "title_text_preserved"
        if kind == "quantity":
            return cleaned, "quantity_and_unit_preserved"
        if kind in {"email", "email_list"}:
            return cleaned.lower(), "lowercase_email"
        if kind in {"phone", "phone_list"}:
            digits = re.sub(r"\D", "", cleaned)
            if cleaned.startswith("+"):
                return f"+{digits}", "explicit_international_prefix"
            if len(digits) == 11 and digits.startswith("1"):
                return f"+{digits}", "explicit_country_code"
            if len(digits) == 10:
                return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}", "country_not_assumed"
            return digits, "digits_only_country_unknown"
        if kind == "currency":
            return (
                {"$": "USD", "€": "EUR", "£": "GBP", "₹": "INR"}.get(
                    cleaned.upper(), cleaned.upper()
                ),
                "currency_code_normalized",
            )
        if kind in {"money", "money_list"}:
            match = re.search(r"-?\d[\d,]*(?:\.\d+)?", cleaned)
            if match:
                try:
                    amount = Decimal(match.group(0).replace(",", ""))
                    return format(amount.quantize(Decimal("0.01")), "f"), "decimal_amount"
                except InvalidOperation:
                    pass
            return cleaned, "amount_preserved_unparsed"
        if kind in {"date", "date_list"}:
            date_cleaned = cleaned.rstrip(" .,:;")
            numeric = re.fullmatch(r"(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})", date_cleaned)
            if numeric and int(numeric.group(1)) <= 12 and int(numeric.group(2)) <= 12:
                return None, "ambiguous_date"
            for date_format in (
                "%Y-%m-%d",
                "%d/%m/%Y",
                "%m/%d/%Y",
                "%d-%m-%Y",
                "%m-%d-%Y",
                "%d/%m/%y",
                "%m/%d/%y",
                "%d-%m-%y",
                "%m-%d-%y",
                "%b %d, %Y",
                "%B %d, %Y",
                "%b %d %Y",
                "%B %d %Y",
            ):
                try:
                    return datetime.strptime(
                        date_cleaned, date_format
                    ).date().isoformat(), "iso_date"
                except ValueError:
                    continue
            return None, "date_unresolved"
        if kind == "account":
            digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
            return f"sha256:{digest}", "sensitive_identifier_hashed"
        if kind in {"reference", "reference_list"}:
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
            {alias for field in schema.fields for alias in field.aliases if len(alias) >= 3},
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

    def _coerce_candidate_value(self, spec: SchemaFieldSpec, value: str) -> str:
        cleaned = re.sub(r"_+", " ", value)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" \t|:;,#=-")
        if not cleaned:
            return ""
        if spec.value_kind in {"date", "date_list"}:
            match = re.search(
                r"(?i)\b(?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|"
                r"\b\d{4}-\d{2}-\d{2}\b|"
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                cleaned,
            )
            return match.group(0).strip() if match else cleaned
        if spec.name == "sample_number":
            match = re.search(r"\b[A-Z0-9][A-Z0-9./-]{1,39}\b", cleaned, re.I)
            return match.group(0) if match else ""
        if spec.name == "product_type":
            cleaned = re.sub(
                r"(?i)\b(\d{2,3})\s*m\s+(?=[A-Za-z])",
                r"\1 mm ",
                cleaned,
            )
            match = re.search(
                r"(?i)\b\d{2,3}\s*mm\s+[A-Za-z][A-Za-z0-9 /-]{1,48}",
                cleaned,
            )
            if not match:
                return ""
            return match.group(0).strip(" .,:;|_-")
        if spec.value_kind == "quantity":
            cleaned = re.sub(r"(?i)\b[iI]bs?\.?", "lbs.", cleaned)
            match = re.search(
                r"(?i)\b\d+(?:[.,]\d+)?\s*(?:lbs?\.?|pounds?|kg|kgs|g|grams?|"
                r"mg|oz|ounces?|litres?|liters?|ml|units?|pcs?\.?)(?=\s|$)",
                cleaned,
            )
            return match.group(0).strip(' "') if match else ""
        if spec.name == "requested_by":
            cleaned = re.split(
                r"(?i)\s+on\s*[,;:]?\s+",
                cleaned,
                maxsplit=1,
            )[0]
        if spec.name == "specifications_written_by":
            cleaned = re.split(
                r"(?i)\s+(?:additional\s+spray|blend|casing|recasing|final\s+flavor)\b",
                cleaned,
                maxsplit=1,
            )[0]
            initials_name = re.search(
                r"\b(?:[A-Z][.,]?\s*){1,4}[A-Z][A-Za-z'\-]{2,}\b",
                cleaned,
            )
            if initials_name:
                cleaned = initials_name.group(0)
        return cleaned.strip(" \t|:;,#=_-")

    def _region_ocr_candidates(
        self,
        line: OCRLine,
        document: VisionDocumentRecord,
        cache: dict[
            tuple[int, float, float, float, float],
            tuple[RegionOCRCandidate, ...],
        ],
    ) -> tuple[RegionOCRCandidate, ...]:
        artifact = next(
            (item for item in document.page_artifacts if item.page_number == line.page),
            None,
        )
        if artifact is None or artifact.native_text_available:
            return ()
        cache_key = (
            line.page,
            round(line.box.x1, 1),
            round(line.box.y1, 1),
            round(line.box.x2, 1),
            round(line.box.y2, 1),
        )
        if cache_key in cache:
            return cache[cache_key]
        source_path = self._artifact_path(artifact.original_image_path)
        if source_path is None:
            cache[cache_key] = ()
            return ()
        try:
            import cv2
            import numpy as np
            import pytesseract
            from PIL import Image, ImageEnhance, ImageOps
            from pytesseract import Output
        except ImportError:
            cache[cache_key] = ()
            return ()

        try:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_command
            page_image = Image.open(source_path).convert("RGB")
            line_height = max(10.0, line.box.y2 - line.box.y1)
            vertical_padding = max(6.0, line_height * 0.30)
            if line.box.y1 < line_height * 2:
                vertical_padding = max(vertical_padding, line_height)
            horizontal_padding = max(24.0, line_height * 2.0)
            if line.box.y1 < line_height * 2:
                horizontal_padding = max(horizontal_padding, line_height * 2.6)
            crop_box = (
                max(0, int(line.box.x1 - horizontal_padding)),
                max(0, int(line.box.y1 - vertical_padding)),
                min(page_image.width, artifact.width),
                min(page_image.height, int(line.box.y2 + vertical_padding)),
            )
            if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
                cache[cache_key] = ()
                return ()
            original = page_image.crop(crop_box)
            grayscale = ImageOps.grayscale(original)
            contrast = ImageEnhance.Contrast(grayscale).enhance(2.0)
            gray_array = np.asarray(grayscale)
            thresholded = cv2.adaptiveThreshold(
                gray_array,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                11,
            )
            variants = {
                "original": original,
                "grayscale": grayscale,
                "adaptive_threshold": Image.fromarray(thresholded),
                "contrast_enhanced": contrast,
            }
            scale = 3 if original.height < 100 else 2
            observed: list[RegionOCRCandidate] = []
            for variant_name, variant in variants.items():
                upscaled = variant.resize(
                    (variant.width * scale, variant.height * scale),
                    Image.Resampling.BICUBIC,
                )
                for psm in (6, 7, 11):
                    data = pytesseract.image_to_data(
                        upscaled,
                        lang=settings.tesseract_languages,
                        config=f"--oem 3 --psm {psm}",
                        output_type=Output.DICT,
                    )
                    recognised_words: list[dict[str, Any]] = []
                    confidences: list[float] = []
                    for index, token in enumerate(data.get("text", [])):
                        token = str(token).strip()
                        if not token:
                            continue
                        try:
                            confidence = max(
                                0.0,
                                min(1.0, float(data["conf"][index]) / 100.0),
                            )
                        except (KeyError, TypeError, ValueError):
                            confidence = 0.0
                        x1 = crop_box[0] + float(data["left"][index]) / scale
                        y1 = crop_box[1] + float(data["top"][index]) / scale
                        x2 = x1 + float(data["width"][index]) / scale
                        y2 = y1 + float(data["height"][index]) / scale
                        recognised_words.append(
                            {
                                "text": token,
                                "confidence": confidence,
                                "box": VisionBoundingBox(
                                    x1=max(0.0, x1),
                                    y1=max(0.0, y1),
                                    x2=max(0.0, x2),
                                    y2=max(0.0, y2),
                                ),
                            }
                        )
                        confidences.append(confidence)
                    text = " ".join(str(item["text"]) for item in recognised_words).strip()
                    if not text:
                        continue
                    observed.append(
                        RegionOCRCandidate(
                            text=text,
                            confidence=(
                                round(sum(confidences) / len(confidences), 4)
                                if confidences
                                else None
                            ),
                            variant=variant_name,
                            psm=psm,
                            bounding_box=VisionBoundingBox(
                                x1=float(crop_box[0]),
                                y1=float(crop_box[1]),
                                x2=float(crop_box[2]),
                                y2=float(crop_box[3]),
                            ),
                            words=tuple(recognised_words),
                        )
                    )
            cache[cache_key] = tuple(observed)
        except Exception:
            # Region re-OCR is an evidence-enhancement pass. The stored full-page
            # OCR remains authoritative when the local executable cannot process it.
            cache[cache_key] = ()
        return cache[cache_key]

    def _artifact_path(self, value: str | None) -> Path | None:
        if not value:
            return None
        raw = Path(value).expanduser()
        candidates = (
            raw,
            Path.cwd() / raw,
            Path(__file__).resolve().parents[3] / raw,
        )
        return next((item for item in candidates if item.is_file()), None)

    def _region_value_box(
        self,
        candidate: RegionOCRCandidate,
        value: str,
    ) -> VisionBoundingBox | None:
        targets = [token for item in value.split() if (token := self._normalize_token(item))]
        words = list(candidate.words)
        normalized_words = [self._normalize_token(str(item.get("text", ""))) for item in words]
        matched: list[VisionBoundingBox] = []
        for start in range(len(words)):
            boxes: list[VisionBoundingBox] = []
            for offset, target in enumerate(targets):
                position = start + offset
                if position >= len(words) or normalized_words[position] != target:
                    break
                box = words[position].get("box")
                if isinstance(box, VisionBoundingBox):
                    boxes.append(box)
            if targets and len(boxes) == len(targets):
                matched = boxes
                break
        if not matched:
            target_set = set(targets)
            matched = [
                item["box"]
                for item, token in zip(words, normalized_words, strict=True)
                if token in target_set and isinstance(item.get("box"), VisionBoundingBox)
            ]
        return self._union(matched)

    def _evidence_conflicts(
        self,
        left: FieldCandidate,
        right: FieldCandidate,
    ) -> bool:
        if left.page != right.page or left.spec.name == right.spec.name:
            return False
        declared_incompatible = (
            right.spec.name in left.spec.incompatible_with
            or left.spec.name in right.spec.incompatible_with
        )
        declared_compatible = (
            right.spec.name in left.spec.allow_evidence_sharing_with
            or left.spec.name in right.spec.allow_evidence_sharing_with
        )
        if declared_compatible and not declared_incompatible:
            return False
        left_words = {word.word_id for word in left.words}
        right_words = {word.word_id for word in right.words}
        if left_words and right_words:
            overlap = len(left_words & right_words) / min(
                len(left_words),
                len(right_words),
            )
            if overlap >= 0.78:
                return True
        left_box = left.value_box or left.evidence_box
        right_box = right.value_box or right.evidence_box
        box_overlap = self._box_overlap_over_smaller(left_box, right_box)
        same_value = self._normalize(left.value) == self._normalize(right.value)
        return box_overlap >= 0.82 or (same_value and box_overlap >= 0.45)

    def _box_overlap_over_smaller(
        self,
        left: VisionBoundingBox,
        right: VisionBoundingBox,
    ) -> float:
        width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
        height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
        intersection = width * height
        left_area = max(1.0, (left.x2 - left.x1) * (left.y2 - left.y1))
        right_area = max(1.0, (right.x2 - right.x1) * (right.y2 - right.y1))
        return intersection / min(left_area, right_area)

    def _words_for_value(
        self,
        value: str,
        words: tuple[OCRWord, ...],
    ) -> tuple[OCRWord, ...]:
        targets = [
            normalized for token in value.split() if (normalized := self._normalize_token(token))
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
            word for word, token in zip(words, normalized_words, strict=True) if token in target_set
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
        return len(cleaned) <= 100 and (
            cleaned.isupper()
            or bool(
                re.search(
                    r"(?i)(?:\b(?:report|statement|letter|notice|confirmation|transmission)\b\s*$|"
                    r"\b(?:registration|application|request|intake|business)\s+form\b)",
                    cleaned,
                )
            )
        )

    def _reliable_title_candidate(
        self,
        value: str,
        document: VisionDocumentRecord,
        *,
        confidence: float | None,
    ) -> bool:
        cleaned = " ".join(value.split()).strip()
        if confidence is not None and confidence < 0.55:
            return False
        if re.match(
            r"(?i)^\s*(?:date|dated|bate|sample\s+(?:no\.?|number|id)|batch\s+size|"
            r"type\s+of\s+cigarette|original\s+request\s+made\s+by|"
            r"sample\s+specifications\s+written\s+by)\s*[:;,#=_-]",
            cleaned,
        ):
            return False
        if re.search(
            r"(?i)\b(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+\d{1,2}.*\d",
            cleaned,
        ):
            return False
        return self._reliable_generic_title(cleaned, document) or bool(
            re.search(
                r"(?i)\b(?:registration|application|request|intake|business)\s+form\b|"
                r"\b(?:invoice|purchase\s+order|receipt|agreement|contract|report|"
                r"statement|letter|notice|confirmation|transmission)\b\s*$",
                cleaned,
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
