"""Open-set key/value and typed-entity discovery over OCR evidence.

This module operates on duck-typed OCR line and word objects.  The application
models satisfy the small attribute contract, while the output records remain
plain dataclasses so downstream layers can consume them without importing
Pydantic or a document-specific schema.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median
from typing import Any

from app.services.vision.semantic_values import (
    SemanticContext,
    SemanticValidation,
    SemanticValueRegistry,
    SemanticValueType,
    semantic_value_registry,
)


@dataclass(frozen=True)
class EvidenceBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if min(self.x1, self.y1, self.x2, self.y2) < 0:
            raise ValueError("Evidence coordinates cannot be negative")
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("Evidence box maxima must follow minima")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_value(cls, value: object) -> EvidenceBox:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(*(float(value[key]) for key in ("x1", "y1", "x2", "y2")))
        return cls(*(float(getattr(value, key)) for key in ("x1", "y1", "x2", "y2")))

    @classmethod
    def union(cls, boxes: Iterable[EvidenceBox]) -> EvidenceBox:
        observed = tuple(boxes)
        if not observed:
            raise ValueError("At least one evidence box is required")
        return cls(
            x1=min(item.x1 for item in observed),
            y1=min(item.y1 for item in observed),
            x2=max(item.x2 for item in observed),
            y2=max(item.y2 for item in observed),
        )


@dataclass(frozen=True)
class CanonicalLabelDefinition:
    canonical_label: str
    aliases: tuple[str, ...]
    expected_types: tuple[SemanticValueType, ...] = ()


@dataclass(frozen=True)
class LabelMapping:
    raw_label: str
    normalized_label: str
    canonical_label: str | None
    confidence: float
    method: str
    matched_alias: str | None = None
    expected_types: tuple[SemanticValueType, ...] = ()

    @property
    def is_known(self) -> bool:
        return self.canonical_label is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_label": self.raw_label,
            "normalized_label": self.normalized_label,
            "canonical_label": self.canonical_label,
            "confidence": self.confidence,
            "method": self.method,
            "matched_alias": self.matched_alias,
            "expected_types": [item.value for item in self.expected_types],
        }


class CanonicalLabelRegistry:
    """Maps surface labels while retaining open-set labels below threshold."""

    def __init__(
        self,
        definitions: Iterable[CanonicalLabelDefinition] = (),
        *,
        fuzzy_threshold: float = 0.88,
        ambiguity_margin: float = 0.035,
    ) -> None:
        if not 0 <= fuzzy_threshold <= 1:
            raise ValueError("fuzzy_threshold must be between zero and one")
        if not 0 <= ambiguity_margin <= 1:
            raise ValueError("ambiguity_margin must be between zero and one")
        self.fuzzy_threshold = fuzzy_threshold
        self.ambiguity_margin = ambiguity_margin
        self._definitions: dict[str, CanonicalLabelDefinition] = {}
        self._aliases: dict[str, tuple[CanonicalLabelDefinition, str]] = {}
        for definition in definitions:
            self.register(definition)

    @property
    def definitions(self) -> tuple[CanonicalLabelDefinition, ...]:
        return tuple(self._definitions.values())

    def register(
        self,
        definition: CanonicalLabelDefinition,
        *,
        replace: bool = False,
    ) -> None:
        canonical = _label_slug(definition.canonical_label)
        if canonical in self._definitions and not replace:
            raise ValueError(f"Canonical label already registered: {canonical}")
        normalized_definition = CanonicalLabelDefinition(
            canonical_label=canonical,
            aliases=tuple(dict.fromkeys((canonical.replace("_", " "), *definition.aliases))),
            expected_types=definition.expected_types,
        )
        if replace and canonical in self._definitions:
            stale = self._definitions[canonical]
            stale_aliases = {_label_key(item) for item in stale.aliases}
            self._aliases = {
                key: value
                for key, value in self._aliases.items()
                if not (value[0].canonical_label == canonical and key in stale_aliases)
            }
        self._definitions[canonical] = normalized_definition
        for alias in normalized_definition.aliases:
            key = _label_key(alias)
            if not key:
                continue
            existing = self._aliases.get(key)
            if existing and existing[0].canonical_label != canonical:
                raise ValueError(
                    f"Alias {alias!r} is already mapped to {existing[0].canonical_label}"
                )
            self._aliases[key] = (normalized_definition, alias)

    def map_label(self, raw_label: str) -> LabelMapping:
        cleaned = _clean_label(raw_label)
        key = _label_key(cleaned)
        generic_label = _label_slug(cleaned) or "unnamed_field"
        exact = self._aliases.get(key)
        if exact:
            definition, alias = exact
            return LabelMapping(
                raw_label=cleaned,
                normalized_label=definition.canonical_label,
                canonical_label=definition.canonical_label,
                confidence=1.0,
                method="exact_alias",
                matched_alias=alias,
                expected_types=definition.expected_types,
            )
        if not key:
            return LabelMapping(cleaned, generic_label, None, 0.0, "open_set")

        candidates: list[tuple[float, CanonicalLabelDefinition, str]] = []
        for alias_key, (definition, alias) in self._aliases.items():
            candidates.append((self._similarity(key, alias_key), definition, alias))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if candidates:
            best_score, definition, alias = candidates[0]
            next_distinct_score = next(
                (
                    score
                    for score, candidate_definition, _ in candidates[1:]
                    if candidate_definition.canonical_label != definition.canonical_label
                ),
                0.0,
            )
            if (
                best_score >= self.fuzzy_threshold
                and best_score - next_distinct_score >= self.ambiguity_margin
            ):
                return LabelMapping(
                    raw_label=cleaned,
                    normalized_label=definition.canonical_label,
                    canonical_label=definition.canonical_label,
                    confidence=round(best_score, 4),
                    method="fuzzy_alias",
                    matched_alias=alias,
                    expected_types=definition.expected_types,
                )
        return LabelMapping(
            raw_label=cleaned,
            normalized_label=generic_label,
            canonical_label=None,
            confidence=0.0,
            method="open_set",
        )

    def _similarity(self, left: str, right: str) -> float:
        if left == right:
            return 1.0
        sequence = SequenceMatcher(None, left, right).ratio()
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        token_overlap = len(left_tokens & right_tokens) / max(len(left_tokens | right_tokens), 1)
        containment = (
            min(len(left), len(right)) / max(len(left), len(right))
            if left in right or right in left
            else 0.0
        )
        return max(sequence, token_overlap, containment * 0.96)


@dataclass(frozen=True)
class GenericDiscoveredField:
    field_id: str
    raw_label: str
    normalized_label: str
    canonical_label: str | None
    raw_value: str
    normalized_value: str
    page: int
    label_bbox: EvidenceBox
    value_bbox: EvidenceBox
    evidence_bbox: EvidenceBox
    confidence: float
    source_engine: str
    semantic_type: SemanticValueType | None
    semantic_confidence: float
    label_mapping_confidence: float
    label_mapping_method: str
    extraction_strategy: str
    source_line_ids: tuple[str, ...]
    label_word_ids: tuple[str, ...]
    value_word_ids: tuple[str, ...]
    review_required: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "field_id": self.field_id,
            "raw_label": self.raw_label,
            "normalized_label": self.normalized_label,
            "canonical_label": self.canonical_label,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "page": self.page,
            "label_bbox": self.label_bbox.as_dict(),
            "value_bbox": self.value_bbox.as_dict(),
            "evidence_bbox": self.evidence_bbox.as_dict(),
            "confidence": self.confidence,
            "source_engine": self.source_engine,
            "semantic_type_if_known": (
                self.semantic_type.value if self.semantic_type is not None else None
            ),
            "semantic_confidence": self.semantic_confidence,
            "label_mapping_confidence": self.label_mapping_confidence,
            "label_mapping_method": self.label_mapping_method,
            "extraction_strategy": self.extraction_strategy,
            "source_line_ids": list(self.source_line_ids),
            "label_word_ids": list(self.label_word_ids),
            "value_word_ids": list(self.value_word_ids),
            "review_required": self.review_required,
        }


@dataclass(frozen=True)
class GenericTypedEntity:
    entity_id: str
    raw_value: str
    normalized_value: str
    semantic_type: SemanticValueType
    page: int
    bounding_box: EvidenceBox
    confidence: float
    source_engine: str
    source_line_ids: tuple[str, ...]
    word_ids: tuple[str, ...]
    contextual_label: str | None = None
    linked_field_id: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "entity_id": self.entity_id,
            "raw_value": self.raw_value,
            "normalized_value": self.normalized_value,
            "semantic_type": self.semantic_type.value,
            "page": self.page,
            "bounding_box": self.bounding_box.as_dict(),
            "confidence": self.confidence,
            "source_engine": self.source_engine,
            "source_line_ids": list(self.source_line_ids),
            "word_ids": list(self.word_ids),
            "contextual_label": self.contextual_label,
            "linked_field_id": self.linked_field_id,
        }


@dataclass(frozen=True)
class GenericDiscoveryResult:
    fields: tuple[GenericDiscoveredField, ...]
    entities: tuple[GenericTypedEntity, ...]
    lines_inspected: int
    words_inspected: int
    source_engine: str

    def as_dict(self) -> dict[str, object]:
        return {
            "generic_fields": [item.as_dict() for item in self.fields],
            "typed_entities": [item.as_dict() for item in self.entities],
            "diagnostics": {
                "lines_inspected": self.lines_inspected,
                "words_inspected": self.words_inspected,
                "generic_fields_discovered": len(self.fields),
                "typed_entities_discovered": len(self.entities),
                "source_engine": self.source_engine,
            },
        }


@dataclass(frozen=True)
class _WordView:
    word_id: str
    text: str
    page: int
    line_id: str
    box: EvidenceBox
    confidence: float | None
    reading_order: int


@dataclass(frozen=True)
class _LineView:
    line_id: str
    text: str
    page: int
    box: EvidenceBox
    confidence: float | None
    reading_order: int
    words: tuple[_WordView, ...]


@dataclass(frozen=True)
class _PairCandidate:
    raw_label: str
    raw_value: str
    page: int
    label_box: EvidenceBox
    value_box: EvidenceBox
    strategy: str
    base_confidence: float
    line_ids: tuple[str, ...]
    label_word_ids: tuple[str, ...]
    value_word_ids: tuple[str, ...]
    ocr_confidence: float | None


class GenericFieldDiscoveryEngine:
    adapter_id = "generic-semantic-discovery"
    model_version = "v1"

    def __init__(
        self,
        *,
        labels: CanonicalLabelRegistry | None = None,
        semantic_values: SemanticValueRegistry | None = None,
        review_threshold: float = 0.8,
    ) -> None:
        self.labels = labels or canonical_label_registry
        self.semantic_values = semantic_values or semantic_value_registry
        self.review_threshold = review_threshold

    @property
    def source_engine(self) -> str:
        return f"{self.adapter_id}-{self.model_version}"

    def discover(
        self,
        lines: Iterable[object],
        words: Iterable[object] = (),
        *,
        document_type: str = "",
    ) -> GenericDiscoveryResult:
        observed_words = self._adapt_words(words)
        observed_lines = self._adapt_lines(lines, observed_words)
        candidates = self._pair_candidates(observed_lines)
        fields = self._to_fields(candidates, document_type=document_type)
        entities = self._typed_entities(
            observed_lines,
            fields,
            document_type=document_type,
        )
        return GenericDiscoveryResult(
            fields=fields,
            entities=entities,
            lines_inspected=len(observed_lines),
            words_inspected=len(observed_words),
            source_engine=self.source_engine,
        )

    def discover_key_values(
        self,
        lines: Iterable[object],
        words: Iterable[object] = (),
        *,
        document_type: str = "",
    ) -> tuple[GenericDiscoveredField, ...]:
        return self.discover(lines, words, document_type=document_type).fields

    def discover_typed_entities(
        self,
        lines: Iterable[object],
        words: Iterable[object] = (),
        *,
        document_type: str = "",
    ) -> tuple[GenericTypedEntity, ...]:
        return self.discover(lines, words, document_type=document_type).entities

    def _adapt_words(self, words: Iterable[object]) -> tuple[_WordView, ...]:
        adapted: list[_WordView] = []
        for index, word in enumerate(words, start=1):
            text = str(_attribute(word, "text", "")).strip()
            box_value = _attribute(word, "bounding_box", _attribute(word, "box", None))
            if not text or box_value is None:
                continue
            adapted.append(
                _WordView(
                    word_id=str(_attribute(word, "word_id", f"anonymous-word-{index}")),
                    text=text,
                    page=max(1, int(_attribute(word, "page", 1))),
                    line_id=str(_attribute(word, "line_id", "")),
                    box=EvidenceBox.from_value(box_value),
                    confidence=_optional_confidence(_attribute(word, "confidence", None)),
                    reading_order=max(1, int(_attribute(word, "reading_order", index))),
                )
            )
        return tuple(sorted(adapted, key=lambda item: (item.page, item.reading_order)))

    def _adapt_lines(
        self,
        lines: Iterable[object],
        words: tuple[_WordView, ...],
    ) -> tuple[_LineView, ...]:
        by_word_id = {item.word_id: item for item in words}
        by_line_id: dict[str, list[_WordView]] = defaultdict(list)
        for word in words:
            by_line_id[word.line_id].append(word)

        adapted: list[_LineView] = []
        for index, original_line in enumerate(lines, start=1):
            record = _attribute(original_line, "record", original_line)
            text = str(_attribute(record, "text", _attribute(original_line, "text", ""))).strip()
            box_value = _attribute(
                record,
                "bounding_box",
                _attribute(original_line, "box", None),
            )
            if not text or box_value is None:
                continue
            line_id = str(_attribute(record, "line_id", f"anonymous-line-{index}"))
            embedded_words = _attribute(original_line, "words", ())
            line_words: tuple[_WordView, ...]
            if embedded_words:
                line_words = self._adapt_words(embedded_words)
            else:
                word_ids = tuple(str(item) for item in _attribute(record, "word_ids", ()))
                selected = [by_word_id[item] for item in word_ids if item in by_word_id]
                if not selected:
                    selected = by_line_id.get(line_id, [])
                line_words = tuple(
                    sorted(selected, key=lambda item: (item.box.x1, item.reading_order))
                )
            adapted.append(
                _LineView(
                    line_id=line_id,
                    text=text,
                    page=max(1, int(_attribute(record, "page", 1))),
                    box=EvidenceBox.from_value(box_value),
                    confidence=_optional_confidence(
                        _attribute(record, "mean_confidence", None)
                    ),
                    reading_order=max(
                        1,
                        int(_attribute(record, "reading_order", index)),
                    ),
                    words=line_words,
                )
            )

        if not adapted and words:
            grouped: dict[tuple[int, str], list[_WordView]] = defaultdict(list)
            for word in words:
                grouped[(word.page, word.line_id)].append(word)
            for index, ((page, line_id), line_words) in enumerate(
                sorted(
                    grouped.items(),
                    key=lambda item: (
                        item[0][0],
                        min(word.reading_order for word in item[1]),
                    ),
                ),
                start=1,
            ):
                ordered = tuple(sorted(line_words, key=lambda item: item.box.x1))
                adapted.append(
                    _LineView(
                        line_id=line_id or f"derived-line-{index}",
                        text=_join_word_text(ordered),
                        page=page,
                        box=EvidenceBox.union(item.box for item in ordered),
                        confidence=_mean_confidence(item.confidence for item in ordered),
                        reading_order=min(item.reading_order for item in ordered),
                        words=ordered,
                    )
                )
        return tuple(
            sorted(adapted, key=lambda item: (item.page, item.reading_order, item.box.y1, item.box.x1))
        )

    def _pair_candidates(self, lines: tuple[_LineView, ...]) -> tuple[_PairCandidate, ...]:
        candidates: list[_PairCandidate] = []
        occupied: set[str] = set()

        for line in lines:
            candidate = self._same_line_candidate(line)
            if candidate is not None:
                candidates.append(candidate)
                occupied.add(line.line_id)

        horizontal = self._horizontal_candidates(lines, occupied)
        candidates.extend(horizontal)
        for candidate in horizontal:
            occupied.update(candidate.line_ids)

        vertical = self._vertical_candidates(lines, occupied)
        candidates.extend(vertical)

        unique: dict[tuple[object, ...], _PairCandidate] = {}
        for candidate in candidates:
            key = (
                candidate.page,
                _label_key(candidate.raw_label),
                candidate.raw_value.casefold(),
                round(candidate.label_box.x1, 1),
                round(candidate.label_box.y1, 1),
                round(candidate.value_box.x1, 1),
                round(candidate.value_box.y1, 1),
            )
            retained = unique.get(key)
            if retained is None or retained.base_confidence < candidate.base_confidence:
                unique[key] = candidate
        return tuple(
            sorted(
                unique.values(),
                key=lambda item: (
                    item.page,
                    item.label_box.y1,
                    item.label_box.x1,
                    -item.base_confidence,
                ),
            )
        )

    def _same_line_candidate(self, line: _LineView) -> _PairCandidate | None:
        explicit = _EXPLICIT_PAIR.fullmatch(line.text)
        if explicit:
            label = _clean_label(explicit.group("label"))
            value = _clean_value(explicit.group("value"))
            if _looks_like_label(label) and value:
                delimiter = explicit.group("delimiter")
                label_box, value_box, label_words, value_words = self._split_line_evidence(
                    line,
                    label,
                    value,
                )
                strategy_by_delimiter = {
                    ":": "same_line_colon",
                    "=": "same_line_equals",
                    "|": "same_line_table_pair",
                }
                return _PairCandidate(
                    raw_label=label,
                    raw_value=value,
                    page=line.page,
                    label_box=label_box,
                    value_box=value_box,
                    strategy=strategy_by_delimiter[delimiter],
                    base_confidence=0.95 if delimiter == ":" else 0.9,
                    line_ids=(line.line_id,),
                    label_word_ids=tuple(item.word_id for item in label_words),
                    value_word_ids=tuple(item.word_id for item in value_words),
                    ocr_confidence=self._candidate_ocr_confidence(label_words + value_words, line),
                )

        geometry = self._word_gap_candidate(line)
        if geometry is not None:
            return geometry

        whitespace = _WHITESPACE_PAIR.fullmatch(line.text)
        if whitespace:
            label = _clean_label(whitespace.group("label"))
            value = _clean_value(whitespace.group("value"))
            if _looks_like_label(label) and value:
                label_box, value_box = _estimated_partition_boxes(
                    line.box,
                    line.text,
                    whitespace.start("value"),
                )
                return _PairCandidate(
                    raw_label=label,
                    raw_value=value,
                    page=line.page,
                    label_box=label_box,
                    value_box=value_box,
                    strategy="same_line_whitespace",
                    base_confidence=0.82,
                    line_ids=(line.line_id,),
                    label_word_ids=(),
                    value_word_ids=(),
                    ocr_confidence=line.confidence,
                )
        return None

    def _word_gap_candidate(self, line: _LineView) -> _PairCandidate | None:
        if len(line.words) < 2:
            return None
        ordered = tuple(sorted(line.words, key=lambda item: item.box.x1))
        typical_height = median(max(item.box.height, 1.0) for item in ordered)
        gaps = [
            (ordered[index + 1].box.x1 - ordered[index].box.x2, index + 1)
            for index in range(len(ordered) - 1)
        ]
        gap, split_at = max(gaps, default=(0.0, 0))
        threshold = max(12.0, typical_height * 1.35)
        if gap < threshold:
            return None
        label_words = ordered[:split_at]
        value_words = ordered[split_at:]
        label = _clean_label(_join_word_text(label_words))
        value = _clean_value(_join_word_text(value_words))
        if not _looks_like_label(label) or not value:
            return None
        return _PairCandidate(
            raw_label=label,
            raw_value=value,
            page=line.page,
            label_box=EvidenceBox.union(item.box for item in label_words),
            value_box=EvidenceBox.union(item.box for item in value_words),
            strategy="same_line_geometry_gap",
            base_confidence=min(0.91, 0.82 + min(gap / max(typical_height, 1.0), 5.0) * 0.018),
            line_ids=(line.line_id,),
            label_word_ids=tuple(item.word_id for item in label_words),
            value_word_ids=tuple(item.word_id for item in value_words),
            ocr_confidence=self._candidate_ocr_confidence(ordered, line),
        )

    def _horizontal_candidates(
        self,
        lines: tuple[_LineView, ...],
        occupied: set[str],
    ) -> tuple[_PairCandidate, ...]:
        retained: list[_PairCandidate] = []
        consumed = set(occupied)
        for label_line in lines:
            label = _clean_label(label_line.text)
            if label_line.line_id in consumed or not _looks_like_label(label):
                continue
            matches: list[tuple[float, _LineView]] = []
            for value_line in lines:
                if (
                    value_line.line_id == label_line.line_id
                    or value_line.line_id in consumed
                    or value_line.page != label_line.page
                    or value_line.box.x1 <= label_line.box.x2
                ):
                    continue
                overlap = _vertical_overlap(label_line.box, value_line.box)
                gap = value_line.box.x1 - label_line.box.x2
                allowed_gap = max(80.0, max(label_line.box.height, value_line.box.height) * 10)
                if overlap < 0.5 or gap > allowed_gap:
                    continue
                value = _clean_value(value_line.text)
                if not _looks_like_value(value):
                    continue
                score = overlap - gap / max(allowed_gap, 1.0) * 0.2
                matches.append((score, value_line))
            if not matches:
                continue
            _, value_line = max(matches, key=lambda item: item[0])
            value = _clean_value(value_line.text)
            retained.append(
                _PairCandidate(
                    raw_label=label,
                    raw_value=value,
                    page=label_line.page,
                    label_box=label_line.box,
                    value_box=value_line.box,
                    strategy="separate_lines_horizontal",
                    base_confidence=0.82,
                    line_ids=(label_line.line_id, value_line.line_id),
                    label_word_ids=tuple(item.word_id for item in label_line.words),
                    value_word_ids=tuple(item.word_id for item in value_line.words),
                    ocr_confidence=_mean_confidence(
                        (label_line.confidence, value_line.confidence)
                    ),
                )
            )
            consumed.update((label_line.line_id, value_line.line_id))
        return tuple(retained)

    def _vertical_candidates(
        self,
        lines: tuple[_LineView, ...],
        occupied: set[str],
    ) -> tuple[_PairCandidate, ...]:
        retained: list[_PairCandidate] = []
        consumed = set(occupied)
        for label_line in lines:
            label = _clean_label(label_line.text)
            if label_line.line_id in consumed or not _looks_like_label(label):
                continue
            possible: list[tuple[float, _LineView]] = []
            for value_line in lines:
                if (
                    value_line.line_id == label_line.line_id
                    or value_line.line_id in consumed
                    or value_line.page != label_line.page
                    or value_line.box.y1 < label_line.box.y2
                ):
                    continue
                vertical_gap = value_line.box.y1 - label_line.box.y2
                allowed_gap = max(35.0, max(label_line.box.height, value_line.box.height) * 2.75)
                if vertical_gap > allowed_gap:
                    continue
                left_delta = abs(value_line.box.x1 - label_line.box.x1)
                overlap = _horizontal_overlap(label_line.box, value_line.box)
                aligned = overlap >= 0.2 or left_delta <= max(
                    30.0,
                    max(label_line.box.height, value_line.box.height) * 2.5,
                )
                value = _clean_value(value_line.text)
                if not aligned or not _looks_like_value(value):
                    continue
                score = 1 - vertical_gap / max(allowed_gap, 1.0) - min(left_delta / 200, 0.2)
                possible.append((score, value_line))
            if not possible:
                continue
            _, value_line = max(possible, key=lambda item: item[0])
            retained.append(
                _PairCandidate(
                    raw_label=label,
                    raw_value=_clean_value(value_line.text),
                    page=label_line.page,
                    label_box=label_line.box,
                    value_box=value_line.box,
                    strategy="label_above_value",
                    base_confidence=0.79,
                    line_ids=(label_line.line_id, value_line.line_id),
                    label_word_ids=tuple(item.word_id for item in label_line.words),
                    value_word_ids=tuple(item.word_id for item in value_line.words),
                    ocr_confidence=_mean_confidence(
                        (label_line.confidence, value_line.confidence)
                    ),
                )
            )
            consumed.update((label_line.line_id, value_line.line_id))
        return tuple(retained)

    def _split_line_evidence(
        self,
        line: _LineView,
        label: str,
        value: str,
    ) -> tuple[EvidenceBox, EvidenceBox, tuple[_WordView, ...], tuple[_WordView, ...]]:
        ordered = tuple(sorted(line.words, key=lambda item: item.box.x1))
        label_key = _label_key(label)
        accumulated: list[str] = []
        for split_at, word in enumerate(ordered, start=1):
            accumulated.append(word.text)
            if _label_key(" ".join(accumulated)) == label_key and split_at < len(ordered):
                label_words = ordered[:split_at]
                value_words = ordered[split_at:]
                return (
                    EvidenceBox.union(item.box for item in label_words),
                    EvidenceBox.union(item.box for item in value_words),
                    label_words,
                    value_words,
                )
        value_start = max(line.text.casefold().find(value.casefold()), len(label))
        label_box, value_box = _estimated_partition_boxes(line.box, line.text, value_start)
        return label_box, value_box, (), ()

    def _candidate_ocr_confidence(
        self,
        words: Sequence[_WordView],
        line: _LineView,
    ) -> float | None:
        return _mean_confidence(item.confidence for item in words) or line.confidence

    def _to_fields(
        self,
        candidates: tuple[_PairCandidate, ...],
        *,
        document_type: str,
    ) -> tuple[GenericDiscoveredField, ...]:
        fields: list[GenericDiscoveredField] = []
        for candidate in candidates:
            mapping = self.labels.map_label(candidate.raw_label)
            context = SemanticContext(
                label=candidate.raw_label,
                surrounding_text=f"{candidate.raw_label}: {candidate.raw_value}",
                document_type=document_type,
            )
            semantic = self.semantic_values.classify_best(
                candidate.raw_value,
                context,
                preferred_types=mapping.expected_types,
                minimum_confidence=0.5,
            )
            normalized_value = (
                semantic.normalized_value
                if semantic is not None and semantic.normalized_value is not None
                else re.sub(r"\s+", " ", candidate.raw_value).strip()
            )
            ocr_factor = 0.85 + 0.15 * (
                candidate.ocr_confidence if candidate.ocr_confidence is not None else 0.75
            )
            confidence = candidate.base_confidence * ocr_factor
            if mapping.is_known:
                confidence += 0.025 * mapping.confidence
            if semantic is not None:
                confidence += 0.025 * semantic.confidence
            confidence = round(min(confidence, 0.99), 4)
            evidence_box = EvidenceBox.union((candidate.label_box, candidate.value_box))
            identity = _identity(
                "field",
                candidate.page,
                mapping.normalized_label,
                normalized_value,
                evidence_box,
            )
            review_required = (
                not mapping.is_known
                or confidence < self.review_threshold
                or semantic is None
                or semantic.confidence < 0.7
            )
            fields.append(
                GenericDiscoveredField(
                    field_id=identity,
                    raw_label=candidate.raw_label,
                    normalized_label=mapping.normalized_label,
                    canonical_label=mapping.canonical_label,
                    raw_value=candidate.raw_value,
                    normalized_value=normalized_value,
                    page=candidate.page,
                    label_bbox=candidate.label_box,
                    value_bbox=candidate.value_box,
                    evidence_bbox=evidence_box,
                    confidence=confidence,
                    source_engine=self.source_engine,
                    semantic_type=semantic.semantic_type if semantic is not None else None,
                    semantic_confidence=semantic.confidence if semantic is not None else 0.0,
                    label_mapping_confidence=mapping.confidence,
                    label_mapping_method=mapping.method,
                    extraction_strategy=candidate.strategy,
                    source_line_ids=candidate.line_ids,
                    label_word_ids=candidate.label_word_ids,
                    value_word_ids=candidate.value_word_ids,
                    review_required=review_required,
                )
            )
        return tuple(fields)

    def _typed_entities(
        self,
        lines: tuple[_LineView, ...],
        fields: tuple[GenericDiscoveredField, ...],
        *,
        document_type: str,
    ) -> tuple[GenericTypedEntity, ...]:
        entities: list[GenericTypedEntity] = []
        seen: set[tuple[object, ...]] = set()
        for field in fields:
            if field.semantic_type is None:
                continue
            key = (
                field.page,
                field.semantic_type,
                field.normalized_value.casefold(),
                round(field.value_bbox.x1, 1),
                round(field.value_bbox.y1, 1),
            )
            seen.add(key)
            entities.append(
                GenericTypedEntity(
                    entity_id=_identity(
                        "entity",
                        field.page,
                        field.semantic_type.value,
                        field.normalized_value,
                        field.value_bbox,
                    ),
                    raw_value=field.raw_value,
                    normalized_value=field.normalized_value,
                    semantic_type=field.semantic_type,
                    page=field.page,
                    bounding_box=field.value_bbox,
                    confidence=round(
                        field.semantic_confidence * 0.65 + field.confidence * 0.35,
                        4,
                    ),
                    source_engine=self.source_engine,
                    source_line_ids=field.source_line_ids,
                    word_ids=field.value_word_ids,
                    contextual_label=field.normalized_label,
                    linked_field_id=field.field_id,
                )
            )

        for line in lines:
            context = SemanticContext(
                surrounding_text=line.text,
                document_type=document_type,
            )
            for span in self.semantic_values.find_spans(line.text, context):
                validation = span.validation
                if validation.normalized_value is None:
                    continue
                box, words = self._span_evidence(line, span.start, span.end)
                key = (
                    line.page,
                    validation.semantic_type,
                    validation.normalized_value.casefold(),
                    round(box.x1, 1),
                    round(box.y1, 1),
                )
                if key in seen:
                    continue
                seen.add(key)
                entities.append(
                    self._entity_from_validation(
                        line,
                        validation,
                        box,
                        words,
                    )
                )
        return tuple(
            sorted(
                entities,
                key=lambda item: (
                    item.page,
                    item.bounding_box.y1,
                    item.bounding_box.x1,
                    item.semantic_type.value,
                ),
            )
        )

    def _entity_from_validation(
        self,
        line: _LineView,
        validation: SemanticValidation,
        box: EvidenceBox,
        words: tuple[_WordView, ...],
    ) -> GenericTypedEntity:
        normalized = validation.normalized_value or validation.raw_value
        return GenericTypedEntity(
            entity_id=_identity(
                "entity",
                line.page,
                validation.semantic_type.value,
                normalized,
                box,
            ),
            raw_value=validation.raw_value,
            normalized_value=normalized,
            semantic_type=validation.semantic_type,
            page=line.page,
            bounding_box=box,
            confidence=validation.confidence,
            source_engine=self.source_engine,
            source_line_ids=(line.line_id,),
            word_ids=tuple(item.word_id for item in words),
        )

    def _span_evidence(
        self,
        line: _LineView,
        start: int,
        end: int,
    ) -> tuple[EvidenceBox, tuple[_WordView, ...]]:
        cursor = 0
        selected: list[_WordView] = []
        for word in sorted(line.words, key=lambda item: item.box.x1):
            word_start = line.text.find(word.text, cursor)
            if word_start < 0:
                continue
            word_end = word_start + len(word.text)
            cursor = word_end
            if word_start < end and word_end > start:
                selected.append(word)
        if selected:
            return EvidenceBox.union(item.box for item in selected), tuple(selected)
        width = max(line.box.width, 1.0)
        text_length = max(len(line.text), 1)
        return (
            EvidenceBox(
                x1=line.box.x1 + width * start / text_length,
                y1=line.box.y1,
                x2=line.box.x1 + width * end / text_length,
                y2=line.box.y2,
            ),
            (),
        )


def _attribute(value: object, name: str, default: Any) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _mean_confidence(values: Iterable[float | None]) -> float | None:
    observed = tuple(item for item in values if item is not None)
    return round(sum(observed) / len(observed), 4) if observed else None


def _join_word_text(words: Sequence[_WordView]) -> str:
    result = ""
    previous: _WordView | None = None
    for word in words:
        if previous is None:
            result = word.text
        else:
            separator = "" if word.text in {".", ",", ":", ";", ")", "]"} else " "
            result += separator + word.text
        previous = word
    return result.strip()


def _clean_label(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t:;|=_-")


def _clean_value(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" \t|=_")


def _label_key(value: str) -> str:
    expanded = value.casefold().replace("#", " number ").replace("&", " and ")
    expanded = re.sub(r"\b(?:no|num|nbr)\.?\b", " number ", expanded)
    expanded = re.sub(r"\bp\s*\.\s*o\s*\.?\b", " purchase order ", expanded)
    expanded = re.sub(r"\binv\.?\b", " invoice ", expanded)
    expanded = re.sub(r"\borg\.?\b", " organisation ", expanded)
    return re.sub(r"[^a-z0-9]+", " ", expanded).strip()


def _label_slug(value: str) -> str:
    return _label_key(value).replace(" ", "_")


def _looks_like_label(value: str) -> bool:
    cleaned = _clean_label(value)
    if not 1 < len(cleaned) <= 80 or not re.search(r"[A-Za-z]", cleaned):
        return False
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9.'/#&()-]*", cleaned)
    if not 1 <= len(words) <= 8:
        return False
    if sum(character.isdigit() for character in cleaned) > max(2, len(cleaned) // 3):
        return False
    if "@" in cleaned or re.match(r"(?i)(?:https?|www\.)", cleaned):
        return False
    if cleaned.endswith(("!", "?")):
        return False
    return True


def _looks_like_value(value: str) -> bool:
    cleaned = _clean_value(value)
    if not cleaned or len(cleaned) > 500:
        return False
    return len(cleaned.split()) <= 30 and not cleaned.endswith("?")


def _vertical_overlap(left: EvidenceBox, right: EvidenceBox) -> float:
    overlap = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    return overlap / max(min(left.height, right.height), 1.0)


def _horizontal_overlap(top: EvidenceBox, bottom: EvidenceBox) -> float:
    overlap = max(0.0, min(top.x2, bottom.x2) - max(top.x1, bottom.x1))
    return overlap / max(min(top.width, bottom.width), 1.0)


def _estimated_partition_boxes(
    box: EvidenceBox,
    text: str,
    value_start: int,
) -> tuple[EvidenceBox, EvidenceBox]:
    ratio = max(0.08, min(0.92, value_start / max(len(text), 1)))
    split_x = box.x1 + box.width * ratio
    return (
        EvidenceBox(box.x1, box.y1, split_x, box.y2),
        EvidenceBox(split_x, box.y1, box.x2, box.y2),
    )


def _identity(
    prefix: str,
    page: int,
    name: str,
    value: str,
    box: EvidenceBox,
) -> str:
    payload = (
        f"{page}|{name}|{value}|{box.x1:.2f}|{box.y1:.2f}|{box.x2:.2f}|{box.y2:.2f}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:14]
    return f"generic-{prefix}-{digest}"


def _definition(
    canonical_label: str,
    *aliases: str,
    expected_types: tuple[SemanticValueType, ...] = (),
) -> CanonicalLabelDefinition:
    return CanonicalLabelDefinition(canonical_label, aliases, expected_types)


DEFAULT_CANONICAL_LABELS = (
    _definition(
        "invoice_number",
        "invoice no",
        "invoice #",
        "invoice number",
        "inv no.",
        "inv #",
        expected_types=(SemanticValueType.DOCUMENT_REFERENCE, SemanticValueType.IDENTIFIER),
    ),
    _definition(
        "purchase_order_number",
        "purchase order",
        "purchase order no",
        "purchase order number",
        "po",
        "po no",
        "po number",
        "p.o.",
        "p.o. no.",
        expected_types=(SemanticValueType.DOCUMENT_REFERENCE, SemanticValueType.IDENTIFIER),
    ),
    _definition(
        "requested_by",
        "requested by",
        "request made by",
        "original request made by",
        "requester",
        "requestor",
        expected_types=(SemanticValueType.PERSON,),
    ),
    _definition(
        "reference_number",
        "reference",
        "reference no",
        "reference number",
        "ref no",
        "ref #",
        expected_types=(SemanticValueType.DOCUMENT_REFERENCE, SemanticValueType.IDENTIFIER),
    ),
    _definition(
        "customer_id",
        "customer id",
        "client id",
        "customer identifier",
        expected_types=(SemanticValueType.IDENTIFIER,),
    ),
    _definition(
        "account_number",
        "account",
        "account no",
        "account number",
        "bank account",
        "bank account number",
        "iban",
        expected_types=(SemanticValueType.BANK_ACCOUNT, SemanticValueType.IDENTIFIER),
    ),
    _definition(
        "date",
        "date",
        "dated",
        "document date",
        "submission date",
        "issue date",
        expected_types=(SemanticValueType.DATE, SemanticValueType.DATETIME),
    ),
    _definition(
        "phone_number",
        "phone",
        "phone no",
        "phone number",
        "telephone",
        "telephone number",
        "mobile",
        "contact number",
        expected_types=(SemanticValueType.PHONE,),
    ),
    _definition(
        "email",
        "email",
        "email address",
        "e-mail",
        expected_types=(SemanticValueType.EMAIL,),
    ),
    _definition(
        "organisation_name",
        "organisation",
        "organisation name",
        "organization",
        "organization name",
        "company",
        "company name",
        "vendor",
        "supplier",
        expected_types=(SemanticValueType.ORGANISATION,),
    ),
    _definition(
        "address",
        "address",
        "postal address",
        "billing address",
        "shipping address",
        expected_types=(SemanticValueType.ADDRESS,),
    ),
    _definition(
        "amount",
        "amount",
        "total amount",
        "grand total",
        "balance",
        expected_types=(SemanticValueType.MONEY, SemanticValueType.NUMBER),
    ),
    _definition(
        "quantity",
        "quantity",
        "qty",
        "units",
        expected_types=(SemanticValueType.QUANTITY, SemanticValueType.NUMBER),
    ),
)


def build_default_canonical_label_registry() -> CanonicalLabelRegistry:
    return CanonicalLabelRegistry(DEFAULT_CANONICAL_LABELS)


canonical_label_registry = build_default_canonical_label_registry()
generic_field_discovery_engine = GenericFieldDiscoveryEngine()


_EXPLICIT_PAIR = re.compile(
    r"\s*(?P<label>[A-Za-z][^:=|\r\n]{0,79}?)\s*(?P<delimiter>[:=|])\s*"
    r"(?P<value>\S(?:.*\S)?)\s*"
)
_WHITESPACE_PAIR = re.compile(
    r"\s*(?P<label>[A-Za-z][^\r\n]{1,79}?)\s{2,}(?P<value>\S(?:.*\S)?)\s*"
)


__all__ = [
    "DEFAULT_CANONICAL_LABELS",
    "CanonicalLabelDefinition",
    "CanonicalLabelRegistry",
    "EvidenceBox",
    "GenericDiscoveredField",
    "GenericDiscoveryResult",
    "GenericFieldDiscoveryEngine",
    "GenericTypedEntity",
    "LabelMapping",
    "build_default_canonical_label_registry",
    "canonical_label_registry",
    "generic_field_discovery_engine",
]
