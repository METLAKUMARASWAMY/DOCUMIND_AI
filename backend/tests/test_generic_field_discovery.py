from __future__ import annotations

from dataclasses import dataclass

import pytest

from app.services.vision.generic_field_discovery import (
    CanonicalLabelRegistry,
    EvidenceBox,
    GenericFieldDiscoveryEngine,
    canonical_label_registry,
)
from app.services.vision.semantic_values import (
    SemanticContext,
    SemanticValueType,
    semantic_value_registry,
)


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Word:
    word_id: str
    text: str
    bounding_box: Box
    page: int
    line_id: str
    reading_order: int
    confidence: float = 0.96


@dataclass(frozen=True)
class Line:
    line_id: str
    text: str
    bounding_box: Box
    page: int
    reading_order: int
    word_ids: tuple[str, ...] = ()
    mean_confidence: float = 0.96


def word(
    word_id: str,
    text: str,
    box: tuple[float, float, float, float],
    *,
    line_id: str,
    reading_order: int,
) -> Word:
    return Word(word_id, text, Box(*box), 1, line_id, reading_order)


@pytest.mark.parametrize("value", ["3/14/90", "2/15/90", "2026-08-10"])
def test_date_shapes_are_globally_rejected_as_phone_numbers(value: str):
    phone = semantic_value_registry.validate(SemanticValueType.PHONE, value)
    date = semantic_value_registry.validate(SemanticValueType.DATE, value)

    assert not phone.valid
    assert phone.reason == "date_shape_excluded_from_phone"
    assert date.valid


def test_date_shaped_ocr_with_duplicate_suffix_is_still_not_a_phone():
    result = semantic_value_registry.validate(SemanticValueType.PHONE, "3/14/90 90")

    assert not result.valid
    assert result.reason == "date_shape_excluded_from_phone"


@pytest.mark.parametrize("value", ["910/535-7707", "202/828-2259"])
def test_fax_numbers_with_slash_separators_remain_valid_phone_values(value: str):
    result = semantic_value_registry.validate(
        SemanticValueType.PHONE,
        value,
        SemanticContext(label="Fax"),
    )
    spans = semantic_value_registry.find_spans(f"Fax: {value}")

    assert result.valid
    assert result.normalized_value == value.replace("/", "").replace("-", "")
    assert any(
        span.validation.semantic_type is SemanticValueType.PHONE and span.text == value
        for span in spans
    )


def test_numeric_values_are_classified_using_account_and_identifier_context():
    account = semantic_value_registry.classify_best(
        "123456789",
        SemanticContext(label="Account Number"),
    )
    identifier = semantic_value_registry.classify_best(
        "123456789",
        SemanticContext(label="Customer ID"),
    )
    phone = semantic_value_registry.classify_best(
        "123456789",
        SemanticContext(label="Phone"),
    )

    assert account is not None
    assert account.semantic_type is SemanticValueType.BANK_ACCOUNT
    assert identifier is not None
    assert identifier.semantic_type is SemanticValueType.IDENTIFIER
    assert phone is not None
    assert phone.semantic_type is SemanticValueType.PHONE


def test_common_aliases_normalize_and_unrelated_labels_stay_open_set():
    expected = {
        "Invoice No.": "invoice_number",
        "Invoice #": "invoice_number",
        "P.O. No.": "purchase_order_number",
        "Original Request Made By": "requested_by",
    }

    for raw_label, canonical in expected.items():
        mapping = canonical_label_registry.map_label(raw_label)
        assert mapping.canonical_label == canonical
        assert mapping.normalized_label == canonical

    fuzzy = canonical_label_registry.map_label("Invioce Number")
    assert fuzzy.canonical_label == "invoice_number"
    assert fuzzy.method == "fuzzy_alias"

    unknown = canonical_label_registry.map_label("Experimental Batch Key")
    assert unknown.canonical_label is None
    assert unknown.normalized_label == "experimental_batch_key"
    assert unknown.method == "open_set"


def test_fuzzy_threshold_is_configurable_and_low_confidence_match_is_not_forced():
    strict = CanonicalLabelRegistry(
        canonical_label_registry.definitions,
        fuzzy_threshold=0.99,
    )

    mapping = strict.map_label("Invioce Number")

    assert mapping.canonical_label is None
    assert mapping.normalized_label == "invioce_number"


def test_discovers_colon_geometry_whitespace_and_label_above_value_pairs():
    words = (
        word("w1", "Invoice", (10, 10, 62, 24), line_id="l1", reading_order=1),
        word("w2", "No.:", (66, 10, 89, 24), line_id="l1", reading_order=2),
        word("w3", "INV-42", (135, 10, 195, 24), line_id="l1", reading_order=3),
        word("w4", "Customer", (10, 40, 74, 54), line_id="l2", reading_order=4),
        word("w5", "ID", (78, 40, 92, 54), line_id="l2", reading_order=5),
        word("w6", "C-900", (180, 40, 230, 54), line_id="l2", reading_order=6),
        word("w7", "Requested", (10, 75, 82, 89), line_id="l3", reading_order=7),
        word("w8", "By", (86, 75, 102, 89), line_id="l3", reading_order=8),
        word("w9", "Priya", (10, 96, 48, 110), line_id="l4", reading_order=9),
        word("w10", "Shah", (52, 96, 88, 110), line_id="l4", reading_order=10),
    )
    lines = (
        Line("l1", "Invoice No.: INV-42", Box(10, 10, 195, 24), 1, 1, ("w1", "w2", "w3")),
        Line("l2", "Customer ID C-900", Box(10, 40, 230, 54), 1, 2, ("w4", "w5", "w6")),
        Line("l3", "Requested By", Box(10, 75, 102, 89), 1, 3, ("w7", "w8")),
        Line("l4", "Priya Shah", Box(10, 96, 88, 110), 1, 4, ("w9", "w10")),
    )

    fields = GenericFieldDiscoveryEngine().discover_key_values(lines, words)
    by_label = {field.normalized_label: field for field in fields}

    assert by_label["invoice_number"].raw_value == "INV-42"
    assert by_label["invoice_number"].extraction_strategy == "same_line_colon"
    assert by_label["customer_id"].raw_value == "C-900"
    assert by_label["customer_id"].extraction_strategy == "same_line_geometry_gap"
    assert by_label["requested_by"].raw_value == "Priya Shah"
    assert by_label["requested_by"].extraction_strategy == "label_above_value"
    assert by_label["requested_by"].semantic_type is SemanticValueType.PERSON


def test_unknown_labels_are_retained_with_typed_value_and_review_marker():
    line = Line(
        "open-line",
        "Experimental Batch Key: ZX-77",
        Box(20, 30, 280, 48),
        1,
        1,
    )

    result = GenericFieldDiscoveryEngine().discover((line,))
    field = result.fields[0]

    assert field.raw_label == "Experimental Batch Key"
    assert field.normalized_label == "experimental_batch_key"
    assert field.canonical_label is None
    assert field.raw_value == "ZX-77"
    assert field.review_required
    assert result.as_dict()["generic_fields"][0]["raw_label"] == field.raw_label


def test_word_evidence_geometry_is_preserved_for_label_value_and_entity():
    words = (
        word("label-1", "Account", (12, 18, 70, 34), line_id="account-line", reading_order=1),
        word("label-2", "Number:", (74, 18, 132, 34), line_id="account-line", reading_order=2),
        word("value-1", "77889900", (190, 18, 270, 34), line_id="account-line", reading_order=3),
    )
    line = Line(
        "account-line",
        "Account Number: 77889900",
        Box(12, 18, 270, 34),
        1,
        1,
        tuple(item.word_id for item in words),
    )

    result = GenericFieldDiscoveryEngine().discover((line,), words)
    field = result.fields[0]
    linked = next(item for item in result.entities if item.linked_field_id == field.field_id)

    assert field.label_bbox == EvidenceBox(12, 18, 132, 34)
    assert field.value_bbox == EvidenceBox(190, 18, 270, 34)
    assert field.evidence_bbox == EvidenceBox(12, 18, 270, 34)
    assert field.label_word_ids == ("label-1", "label-2")
    assert field.value_word_ids == ("value-1",)
    assert linked.bounding_box == field.value_bbox
    assert linked.semantic_type is SemanticValueType.BANK_ACCOUNT
