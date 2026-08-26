from __future__ import annotations

import json
import re
from importlib import resources
from pathlib import Path

import pytest

from app.core.config import settings
from app.models.schemas import TextAnalysisRequest
from app.models.vision import OCRLineRecord, VisionBoundingBox, VisionDocumentType
from app.services.vision.deterministic_extraction import OCRLine, deterministic_schema_extractor
from app.services.vision.document_classification import hybrid_document_classifier
from app.services.vision.schema_config import (
    DEFAULT_RESOURCE_NAME,
    DEFAULT_RESOURCE_PACKAGE,
    SchemaConfigurationError,
)
from app.services.vision.schema_registry import (
    SCHEMAS,
    SchemaFieldSpec,
    VisionSchemaRegistry,
    vision_schema_registry,
)
from app.services.vision.service import VisionAIService


def _catalog_payload() -> dict[str, object]:
    resource = resources.files(DEFAULT_RESOURCE_PACKAGE).joinpath(DEFAULT_RESOURCE_NAME)
    return json.loads(resource.read_text(encoding="utf-8"))


def _write_catalog(tmp_path: Path, payload: dict[str, object]) -> Path:
    destination = tmp_path / "catalog.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def test_packaged_catalog_is_versioned_validated_and_runtime_complete() -> None:
    assert vision_schema_registry.catalog_version == "2026.08.13"
    assert vision_schema_registry.catalog_source.startswith("package:")
    assert len(vision_schema_registry.catalog_fingerprint) == 64
    assert {
        VisionDocumentType.invoice,
        VisionDocumentType.purchase_order,
        VisionDocumentType.contract,
        VisionDocumentType.business_form,
        VisionDocumentType.sample_form,
        VisionDocumentType.product_specification,
        VisionDocumentType.email,
        VisionDocumentType.statement,
    } <= set(SCHEMAS)

    for schema in vision_schema_registry.registered_schemas():
        assert schema.catalog_version == vision_schema_registry.catalog_version
        assert schema.version.startswith("v")
        assert schema.classification_hints
        assert schema.applicable_modules
        assert len({field.name for field in schema.fields}) == len(schema.fields)
        for field in schema.fields:
            assert field.aliases
            assert field.validators
            assert field.normalizers
            assert field.extraction_strategies
            assert field.semantic_entity_type


@pytest.mark.parametrize(
    ("mutator", "expected_path"),
    [
        (
            lambda payload: payload.__setitem__("format_version", 999),
            "$.format_version",
        ),
        (
            lambda payload: payload["schemas"][0].__setitem__("requiredd", True),
            "$.schemas[0]",
        ),
        (
            lambda payload: payload["value_kinds"]["text"].__setitem__(
                "normalizer", "invented_normalizer"
            ),
            "$.value_kinds.text.normalizer",
        ),
        (
            lambda payload: payload["schemas"][6]["document_types"].append("invoice"),
            "$.schemas",
        ),
    ],
)
def test_invalid_catalog_fails_explicitly_at_startup(
    tmp_path: Path,
    mutator,
    expected_path: str,
) -> None:
    payload = _catalog_payload()
    mutator(payload)
    path = _write_catalog(tmp_path, payload)

    with pytest.raises(SchemaConfigurationError, match=re.escape(expected_path)):
        VisionSchemaRegistry(catalog_path=path)


def test_schema_selection_modes_are_mutually_exclusive_and_complete() -> None:
    for document_type in VisionDocumentType:
        selection = vision_schema_registry.resolve(document_type, prefer_key_value=True)
        assert selection.mode in {"A", "B", "C"}
        assert selection.schema_status in {"registered", "inferred", "generic"}
        assert selection.selected_schema in selection.path
        if vision_schema_registry.get(document_type) is not None:
            assert selection.mode == "A"
            assert selection.source == "exact"
            assert selection.exact_schema == selection.selected_schema
        elif document_type in {VisionDocumentType.generic_form, VisionDocumentType.legal_form}:
            assert selection.mode == "B"
            assert selection.source == "parent"
            assert selection.fallback_schema == selection.selected_schema
        else:
            assert selection.mode == "C"
            assert selection.source.startswith("generic")
            assert selection.exact_schema is None
            assert selection.fallback_schema is None

    for prefer_key_value in (False, True):
        unknown = vision_schema_registry.resolve("not-a-registered-type", prefer_key_value=prefer_key_value)
        assert unknown.mode == "C"
        assert unknown.schema_status == "generic"


def test_calendar_date_phone_exclusion_holds_for_generated_valid_dates() -> None:
    date_spec = SchemaFieldSpec("date", "Date", ("date",), value_kind="date")
    phone_spec = SchemaFieldSpec("phone", "Phone", ("phone",), value_kind="phone")
    for month in range(1, 13):
        for day in range(1, 29):
            for year in ("24", "90", "2024", "2099"):
                for separator in ("/", "-"):
                    value = f"{month}{separator}{day}{separator}{year}"
                    assert vision_schema_registry.value_is_valid(date_spec, value), value
                    valid, reason = vision_schema_registry.validate_value(phone_spec, value)
                    assert not valid, value
                    assert reason == "calendar_date_is_not_phone"


@pytest.mark.parametrize(
    ("document_type", "field_name", "alias", "value"),
    [
        (VisionDocumentType.invoice, "invoice_number", "Invoice Number", "INV-X9"),
        (VisionDocumentType.invoice, "purchase_order_number", "PO Number", "PO-77"),
        (VisionDocumentType.business_form, "requested_by", "Request Made By", "A. Person"),
        (VisionDocumentType.business_form, "sample_number", "Requisition No", "REQ-8"),
        (VisionDocumentType.statement, "closing_balance", "Closing Amount", "USD 8.00"),
    ],
)
def test_configured_aliases_survive_case_spacing_and_delimiter_variations(
    document_type: VisionDocumentType,
    field_name: str,
    alias: str,
    value: str,
) -> None:
    spec = vision_schema_registry.get(document_type).field(field_name)
    assert spec is not None
    for delimiter in (":", " =", " ;"):
        varied_alias = "   ".join(alias.upper().split())
        text = f"{varied_alias}{delimiter} {value}"
        record = OCRLineRecord(
            line_id="generated-label-line",
            text=text,
            page=1,
            bounding_box=VisionBoundingBox(x1=0, y1=0, x2=500, y2=30),
            reading_order=1,
        )
        match = deterministic_schema_extractor._match_label(OCRLine(record, ()), spec)
        assert match is not None, text
        assert match.value == value


def test_classifier_uses_schema_catalog_hints_for_unseen_statement_wording() -> None:
    result = hybrid_document_classifier.classify(
        "ACCOUNT ACTIVITY\nStatement Period: August 2026\nOpening Balance: USD 20.00",
        "opaque-upload-name.bin",
    )

    assert result.predicted_type == VisionDocumentType.statement
    assert result.schema_mode == "A"
    assert any(signal.signal_type == "schema_registry_hint" for signal in result.signals)


def test_unseen_statement_variant_uses_configured_schema_without_engine_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    document = VisionAIService().process_text(
        TextAnalysisRequest(
            filename="opaque-ledger-upload.txt",
            text=(
                "ACCOUNT ACTIVITY\n"
                "Statement Period: August 2026\n"
                "Statement Ref.: ST-900\n"
                "Period From: 08/01/2026\n"
                "Period To: 08/31/2026\n"
                "Opening Balance: USD 100.00\n"
                "Closing Amount: USD 250.40\n"
            ),
        ),
        tenant_id="tenant-schema-holdout",
    )
    fields = {field.name: field for field in document.extracted_fields}

    assert document.document_type == VisionDocumentType.statement
    assert document.schema_profile.schema_id == "statement"
    assert document.schema_profile.mode in {"A", "B"}
    assert fields["statement_reference"].value == "ST-900"
    assert fields["period_start"].value == "08/01/2026"
    assert fields["period_end"].value == "08/31/2026"
    assert fields["opening_balance"].normalized_value == "100.00"
    assert fields["closing_balance"].normalized_value == "250.40"
    assert all(field.evidence.bounding_box is not None for field in fields.values())
