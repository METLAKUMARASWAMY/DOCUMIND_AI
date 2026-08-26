from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.core.config import settings
from app.models.schemas import TextAnalysisRequest
from app.models.vision import VisionDocumentRecord, VisionDocumentType
from app.services.vision.deterministic_extraction import deterministic_schema_extractor
from app.services.vision.document_classification import hybrid_document_classifier
from app.services.vision.ocr_adapters import TesseractOCRAdapter
from app.services.vision.ocr_lines import ocr_line_reconstructor
from app.services.vision.preprocessing import DocumentPreprocessor
from app.services.vision.retrieval_adapters import HybridVector
from app.services.vision.schema_registry import SchemaFieldSpec, vision_schema_registry
from app.services.vision.service import VisionAIService


def _fixture(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "var" / "test_fixtures" / name


def _extract(source: Path, tmp_path: Path, *, filename: str) -> VisionDocumentRecord:
    content = source.read_bytes()
    preprocessing = DocumentPreprocessor(tmp_path).process(
        "generalization-regression", filename, content
    )
    document = VisionDocumentRecord(
        document_id="generalization-regression",
        case_id="generalization-regression-case",
        filename=filename,
        file_hash=hashlib.sha256(content).hexdigest(),
        document_type=VisionDocumentType.unknown,
        page_artifacts=preprocessing.artifacts,
        quality=preprocessing.quality,
        created_at="2026-08-12T00:00:00+00:00",
        updated_at="2026-08-12T00:00:00+00:00",
    )
    document.ocr_words = TesseractOCRAdapter().extract_words(document)
    document.ocr_lines = ocr_line_reconstructor.reconstruct(document.ocr_words)
    classification = hybrid_document_classifier.classify(
        "\n".join(line.text for line in document.ocr_lines),
        document.filename,
        lines=document.ocr_lines,
    )
    document.document_type = classification.predicted_type
    document.metadata["classification"] = classification.as_dict()
    document.extracted_fields = deterministic_schema_extractor.extract(document)
    return document


@pytest.mark.parametrize(
    "value",
    ["3/14/90", "2/15/90", "12/03/2026", "2026-08-10", "3/14/90 90-B-1"],
)
def test_calendar_dates_never_validate_as_phone(value: str) -> None:
    phone = SchemaFieldSpec("phone", "Phone", ("phone",), value_kind="phone")
    valid, reason = vision_schema_registry.validate_value(phone, value)
    assert valid is False
    assert reason == "calendar_date_is_not_phone"


def test_regression_form_uses_semantics_after_filename_change(tmp_path: Path) -> None:
    document = _extract(
        _fixture("87093315_87093318.png"),
        tmp_path,
        filename="previously-unseen-name.png",
    )
    fields = {field.name: field for field in document.extracted_fields}

    assert document.document_type == VisionDocumentType.business_form
    assert document.schema_profile.mode == "A"
    assert document.schema_profile.schema_status == "registered"
    assert {
        "sample_number": "1194-90",
        "date": "3/14/90",
        "product_type": "100 mm Filter",
        "batch_size": "47.5 lbs.",
        "requested_by": "J.H. Bell",
        "request_date": "2/15/90",
        "purpose": "Cigarette Modification B-451",
        "specifications_written_by": "C.W. Lassiter",
    } == {name: fields[name].value for name in fields}
    assert fields["date"].normalized_value == "1990-03-14"
    assert fields["request_date"].normalized_value == "1990-02-15"
    assert all(field.evidence.bounding_box is not None for field in fields.values())
    assert "visible_phone_numbers" not in fields
    assert document.metadata["ocr_consensus"][0]["selected_variant"] in {
        "original",
        "processed",
        "grayscale",
        "enhanced",
    }


def test_regression_form_full_pipeline_emits_exact_source_backed_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    service = VisionAIService()
    source = _fixture("87093315_87093318.png")
    response = service.queue_documents(
        [("fully-renamed-source.png", source.read_bytes(), {})],
        actor="regression-test",
        process_inline=True,
        tenant_id="tenant-generalization-regression",
    )
    document = service.get_document(
        response.document_ids[0],
        "tenant-generalization-regression",
    )
    fields = {field.name: field for field in document.extracted_fields}

    assert document.document_type == VisionDocumentType.business_form
    assert document.schema_profile.mode == "A"
    assert {
        "sample_number": "1194-90",
        "date": "3/14/90",
        "product_type": "100 mm Filter",
        "batch_size": "47.5 lbs.",
        "requested_by": "J.H. Bell",
        "request_date": "2/15/90",
        "purpose": "Cigarette Modification B-451",
        "specifications_written_by": "C.W. Lassiter",
    } == {name: field.value for name, field in fields.items()}
    assert fields["date"].normalized_value == "1990-03-14"
    assert fields["request_date"].normalized_value == "1990-02-15"
    assert all(field.evidence.bounding_box is not None for field in fields.values())
    assert all(field.evidence.document_id == document.document_id for field in fields.values())
    assert not any(entity.semantic_type.value == "PHONE" for entity in document.typed_entities)


def test_00093726_generic_layer_rejects_geometry_fragments_without_losing_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    service = VisionAIService()
    source = _fixture("00093726.png")
    response = service.queue_documents(
        [("unseen-business-form-name.png", source.read_bytes(), {})],
        actor="generic-filter-regression-test",
        process_inline=True,
        tenant_id="tenant-generic-filter-regression",
    )
    document = service.get_document(
        response.document_ids[0],
        "tenant-generic-filter-regression",
    )

    assert {field.name for field in document.extracted_fields} == {
        "date",
        "sample_number",
        "product_type",
        "batch_size",
        "requested_by",
        "request_date",
        "specifications_written_by",
    }
    assert document.schema_profile.mode == "A"
    assert not {
        "original_request_made_by_dr_a_w_spears",
        "pr_i_j",
        "yn",
    } & {field.normalized_label for field in document.generic_fields}
    diagnostics = document.metadata["generic_discovery"]
    assert diagnostics["candidates_rejected"] >= 3
    rejected = {
        item["normalized_label"]: item["reason"] for item in diagnostics["rejected_candidates"]
    }
    assert {
        "original_request_made_by_dr_a_w_spears",
        "pr_i_j",
        "yn",
    } <= set(rejected)
    assert {rejected[name] for name in {"pr_i_j", "yn"}} == {"ocr_label_fragment"}
    assert (
        rejected["original_request_made_by_dr_a_w_spears"]
        == ("known_schema_line_value_leakage",)[0]
    )


def test_classification_is_invariant_to_filename() -> None:
    text = "INVOICE\nInvoice Number: INV-22\nAmount Due: USD 50.00"
    first = hybrid_document_classifier.classify(text, "invoice-looking-name.pdf")
    second = hybrid_document_classifier.classify(text, "renamed-unrelated.bin")
    assert first.predicted_type == second.predicted_type == VisionDocumentType.invoice
    assert first.confidence == second.confidence


def test_invoice_heading_outweighs_later_purchase_order_reference() -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "dataset"
        / "documind_ai_synthetic_dataset"
        / "documents"
        / "case-0003"
        / "invoice.txt"
    )

    classification = hybrid_document_classifier.classify(
        source.read_text(encoding="utf-8"),
        "semantically-unrelated-name.txt",
    )

    assert classification.predicted_type == VisionDocumentType.invoice
    assert classification.signals[0].signal_type == "title_keyword"
    assert classification.signals[0].value == "INVOICE"


def test_active_bge_mapper_conservatively_maps_only_unambiguous_schema_labels() -> None:
    class FakeBGE:
        adapter_id = "bge-m3-embedding"

        def embed_many(self, texts: list[str]) -> list[HybridVector]:
            vectors = []
            for text in texts:
                normalized = text.casefold()
                if "supplier designation" in normalized or "vendor name" in normalized:
                    dense = [1.0, 0.0, 0.0]
                elif normalized.startswith("canonical document field:"):
                    dense = [0.0, 1.0, 0.0]
                else:
                    dense = [0.0, 0.0, 1.0]
                vectors.append(HybridVector(dense=dense, sparse={}))
            return vectors

    service = VisionAIService()
    service.retrieval_index.embedding = FakeBGE()
    schema = vision_schema_registry.get(VisionDocumentType.invoice)

    mappings, diagnostics = service._embedding_schema_labels(
        ["Supplier Designation", "Warehouse Zone"],
        schema,
    )

    assert mappings == {"Supplier Designation": ("vendor_name", 1.0)}
    assert diagnostics["status"] == "completed"
    assert diagnostics["labels_considered"] == 2
    assert diagnostics["labels_mapped"] == 1


@pytest.mark.parametrize("value", ["3/14/90", "2/15/90", "2026-08-10"])
def test_raw_ocr_phone_fallback_also_rejects_calendar_dates(value: str) -> None:
    service = VisionAIService()
    document = service.process_text(
        TextAnalysisRequest(
            filename="unknown-date-label.txt",
            text=f"Phone: {value}\n",
        ),
        tenant_id="tenant-raw-ocr-phone-regression",
    )
    document.extracted_fields = []

    assert service._ocr_lookup("What is the phone number?", [document], 5) == []
