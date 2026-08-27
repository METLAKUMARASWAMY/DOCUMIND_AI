from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

from app.models.vision import (
    OCRWord,
    VisionBoundingBox,
    VisionDocumentRecord,
    VisionDocumentType,
)
from app.services.vision.deterministic_extraction import (
    FieldCandidateResolver,
    deterministic_schema_extractor,
)
from app.services.vision.document_classification import hybrid_document_classifier
from app.services.vision.ocr_adapters import TesseractOCRAdapter
from app.services.vision.ocr_lines import ocr_line_reconstructor
from app.services.vision.preprocessing import DocumentPreprocessor
from app.services.vision.schema_registry import (
    SchemaFieldSpec,
    vision_schema_registry,
)


def _source_fixture() -> Path:
    backend_root = Path(__file__).resolve().parents[2]
    fixture = backend_root / "var" / "test_fixtures" / "00093726.png"
    assert fixture.is_file(), "The required 00093726.png acceptance fixture is missing"
    return fixture


@pytest.fixture(scope="module")
def extracted_fixture(tmp_path_factory: pytest.TempPathFactory) -> VisionDocumentRecord:
    source = _source_fixture()
    content = source.read_bytes()
    preprocessor = DocumentPreprocessor(tmp_path_factory.mktemp("business-form-v3-artifacts"))
    preprocessing = preprocessor.process(
        "vision-doc-business-form-fixture",
        "00093726.png",
        content,
    )
    document = VisionDocumentRecord(
        document_id="vision-doc-business-form-fixture",
        case_id="vision-case-business-form-fixture",
        filename="00093726.png",
        file_hash=hashlib.sha256(content).hexdigest(),
        document_type=VisionDocumentType.unknown,
        page_artifacts=preprocessing.artifacts,
        quality=preprocessing.quality,
        created_at="2026-08-06T00:00:00+00:00",
        updated_at="2026-08-06T00:00:00+00:00",
    )
    ocr = TesseractOCRAdapter()
    assert ocr.status().status == "active"
    document.ocr_words = ocr.extract_words(document)
    document.ocr_lines = ocr_line_reconstructor.reconstruct(document.ocr_words)
    classification = hybrid_document_classifier.classify(
        "\n".join(line.text for line in document.ocr_lines),
        document.filename,
        lines=document.ocr_lines,
        layout_regions=document.layout_regions,
    )
    document.document_type = classification.predicted_type
    document.metadata["classification"] = classification.as_dict()
    document.extracted_fields = deterministic_schema_extractor.extract(document)
    return document


def test_00093726_date_is_corrected_from_roi_evidence_not_promoted_to_title(
    extracted_fixture: VisionDocumentRecord,
) -> None:
    fields = {field.name: field for field in extracted_fixture.extracted_fields}

    assert extracted_fixture.document_type == VisionDocumentType.business_form
    assert fields["date"].value == "September 21, 1976"
    assert fields["date"].normalized_value == "1976-09-21"
    assert "document_title" not in fields
    assert "form_title" not in fields

    provenance = fields["date"].evidence.metadata
    raw_ocr = provenance["raw_ocr"]
    assert "september 21" in raw_ocr.lower()
    assert raw_ocr != provenance["corrected_ocr"]
    assert provenance["corrected_ocr"] == "Date: September 21, 1976"
    assert provenance["correction_method"] == "roi_reocr_and_date_validation"
    assert provenance["correction_confidence"] is not None
    assert len(provenance["candidates"]) >= 3
    assert provenance["label_box"] is not None
    assert provenance["value_box"] is not None

    consensus = extracted_fixture.metadata["ocr_consensus"][0]
    assert consensus["selection_method"] == "confidence_character_semantic_consensus_v1"
    assert consensus["selected_variant"] in {
        candidate["variant"] for candidate in consensus["candidates"]
    }
    assert len(consensus["candidates"]) >= 2


def test_00093726_extracts_six_business_fields_with_source_evidence(
    extracted_fixture: VisionDocumentRecord,
) -> None:
    fields = {field.name: field for field in extracted_fixture.extracted_fields}
    expected = {
        "date": "September 21, 1976",
        "sample_number": "6030",
        "product_type": "85 mm Filter",
        "batch_size": "50 lbs.",
        "requested_by": "Dr. A. W. Spears",
        "specifications_written_by": "W. E. Routh",
    }

    assert {name: fields[name].value for name in expected} == expected
    for name in expected:
        evidence = fields[name].evidence
        assert evidence.source_text
        assert evidence.page_number == 1
        assert evidence.bounding_box is not None
        assert evidence.metadata["evidence_box"] is not None
        assert fields[name].model_version == "v3"


def test_uncertain_date_preserves_raw_ocr_and_remains_unresolved() -> None:
    document = _document_from_lines(["Date: September 21, 11/6"])
    fields = deterministic_schema_extractor.extract(document)
    date = next(field for field in fields if field.name == "date")

    assert date.value == "September 21, 11/6"
    assert date.raw_value == "September 21, 11/6"
    assert date.normalized_value is None
    assert date.validation_status == "ambiguous"
    assert date.evidence.metadata["raw_ocr"] == "Date: September 21, 11/6"
    assert date.evidence.metadata["corrected_ocr"] is None

    completeness = vision_schema_registry.summarize(
        document.document_type,
        fields,
        extraction_diagnostics=document.metadata["extraction_diagnostics"],
    )
    assert "date" in completeness.missing_required_fields


def test_candidate_resolver_prevents_incompatible_fields_sharing_evidence() -> None:
    box = VisionBoundingBox(x1=10, y1=10, x2=210, y2=35)
    word = OCRWord(
        word_id="word-shared-evidence",
        text="Shared Evidence",
        confidence=0.95,
        polygon=[[10, 10], [210, 10], [210, 35], [10, 35]],
        bounding_box=box,
        page=1,
        line_id="line-shared-evidence",
        block_id="block-shared-evidence",
        reading_order=1,
    )
    required = SchemaFieldSpec(
        name="date",
        display_name="Date",
        aliases=("date",),
        required=True,
    )
    incompatible = SchemaFieldSpec(
        name="document_title",
        display_name="Document Title",
        aliases=("document title",),
    )
    primary = deterministic_schema_extractor._candidate_from_boxes(
        required,
        "Shared Evidence",
        "Shared Evidence",
        1,
        box,
        box,
        (word,),
        "same_line_label_value",
        0.95,
    )
    duplicate = deterministic_schema_extractor._candidate_from_boxes(
        incompatible,
        "Shared Evidence",
        "Shared Evidence",
        1,
        box,
        box,
        (word,),
        "same_line_label_value",
        0.95,
    )

    retained, rejected = FieldCandidateResolver().resolve(
        [duplicate, primary],
        score=deterministic_schema_extractor._candidate_score,
        conflicts=deterministic_schema_extractor._evidence_conflicts,
    )

    assert [candidate.spec.name for candidate in retained] == ["date"]
    assert rejected[0]["field"] == "document_title"
    assert rejected[0]["reason"] == "duplicate_evidence_conflict"
    assert rejected[0]["conflicts_with"] == "date"


def test_review_counts_share_one_internally_consistent_contract(
    extracted_fixture: VisionDocumentRecord,
) -> None:
    completeness = vision_schema_registry.summarize(
        extracted_fixture.document_type,
        extracted_fixture.extracted_fields,
        extraction_diagnostics={
            **extracted_fixture.metadata["extraction_diagnostics"],
            "duplicates_rejected": 1,
        },
    )

    assert completeness.required_expected_count == 6
    assert completeness.required_matched_count == 6
    assert completeness.required_missing_count == 0
    assert completeness.optional_extracted_count == len(completeness.optional_extracted_fields)
    assert completeness.optional_extracted_count == 1
    assert completeness.total_extracted_count == len(extracted_fixture.extracted_fields)
    assert completeness.total_extracted_count == (
        completeness.required_matched_count + completeness.optional_extracted_count
    )
    assert completeness.duplicates_rejected_count == 1
    assert completeness.disputed_count == 0
    assert completeness.required_expected_count == (
        completeness.required_matched_count + completeness.required_missing_count
    )
    assert completeness.expected_required_count == completeness.required_expected_count
    assert completeness.extracted_count == completeness.total_extracted_count


def _document_from_lines(lines: list[str]) -> VisionDocumentRecord:
    words: list[OCRWord] = []
    reading_order = 1
    for line_index, line in enumerate(lines, start=1):
        cursor = 10.0
        for token in re.findall(r"\S+", line):
            width = max(12.0, len(token) * 8.0)
            box = VisionBoundingBox(
                x1=cursor,
                y1=float(line_index * 30),
                x2=cursor + width,
                y2=float(line_index * 30 + 18),
            )
            words.append(
                OCRWord(
                    text=token,
                    confidence=0.45,
                    polygon=[
                        [box.x1, box.y1],
                        [box.x2, box.y1],
                        [box.x2, box.y2],
                        [box.x1, box.y2],
                    ],
                    bounding_box=box,
                    page=1,
                    line_id=f"line-{line_index}",
                    block_id="block-1",
                    reading_order=reading_order,
                )
            )
            cursor += width + 8
            reading_order += 1
    return VisionDocumentRecord(
        document_id="vision-doc-uncertain-date",
        case_id="vision-case-uncertain-date",
        filename="uncertain-date.txt",
        file_hash="a" * 64,
        document_type=VisionDocumentType.business_form,
        ocr_words=words,
        created_at="2026-08-06T00:00:00+00:00",
        updated_at="2026-08-06T00:00:00+00:00",
    )
