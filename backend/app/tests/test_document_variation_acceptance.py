from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from app.core.config import settings
from app.models.vision import VisionDocumentRecord
from app.services.vision.document_classification import hybrid_document_classifier
from app.services.vision.generic_field_discovery import generic_field_discovery_engine
from app.services.vision.ocr_adapters import TesseractOCRAdapter
from app.services.vision.ocr_lines import ocr_line_reconstructor
from app.services.vision.preprocessing import DocumentPreprocessor
from app.services.vision.service import VisionAIService


def _font(size: int = 52) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _scan(text: str, *, foreground: int = 20, background: int = 250) -> Image.Image:
    image = Image.new("L", (1400, 620), color=background)
    draw = ImageDraw.Draw(image)
    draw.text((90, 110), "DOCUMENT VARIATION TEST", fill=foreground, font=_font(58))
    draw.text((90, 260), text, fill=foreground, font=_font())
    draw.text((90, 390), "Reviewer Note: source backed", fill=foreground, font=_font(40))
    return image.convert("RGB")


def _encoded(image: Image.Image, format_name: str, **options: object) -> bytes:
    stream = BytesIO()
    image.save(stream, format=format_name, **options)
    return stream.getvalue()


def _ocr(
    tmp_path: Path,
    *,
    filename: str,
    content: bytes,
) -> tuple[VisionDocumentRecord, str]:
    preprocessed = DocumentPreprocessor(tmp_path / "artifacts").process(
        "variation-document",
        filename,
        content,
    )
    document = VisionDocumentRecord(
        document_id="variation-document",
        case_id="variation-case",
        filename=filename,
        file_hash="a" * 64,
        page_artifacts=preprocessed.artifacts,
        quality=preprocessed.quality,
        page_count=len(preprocessed.artifacts),
        created_at="2026-08-13T00:00:00+00:00",
        updated_at="2026-08-13T00:00:00+00:00",
    )
    adapter = TesseractOCRAdapter()
    if adapter.status().status != "active":
        pytest.skip("Tesseract is required for image-variation acceptance")
    document.ocr_words = adapter.extract_words(document, preprocessed.native_words)
    document.ocr_lines = ocr_line_reconstructor.reconstruct(document.ocr_words)
    text = " ".join(word.text for word in document.ocr_words).casefold()
    return document, text


@pytest.mark.parametrize("variant", ["rotated_png", "compressed_low_contrast_jpeg"])
def test_degraded_scans_keep_meaningful_source_backed_output(
    variant: str,
    tmp_path: Path,
) -> None:
    image = _scan("Calibration Token: VARIANT-731")
    if variant == "rotated_png":
        image = image.rotate(
            2.5, resample=Image.Resampling.BICUBIC, expand=False, fillcolor="white"
        )
        filename = "unseen-rotated.png"
        content = _encoded(image, "PNG")
    else:
        image = (
            ImageEnhance.Contrast(image).enhance(0.42).filter(ImageFilter.GaussianBlur(radius=0.7))
        )
        filename = "unseen-low-contrast.jpg"
        content = _encoded(image, "JPEG", quality=32, optimize=True)

    document, text = _ocr(tmp_path, filename=filename, content=content)

    assert "variant" in text
    assert "731" in text
    assert document.metadata["ocr_consensus"][0]["selected_variant"] in {
        "original",
        "processed",
        "grayscale",
        "enhanced",
    }
    assert {"denoised", "shadow_removed", "contrast_enhanced", "adaptive_thresholded"} <= set(
        document.quality[0].preprocessing_steps
    )
    discovered = generic_field_discovery_engine.discover(
        document.ocr_lines,
        document.ocr_words,
        document_type="unknown",
    )
    assert any(
        "calibration" in field.normalized_label
        and "731" in field.raw_value
        and field.evidence_bbox.x2 > field.evidence_bbox.x1
        and field.evidence_bbox.y2 > field.evidence_bbox.y1
        for field in discovered.fields
    )


@pytest.mark.parametrize(("filename", "format_name"), [("scan.tiff", "TIFF"), ("scan.pdf", "PDF")])
def test_scanned_multi_page_tiff_and_pdf_preserve_pages_and_evidence(
    filename: str,
    format_name: str,
    tmp_path: Path,
) -> None:
    first = _scan("Page Token: PAGEALPHA-101")
    second = _scan("Page Token: PAGEBETA-202")
    stream = BytesIO()
    first.save(stream, format=format_name, save_all=True, append_images=[second], resolution=150)

    document, text = _ocr(tmp_path, filename=filename, content=stream.getvalue())

    assert document.page_count == 2
    assert {word.page for word in document.ocr_words} == {1, 2}
    assert "pagealpha" in text and "101" in text
    assert "pagebeta" in text and "202" in text
    assert len(document.metadata["ocr_consensus"]) == 2
    assert all(
        artifact.source_kind in {"uploaded_image", "pdf_render"}
        for artifact in document.page_artifacts
    )


def test_repository_holdout_pdf_is_genuinely_multi_page_and_not_fixture_tuned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (
        Path(__file__).resolve().parents[3]
        / "dataset"
        / "documind_ai_synthetic_dataset"
        / "documents"
        / "case-0003"
        / "Research_papers"
        / "1.pdf"
    )
    assert source.is_file()
    result = DocumentPreprocessor(tmp_path / "holdout-artifacts").process(
        "unseen-research-holdout",
        "renamed-unseen-research.pdf",
        source.read_bytes(),
    )

    assert len(result.artifacts) == 10
    assert all(artifact.source_kind == "pdf_render" for artifact in result.artifacts)
    assert sum(len(text.strip()) for text in result.page_text.values()) > 1000
    assert len({word["page"] for word in result.native_words}) >= 5
    assert all(report.image_quality_score >= 0 for report in result.quality)
    classification = hybrid_document_classifier.classify(
        "\n".join(result.page_text.values()),
        "renamed-unseen-research.pdf",
    )
    assert classification.predicted_type.value == "unknown"
    assert classification.schema_mode == "C"
    assert classification.signals[0].signal_type == "ambiguous_candidate_abstention"

    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "holdout-uploads"))
    monkeypatch.setattr(settings, "enable_neo4j", False)
    monkeypatch.setattr(settings, "enable_qdrant", False)
    service = VisionAIService()
    response = service.queue_documents(
        [("opaque-holdout-7f3a.pdf", source.read_bytes(), {})],
        actor="holdout-verifier",
        process_inline=True,
        tenant_id="tenant-unseen-holdout",
    )
    document = service.get_document(response.document_ids[0], "tenant-unseen-holdout")

    assert document.document_type.value == "unknown"
    assert document.page_count == 10
    assert document.schema_profile.mode == "C"
    assert document.generic_fields
    assert document.typed_entities
    assert document.evidence_chunks
    assert document.graph.nodes and document.graph.edges
    assert document.metadata["extraction"]["errors"] == []
