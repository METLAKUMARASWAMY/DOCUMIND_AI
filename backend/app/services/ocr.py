from app.models.schemas import Evidence


class OCRLayoutService:
    """Adapter boundary for PaddleOCR, EasyOCR, Tesseract, LayoutLMv3, and DocFormer."""

    def analyze_layout(self, document_id: str, filename: str, text: str) -> dict:
        lower = text.lower()
        regions = []
        if "|" in text or "\t" in text:
            regions.append("table")
        if "signature" in lower or "signed" in lower:
            regions.append("signature")
        if "stamp" in lower or "seal" in lower:
            regions.append("stamp")
        if "invoice" in lower or "purchase order" in lower:
            regions.append("form")

        return {
            "document_id": document_id,
            "filename": filename,
            "engine": "local-layout-adapter",
            "regions": regions or ["text_block"],
            "evidence": Evidence(
                document_id=document_id,
                source_document=filename,
                page_number=1,
                snippet=text[:220],
                confidence=0.74,
                reasoning="Detected layout regions from textual and structural cues.",
            ).model_dump(),
        }

