import re

from app.models.schemas import DocumentRecord, EvidenceChunk

sentence_boundary = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class EvidenceChunkingService:
    def chunk_documents(self, case_id: str, documents: list[DocumentRecord]) -> list[EvidenceChunk]:
        chunks: list[EvidenceChunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(case_id, document))
        return chunks

    def chunk_document(
        self, case_id: str, document: DocumentRecord, window: int = 700
    ) -> list[EvidenceChunk]:
        text = " ".join(document.text.split())
        if not text:
            text = "No extracted text available; OCR review required."
        pieces = self._semantic_windows(text, window)
        return [
            EvidenceChunk(
                case_id=case_id,
                document_id=document.document_id,
                source_document=document.filename,
                page_number=index + 1,
                text=piece,
                extraction_method="parser_chunk",
                confidence=document.metadata.get("parse_confidence", 0.76),
                metadata={"document_type": document.document_type.value},
            )
            for index, piece in enumerate(pieces)
        ]

    def _semantic_windows(self, text: str, window: int) -> list[str]:
        sentences = [item.strip() for item in sentence_boundary.split(text) if item.strip()]
        if not sentences:
            return [text[:window]]
        chunks = []
        current = ""
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= window:
                current = f"{current} {sentence}".strip()
                continue
            if current:
                chunks.append(current)
            current = sentence
        if current:
            chunks.append(current)
        return chunks or [text[:window]]
