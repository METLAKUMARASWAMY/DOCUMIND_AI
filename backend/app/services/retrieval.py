from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.models.schemas import Evidence, EvidenceChunk, QueryCitation
from app.services.embedding import LocalEmbeddingService


@dataclass
class IndexedChunk:
    case_id: str
    document_id: str
    source_document: str
    page_number: int
    chunk_id: str
    text: str
    vector: Counter
    confidence: float
    extraction_method: str
    evidence_id: str | None = None
    linked_findings: list[str] = field(default_factory=list)


class VectorIntelligenceLayer:
    """Hybrid lexical/vector retrieval with Chroma/FAISS/pgvector-ready boundaries."""

    def __init__(self, embedding_service: LocalEmbeddingService | None = None) -> None:
        self.embedding = embedding_service or LocalEmbeddingService()
        self._chunks: list[IndexedChunk] = []

    def clear_case(self, case_id: str) -> None:
        self._chunks = [chunk for chunk in self._chunks if chunk.case_id != case_id]

    def index_chunks(self, chunks: list[EvidenceChunk]) -> None:
        for chunk in chunks:
            self._chunks.append(
                IndexedChunk(
                    case_id=chunk.case_id,
                    document_id=chunk.document_id,
                    source_document=chunk.source_document,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    vector=self.embedding.embed(chunk.text),
                    confidence=chunk.confidence,
                    extraction_method=chunk.extraction_method,
                    linked_findings=list(chunk.linked_findings),
                )
            )

    def index_evidence(self, case_id: str, evidence: list[Evidence]) -> None:
        for item in evidence:
            chunk_id = item.chunk_id or f"evchunk-{item.evidence_id}"
            evidence_text = " ".join(
                part
                for part in [
                    item.reasoning,
                    item.text or item.snippet,
                    " ".join(item.linked_findings),
                ]
                if part
            )
            self._chunks.append(
                IndexedChunk(
                    case_id=case_id,
                    document_id=item.document_id,
                    source_document=item.source_document,
                    page_number=item.page_number,
                    chunk_id=chunk_id,
                    text=evidence_text,
                    vector=self.embedding.embed(evidence_text),
                    confidence=item.confidence,
                    extraction_method=item.extraction_method,
                    evidence_id=item.evidence_id,
                    linked_findings=list(item.linked_findings),
                )
            )

    def search(self, question: str, case_id: str | None = None, top_k: int = 5) -> list[QueryCitation]:
        query_vector = self.embedding.embed(question)
        scored = []
        for chunk in self._chunks:
            if case_id and chunk.case_id != case_id:
                continue
            vector_score = self.embedding.cosine(query_vector, chunk.vector)
            keyword_score = self.embedding.keyword_overlap(question, chunk.text)
            finding_boost = 0.32 if chunk.linked_findings else 0.0
            if chunk.linked_findings and any(term in question.lower() for term in ("exception", "review", "blocked")):
                finding_boost += 0.18
            score = vector_score * 0.72 + keyword_score * 0.38 + finding_boost
            if vector_score < 0.08 and keyword_score >= 0.18:
                score += 0.18
            if score > 0:
                scored.append((score, vector_score, keyword_score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        reranked = self._rerank(question, scored)
        return [
            QueryCitation(
                document_id=chunk.document_id,
                source_document=chunk.source_document,
                page_number=chunk.page_number,
                chunk_id=chunk.chunk_id,
                evidence_id=chunk.evidence_id,
                evidence_text=chunk.text,
                snippet=chunk.text[:420],
                score=round(min(score, 1.0), 4),
                confidence=round(min(0.98, chunk.confidence * 0.55 + min(score, 1.0) * 0.45), 3),
                retrieval_method="hybrid_vector_keyword",
                linked_findings=chunk.linked_findings,
            )
            for score, _, _, chunk in reranked[:top_k]
        ]

    def facets(self) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for chunk in self._chunks:
            counts[chunk.case_id] += 1
        return dict(counts)

    def _rerank(self, question: str, scored: list[tuple]) -> list[tuple]:
        lower_question = question.lower()
        priority_terms = (
            "exception",
            "review",
            "indemnity",
            "payment",
            "override",
            "anomaly",
            "missing",
            "fraud",
            "compliance",
        )
        reranked = []
        for score, vector_score, keyword_score, chunk in scored:
            text = chunk.text.lower()
            priority_hits = sum(1 for term in priority_terms if term in lower_question and term in text)
            reranked.append((score + priority_hits * 0.045, vector_score, keyword_score, chunk))
        reranked.sort(key=lambda item: item[0], reverse=True)
        return reranked
