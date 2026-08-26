from dataclasses import dataclass
from typing import Any

from app.legacy.embedding import LocalEmbeddingService
from app.models.schemas import FinanceDocumentType, FinanceWorkflowStatus, RiskLevel, VectorRecord


@dataclass
class FinanceVectorHit:
    record: VectorRecord
    score: float


class FinanceVectorStore:
    """Local Chroma-compatible vector store boundary for finance workflow records."""

    def __init__(self, embedding_service: LocalEmbeddingService | None = None) -> None:
        self.embedding = embedding_service or LocalEmbeddingService()
        self._records: list[VectorRecord] = []
        self._vectors: dict[str, Any] = {}

    def upsert(self, record: VectorRecord) -> VectorRecord:
        self._records = [item for item in self._records if item.record_id != record.record_id]
        self._records.append(record)
        self._vectors[record.record_id] = self.embedding.embed(record.text)
        return record

    def upsert_many(self, records: list[VectorRecord]) -> list[VectorRecord]:
        return [self.upsert(record) for record in records]

    def document_records(self, document_id: str) -> list[VectorRecord]:
        return [record for record in self._records if record.document_id == document_id]

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: str | None = None,
        source_type: str | None = None,
        risk_level: RiskLevel | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = self.embedding.embed(query)
        hits: list[FinanceVectorHit] = []
        for record in self._records:
            if document_id and record.document_id != document_id:
                continue
            if source_type and record.source_type != source_type:
                continue
            if risk_level and record.risk_level != risk_level:
                continue
            vector = self._vectors.get(record.record_id) or self.embedding.embed(record.text)
            vector_score = self.embedding.cosine(query_vector, vector)
            keyword_score = self.embedding.keyword_overlap(query, record.text)
            metadata_boost = self._metadata_boost(query, record)
            score = vector_score * 0.68 + keyword_score * 0.42 + metadata_boost
            if score > 0:
                hits.append(FinanceVectorHit(record=record, score=round(min(score, 1.0), 4)))
        hits.sort(key=lambda item: item.score, reverse=True)
        return [
            {
                **hit.record.model_dump(mode="json"),
                "score": hit.score,
                "retrieval_method": "local_chroma_compatible_hybrid",
            }
            for hit in hits[:top_k]
        ]

    def similar_risk_cases(self, risk_level: RiskLevel | None = None, top_k: int = 8) -> list[dict[str, Any]]:
        grouped: dict[str, dict[str, Any]] = {}
        for record in self._records:
            if risk_level and record.risk_level != risk_level:
                continue
            if record.source_type not in {"risk_flag", "human_review_decision", "finance_narrative"}:
                continue
            current = grouped.setdefault(
                record.document_id,
                {
                    "document_id": record.document_id,
                    "document_type": record.document_type.value,
                    "file_name": record.file_name,
                    "risk_level": record.risk_level.value,
                    "review_status": record.review_status.value,
                    "signals": [],
                    "record_count": 0,
                },
            )
            current["signals"].append(record.text[:260])
            current["record_count"] += 1
        results = sorted(grouped.values(), key=lambda item: item["record_count"], reverse=True)
        return results[:top_k]

    def facets(self) -> dict[str, Any]:
        source_counts: dict[str, int] = {}
        document_counts: dict[str, int] = {}
        for record in self._records:
            source_counts[record.source_type] = source_counts.get(record.source_type, 0) + 1
            document_counts[record.document_id] = document_counts.get(record.document_id, 0) + 1
        return {
            "record_count": len(self._records),
            "source_type_counts": source_counts,
            "document_counts": document_counts,
            "backend": "local_chroma_compatible",
            "embedding_model": self.embedding.model_name,
        }

    def _metadata_boost(self, query: str, record: VectorRecord) -> float:
        lower = query.lower()
        boost = 0.0
        if record.document_type != FinanceDocumentType.unknown and record.document_type.value.replace("_", " ") in lower:
            boost += 0.08
        if record.source_type.replace("_", " ") in lower:
            boost += 0.08
        if record.risk_level in {RiskLevel.high, RiskLevel.critical} and any(
            term in lower for term in ("risk", "fraud", "bank", "review", "escalate")
        ):
            boost += 0.1
        if record.review_status in {FinanceWorkflowStatus.pending_review, FinanceWorkflowStatus.escalated} and "review" in lower:
            boost += 0.08
        return boost
