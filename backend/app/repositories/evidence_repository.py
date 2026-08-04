from app.models.schemas import Evidence, EvidenceChunk


class EvidenceRepository:
    def __init__(self) -> None:
        self._chunks_by_case: dict[str, list[EvidenceChunk]] = {}
        self._evidence_by_case: dict[str, list[Evidence]] = {}

    def save_chunks(self, case_id: str, chunks: list[EvidenceChunk]) -> list[EvidenceChunk]:
        self._chunks_by_case[case_id] = chunks
        return chunks

    def save_evidence(self, case_id: str, evidence: list[Evidence]) -> list[Evidence]:
        self._evidence_by_case[case_id] = evidence
        return evidence

    def chunks_for_case(self, case_id: str) -> list[EvidenceChunk]:
        return list(self._chunks_by_case.get(case_id, []))

    def evidence_for_case(self, case_id: str) -> list[Evidence]:
        return list(self._evidence_by_case.get(case_id, []))
