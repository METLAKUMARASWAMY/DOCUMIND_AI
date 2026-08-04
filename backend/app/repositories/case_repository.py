from app.models.schemas import AnalysisResponse


class CaseRepository:
    def __init__(self) -> None:
        self._cases: dict[str, AnalysisResponse] = {}

    def save(self, analysis: AnalysisResponse) -> AnalysisResponse:
        self._cases[analysis.case_id] = analysis
        return analysis

    def get(self, case_id: str) -> AnalysisResponse | None:
        return self._cases.get(case_id)

    def list(self) -> list[AnalysisResponse]:
        return list(self._cases.values())
