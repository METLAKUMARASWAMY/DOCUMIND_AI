from app.models.schemas import ComplianceFinding, FraudSignal


class FindingRepository:
    def __init__(self) -> None:
        self._compliance: dict[str, list[ComplianceFinding]] = {}
        self._fraud: dict[str, list[FraudSignal]] = {}

    def save_compliance(self, case_id: str, findings: list[ComplianceFinding]) -> list[ComplianceFinding]:
        self._compliance[case_id] = findings
        return findings

    def save_fraud(self, case_id: str, signals: list[FraudSignal]) -> list[FraudSignal]:
        self._fraud[case_id] = signals
        return signals

    def compliance_for_case(self, case_id: str) -> list[ComplianceFinding]:
        return list(self._compliance.get(case_id, []))

    def fraud_for_case(self, case_id: str) -> list[FraudSignal]:
        return list(self._fraud.get(case_id, []))
