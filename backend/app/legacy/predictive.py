from app.models.schemas import ComplianceFinding, FraudSignal, RiskLevel


class PredictiveIntelligenceService:
    def score_risk(self, compliance_score: int, fraud_signals: list[FraudSignal], findings: list[ComplianceFinding]) -> int:
        severity_weight = {
            RiskLevel.low: 4,
            RiskLevel.medium: 9,
            RiskLevel.high: 17,
            RiskLevel.critical: 28,
        }
        weighted_findings = sum(max(severity_weight[finding.severity], finding.score_impact) for finding in findings)
        weighted_fraud = sum(max(severity_weight[signal.severity], signal.score_impact) * signal.probability for signal in fraud_signals)
        compliance_penalty = max(0, 100 - compliance_score) * 0.45
        return min(100, round(weighted_findings + weighted_fraud + compliance_penalty))

    def predict_approval_delay(self, risk_score: int, open_findings: int) -> float:
        return min(0.95, round(0.12 + risk_score / 140 + open_findings * 0.05, 3))
