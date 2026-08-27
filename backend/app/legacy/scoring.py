from app.models.schemas import ComplianceFinding, FraudSignal, RiskLevel, WorkflowAction

COMPLIANCE_PENALTY = {
    RiskLevel.critical: 25,
    RiskLevel.high: 18,
    RiskLevel.medium: 10,
    RiskLevel.low: 5,
}


class ScoringService:
    def compliance_score(self, findings: list[ComplianceFinding]) -> int:
        return max(0, 100 - sum(COMPLIANCE_PENALTY[finding.severity] for finding in findings))

    def fraud_probability_score(self, signals: list[FraudSignal]) -> int:
        score = 0
        titles = " | ".join(signal.title.lower() for signal in signals)
        if "duplicate invoice" in titles or "possible duplicate invoice" in titles:
            score += 35
        if "bank account change" in titles or "new bank" in titles:
            score += 30
        if "manual override" in titles:
            score += 25
        if "amount does not match" in titles or "amount mismatch" in titles:
            score += 25
        if "urgent" in titles or "wire request" in titles:
            score += 15
        if "off-cycle" in titles:
            score += 10
        if "vendor anomaly" in titles:
            score += 20
        return min(95, score)

    def risk_score(
        self,
        compliance_score: int,
        fraud_probability_score: int,
        workflow_action: WorkflowAction,
    ) -> int:
        compliance_penalty = 100 - compliance_score
        workflow_penalty = {
            WorkflowAction.approve: 0,
            WorkflowAction.request_changes: 10,
            WorkflowAction.escalate_to_legal: 18,
            WorkflowAction.escalate_to_finance: 18,
            WorkflowAction.escalate_to_compliance: 18,
            WorkflowAction.hold_payment: 30,
            WorkflowAction.reject: 25,
        }[workflow_action]
        return min(100, compliance_penalty + fraud_probability_score + workflow_penalty)

    def risk_severity(self, risk_score: int) -> str:
        if risk_score >= 75:
            return "critical"
        if risk_score >= 50:
            return "high"
        if risk_score >= 25:
            return "medium"
        return "low"

    def approval_blocked(
        self,
        findings: list[ComplianceFinding],
        fraud_probability_score: int,
        risk_score: int,
        workflow_action: WorkflowAction,
    ) -> tuple[bool, list[str]]:
        reasons = []
        if any(finding.severity == RiskLevel.critical for finding in findings):
            reasons.append("Critical compliance finding is open.")
        if any(
            finding.severity == RiskLevel.high and finding.approval_blocking for finding in findings
        ):
            reasons.append("High approval-blocking finding is open.")
        if fraud_probability_score >= 60:
            reasons.append("Fraud probability is above 60%.")
        if risk_score >= 75:
            reasons.append("Risk score is critical.")
        if workflow_action == WorkflowAction.hold_payment:
            reasons.append("Workflow recommends holding payment.")
        return bool(reasons), reasons
