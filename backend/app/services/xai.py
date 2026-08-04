from app.models.schemas import AgentDecision, ComplianceFinding, FraudSignal


class ExplainabilityLayer:
    def package(
        self,
        decisions: list[AgentDecision],
        findings: list[ComplianceFinding],
        fraud_signals: list[FraudSignal],
    ) -> dict:
        return {
            "decision_count": len(decisions),
            "evidence_count": sum(len(decision.evidence) for decision in decisions)
            + sum(len(finding.evidence) for finding in findings)
            + sum(len(signal.evidence) for signal in fraud_signals),
            "confidence_method": "Weighted confidence from parser quality, retrieval similarity, rule coverage, and agent agreement.",
            "reasoning_policy": "Every alert must include source document, page number, evidence snippet, confidence, and recommendation.",
        }

