from app.models.schemas import ComplianceFinding, Evidence, FraudSignal, RiskLevel, WorkflowAction
from app.services.scoring import ScoringService


def _evidence() -> Evidence:
    return Evidence(document_id="doc-1", source_document="contract.txt", text="evidence", confidence=0.8)


def test_compliance_score_calculation():
    scoring = ScoringService()
    findings = [
        ComplianceFinding(
            finding_id="c1",
            title="Missing indemnity requirement",
            severity=RiskLevel.high,
            framework="custom",
            clause_or_policy="indemnity",
            evidence=[_evidence()],
            recommendation="Add clause.",
            approval_blocking=True,
        ),
        ComplianceFinding(
            finding_id="c2",
            title="Missing audit rights",
            severity=RiskLevel.medium,
            framework="SOC2",
            clause_or_policy="audit rights",
            evidence=[_evidence()],
            recommendation="Add audit rights.",
        ),
    ]

    assert scoring.compliance_score(findings) == 72


def test_fraud_risk_and_approval_gate_calculation():
    scoring = ScoringService()
    signals = [
        FraudSignal(
            signal_id="f1",
            title="Vendor bank account change request",
            severity=RiskLevel.high,
            probability=0.3,
            evidence=[_evidence()],
            recommendation="Verify.",
            score_impact=30,
        ),
        FraudSignal(
            signal_id="f2",
            title="Manual override without approval",
            severity=RiskLevel.high,
            probability=0.25,
            evidence=[_evidence()],
            recommendation="Hold.",
            score_impact=25,
        ),
    ]

    fraud_score = scoring.fraud_probability_score(signals)
    risk_score = scoring.risk_score(82, fraud_score, WorkflowAction.hold_payment)
    blocked, reasons = scoring.approval_blocked([], fraud_score, risk_score, WorkflowAction.hold_payment)

    assert fraud_score == 55
    assert risk_score == 100
    assert scoring.risk_severity(risk_score) == "critical"
    assert blocked is True
    assert reasons
