from typing import ClassVar

from app.models.schemas import (
    AgentDecision,
    ComplianceFinding,
    FraudSignal,
    RiskLevel,
    WorkflowAction,
    WorkflowDecisionRequest,
    WorkflowState,
)


class WorkflowEngine:
    all_actions: ClassVar[list[WorkflowAction]] = [
        WorkflowAction.approve,
        WorkflowAction.reject,
        WorkflowAction.request_changes,
        WorkflowAction.escalate_to_legal,
        WorkflowAction.escalate_to_finance,
        WorkflowAction.escalate_to_compliance,
        WorkflowAction.hold_payment,
    ]

    def assign(
        self,
        case_id: str,
        risk_score: int,
        fraud_probability: float,
        compliance_score: int,
        compliance_findings: list[ComplianceFinding] | None = None,
        fraud_signals: list[FraudSignal] | None = None,
        agent_decisions: list[AgentDecision] | None = None,
        evidence_count: int = 0,
    ) -> WorkflowState:
        compliance_findings = compliance_findings or []
        fraud_signals = fraud_signals or []
        agent_decisions = agent_decisions or []
        blocked_reasons = self.blocked_reasons(
            risk_score,
            fraud_probability,
            compliance_findings,
            fraud_signals,
            agent_decisions,
            evidence_count,
        )
        approval_blocked = bool(blocked_reasons)

        if any(signal.severity in {RiskLevel.high, RiskLevel.critical} for signal in fraud_signals) or fraud_probability >= 0.55:
            return WorkflowState(
                case_id=case_id,
                status="payment_hold",
                assigned_team="Risk Operations",
                next_action="Hold payment and complete fraud/risk review",
                recommended_action=WorkflowAction.hold_payment,
                allowed_actions=self._allowed(approval_blocked),
                approval_required=True,
                approval_blocked=approval_blocked,
                blocked_reasons=blocked_reasons,
            )
        if any(finding.severity in {RiskLevel.high, RiskLevel.critical} for finding in compliance_findings):
            return WorkflowState(
                case_id=case_id,
                status="compliance_review",
                assigned_team="Compliance",
                next_action="Resolve approval-blocking compliance findings",
                recommended_action=WorkflowAction.request_changes,
                allowed_actions=self._allowed(approval_blocked),
                approval_required=True,
                approval_blocked=approval_blocked,
                blocked_reasons=blocked_reasons,
            )
        if self._legal_review_required(agent_decisions):
            return WorkflowState(
                case_id=case_id,
                status="legal_review",
                assigned_team="Legal",
                next_action="Complete legal review before business approval",
                recommended_action=WorkflowAction.escalate_to_legal,
                allowed_actions=self._allowed(approval_blocked),
                approval_required=True,
                approval_blocked=approval_blocked,
                blocked_reasons=blocked_reasons,
            )
        if risk_score >= 55 or compliance_score < 85:
            return WorkflowState(
                case_id=case_id,
                status="changes_requested",
                assigned_team="Business Owner",
                next_action="Request changes and attach missing support",
                recommended_action=WorkflowAction.request_changes,
                allowed_actions=self._allowed(approval_blocked),
                approval_required=True,
                approval_blocked=approval_blocked,
                blocked_reasons=blocked_reasons,
            )
        return WorkflowState(
            case_id=case_id,
            status="ready_for_approval",
            assigned_team="Business Owner",
            next_action="Approve or request changes",
            recommended_action=WorkflowAction.approve,
            allowed_actions=self._allowed(False),
            approval_required=True,
            approval_blocked=False,
            blocked_reasons=[],
        )

    def record_decision(
        self,
        request: WorkflowDecisionRequest,
        workflow: WorkflowState | None = None,
    ) -> dict:
        if workflow and workflow.approval_blocked and request.decision == WorkflowAction.approve:
            return {
                "case_id": request.case_id,
                "approver": request.approver,
                "decision": request.decision,
                "comment": request.comment,
                "status": "blocked",
                "blocked_reasons": workflow.blocked_reasons,
            }
        return {
            "case_id": request.case_id,
            "approver": request.approver,
            "decision": request.decision,
            "comment": request.comment,
            "status": "recorded",
        }

    def blocked_reasons(
        self,
        risk_score: int,
        fraud_probability: float,
        compliance_findings: list[ComplianceFinding],
        fraud_signals: list[FraudSignal],
        agent_decisions: list[AgentDecision],
        evidence_count: int,
    ) -> list[str]:
        reasons = []
        if any(finding.approval_blocking for finding in compliance_findings):
            reasons.append("High or critical compliance finding is open.")
        if fraud_probability >= 0.55:
            reasons.append("Fraud probability exceeds approval threshold.")
        if risk_score >= 75:
            reasons.append("Risk score is critical.")
        if any(signal.severity in {RiskLevel.high, RiskLevel.critical} for signal in fraud_signals):
            reasons.append("High or critical fraud signal requires investigation.")
        if evidence_count == 0:
            reasons.append("Required evidence is missing.")
        if self._legal_review_required(agent_decisions):
            reasons.append("Legal review is required.")
        return reasons

    def _allowed(self, approval_blocked: bool) -> list[WorkflowAction]:
        if approval_blocked:
            return [
                WorkflowAction.request_changes,
                WorkflowAction.escalate_to_legal,
                WorkflowAction.escalate_to_finance,
                WorkflowAction.escalate_to_compliance,
                WorkflowAction.hold_payment,
                WorkflowAction.reject,
            ]
        return self.all_actions

    def _legal_review_required(self, decisions: list[AgentDecision]) -> bool:
        return any(
            decision.agent_name == "Legal Agent" and "required" in decision.decision.lower()
            for decision in decisions
        )
