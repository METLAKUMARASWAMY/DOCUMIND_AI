from __future__ import annotations

from pydantic import BaseModel, Field

from app.models.schemas import UserRole
from app.models.vision import (
    ReviewerStatus,
    VisionDocumentRecord,
    VisionReviewDecisionRequest,
    VisionRiskSeverity,
)
from app.services.vision.schema_registry import vision_schema_registry


class ApprovalPolicyBlocker(BaseModel):
    code: str
    message: str
    overrideable: bool = False
    affected_fields: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class ApprovalPolicyEvaluation(BaseModel):
    allowed: bool = False
    mode: str = "blocked"
    blockers: list[ApprovalPolicyBlocker] = Field(default_factory=list)
    observed_conditions: list[ApprovalPolicyBlocker] = Field(default_factory=list)
    manual_override_available: bool = False
    manual_override_required: bool = False
    required_confirmed_fields: list[str] = Field(default_factory=list)
    required_evidence_ids: list[str] = Field(default_factory=list)
    confidence_threshold: float
    message: str
    fraud_second_reviewer_required: bool = False


class ApprovalPolicyViolation(ValueError):
    def __init__(self, evaluation: ApprovalPolicyEvaluation) -> None:
        self.evaluation = evaluation
        super().__init__(evaluation.message)


class ReviewDecisionValidationError(ValueError):
    pass


class VisionApprovalPolicyService:
    def evaluate(
        self,
        document: VisionDocumentRecord,
        *,
        request: VisionReviewDecisionRequest | None,
        actor_role: UserRole,
        confidence_threshold: float,
    ) -> ApprovalPolicyEvaluation:
        schema = vision_schema_registry.get(document.document_type)
        required_fields = [item.name for item in schema.required_fields] if schema else []
        selected_evidence = set(request.evidence_ids if request else [])
        conditions: list[ApprovalPolicyBlocker] = []

        if not document.extracted_fields:
            conditions.append(
                ApprovalPolicyBlocker(
                    code="no_structured_fields",
                    message="No structured fields were extracted.",
                    overrideable=True,
                    affected_fields=required_fields,
                )
            )
        if document.field_completeness.schema_available:
            missing = document.field_completeness.missing_required_fields
            if missing:
                conditions.append(
                    ApprovalPolicyBlocker(
                        code="required_fields_unresolved",
                        message="Required schema fields remain unresolved.",
                        overrideable=True,
                        affected_fields=missing,
                    )
                )

        confidence = document.confidence_summary.mean_field_confidence
        if confidence is None:
            conditions.append(
                ApprovalPolicyBlocker(
                    code="confidence_unavailable",
                    message="Field confidence is unavailable.",
                    overrideable=True,
                )
            )
        elif confidence < confidence_threshold:
            conditions.append(
                ApprovalPolicyBlocker(
                    code="confidence_below_threshold",
                    message=(
                        f"Mean field confidence is below the configured "
                        f"{confidence_threshold:.0%} approval threshold."
                    ),
                    overrideable=True,
                )
            )

        critical = [
            risk
            for risk in document.risk_indicators
            if risk.severity == VisionRiskSeverity.critical
        ]
        if critical:
            conditions.append(
                ApprovalPolicyBlocker(
                    code="critical_validation_failure",
                    message="A critical validation failure must be resolved before approval.",
                    affected_fields=sorted(
                        {name for risk in critical for name in risk.affected_fields}
                    ),
                    evidence_ids=sorted(
                        {item.evidence_id for risk in critical for item in risk.evidence}
                    ),
                )
            )

        bank_mismatches = [
            risk
            for risk in document.risk_indicators
            if "bank" in risk.title.lower() and "mismatch" in risk.title.lower()
        ]
        if bank_mismatches:
            conditions.append(
                ApprovalPolicyBlocker(
                    code="bank_account_mismatch",
                    message="A bank-account mismatch must be resolved before approval.",
                    affected_fields=sorted(
                        {name for risk in bank_mismatches for name in risk.affected_fields}
                    ),
                    evidence_ids=sorted(
                        {item.evidence_id for risk in bank_mismatches for item in risk.evidence}
                    ),
                )
            )

        critical_duplicates = [
            item
            for item in document.duplicate_candidates
            if item.matched_document_id
            and item.status == "exact_duplicate"
        ]
        if critical_duplicates:
            conditions.append(
                ApprovalPolicyBlocker(
                    code="critical_duplicate_document",
                    message="A critical duplicate candidate must be resolved before approval.",
                    evidence_ids=sorted(
                        {
                            evidence.evidence_id
                            for duplicate in critical_duplicates
                            for evidence in duplicate.evidence
                        }
                    ),
                )
            )

        required_evidence = sorted(
            {
                field.evidence.evidence_id
                for field in document.extracted_fields
                if field.required
                and field.reviewer_status
                not in {ReviewerStatus.accepted, ReviewerStatus.edited}
            }
        )
        missing_evidence = sorted(set(required_evidence) - selected_evidence)
        if missing_evidence:
            conditions.append(
                ApprovalPolicyBlocker(
                    code="mandatory_evidence_not_reviewed",
                    message="Mandatory field evidence has not been selected and reviewed.",
                    overrideable=True,
                    evidence_ids=missing_evidence,
                )
            )

        comment = request.comment.strip() if request else ""
        if len(comment) < 10:
            conditions.append(
                ApprovalPolicyBlocker(
                    code="reviewer_comment_required",
                    message="A reviewer comment of at least 10 characters is required.",
                    overrideable=True,
                )
            )

        if not conditions:
            return ApprovalPolicyEvaluation(
                allowed=True,
                mode="ordinary",
                confidence_threshold=confidence_threshold,
                message="Approval policy requirements are satisfied.",
            )

        manual_requested = bool(request and request.manual_verification_completed)
        hard_blockers = [item for item in conditions if not item.overrideable]
        manual_requirements: list[ApprovalPolicyBlocker] = []
        confirmed_fields = {
            name: value.strip()
            for name, value in (request.confirmed_fields if request else {}).items()
            if value.strip()
        }
        missing_confirmations = sorted(set(required_fields) - set(confirmed_fields))
        if actor_role not in {
            UserRole.admin,
            UserRole.reviewer,
            UserRole.senior_reviewer,
            UserRole.compliance_officer,
        }:
            manual_requirements.append(
                ApprovalPolicyBlocker(
                    code="manual_override_role_required",
                    message="Reviewer or admin role is required for manual approval.",
                )
            )
        if not request or not request.manual_verification_completed:
            manual_requirements.append(
                ApprovalPolicyBlocker(
                    code="manual_verification_confirmation_required",
                    message="Complete manual verification must be explicitly confirmed.",
                )
            )
        if missing_confirmations:
            manual_requirements.append(
                ApprovalPolicyBlocker(
                    code="critical_fields_confirmation_required",
                    message="Every required schema field must be entered or confirmed.",
                    affected_fields=missing_confirmations,
                )
            )
        if not request or len(request.approval_reason.strip()) < 10:
            manual_requirements.append(
                ApprovalPolicyBlocker(
                    code="approval_reason_required",
                    message="A manual approval reason of at least 10 characters is required.",
                )
            )
        if not request or not request.confirmation_acknowledged:
            manual_requirements.append(
                ApprovalPolicyBlocker(
                    code="second_confirmation_required",
                    message="The second approval confirmation must be acknowledged.",
                )
            )

        if manual_requested:
            remaining = [*hard_blockers, *manual_requirements]
            if not remaining:
                return ApprovalPolicyEvaluation(
                    allowed=True,
                    mode="manual_override",
                    observed_conditions=conditions,
                    manual_override_available=True,
                    manual_override_required=True,
                    required_confirmed_fields=required_fields,
                    required_evidence_ids=required_evidence,
                    confidence_threshold=confidence_threshold,
                    message="Manual approval requirements are satisfied.",
                )
            blockers = remaining
        else:
            blockers = conditions

        manual_available = (
            actor_role
            in {
                UserRole.admin,
                UserRole.reviewer,
                UserRole.senior_reviewer,
                UserRole.compliance_officer,
            }
            and not hard_blockers
        )
        manual_condition_codes = {
            "no_structured_fields",
            "required_fields_unresolved",
            "confidence_unavailable",
            "confidence_below_threshold",
        }
        return ApprovalPolicyEvaluation(
            blockers=blockers,
            observed_conditions=conditions,
            manual_override_available=manual_available,
            manual_override_required=any(
                item.code in manual_condition_codes for item in conditions
            ),
            required_confirmed_fields=required_fields,
            required_evidence_ids=required_evidence,
            confidence_threshold=confidence_threshold,
            message=(
                "No structured fields were extracted. Approval requires complete manual "
                "verification and a recorded justification."
                if any(item.code == "no_structured_fields" for item in conditions)
                else blockers[0].message
            ),
        )


vision_approval_policy_service = VisionApprovalPolicyService()
