import type { ApprovalPolicy, ApprovalPolicyBlocker } from "../types/api";

interface ApprovalControlInput {
  policy?: ApprovalPolicy;
  comment: string;
  selectedEvidenceIds: string[];
  manualVerification: boolean;
  approvalReason: string;
  requiredFieldValuesComplete: boolean;
}

interface ApprovalControlResult {
  ready: boolean;
  hardBlockers: ApprovalPolicyBlocker[];
}

export function reviewApprovalControl(input: ApprovalControlInput): ApprovalControlResult {
  const { policy } = input;
  if (!policy) return { ready: false, hardBlockers: [] };
  const hardBlockers = policy.observed_conditions.filter((item) => !item.overrideable);
  if (hardBlockers.length) return { ready: false, hardBlockers };

  const commentReady = input.comment.trim().length >= 10;
  if (policy.manual_override_required) {
    return {
      hardBlockers,
      ready: Boolean(
        policy.manual_override_available
        && input.manualVerification
        && input.approvalReason.trim().length >= 10
        && commentReady
        && input.requiredFieldValuesComplete
      ),
    };
  }

  const reviewed = policy.required_evidence_ids.every((id) =>
    input.selectedEvidenceIds.includes(id));
  return { hardBlockers, ready: commentReady && reviewed };
}
