import { describe, expect, it } from "vitest";

import type { ApprovalPolicy } from "../types/api";
import { reviewApprovalControl, secondReviewerReady } from "./reviewPolicy";

const basePolicy: ApprovalPolicy = {
  allowed: false,
  mode: "blocked",
  blockers: [],
  observed_conditions: [],
  manual_override_available: true,
  manual_override_required: false,
  required_confirmed_fields: [],
  required_evidence_ids: ["evidence-1"],
  confidence_threshold: 0.7,
  message: "Evidence required",
};

describe("review approval controls", () => {
  it("stays disabled until ordinary evidence and comment requirements are met", () => {
    const incomplete = reviewApprovalControl({
      policy: basePolicy,
      comment: "Checked source evidence.",
      selectedEvidenceIds: [],
      manualVerification: false,
      approvalReason: "",
      requiredFieldValuesComplete: true,
    });
    const complete = reviewApprovalControl({
      policy: basePolicy,
      comment: "Checked source evidence.",
      selectedEvidenceIds: ["evidence-1"],
      manualVerification: false,
      approvalReason: "",
      requiredFieldValuesComplete: true,
    });

    expect(incomplete.ready).toBe(false);
    expect(complete.ready).toBe(true);
  });

  it("requires the complete manual verification contract", () => {
    const policy = { ...basePolicy, manual_override_required: true };
    const incomplete = reviewApprovalControl({
      policy,
      comment: "Manual source check completed.",
      selectedEvidenceIds: [],
      manualVerification: true,
      approvalReason: "Recorded reviewer justification.",
      requiredFieldValuesComplete: false,
    });
    const complete = reviewApprovalControl({
      policy,
      comment: "Manual source check completed.",
      selectedEvidenceIds: [],
      manualVerification: true,
      approvalReason: "Recorded reviewer justification.",
      requiredFieldValuesComplete: true,
    });

    expect(incomplete.ready).toBe(false);
    expect(complete.ready).toBe(true);
  });

  it("never enables approval while a hard blocker remains", () => {
    const result = reviewApprovalControl({
      policy: {
        ...basePolicy,
        observed_conditions: [{
          code: "bank_account_mismatch",
          message: "Resolve mismatch",
          overrideable: false,
          affected_fields: [],
          evidence_ids: [],
        }],
      },
      comment: "Checked source evidence.",
      selectedEvidenceIds: ["evidence-1"],
      manualVerification: false,
      approvalReason: "",
      requiredFieldValuesComplete: true,
    });

    expect(result.ready).toBe(false);
    expect(result.hardBlockers[0].code).toBe("bank_account_mismatch");
  });
});

describe("second reviewer controls", () => {
  it("requires a valid, approved reviewer distinct from the current actor", () => {
    expect(secondReviewerReady("second@example.com", "actor@example.com", true)).toBe(true);
    expect(secondReviewerReady("actor@example.com", "ACTOR@example.com", true)).toBe(false);
    expect(secondReviewerReady("not-an-email", "actor@example.com", true)).toBe(false);
    expect(secondReviewerReady("second@example.com", "actor@example.com", false)).toBe(false);
  });
});
