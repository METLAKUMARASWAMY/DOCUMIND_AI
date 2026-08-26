import { describe, expect, it } from "vitest";

import { toneFor } from "./StatusChip";

/**
 * Regression tests for the status-vocabulary drift found in the 2026 audit.
 *
 * The backend feeds nine distinct status vocabularies through StatusChip. Only
 * 20 keys were mapped, so `error`, `failed`, `unhealthy` and `blocked` all
 * rendered in the same neutral grey as `not_run` — a failed model smoke test was
 * visually indistinguishable from one that had never been run. That is a UI
 * honesty defect, not a cosmetic one.
 */

// Every status string the backend can emit into a StatusChip, by source.
const BACKEND_VOCABULARY = {
  // VisionProcessingStatus (app/models/vision.py)
  processing: [
    "UPLOADED",
    "QUEUED",
    "PREPROCESSING",
    "OCR_COMPLETED",
    "EXTRACTION_COMPLETED",
    "VALIDATION_COMPLETED",
    "INDEXED",
    "REVIEW_REQUIRED",
    "UNDER_REVIEW",
    "APPROVED",
    "REJECTED",
    "CONFIRMED_ERROR",
    "CONFIRMED_FRAUD",
    "COMPLETED",
  ],
  // ReviewerStatus
  reviewer: ["unverified", "accepted", "edited", "rejected", "confirmed_error", "confirmed_fraud"],
  // VisionRiskSeverity
  severity: ["info", "low", "medium", "high", "critical"],
  // model_registry runtime + health + smoke statuses
  model: [
    "active",
    "adapter_pending",
    "degraded",
    "disabled",
    "error",
    "failed",
    "healthy",
    "misconfigured",
    "not_run",
    "passed",
    "unavailable",
    "unhealthy",
    "unknown",
  ],
  // VisionAskResponse.answer_status
  answer: ["supported", "partially_supported", "clarification_needed", "not_found"],
} as const;

const ALL_VALUES = Object.values(BACKEND_VOCABULARY).flat();

/** Statuses that mean something went wrong. None may render neutral. */
const FAILURE_VALUES = [
  "REJECTED",
  "CONFIRMED_ERROR",
  "CONFIRMED_FRAUD",
  "rejected",
  "confirmed_error",
  "confirmed_fraud",
  "error",
  "failed",
  "unhealthy",
  "unavailable",
  "critical",
  "blocked",
];

/** Statuses that mean work is under way. None may render neutral. */
const IN_FLIGHT_VALUES = [
  "UPLOADED",
  "QUEUED",
  "PREPROCESSING",
  "OCR_COMPLETED",
  "EXTRACTION_COMPLETED",
  "VALIDATION_COMPLETED",
  "INDEXED",
];

describe("StatusChip tone mapping", () => {
  it("maps every status the backend can emit", () => {
    const unmapped = ALL_VALUES.filter((value) => toneFor(value) === "default");
    // Only genuinely neutral states may be grey.
    const allowedNeutral = ["disabled", "unknown", "not_run", "not_found"];
    const unexpected = unmapped.filter((value) => !allowedNeutral.includes(value));
    expect(unexpected).toEqual([]);
  });

  it.each(FAILURE_VALUES)("renders %s as a failure tone, never neutral", (value) => {
    expect(toneFor(value)).toBe("error");
  });

  it.each(IN_FLIGHT_VALUES)("renders %s as in-progress, never neutral", (value) => {
    expect(toneFor(value)).toBe("info");
  });

  it("distinguishes a failed smoke test from one that never ran", () => {
    expect(toneFor("failed")).toBe("error");
    expect(toneFor("not_run")).toBe("default");
    expect(toneFor("failed")).not.toBe(toneFor("not_run"));
  });

  it("distinguishes review_required from approved", () => {
    expect(toneFor("REVIEW_REQUIRED")).toBe("warning");
    expect(toneFor("APPROVED")).toBe("success");
  });

  it("is case-insensitive so casing drift cannot neutralise a failure", () => {
    expect(toneFor("FAILED")).toBe(toneFor("failed"));
    expect(toneFor("Approved")).toBe(toneFor("APPROVED"));
    expect(toneFor("  error  ")).toBe("error");
  });

  it("falls back to neutral for genuinely unknown values", () => {
    expect(toneFor("something-nobody-defined")).toBe("default");
  });

  it("flags an abstained answer rather than showing it as success", () => {
    expect(toneFor("clarification_needed")).toBe("warning");
    expect(toneFor("partially_supported")).toBe("warning");
    expect(toneFor("supported")).toBe("success");
  });
});
