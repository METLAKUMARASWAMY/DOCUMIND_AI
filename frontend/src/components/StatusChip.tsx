import { Chip } from "@mui/material";

type Tone = "default" | "primary" | "secondary" | "error" | "info" | "success" | "warning";

/**
 * Status vocabulary shared with the backend.
 *
 * The backend emits several vocabularies through this one component: document
 * processing status (UPPER_SNAKE), reviewer status, review priority, risk
 * severity, model runtime status, smoke-test status, report status and answer
 * status (all lower_snake). Previously only 20 keys were mapped, so `error`,
 * `failed`, `unhealthy` and `blocked` fell through to neutral grey — a failed
 * model smoke test looked exactly like one that had never run.
 *
 * Lookup is case-insensitive so a casing difference can never silently turn a
 * failure into a neutral chip.
 */
const TONES: Record<string, Tone> = {
  // --- document processing lifecycle -----------------------------------
  uploaded: "info",
  queued: "info",
  preprocessing: "info",
  ocr_completed: "info",
  extraction_completed: "info",
  validation_completed: "info",
  indexed: "info",
  processing: "info",
  review_required: "warning",
  under_review: "warning",
  approved: "success",
  completed: "success",
  rejected: "error",
  confirmed_error: "error",
  confirmed_fraud: "error",
  failed: "error",

  // --- reviewer status --------------------------------------------------
  unverified: "warning",
  accepted: "success",
  edited: "info",

  // --- severity / priority ----------------------------------------------
  critical: "error",
  high: "warning",
  medium: "info",
  low: "success",
  info: "info",
  urgent: "error",
  normal: "info",

  // --- model runtime / infrastructure status ----------------------------
  active: "success",
  healthy: "success",
  ready_for_connection: "info",
  degraded: "warning",
  adapter_pending: "warning",
  misconfigured: "warning",
  stale: "warning",
  unavailable: "error",
  unhealthy: "error",
  error: "error",
  blocked: "error",
  disabled: "default",
  unknown: "default",
  not_run: "default",
  not_available: "default",
  optional: "default",

  // --- smoke test / validation results ----------------------------------
  passed: "success",
  valid: "success",
  skipped: "default",
  invalid: "error",

  // --- answer status ----------------------------------------------------
  supported: "success",
  partially_supported: "warning",
  clarification_needed: "warning",
  not_found: "default",
  abstained: "warning",

  // --- report status ----------------------------------------------------
  generated: "success",
  archived: "default",
  superseded: "default",
};

/** Values rendered as a solid chip because they are terminal outcomes. */
const FILLED = new Set([
  "approved",
  "completed",
  "active",
  "healthy",
  "passed",
  "rejected",
  "failed",
  "error",
  "unhealthy",
  "blocked",
  "confirmed_error",
  "confirmed_fraud",
  "critical",
]);

export function toneFor(value: string): Tone {
  return TONES[value.trim().toLowerCase()] ?? "default";
}

export function StatusChip({
  value,
  size = "small",
}: {
  value: string;
  size?: "small" | "medium";
}) {
  const key = value.trim().toLowerCase();
  const tone = toneFor(value);
  return (
    <Chip
      label={value.replaceAll("_", " ")}
      color={tone}
      size={size}
      variant={FILLED.has(key) ? "filled" : "outlined"}
    />
  );
}
