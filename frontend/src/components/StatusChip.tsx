import { Chip } from "@mui/material";

const tones: Record<string, "default" | "primary" | "secondary" | "error" | "info" | "success" | "warning"> = {
  APPROVED: "success",
  COMPLETED: "success",
  INDEXED: "info",
  REVIEW_REQUIRED: "warning",
  UNDER_REVIEW: "warning",
  REJECTED: "error",
  CONFIRMED_ERROR: "error",
  CONFIRMED_FRAUD: "error",
  critical: "error",
  high: "warning",
  medium: "info",
  low: "success",
  active: "success",
  degraded: "warning",
  unavailable: "error",
  adapter_pending: "warning",
  disabled: "default",
  supported: "success",
  not_found: "default",
};

export function StatusChip({ value, size = "small" }: { value: string; size?: "small" | "medium" }) {
  return (
    <Chip
      label={value.replaceAll("_", " ")}
      color={tones[value] || "default"}
      size={size}
      variant={value === "APPROVED" || value === "COMPLETED" ? "filled" : "outlined"}
    />
  );
}
