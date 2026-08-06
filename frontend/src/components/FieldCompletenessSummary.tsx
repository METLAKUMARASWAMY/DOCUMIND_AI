import { Box, Typography } from "@mui/material";

import type { VisionDocument } from "../types/api";

export function FieldCompletenessSummary({
  completeness,
}: {
  completeness: VisionDocument["field_completeness"];
}) {
  const items = [
    ["Required expected", completeness.required_expected_count],
    ["Required matched", completeness.required_matched_count],
    ["Required missing", completeness.required_missing_count],
    ["Optional extracted", completeness.optional_extracted_count],
    ["Total extracted", completeness.total_extracted_count],
    ["Duplicates rejected", completeness.duplicates_rejected_count],
    ["Disputed", completeness.disputed_count],
  ] as const;

  return (
    <Box className="field-completeness-summary" aria-label="Field completeness">
      {items.map(([label, value]) => (
        <Box className="field-completeness-item" key={label}>
          <Typography variant="caption" color="text.secondary">{label}</Typography>
          <Typography fontWeight={700}>{value ?? "Unavailable"}</Typography>
        </Box>
      ))}
    </Box>
  );
}
