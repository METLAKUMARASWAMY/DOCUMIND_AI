import { Alert, Box, Checkbox, FormControlLabel, Typography } from "@mui/material";

import type { BoundingBox, GenericDiscoveredField } from "../../types/api";
import { StatusChip } from "../StatusChip";
import { SectionHeading } from "./SectionHeading";
import { humanize, maskedGenericValue } from "./values";

interface DiscoveredAttributesProps {
  fields: GenericDiscoveredField[];
  onFocus?: (page: number, box: BoundingBox, id: string) => void;
  selectedEvidenceIds: string[];
  onToggleEvidence?: (evidenceId: string) => void;
}

/**
 * Schema-independent label/value observations, each anchored to a box on the
 * page so a reviewer can jump straight to where the value was read.
 */
export function DiscoveredAttributes({
  fields,
  onFocus,
  selectedEvidenceIds,
  onToggleEvidence,
}: DiscoveredAttributesProps) {
  return (
    <Box className="generic-section">
      <SectionHeading title="Discovered attributes" count={fields.length} />
      {!fields.length ? (
        <Alert severity="info">No schema-independent label/value observations were found.</Alert>
      ) : null}
      {fields.map((field) => {
        const box = field.evidence.bounding_box || field.value_bbox;
        return (
          <Box className="generic-field-row" key={field.field_id}>
            <button
              type="button"
              className="generic-field-main"
              onClick={() =>
                onFocus?.(field.evidence.page_number || field.page, box, field.field_id)
              }
            >
              <Box>
                <Typography variant="caption" color="text.secondary">
                  {field.raw_label || humanize(field.normalized_label)}
                </Typography>
                <Typography fontWeight={650} className="field-value">
                  {maskedGenericValue(field)}
                </Typography>
                {field.normalized_value && field.normalized_value !== field.raw_value ? (
                  <Typography variant="caption" color="text.secondary">
                    Normalized: {maskedGenericValue(field, true)}
                  </Typography>
                ) : null}
              </Box>
              <Box className="field-meta">
                <Typography variant="caption">{Math.round(field.confidence * 100)}%</Typography>
                <StatusChip
                  value={
                    field.semantic_type_if_known ||
                    (field.review_required ? "review required" : "untyped")
                  }
                />
              </Box>
            </button>
            {onToggleEvidence ? (
              <FormControlLabel
                className="generic-evidence-control"
                control={
                  <Checkbox
                    size="small"
                    checked={selectedEvidenceIds.includes(field.evidence.evidence_id)}
                    onChange={() => onToggleEvidence(field.evidence.evidence_id)}
                  />
                }
                label="Evidence reviewed"
              />
            ) : null}
          </Box>
        );
      })}
    </Box>
  );
}
