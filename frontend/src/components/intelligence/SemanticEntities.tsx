import { Box, Typography } from "@mui/material";

import type { BoundingBox, VisionDocument } from "../../types/api";
import { StatusChip } from "../StatusChip";
import { SectionHeading } from "./SectionHeading";
import { humanize, maskedSemanticValue, resolveEvidence } from "./values";

interface SemanticEntitiesProps {
  document: VisionDocument;
  onFocus?: (page: number, box: BoundingBox, id: string) => void;
}

/** Reusable typed entities (dates, amounts, accounts) resolved from the page. */
export function SemanticEntities({ document, onFocus }: SemanticEntitiesProps) {
  const entities = document.typed_entities ?? [];

  return (
    <Box className="generic-section">
      <SectionHeading title="Semantic entities" count={entities.length} />
      {!entities.length ? (
        <Typography color="text.secondary">
          No reusable semantic entities were identified.
        </Typography>
      ) : null}
      {entities.map((entity) => {
        const evidence = resolveEvidence(document, entity.evidence_ids);
        return (
          <button
            type="button"
            className="semantic-row"
            key={entity.entity_id}
            disabled={!evidence?.bounding_box}
            onClick={() =>
              evidence?.bounding_box &&
              onFocus?.(evidence.page_number, evidence.bounding_box, entity.entity_id)
            }
          >
            <Box>
              <Typography fontWeight={650}>
                {maskedSemanticValue(
                  entity.semantic_type,
                  entity.normalized_value || entity.raw_value,
                )}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {humanize(entity.semantic_type)} · {Math.round(entity.confidence * 100)}% confidence
              </Typography>
            </Box>
            <StatusChip
              value={
                entity.validator_valid
                  ? entity.review_required
                    ? "review required"
                    : "validated"
                  : "invalid"
              }
            />
          </button>
        );
      })}
    </Box>
  );
}
