import { Box, Typography } from "@mui/material";

import type { BoundingBox, VisionDocument } from "../../types/api";
import { StatusChip } from "../StatusChip";
import { SectionHeading } from "./SectionHeading";
import { humanize, maskedSemanticValue, resolveEvidence } from "./values";

interface VerifiedFactsProps {
  document: VisionDocument;
  onFocus?: (page: number, box: BoundingBox, id: string) => void;
}

/** Facts a reviewer has confirmed, carrying who confirmed them. */
export function VerifiedFacts({ document, onFocus }: VerifiedFactsProps) {
  const facts = document.verified_facts ?? [];

  return (
    <Box className="generic-section">
      <SectionHeading title="Verified facts" count={facts.length} />
      {!facts.length ? (
        <Typography color="text.secondary">
          No reviewer-verified facts have been recorded.
        </Typography>
      ) : null}
      {facts.map((fact) => {
        const evidence = resolveEvidence(document, [fact.evidence_id]);
        return (
          <button
            type="button"
            className="semantic-row"
            key={fact.fact_id}
            disabled={!evidence?.bounding_box}
            onClick={() =>
              evidence?.bounding_box &&
              onFocus?.(evidence.page_number, evidence.bounding_box, fact.fact_id)
            }
          >
            <Box>
              <Typography variant="caption" color="text.secondary">
                {humanize(fact.name)}
              </Typography>
              <Typography fontWeight={650}>
                {maskedSemanticValue(fact.semantic_type, fact.normalized_value || fact.value)}
              </Typography>
            </Box>
            <Box className="field-meta">
              <StatusChip value={fact.reviewer_status} />
              {fact.reviewer ? <Typography variant="caption">{fact.reviewer}</Typography> : null}
            </Box>
          </button>
        );
      })}
    </Box>
  );
}
