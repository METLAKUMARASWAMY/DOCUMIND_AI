import { Box } from "@mui/material";

import type { BoundingBox, VisionDocument } from "../types/api";
import { DiscoveredAttributes } from "./intelligence/DiscoveredAttributes";
import { SchemaProfileSummary } from "./intelligence/SchemaProfileSummary";
import { SemanticEntities } from "./intelligence/SemanticEntities";
import { VerifiedFacts } from "./intelligence/VerifiedFacts";

export { humanize } from "./intelligence/values";

interface GenericIntelligencePanelProps {
  document: VisionDocument;
  onFocus?: (page: number, box: BoundingBox, id: string) => void;
  selectedEvidenceIds?: string[];
  onToggleEvidence?: (evidenceId: string) => void;
}

/**
 * Everything the pipeline learned about a document without being told a schema.
 *
 * This is composition only: each section owns its own rendering, so a change to
 * one no longer means reading past the other three.
 */
export function GenericIntelligencePanel({
  document,
  onFocus,
  selectedEvidenceIds = [],
  onToggleEvidence,
}: GenericIntelligencePanelProps) {
  return (
    <Box className="generic-intelligence">
      <SchemaProfileSummary profile={document.schema_profile} />
      <DiscoveredAttributes
        fields={document.generic_fields ?? []}
        onFocus={onFocus}
        selectedEvidenceIds={selectedEvidenceIds}
        onToggleEvidence={onToggleEvidence}
      />
      <SemanticEntities document={document} onFocus={onFocus} />
      <VerifiedFacts document={document} onFocus={onFocus} />
    </Box>
  );
}
