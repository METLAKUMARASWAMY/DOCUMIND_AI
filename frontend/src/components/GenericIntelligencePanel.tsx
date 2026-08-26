import { Alert, Box, Checkbox, FormControlLabel, Typography } from "@mui/material";

import type { BoundingBox, Evidence, GenericDiscoveredField, VisionDocument } from "../types/api";
import { StatusChip } from "./StatusChip";

interface GenericIntelligencePanelProps {
  document: VisionDocument;
  onFocus?: (page: number, box: BoundingBox, id: string) => void;
  selectedEvidenceIds?: string[];
  onToggleEvidence?: (evidenceId: string) => void;
}

export function GenericIntelligencePanel({
  document,
  onFocus,
  selectedEvidenceIds = [],
  onToggleEvidence,
}: GenericIntelligencePanelProps) {
  const genericFields = document.generic_fields ?? [];
  const typedEntities = document.typed_entities ?? [];
  const verifiedFacts = document.verified_facts ?? [];
  const profile = document.schema_profile;

  return (
    <Box className="generic-intelligence">
      {profile ? (
        <Box className="schema-profile-summary">
          <Box>
            <Typography variant="overline" color="text.secondary">
              Schema profile
            </Typography>
            <Typography variant="h4">
              Mode {profile.mode} · {humanize(profile.schema_status)}
            </Typography>
          </Box>
          <StatusChip value={profile.schema_status} />
          <Typography variant="caption" color="text.secondary">
            {profile.schema_id ? `${profile.schema_id} · ` : ""}
            {profile.schema_version} · selected by {humanize(profile.selection_source)}
          </Typography>
        </Box>
      ) : null}

      <GenericFields
        fields={genericFields}
        onFocus={onFocus}
        selectedEvidenceIds={selectedEvidenceIds}
        onToggleEvidence={onToggleEvidence}
      />

      <Box className="generic-section">
        <Box className="section-heading compact-heading">
          <Typography variant="h4">Semantic entities</Typography>
          <Typography variant="caption">{typedEntities.length}</Typography>
        </Box>
        {!typedEntities.length ? (
          <Typography color="text.secondary">
            No reusable semantic entities were identified.
          </Typography>
        ) : null}
        {typedEntities.map((entity) => {
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
                  {humanize(entity.semantic_type)} · {Math.round(entity.confidence * 100)}%
                  confidence
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

      <Box className="generic-section">
        <Box className="section-heading compact-heading">
          <Typography variant="h4">Verified facts</Typography>
          <Typography variant="caption">{verifiedFacts.length}</Typography>
        </Box>
        {!verifiedFacts.length ? (
          <Typography color="text.secondary">
            No reviewer-verified facts have been recorded.
          </Typography>
        ) : null}
        {verifiedFacts.map((fact) => {
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
    </Box>
  );
}

function GenericFields({
  fields,
  onFocus,
  selectedEvidenceIds,
  onToggleEvidence,
}: {
  fields: GenericDiscoveredField[];
  onFocus?: (page: number, box: BoundingBox, id: string) => void;
  selectedEvidenceIds: string[];
  onToggleEvidence?: (evidenceId: string) => void;
}) {
  return (
    <Box className="generic-section">
      <Box className="section-heading compact-heading">
        <Typography variant="h4">Discovered attributes</Typography>
        <Typography variant="caption">{fields.length}</Typography>
      </Box>
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

function resolveEvidence(document: VisionDocument, evidenceIds: string[]): Evidence | undefined {
  const wanted = new Set(evidenceIds);
  return [
    ...(document.extracted_fields ?? []).map((field) => field.evidence),
    ...(document.generic_fields ?? []).map((field) => field.evidence),
  ].find((evidence) => wanted.has(evidence.evidence_id));
}

export function humanize(value: string): string {
  return value
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function maskedGenericValue(field: GenericDiscoveredField, normalized = false): string {
  return maskedSemanticValue(
    field.semantic_type_if_known,
    normalized ? field.normalized_value : field.raw_value,
  );
}

function maskedSemanticValue(semanticType: string | null | undefined, value: string): string {
  if (semanticType !== "BANK_ACCOUNT") return value;
  return `${"*".repeat(Math.max(4, value.length - 4))}${value.slice(-4)}`;
}
