import type { Evidence, GenericDiscoveredField, VisionDocument } from "../../types/api";

/** Turns a machine label such as `new_vendor_REFERENCE` into `New Vendor Reference`. */
export function humanize(value: string): string {
  return value
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

/**
 * Masks values the backend classifies as sensitive.
 *
 * This mirrors the backend's own masking rather than replacing it — the backend
 * is the authority, and a value that arrives already masked is left alone.
 */
export function maskedSemanticValue(
  semanticType: string | null | undefined,
  value: string,
): string {
  if (semanticType !== "BANK_ACCOUNT") return value;
  return `${"*".repeat(Math.max(4, value.length - 4))}${value.slice(-4)}`;
}

export function maskedGenericValue(field: GenericDiscoveredField, normalized = false): string {
  return maskedSemanticValue(
    field.semantic_type_if_known,
    normalized ? field.normalized_value : field.raw_value,
  );
}

/** Finds the evidence record a semantic entity or verified fact points at. */
export function resolveEvidence(
  document: VisionDocument,
  evidenceIds: string[],
): Evidence | undefined {
  const wanted = new Set(evidenceIds);
  return [
    ...(document.extracted_fields ?? []).map((field) => field.evidence),
    ...(document.generic_fields ?? []).map((field) => field.evidence),
  ].find((evidence) => wanted.has(evidence.evidence_id));
}
