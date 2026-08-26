import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { VisionDocument } from "../types/api";
import { GenericIntelligencePanel, humanize } from "./GenericIntelligencePanel";

const valueBox = { x1: 110, y1: 210, x2: 310, y2: 250 };
const evidence = {
  evidence_id: "evidence-dynamic",
  document_id: "document-1",
  page_number: 2,
  source_text: "Novel Attribute: AC-204",
  bounding_box: valueBox,
  model_confidence: 0.91,
  model_name: "generic-discovery",
  model_version: "v1",
  processing_timestamp: "2026-08-12T00:00:00Z",
  reviewer_status: "unverified",
};
const document = {
  schema_profile: {
    mode: "B",
    schema_status: "inferred",
    schema_id: "schema-dynamic",
    schema_version: "v3",
    selection_source: "layout_inference",
  },
  extracted_fields: [],
  generic_fields: [
    {
      field_id: "generic-dynamic",
      raw_label: "Novel Attribute",
      normalized_label: "novel_attribute",
      raw_value: "AC-204",
      normalized_value: "AC-204",
      page: 2,
      label_bbox: { x1: 10, y1: 210, x2: 100, y2: 250 },
      value_bbox: valueBox,
      confidence: 0.91,
      source_engine: "geometry",
      semantic_type_if_known: "IDENTIFIER",
      semantic_mapping_confidence: 0.88,
      evidence,
      alternatives: [],
      review_required: false,
      metadata: {},
    },
  ],
  typed_entities: [
    {
      entity_id: "entity-dynamic",
      raw_value: "AC-204",
      normalized_value: "AC-204",
      semantic_type: "IDENTIFIER",
      confidence: 0.88,
      validator_valid: true,
      validation_reasons: [],
      context: {},
      evidence_ids: ["evidence-dynamic"],
      alternatives: [],
      conflict_ids: [],
      review_required: false,
    },
  ],
  verified_facts: [
    {
      fact_id: "fact-dynamic",
      field_id: "generic-dynamic",
      name: "novel_attribute",
      value: "AC-204",
      normalized_value: "AC-204",
      semantic_type: "IDENTIFIER",
      evidence_id: "evidence-dynamic",
      reviewer_status: "accepted",
      reviewer: "reviewer@example.com",
    },
  ],
} as unknown as VisionDocument;

afterEach(cleanup);

describe("GenericIntelligencePanel", () => {
  it("renders inferred schema data without a predefined field catalog", () => {
    render(<GenericIntelligencePanel document={document} />);

    expect(screen.getByText("Mode B · Inferred")).toBeInTheDocument();
    expect(screen.getAllByText("Novel Attribute")).toHaveLength(2);
    expect(screen.getAllByText("AC-204")).toHaveLength(3);
    expect(screen.getByText("Identifier · 88% confidence")).toBeInTheDocument();
    expect(screen.getByText("reviewer@example.com")).toBeInTheDocument();
  });

  it("links discovered evidence to the source region and review selection", () => {
    const onFocus = vi.fn();
    const onToggleEvidence = vi.fn();
    render(
      <GenericIntelligencePanel
        document={document}
        onFocus={onFocus}
        onToggleEvidence={onToggleEvidence}
      />,
    );

    fireEvent.click(screen.getAllByText("Novel Attribute")[0]);
    expect(onFocus).toHaveBeenCalledWith(2, valueBox, "generic-dynamic");

    fireEvent.click(screen.getByLabelText("Evidence reviewed"));
    expect(onToggleEvidence).toHaveBeenCalledWith("evidence-dynamic");
  });

  it("humanizes arbitrary machine labels", () => {
    expect(humanize("new_vendor_REFERENCE")).toBe("New Vendor Reference");
  });
});
