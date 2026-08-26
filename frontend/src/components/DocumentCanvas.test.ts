import { describe, expect, it } from "vitest";

import type { VisionDocument } from "../types/api";
import { boxStyle, buildOverlays, type Layer } from "./DocumentCanvas";

const box = { x1: 100, y1: 200, x2: 300, y2: 500 };
const document = {
  ocr_words: [{
    word_id: "word-1",
    page: 1,
    text: "Invoice",
    confidence: 0.6,
    review_required: true,
    bounding_box: box,
  }],
  layout_regions: [{
    region_id: "region-1",
    page_number: 1,
    region_type: "heading",
    text: "Invoice",
    bounding_box: box,
  }],
  extracted_fields: [{
    field_id: "field-1",
    name: "invoice_number",
    display_name: "Invoice Number",
    value: "INV-1",
    reviewed_value: null,
    confidence: 0.6,
    reviewer_status: "unverified",
    evidence: { page_number: 1, bounding_box: box },
  }],
  tables: [{ table_id: "table-1", page_number: 1, rows: [["A"]], bounding_box: box }],
  visual_detections: [{ detection_id: "visual-1", page_number: 1, detection_class: "signature", confidence: 0.8, bounding_box: box }],
  risk_indicators: [{ risk_id: "risk-1", title: "Mismatch", description: "Values differ", evidence: [{ evidence_id: "risk-evidence", page_number: 1, bounding_box: box }] }],
  tamper_analysis: [{ page_number: 1, suspicious_regions: [{ region_id: "tamper-1", manipulation_class: "copy_move", screening_score: 0.75, bounding_box: box }] }],
} as unknown as VisionDocument;

function layers(enabled: Layer): Record<Layer, boolean> {
  return {
    ocr: false,
    layout: false,
    fields: false,
    tables: false,
    visual: false,
    lowConfidence: false,
    risk: false,
    tamper: false,
    [enabled]: true,
  };
}

describe("document canvas overlays", () => {
  it.each<Layer>(["ocr", "layout", "fields", "tables", "visual", "risk", "tamper"])(
    "toggles %s independently",
    (layer) => {
      const result = buildOverlays(document, 1, layers(layer));
      expect(result.length).toBe(1);
      expect(result[0].layer).toBe(layer);
    },
  );

  it("isolates low-confidence OCR and field regions", () => {
    const result = buildOverlays(document, 1, layers("lowConfidence"));
    expect(result).toHaveLength(2);
    expect(result.every((item) => item.layer === "lowConfidence")).toBe(true);
  });

  it("projects schema-independent attributes and their review state", () => {
    const genericDocument = {
      ...document,
      generic_fields: [{
        field_id: "generic-1",
        raw_label: "Unseen Attribute",
        normalized_label: "unseen_attribute",
        raw_value: "VALUE-7",
        normalized_value: "VALUE-7",
        page: 1,
        confidence: 0.65,
        review_required: true,
        value_bbox: box,
        evidence: { evidence_id: "generic-evidence", page_number: 1, bounding_box: box },
      }],
      verified_facts: [{ evidence_id: "generic-evidence" }],
    } as unknown as VisionDocument;

    const fieldOverlays = buildOverlays(genericDocument, 1, layers("fields"));
    expect(fieldOverlays).toHaveLength(2);
    expect(fieldOverlays[1]).toMatchObject({
      id: "generic-1",
      label: "Unseen Attribute: VALUE-7",
      variant: "verified",
    });

    expect(buildOverlays(genericDocument, 1, layers("lowConfidence"))).toHaveLength(3);
  });

  it("keeps coordinates proportional across rendered sizes", () => {
    expect(boxStyle(box, 1000, 2000)).toEqual({
      left: "10%",
      top: "10%",
      width: "20%",
      height: "15%",
    });
  });
});
