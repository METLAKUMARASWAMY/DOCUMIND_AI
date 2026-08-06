import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { VisionDocument } from "../types/api";
import { FieldCompletenessSummary } from "./FieldCompletenessSummary";

const completeness: VisionDocument["field_completeness"] = {
  schema_available: true,
  document_type: "business_form",
  required_fields: ["date", "sample_number", "product_type"],
  optional_fields: ["document_title"],
  extracted_fields: ["date", "document_title"],
  missing_required_fields: ["sample_number", "product_type"],
  invalid_fields: [],
  disputed_fields: [],
  manually_resolved_fields: [],
  required_matched_fields: ["date"],
  optional_extracted_fields: ["document_title"],
  duplicate_fields: ["document_title"],
  required_expected_count: 3,
  required_matched_count: 1,
  required_missing_count: 2,
  optional_extracted_count: 1,
  total_extracted_count: 2,
  duplicates_rejected_count: 1,
  disputed_count: 0,
  expected_required_count: 3,
  extracted_count: 2,
  unresolved_required_count: 2,
};

describe("FieldCompletenessSummary", () => {
  it("renders all seven backend-calculated counts without recomputing them", () => {
    render(<FieldCompletenessSummary completeness={completeness} />);

    const summary = screen.getByLabelText("Field completeness");
    for (const label of [
      "Required expected",
      "Required matched",
      "Required missing",
      "Optional extracted",
      "Total extracted",
      "Duplicates rejected",
      "Disputed",
    ]) {
      expect(summary).toHaveTextContent(label);
    }
    expect(summary).toHaveTextContent("Required expected3");
    expect(summary).toHaveTextContent("Required matched1");
    expect(summary).toHaveTextContent("Required missing2");
    expect(summary).toHaveTextContent("Duplicates rejected1");
  });
});
