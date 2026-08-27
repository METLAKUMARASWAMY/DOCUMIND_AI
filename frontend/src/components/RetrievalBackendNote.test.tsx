import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { RetrievalBackendNote } from "./RetrievalBackendNote";
import type { RetrievalTrace } from "../types/api";

/**
 * The UI must not imply a semantic capability the backend does not have.
 * Measured: with feature hashing, "who owns the property being let" scores
 * exactly 0.0000 against "Landlord: Brackenfield Estates".
 */
const base: RetrievalTrace = {
  trace_id: "retrieval-1",
  sources_checked: [],
  candidate_counts: {},
  selected_evidence: [],
  retrieval_duration_ms: 12,
  retriever_versions: {},
};

describe("RetrievalBackendNote", () => {
  it("renders nothing when the trace does not name a backend", () => {
    const { container } = render(<RetrievalBackendNote trace={base} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing without a trace at all", () => {
    const { container } = render(<RetrievalBackendNote />);
    expect(container).toBeEmptyDOMElement();
  });

  it("calls a hashing backend keyword matching, never semantic", () => {
    render(
      <RetrievalBackendNote
        trace={{
          ...base,
          embedding_backend: "local-hashing-embedding",
          embedding_model: "feature-hashing-v1",
          semantic_matching: false,
        }}
      />,
    );

    const text = document.body.textContent ?? "";
    expect(text).toContain("Keyword matching");
    expect(text).toContain("feature-hashing-v1");
    expect(text).not.toContain("Semantic matching");
  });

  it("says semantic only when the backend reports semantic matching", () => {
    render(
      <RetrievalBackendNote
        trace={{
          ...base,
          embedding_backend: "bge-m3-embedding",
          embedding_model: "BAAI/bge-m3",
          semantic_matching: true,
        }}
      />,
    );

    expect(document.body.textContent).toContain("Semantic matching");
    expect(document.body.textContent).toContain("BAAI/bge-m3");
  });

  it("treats a missing semantic flag as not semantic", () => {
    render(<RetrievalBackendNote trace={{ ...base, embedding_backend: "unknown-backend" }} />);

    expect(document.body.textContent).toContain("Keyword matching");
  });

  it("falls back to the adapter id when no model version is given", () => {
    render(<RetrievalBackendNote trace={{ ...base, embedding_backend: "some-adapter" }} />);
    expect(screen.getByText(/some-adapter/)).toBeTruthy();
  });
});
