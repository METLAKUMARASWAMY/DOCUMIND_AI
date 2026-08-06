import { describe, expect, it } from "vitest";

import type { GraphProjection } from "../types/api";
import { filterGraphProjection, truncateGraphLabel } from "./graphProjection";

const graph: GraphProjection = {
  nodes: [
    { id: "document", type: "Document", label: "Document" },
    { id: "vendor", type: "Vendor", label: "Vendor" },
    { id: "field", type: "ExtractedField", label: "Vendor Name" },
  ],
  edges: [
    { id: "verified", source: "document", target: "vendor", type: "ISSUED_BY", status: "verified" },
    { id: "proposed", source: "vendor", target: "field", type: "MATCHES", status: "proposed" },
  ],
  evidence_policy: "Source linked",
  schema_version: "v2",
  preferred_backend: "neo4j",
  active_backend: "local_evidence_graph",
};

describe("graph projection helpers", () => {
  it("caps display labels without changing short labels", () => {
    expect(truncateGraphLabel("Short label")).toBe("Short label");
    expect(truncateGraphLabel("A very long graph node label that cannot cover the canvas", 24)).toBe("A very long graph nod...");
  });

  it("filters edges with their visible endpoints", () => {
    expect(filterGraphProjection(graph, "all", "proposed").edges.map((edge) => edge.id)).toEqual(["proposed"]);
    expect(filterGraphProjection(graph, "all", "all", "ISSUED_BY").edges.map((edge) => edge.id)).toEqual(["verified"]);
    expect(filterGraphProjection(graph, "Vendor", "all")).toEqual({ nodes: graph.nodes, edges: graph.edges });
  });
});
