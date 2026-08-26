import type { GraphEdge, GraphNode, GraphProjection } from "../types/api";

export function truncateGraphLabel(value: string, limit = 30): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (normalized.length <= limit) return normalized;
  return `${normalized.slice(0, Math.max(0, limit - 3))}...`;
}

export function filterGraphProjection(
  graph: GraphProjection,
  nodeType: string,
  edgeStatus: string,
  relationshipType = "all",
): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const statusEdges = graph.edges.filter(
    (edge) =>
      (edgeStatus === "all" || edge.status === edgeStatus) &&
      (relationshipType === "all" || edge.type === relationshipType),
  );
  if (nodeType === "all") return { nodes: graph.nodes, edges: statusEdges };

  const matchingIds = new Set(
    graph.nodes.filter((node) => node.type === nodeType).map((node) => node.id),
  );
  const edges = statusEdges.filter(
    (edge) => matchingIds.has(edge.source) || matchingIds.has(edge.target),
  );
  const visibleIds = new Set(matchingIds);
  edges.forEach((edge) => {
    visibleIds.add(edge.source);
    visibleIds.add(edge.target);
  });
  const nodes = graph.nodes.filter((node) => visibleIds.has(node.id));
  return { nodes, edges };
}
