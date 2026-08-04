import {
  Alert,
  Box,
  Button,
  Checkbox,
  FormControlLabel,
  IconButton,
  InputAdornment,
  MenuItem,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Core, EventObject, StylesheetStyle } from "cytoscape";
import { BadgeCheck, ExternalLink, Maximize2, RotateCcw, Route, Search } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import CytoscapeComponent from "react-cytoscapejs";

import { api } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { ErrorState } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";
import { useNavigate } from "../router";
import type { Citation, GraphEdge, GraphNode } from "../types/api";
import { filterGraphProjection, truncateGraphLabel } from "../utils/graphProjection";

type GraphSelection =
  | { kind: "node"; data: GraphNode }
  | { kind: "edge"; data: GraphEdge }
  | null;

export function GraphExplorerPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { user } = useAuth();
  const canVerify = ["ADMIN", "REVIEWER", "SENIOR_REVIEWER", "COMPLIANCE_OFFICER"].includes(user?.role || "");
  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const [documentId, setDocumentId] = useState("");
  const [question, setQuestion] = useState("");
  const [search, setSearch] = useState("");
  const [nodeType, setNodeType] = useState("all");
  const [edgeStatus, setEdgeStatus] = useState("all");
  const [pathStart, setPathStart] = useState("");
  const [pathEnd, setPathEnd] = useState("");
  const [selection, setSelection] = useState<GraphSelection>(null);
  const [cy, setCy] = useState<Core | null>(null);
  const [verifyComment, setVerifyComment] = useState("");
  const [evidenceAcknowledged, setEvidenceAcknowledged] = useState(false);

  const activeDocumentId = documentId || documents.data?.documents[0]?.document_id || "";

  const graph = useQuery({
    queryKey: ["graph", activeDocumentId],
    queryFn: () => api.graphDocument(activeDocumentId),
    enabled: Boolean(activeDocumentId),
  });
  const relationshipQuery = useMutation({
    mutationFn: () => api.graphQuery(question, activeDocumentId || undefined),
  });
  const verify = useMutation({
    mutationFn: () => {
      if (selection?.kind !== "edge") throw new Error("Select a relationship");
      return api.verifyGraphRelationship(selection.data.id, verifyComment, evidenceAcknowledged);
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({ queryKey: ["graph", activeDocumentId] });
      setSelection(null);
      setVerifyComment("");
      setEvidenceAcknowledged(false);
    },
  });

  const nodeTypes = useMemo(
    () => Array.from(new Set(graph.data?.nodes.map((node) => node.type) || [])).sort(),
    [graph.data],
  );
  const projection = useMemo(
    () => (graph.data ? filterGraphProjection(graph.data, nodeType, edgeStatus) : { nodes: [], edges: [] }),
    [edgeStatus, graph.data, nodeType],
  );
  const elements = useMemo(() => {
    const nodes = projection.nodes.map((node) => ({
      data: {
        ...node,
        id: String(node.id),
        label: primaryNodeType(node.type)
          ? truncateGraphLabel(String(node.display_label || node.label || node.id))
          : "",
        fullLabel: String(node.label || node.id),
      },
    }));
    const edges = projection.edges.map((edge) => ({
      data: {
        ...edge,
        id: String(edge.id),
        source: String(edge.source),
        target: String(edge.target),
        label: String(edge.label || edge.type),
        status: String(edge.status || "proposed"),
      },
    }));
    return [...nodes, ...edges];
  }, [projection]);

  useEffect(() => {
    if (!cy || cy.destroyed()) return;
    const onNodeTap = (event: EventObject) => {
      setSelection({ kind: "node", data: event.target.data() as GraphNode });
    };
    const onEdgeTap = (event: EventObject) => {
      setSelection({ kind: "edge", data: event.target.data() as GraphEdge });
      setVerifyComment("");
      setEvidenceAcknowledged(false);
    };
    const onCanvasTap = (event: EventObject) => {
      if (event.target === cy) setSelection(null);
    };
    cy.on("tap", "node", onNodeTap);
    cy.on("tap", "edge", onEdgeTap);
    cy.on("tap", onCanvasTap);
    return () => {
      cy.off("tap", "node", onNodeTap);
      cy.off("tap", "edge", onEdgeTap);
      cy.off("tap", onCanvasTap);
    };
  }, [cy]);

  useEffect(() => {
    if (!cy || cy.destroyed()) return;
    cy.nodes().removeClass("search-match");
    const needle = search.trim().toLowerCase();
    if (!needle) return;
    cy.nodes()
      .filter((node) => String(node.data("fullLabel") || "").toLowerCase().includes(needle))
      .addClass("search-match");
  }, [cy, search, elements]);

  const fitGraph = () => {
    if (cy && !cy.destroyed()) cy.fit(undefined, 44);
  };
  const resetLayout = () => {
    if (!cy || cy.destroyed()) return;
    setSearch("");
    setNodeType("all");
    setEdgeStatus("all");
    setPathStart("");
    setPathEnd("");
    cy.elements().removeClass("path-highlight search-match");
    cy.layout({ name: "cose", animate: false, fit: true, padding: 44 }).run();
  };
  const highlightPath = () => {
    if (!cy || cy.destroyed() || !pathStart || !pathEnd) return;
    cy.elements().removeClass("path-highlight");
    const result = cy.elements().aStar({
      root: cy.$id(pathStart),
      goal: cy.$id(pathEnd),
      directed: false,
    });
    if (result.found) {
      result.path.addClass("path-highlight");
      cy.fit(result.path, 70);
    }
  };

  const openEvidence = (edge: GraphEdge) => {
    if (!edge.document_id) return;
    const citation: Citation = {
      document_id: edge.document_id,
      page_number: edge.source_page || 1,
      source_text: edge.evidence_text || String(edge.type),
      bounding_boxes: edge.bounding_boxes || [],
      evidence_id: edge.evidence_id,
      model_confidence: typeof edge.confidence === "number" ? edge.confidence : 0,
      reviewer_status: String(edge.review_status || "unverified"),
    };
    navigate(`/documents/${edge.document_id}`, { state: { citation } });
  };

  const queryAnswer = relationshipQuery.data
    ? String((relationshipQuery.data.answer as Record<string, unknown>)?.answer || "")
    : "";

  return (
    <Box className="page-stack graph-page">
      <PageHeader title="Knowledge graph" />
      <Box className="graph-query-bar surface">
        <TextField select label="Document graph" value={activeDocumentId} onChange={(event) => setDocumentId(event.target.value)} size="small">
          <MenuItem value="">Select document</MenuItem>
          {documents.data?.documents.map((document) => <MenuItem key={document.document_id} value={document.document_id}>{document.filename}</MenuItem>)}
        </TextField>
        <TextField placeholder="Ask a relationship question" value={question} onChange={(event) => setQuestion(event.target.value)} size="small" />
        <Button variant="contained" startIcon={<Search size={16} />} disabled={!question.trim() || relationshipQuery.isPending} onClick={() => relationshipQuery.mutate()}>Query graph</Button>
      </Box>
      {graph.error ? <ErrorState error={graph.error} /> : null}
      {relationshipQuery.error ? <Alert severity="error">{relationshipQuery.error.message}</Alert> : null}
      {queryAnswer ? <Alert severity="info">{queryAnswer}</Alert> : null}
      <Box className="graph-workspace surface">
        <Box className="graph-controls">
          <TextField
            placeholder="Search nodes"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            size="small"
            slotProps={{ input: { startAdornment: <InputAdornment position="start"><Search size={16} /></InputAdornment> } }}
          />
          <TextField select label="Node type" value={nodeType} onChange={(event) => setNodeType(event.target.value)} size="small">
            <MenuItem value="all">All types</MenuItem>
            {nodeTypes.map((type) => <MenuItem value={type} key={type}>{type}</MenuItem>)}
          </TextField>
          <TextField select label="Relationship" value={edgeStatus} onChange={(event) => setEdgeStatus(event.target.value)} size="small">
            <MenuItem value="all">All statuses</MenuItem>
            <MenuItem value="verified">Verified</MenuItem>
            <MenuItem value="proposed">Proposed</MenuItem>
          </TextField>
          <Box className="graph-icon-actions">
            <Tooltip title="Fit graph"><IconButton onClick={fitGraph}><Maximize2 size={18} /></IconButton></Tooltip>
            <Tooltip title="Reset layout"><IconButton onClick={resetLayout}><RotateCcw size={18} /></IconButton></Tooltip>
          </Box>
        </Box>
        <Box className="graph-path-controls">
          <TextField select label="Path from" value={pathStart} onChange={(event) => setPathStart(event.target.value)} size="small">
            <MenuItem value="">Select node</MenuItem>
            {projection.nodes.map((node) => <MenuItem value={node.id} key={node.id}>{truncateGraphLabel(node.label, 42)}</MenuItem>)}
          </TextField>
          <TextField select label="Path to" value={pathEnd} onChange={(event) => setPathEnd(event.target.value)} size="small">
            <MenuItem value="">Select node</MenuItem>
            {projection.nodes.map((node) => <MenuItem value={node.id} key={node.id}>{truncateGraphLabel(node.label, 42)}</MenuItem>)}
          </TextField>
          <Button variant="outlined" startIcon={<Route size={16} />} disabled={!pathStart || !pathEnd} onClick={highlightPath}>Highlight path</Button>
        </Box>
        <Box className="graph-main">
          <Box className="graph-canvas">
            {elements.length ? (
              <CytoscapeComponent
                elements={elements}
                layout={{ name: "cose", animate: false, fit: true, padding: 44 }}
                stylesheet={graphStyles}
                style={{ width: "100%", height: "100%" }}
                minZoom={0.35}
                maxZoom={2.5}
                cy={setCy}
              />
            ) : <Box className="graph-empty"><Typography variant="h3">Select a document</Typography></Box>}
          </Box>
          {selection ? (
            <Box className="graph-details">
              {selection.kind === "node" ? (
                <NodeDetails node={selection.data} onOpen={() => selection.data.document_id && navigate(`/documents/${selection.data.document_id}`)} />
              ) : (
                <EdgeDetails
                  edge={selection.data}
                  canVerify={canVerify}
                  comment={verifyComment}
                  acknowledged={evidenceAcknowledged}
                  pending={verify.isPending}
                  error={verify.error?.message}
                  onComment={setVerifyComment}
                  onAcknowledged={setEvidenceAcknowledged}
                  onOpen={() => openEvidence(selection.data)}
                  onVerify={() => verify.mutate()}
                />
              )}
            </Box>
          ) : null}
        </Box>
        <GraphLegend nodeTypes={nodeTypes} />
      </Box>
    </Box>
  );
}

function NodeDetails({ node, onOpen }: { node: GraphNode; onOpen: () => void }) {
  return (
    <Box className="graph-detail-content">
      <Typography variant="overline">{node.type}</Typography>
      <Typography variant="h3">{node.label}</Typography>
      {node.status ? <StatusChip value={String(node.status)} /> : null}
      {node.source_page ? <Typography variant="body2" color="text.secondary">Page {node.source_page}</Typography> : null}
      <Button variant="outlined" startIcon={<ExternalLink size={16} />} disabled={!node.document_id} onClick={onOpen}>Open source</Button>
    </Box>
  );
}

function EdgeDetails({
  edge,
  canVerify,
  comment,
  acknowledged,
  pending,
  error,
  onComment,
  onAcknowledged,
  onOpen,
  onVerify,
}: {
  edge: GraphEdge;
  canVerify: boolean;
  comment: string;
  acknowledged: boolean;
  pending: boolean;
  error?: string;
  onComment: (value: string) => void;
  onAcknowledged: (value: boolean) => void;
  onOpen: () => void;
  onVerify: () => void;
}) {
  const proposed = edge.status === "proposed" && edge.verification_eligible;
  return (
    <Box className="graph-detail-content">
      <Typography variant="overline">Relationship</Typography>
      <Typography variant="h3">{edge.type.replaceAll("_", " ")}</Typography>
      <StatusChip value={edge.status || "proposed"} />
      <Typography variant="body2">{edge.evidence_text || "No evidence text available."}</Typography>
      <Typography variant="caption" color="text.secondary">
        Page {edge.source_page || 1} | {typeof edge.confidence === "number" ? `${Math.round(edge.confidence * 100)}%` : "Confidence not available"}
      </Typography>
      <Button variant="outlined" startIcon={<ExternalLink size={16} />} disabled={!edge.document_id} onClick={onOpen}>Open evidence</Button>
      {proposed && canVerify ? (
        <Box className="graph-verification">
          <TextField multiline minRows={2} label="Reviewer comment" value={comment} onChange={(event) => onComment(event.target.value)} />
          <FormControlLabel control={<Checkbox checked={acknowledged} onChange={(event) => onAcknowledged(event.target.checked)} />} label="Evidence matches the source region" />
          {error ? <Alert severity="error">{error}</Alert> : null}
          <Button variant="contained" startIcon={<BadgeCheck size={16} />} disabled={!acknowledged || comment.trim().length < 10 || pending} onClick={onVerify}>Verify relationship</Button>
        </Box>
      ) : null}
    </Box>
  );
}

function GraphLegend({ nodeTypes }: { nodeTypes: string[] }) {
  return (
    <Box className="graph-legend" aria-label="Graph legend">
      {nodeTypes.map((type) => <Box className="legend-item" key={type}><span className={`graph-swatch swatch-${type}`} />{type}</Box>)}
      <Box className="legend-item"><span className="edge-swatch edge-verified" />Verified</Box>
      <Box className="legend-item"><span className="edge-swatch edge-proposed" />Proposed</Box>
      <Box className="legend-item"><span className="edge-swatch" style={{ backgroundColor: "#b42318" }} />Conflict</Box>
    </Box>
  );
}

const graphStyles: StylesheetStyle[] = [
  { selector: "node", style: { "background-color": "#46707c", label: "data(label)", color: "#18232d", "font-size": 10, "font-weight": 600, "text-valign": "bottom", "text-margin-y": 7, "text-wrap": "ellipsis", "text-max-width": "100px", width: 28, height: 28, "border-width": 2, "border-color": "#ffffff" } },
  { selector: "node[type = 'Document']", style: { "background-color": "#12324a", shape: "round-rectangle", width: 38, height: 32 } },
  { selector: "node[type = 'Page']", style: { "background-color": "#8796a1", shape: "rectangle", width: 26, height: 32 } },
  { selector: "node[type = 'Organisation'], node[type = 'Vendor'], node[type = 'Customer']", style: { "background-color": "#087f78", shape: "round-rectangle", width: 34, height: 30 } },
  { selector: "node[type = 'Person']", style: { "background-color": "#3568a8", shape: "ellipse" } },
  { selector: "node[type = 'ContactMethod'], node[type = 'Address']", style: { "background-color": "#5b7f3b", shape: "diamond" } },
  { selector: "node[type = 'Date'], node[type = 'DocumentTitle'], node[type = 'ReferenceNumber']", style: { "background-color": "#6b7280", shape: "round-rectangle" } },
  { selector: "node[type = 'Invoice'], node[type = 'PurchaseOrder'], node[type = 'Contract'], node[type = 'Payment']", style: { "background-color": "#c28a13", shape: "round-rectangle" } },
  { selector: "node[type = 'BankAccount'], node[type = 'TaxIdentifier']", style: { "background-color": "#7b5ca7", shape: "hexagon" } },
  { selector: "node[type = 'Signature']", style: { "background-color": "#b45f83", shape: "star" } },
  { selector: "node[type = 'RiskIndicator']", style: { "background-color": "#b42318", shape: "triangle" } },
  { selector: "node[type = 'ExtractedField']", style: { "background-color": "#c5ccd1", shape: "round-tag", width: 22, height: 22 } },
  { selector: "node.search-match", style: { "border-width": 5, "border-color": "#f2b705" } },
  { selector: "node:selected", style: { "border-width": 5, "border-color": "#18232d", label: "data(fullLabel)" } },
  { selector: "edge", style: { width: 1.7, "line-color": "#70808c", "target-arrow-color": "#70808c", "target-arrow-shape": "triangle", "curve-style": "bezier", label: "" } },
  { selector: "edge:selected", style: { label: "data(label)", "font-size": 8, color: "#4d5b65", "text-background-color": "#ffffff", "text-background-opacity": 0.9, "text-background-padding": "2px" } },
  { selector: "edge[status = 'verified']", style: { "line-color": "#287a54", "target-arrow-color": "#287a54" } },
  { selector: "edge[status = 'proposed']", style: { "line-style": "dashed", "line-color": "#b54708", "target-arrow-color": "#b54708" } },
  { selector: ".path-highlight", style: { "line-color": "#d4382d", "target-arrow-color": "#d4382d", "border-color": "#d4382d", "border-width": 5, width: 4, "z-index": 20 } },
];

function primaryNodeType(type: string) {
  return new Set([
    "Contract",
    "ContactMethod",
    "Customer",
    "Document",
    "DocumentTitle",
    "Date",
    "Invoice",
    "Organisation",
    "Payment",
    "Person",
    "PurchaseOrder",
    "ReferenceNumber",
    "RiskIndicator",
    "Vendor",
  ]).has(type);
}
