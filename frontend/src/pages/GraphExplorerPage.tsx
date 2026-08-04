import { Alert, Box, Button, MenuItem, TextField, Typography } from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { useMemo, useState } from "react";
import CytoscapeComponent from "react-cytoscapejs";
import type { StylesheetStyle } from "cytoscape";

import { api } from "../api/client";
import { ErrorState } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";

export function GraphExplorerPage() {
  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const [documentId, setDocumentId] = useState("");
  const [question, setQuestion] = useState("");
  const graph = useQuery({
    queryKey: ["graph", documentId],
    queryFn: () => api.graphDocument(documentId),
    enabled: Boolean(documentId),
  });
  const query = useMutation({ mutationFn: () => api.graphQuery(question, documentId || undefined) });
  const elements = useMemo(() => {
    if (!graph.data) return [];
    const nodes = graph.data.nodes.map((node) => ({ data: { id: String(node.id), label: String(node.label || node.value || node.id), type: String(node.type || "entity") } }));
    const edges = graph.data.edges.map((edge) => ({ data: { id: edge.id, source: edge.source, target: edge.target, label: edge.type, status: String(edge.status || "proposed") } }));
    return [...nodes, ...edges];
  }, [graph.data]);

  return (
    <Box className="page-stack graph-page">
      <PageHeader title="Knowledge graph" subtitle="Evidence-linked entities and reviewed or proposed relationships." />
      <Box className="graph-toolbar surface">
        <TextField select label="Document graph" value={documentId} onChange={(event) => setDocumentId(event.target.value)} size="small">
          <MenuItem value="">Select document</MenuItem>
          {documents.data?.documents.map((document) => <MenuItem key={document.document_id} value={document.document_id}>{document.filename}</MenuItem>)}
        </TextField>
        <TextField placeholder="Ask a relationship question" value={question} onChange={(event) => setQuestion(event.target.value)} size="small" />
        <Button variant="contained" startIcon={<Search size={16} />} disabled={!question.trim()} onClick={() => query.mutate()}>Query graph</Button>
      </Box>
      {graph.error ? <ErrorState error={graph.error} /> : null}
      {query.error ? <Alert severity="error">{query.error.message}</Alert> : null}
      {query.data ? <Alert severity="info">{String(((query.data.answer as Record<string, unknown>)?.answer as string) || "Graph evidence returned")}</Alert> : null}
      <Box className="graph-workspace surface">
        {elements.length ? (
          <CytoscapeComponent
            elements={elements}
            layout={{ name: "cose", animate: false, fit: true, padding: 40 }}
            stylesheet={graphStyles}
            style={{ width: "100%", height: "100%" }}
            wheelSensitivity={0.25}
          />
        ) : <Box className="graph-empty"><Typography variant="h3">Select a document graph</Typography><Typography color="text.secondary">Approved edges remain verified; model-proposed edges stay visibly proposed.</Typography></Box>}
      </Box>
    </Box>
  );
}

const graphStyles: StylesheetStyle[] = [
  { selector: "node", style: { "background-color": "#087f78", label: "data(label)", color: "#18232d", "font-size": 10, "text-valign": "bottom", "text-margin-y": 7, width: 24, height: 24 } },
  { selector: "node[type = 'document']", style: { "background-color": "#12324a", shape: "round-rectangle", width: 34, height: 28 } },
  { selector: "edge", style: { width: 1.5, "line-color": "#8d9aa5", "target-arrow-color": "#8d9aa5", "target-arrow-shape": "triangle", "curve-style": "bezier", label: "data(label)", "font-size": 8, color: "#596673" } },
  { selector: "edge[status = 'proposed']", style: { "line-style": "dashed", "line-color": "#b54708", "target-arrow-color": "#b54708" } },
];
