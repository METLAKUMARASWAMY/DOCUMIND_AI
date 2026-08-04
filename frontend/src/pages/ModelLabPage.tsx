import { Alert, Box, Button, MenuItem, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { FlaskConical } from "lucide-react";
import { useState } from "react";

import { api } from "../api/client";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";

export function ModelLabPage() {
  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const [documentId, setDocumentId] = useState("");
  const comparison = useMutation({ mutationFn: () => api.modelComparison(documentId) });
  const rows = ((comparison.data?.field_output || []) as Array<Record<string, unknown>>);
  return (
    <Box className="page-stack">
      <PageHeader title="Model comparison laboratory" subtitle="Observed outputs and routing status for one processed document." />
      <Box className="lab-toolbar surface"><TextField select size="small" label="Document" value={documentId} onChange={(event) => setDocumentId(event.target.value)}><MenuItem value="">Select document</MenuItem>{documents.data?.documents.map((document) => <MenuItem key={document.document_id} value={document.document_id}>{document.filename}</MenuItem>)}</TextField><Button variant="contained" startIcon={<FlaskConical size={16} />} disabled={!documentId || comparison.isPending} onClick={() => comparison.mutate()}>Compare outputs</Button></Box>
      {comparison.error ? <Alert severity="error">{comparison.error.message}</Alert> : null}
      {comparison.data ? <><Alert severity="info">{String(comparison.data.note)}</Alert><Box className="surface table-surface"><Table><TableHead><TableRow><TableCell>Field</TableCell><TableCell>Primary value</TableCell><TableCell>Confidence</TableCell><TableCell>LayoutLMv3</TableCell><TableCell>Donut</TableCell><TableCell>Review</TableCell></TableRow></TableHead><TableBody>{rows.map((row, index) => <TableRow key={String(row.field_id || index)}><TableCell>{String(row.field_name || row.name)}</TableCell><TableCell>{String(row.value)}</TableCell><TableCell>{Math.round(Number(row.confidence || 0) * 100)}%</TableCell><TableCell><StatusChip value={String(row.layoutlmv3_status || row.layoutlmv3_backend || "disabled")} /></TableCell><TableCell><StatusChip value={String(row.donut_status || row.donut_backend || "disabled")} /></TableCell><TableCell>{row.review_required ? "Required" : "Not required"}</TableCell></TableRow>)}</TableBody></Table></Box><Box className="json-metrics surface"><Typography variant="h3">Fallback routing</Typography><pre>{JSON.stringify(comparison.data.fallback_routing, null, 2)}</pre></Box></> : null}
    </Box>
  );
}
