import { Alert, Box, Button, Chip, MenuItem, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Calculator, FileText, RefreshCw } from "lucide-react";
import { useState } from "react";

import { api } from "../api/client";
import { EmptyState, ErrorState, LoadingRows } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";

const narrativeTypes = [
  "monthly_invoice_summary",
  "vendor_payment_summary",
  "tax_summary",
  "outstanding_payment_summary",
  "anomaly_summary",
  "approval_status_summary",
  "contract_exposure_summary",
];

export function FinanceNarrativesPage() {
  const queryClient = useQueryClient();
  const narratives = useQuery({ queryKey: ["narratives"], queryFn: api.narratives });
  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const [type, setType] = useState(narrativeTypes[0]);
  const [vendor, setVendor] = useState("");
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const [documentIds, setDocumentIds] = useState<string[]>([]);
  const generate = useMutation({
    mutationFn: (payload: Record<string, unknown>) => api.generateNarrative(payload),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["narratives"] }),
  });
  const requestPayload = (newVersion = false) => ({ narrative_type: type, vendor_name: vendor || null, date_from: dateFrom || null, date_to: dateTo || null, document_ids: documentIds, generate_new_version: newVersion });

  return (
    <Box className="page-stack">
      <PageHeader title="Finance narratives" />
      <Box className="narrative-layout">
        <Box className="surface config-panel narrative-config">
          <Typography variant="h3">Narrative scope</Typography>
          <TextField select label="Narrative type" value={type} onChange={(event) => setType(event.target.value)} fullWidth>{narrativeTypes.map((value) => <MenuItem key={value} value={value}>{value.replaceAll("_", " ")}</MenuItem>)}</TextField>
          <TextField label="Vendor name" value={vendor} onChange={(event) => setVendor(event.target.value)} fullWidth />
          <Box className="two-column-fields"><TextField label="From" type="date" value={dateFrom} onChange={(event) => setDateFrom(event.target.value)} InputLabelProps={{ shrink: true }} /><TextField label="To" type="date" value={dateTo} onChange={(event) => setDateTo(event.target.value)} InputLabelProps={{ shrink: true }} /></Box>
          <TextField select SelectProps={{ multiple: true }} label="Documents" value={documentIds} onChange={(event) => setDocumentIds(typeof event.target.value === "string" ? event.target.value.split(",") : event.target.value)} fullWidth>{documents.data?.documents.map((document) => <MenuItem key={document.document_id} value={document.document_id}>{document.filename} ({document.status})</MenuItem>)}</TextField>
          {generate.error ? <Alert severity="error">{generate.error.message}</Alert> : null}
          <Button variant="contained" startIcon={<Calculator size={17} />} disabled={generate.isPending} onClick={() => generate.mutate(requestPayload())}>{generate.isPending ? "Calculating" : "Generate or reuse"}</Button>
          <Button variant="outlined" startIcon={<RefreshCw size={17} />} disabled={generate.isPending} onClick={() => generate.mutate(requestPayload(true))}>Generate new version</Button>
        </Box>
        <Box className="narrative-list">
          {narratives.isLoading ? <LoadingRows /> : narratives.error ? <ErrorState error={narratives.error} /> : !narratives.data?.narratives.length ? <EmptyState title="No narratives yet" detail="Generate a summary from approved, reviewed document values." /> : narratives.data.narratives.map((narrative) => (
            <Box className="surface narrative-result" key={narrative.narrative_id}>
              <Box className="section-heading"><Box><Typography variant="h3">{narrative.narrative_type.replaceAll("_", " ")} | v{narrative.version}</Typography><Typography variant="caption">{new Date(narrative.created_at).toLocaleString()} | source {narrative.source_data_version.slice(0, 12)}</Typography></Box><Box className="narrative-status"><Chip color="success" variant="outlined" label="Verified values" /><Chip variant="outlined" label={narrative.status} /></Box></Box>
              <Typography className="narrative-copy">{narrative.narrative}</Typography>
              <Table size="small"><TableHead><TableRow><TableCell>Calculated value</TableCell><TableCell>Sources</TableCell><TableCell>Function</TableCell></TableRow></TableHead><TableBody>{narrative.calculations.map((trace) => <TableRow key={trace.trace_id}><TableCell><Typography fontWeight={700}>{formatTraceValue(trace.currency, trace.value)}</Typography><Typography variant="caption">{trace.label}</Typography></TableCell><TableCell>{trace.source_document_ids.length} document(s)</TableCell><TableCell><code>{trace.calculation_function}</code></TableCell></TableRow>)}</TableBody></Table>
              <Box className="narrative-footer"><FileText size={15} /><Typography variant="caption">{narrative.citations.length} evidence citation(s)</Typography><Button size="small" startIcon={<RefreshCw size={14} />} disabled={generate.isPending} onClick={() => generate.mutate({ ...narrative.request_scope, generate_new_version: true })}>New version</Button></Box>
            </Box>
          ))}
        </Box>
      </Box>
    </Box>
  );
}

function formatTraceValue(currency: string | null | undefined, value: string): string {
  const numeric = Number(value);
  if (!currency || !Number.isFinite(numeric)) return value;
  try {
    return new Intl.NumberFormat(currency === "INR" ? "en-IN" : "en-US", {
      style: "currency",
      currency,
      minimumFractionDigits: 2,
    }).format(numeric);
  } catch {
    return `${currency} ${value}`;
  }
}
