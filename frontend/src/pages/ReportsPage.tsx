import { Alert, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, IconButton, MenuItem, Table, TableBody, TableCell, TableHead, TableRow, TextField, Tooltip, Typography } from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Archive, Download, Eye, FilePlus2, RefreshCw } from "lucide-react";
import { useMemo, useState } from "react";

import { api, downloadReport } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { EmptyState, ErrorState, LoadingRows } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";
import type { ReportRecord } from "../types/api";

const reportTypes = ["document_intelligence_report", "risk_screening_report", "contract_review_report", "vendor_summary", "finance_narrative_report", "model_evaluation_report"];

export function ReportsPage() {
  const { user } = useAuth();
  const canManage = ["ADMIN", "REVIEWER", "SENIOR_REVIEWER", "COMPLIANCE_OFFICER"].includes(user?.role || "");
  const queryClient = useQueryClient();
  const reports = useQuery({ queryKey: ["reports"], queryFn: api.reports });
  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const narratives = useQuery({ queryKey: ["narratives"], queryFn: api.narratives });
  const [type, setType] = useState(reportTypes[0]);
  const [documentId, setDocumentId] = useState("");
  const [vendor, setVendor] = useState("");
  const [narrativeId, setNarrativeId] = useState("");
  const documentNames = useMemo(() => new Map(documents.data?.documents.map((item) => [item.document_id, item.filename]) || []), [documents.data]);
  const generate = useMutation({
    mutationFn: (payload: Record<string, unknown>) => api.generateReport(payload),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["reports"] }),
  });
  const archive = useMutation({
    mutationFn: api.archiveReport,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["reports"] }),
  });
  const requestPayload = (newVersion = false) => ({
    report_type: type,
    document_id: documentId || null,
    vendor_name: vendor || null,
    narrative_id: narrativeId || null,
    generate_new_version: newVersion,
  });
  const regenerate = (report: ReportRecord) => {
    generate.mutate({ ...report.request_scope, generate_new_version: true });
  };

  return (
    <Box className="page-stack">
      <PageHeader title="Reports" />
      <Box className="report-builder surface">
        <TextField select size="small" label="Report type" value={type} onChange={(event) => setType(event.target.value)}>{reportTypes.map((value) => <MenuItem key={value} value={value}>{value.replaceAll("_", " ")}</MenuItem>)}</TextField>
        {type.includes("document") || type.includes("risk") || type.includes("contract") ? <TextField select size="small" label="Document" value={documentId} onChange={(event) => setDocumentId(event.target.value)}><MenuItem value="">Select document</MenuItem>{documents.data?.documents.map((document) => <MenuItem key={document.document_id} value={document.document_id}>{document.filename}</MenuItem>)}</TextField> : null}
        {type === "vendor_summary" ? <TextField size="small" label="Vendor" value={vendor} onChange={(event) => setVendor(event.target.value)} /> : null}
        {type === "finance_narrative_report" ? <TextField select size="small" label="Narrative" value={narrativeId} onChange={(event) => setNarrativeId(event.target.value)}><MenuItem value="">Select narrative</MenuItem>{narratives.data?.narratives.map((item) => <MenuItem key={item.narrative_id} value={item.narrative_id}>{item.narrative_type} | v{item.version}</MenuItem>)}</TextField> : null}
        <Button variant="contained" startIcon={<FilePlus2 size={16} />} disabled={generate.isPending || type === "model_evaluation_report"} onClick={() => generate.mutate(requestPayload())}>Generate or reuse</Button>
        <Button variant="outlined" startIcon={<RefreshCw size={16} />} disabled={generate.isPending || type === "model_evaluation_report"} onClick={() => generate.mutate(requestPayload(true))}>New version</Button>
      </Box>
      {type === "model_evaluation_report" ? <Alert severity="info">A labelled evaluation run with dataset provenance is required.</Alert> : null}
      {generate.data ? <Alert severity="success">{generate.data.title} v{generate.data.version} is ready.</Alert> : null}
      {generate.error ? <Alert severity="error">{generate.error.message}</Alert> : null}
      {archive.error ? <Alert severity="error">{archive.error.message}</Alert> : null}
      <Box className="surface table-surface">
        {reports.isLoading ? <LoadingRows /> : reports.error ? <ErrorState error={reports.error} /> : !reports.data?.reports.length ? <EmptyState title="No reports generated" detail="Choose a report type and evidence scope." /> : (
          <Table><TableHead><TableRow><TableCell>Report</TableCell><TableCell>Type</TableCell><TableCell>Documents</TableCell><TableCell>Version</TableCell><TableCell>Source snapshot</TableCell><TableCell>Created</TableCell><TableCell align="right">Actions</TableCell></TableRow></TableHead><TableBody>{reports.data.reports.map((report) => <ReportRow key={report.report_id} report={report} documentNames={documentNames} canManage={canManage} onRegenerate={regenerate} onArchive={(item) => { if (window.confirm(`Archive ${item.title} v${item.version}?`)) archive.mutate(item.report_id); }} />)}</TableBody></Table>
        )}
      </Box>
    </Box>
  );
}

function ReportRow({ report, documentNames, canManage, onRegenerate, onArchive }: { report: ReportRecord; documentNames: Map<string, string>; canManage: boolean; onRegenerate: (report: ReportRecord) => void; onArchive: (report: ReportRecord) => void }) {
  const [busy, setBusy] = useState(false);
  const [sourcesOpen, setSourcesOpen] = useState(false);
  return (
    <>
    <TableRow>
      <TableCell><Typography fontWeight={700}>{report.title}</Typography><Typography variant="caption" color="text.secondary">{report.report_id}</Typography></TableCell>
      <TableCell>{report.report_type.replaceAll("_", " ")}</TableCell>
      <TableCell>{report.document_ids.length ? <><Typography variant="body2">{report.document_ids.length} {report.document_ids.length === 1 ? "document" : "documents"}</Typography><Button size="small" startIcon={<Eye size={14} />} onClick={() => setSourcesOpen(true)}>View sources</Button></> : <Typography variant="body2">No document scope</Typography>}</TableCell>
      <TableCell><Typography fontWeight={700}>v{report.version}</Typography><StatusChip value={report.status} /></TableCell>
      <TableCell><Tooltip title={report.source_data_version}><code>{report.source_data_version.slice(0, 12)}</code></Tooltip></TableCell>
      <TableCell>{new Date(report.created_at).toLocaleString()}<Typography variant="caption" display="block" color="text.secondary">{report.generated_by}</Typography></TableCell>
      <TableCell align="right">
        <Tooltip title="Download PDF"><span><IconButton disabled={busy} onClick={() => { setBusy(true); void downloadReport(report).finally(() => setBusy(false)); }}><Download size={17} /></IconButton></span></Tooltip>
        {canManage ? <Tooltip title="Generate new version"><IconButton onClick={() => onRegenerate(report)}><RefreshCw size={17} /></IconButton></Tooltip> : null}
        {canManage && report.status !== "archived" ? <Tooltip title="Archive report"><IconButton color="warning" onClick={() => onArchive(report)}><Archive size={17} /></IconButton></Tooltip> : null}
      </TableCell>
    </TableRow>
    <Dialog open={sourcesOpen} onClose={() => setSourcesOpen(false)} maxWidth="sm" fullWidth><DialogTitle>Report sources</DialogTitle><DialogContent><Box className="report-document-list">{report.document_ids.map((id) => <Typography variant="body2" key={id}>{documentNames.get(id) || id}</Typography>)}</Box></DialogContent><DialogActions><Button onClick={() => setSourcesOpen(false)}>Close</Button></DialogActions></Dialog>
    </>
  );
}
