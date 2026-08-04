import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Divider, LinearProgress, Tab, Tabs, Table, TableBody, TableCell, TableHead, TableRow, Typography } from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ChevronDown, ExternalLink, RefreshCw } from "lucide-react";
import { useMemo, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "../router";

import { api } from "../api/client";
import { DocumentCanvas, type FocusRegion } from "../components/DocumentCanvas";
import { ErrorState, LoadingRows } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";
import type { BoundingBox, Citation, ExtractedField } from "../types/api";

export function DocumentViewerPage() {
  const { documentId = "" } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const citation = (location.state as { citation?: Citation } | null)?.citation;
  const documentQuery = useQuery({ queryKey: ["document", documentId], queryFn: () => api.document(documentId) });
  const reprocess = useMutation({
    mutationFn: api.reprocessDocument,
    onSuccess: (result) => {
      navigate(`/processing/${result.job_id}`, {
        state: { documentIds: result.document_ids },
      });
    },
  });
  const [page, setPage] = useState(citation?.page_number || 1);
  const [tab, setTab] = useState(0);
  const [focus, setFocus] = useState<FocusRegion | null>(citation?.bounding_boxes?.[0] ? { page: citation.page_number, box: citation.bounding_boxes[0], id: citation.evidence_id || undefined } : null);
  const document = documentQuery.data;
  const meanConfidence = useMemo(() => document?.extracted_fields.length ? document.extracted_fields.reduce((sum, item) => sum + item.confidence, 0) / document.extracted_fields.length : 0, [document]);

  if (documentQuery.isLoading) return <Box className="page-stack"><PageHeader title="Document intelligence" /><LoadingRows rows={10} /></Box>;
  if (documentQuery.error || !document) return <Box className="page-stack"><PageHeader title="Document intelligence" /><ErrorState error={documentQuery.error || new Error("Document not found")} /></Box>;

  const focusField = (field: ExtractedField) => {
    if (!field.evidence.bounding_box) return;
    setPage(field.evidence.page_number);
    setFocus({ page: field.evidence.page_number, box: field.evidence.bounding_box, id: field.field_id });
  };

  return (
    <Box className="page-stack viewer-page">
      <PageHeader
        title={document.filename}
        subtitle={`${document.document_type.replaceAll("_", " ")} | ${document.page_count} page(s) | ${Math.round(meanConfidence * 100)}% mean field confidence`}
        actions={<><StatusChip value={document.status} size="medium" />{document.review_task ? <Button component={Link} to={`/reviews/${document.review_task.review_id}`} variant="contained">Open review</Button> : null}<Button startIcon={<RefreshCw size={16} />} disabled={reprocess.isPending} onClick={() => reprocess.mutate(document.document_id)}>{reprocess.isPending ? "Reprocessing" : "Reprocess"}</Button></>}
      />
      {citation ? <Alert severity="info">Opened from answer citation on page {citation.page_number}.</Alert> : null}
      {reprocess.error ? <Alert severity="error">{reprocess.error.message}</Alert> : null}
      <Box className="intelligence-workspace">
        <Box className="viewer-pane surface">
          <DocumentCanvas document={document} page={page} onPageChange={setPage} focus={focus} onRegionClick={setFocus} />
        </Box>
        <Box className="intelligence-panel surface">
          <Tabs value={tab} onChange={(_, value) => setTab(value)} variant="scrollable" scrollButtons="auto">
            <Tab label={`Fields ${document.extracted_fields.length}`} />
            <Tab label={`Tables ${document.tables.length}`} />
            <Tab label={`Risk ${document.risk_indicators.length}`} />
            <Tab label="Models" />
          </Tabs>
          <Divider />
          <Box className="panel-scroll">
            {tab === 0 ? (
              <Box className="field-list">
                {document.extracted_fields.map((field) => (
                  <button type="button" className={`field-row${focus?.id === field.field_id ? " selected" : ""}`} key={field.field_id} onClick={() => focusField(field)}>
                    <Box><Typography variant="caption" color="text.secondary">{field.name.replaceAll("_", " ")}</Typography><Typography fontWeight={650} className="field-value">{masked(field)}</Typography></Box>
                    <Box className="field-meta"><Typography variant="caption">{Math.round(field.confidence * 100)}%</Typography><StatusChip value={field.reviewer_status} /></Box>
                  </button>
                ))}
              </Box>
            ) : null}
            {tab === 1 ? document.tables.length ? document.tables.map((table) => (
              <Box key={table.table_id} className="table-result">
                <Box className="section-heading"><Typography variant="h4">Page {table.page_number}</Typography><Typography variant="caption">{Math.round(table.confidence * 100)}% | {table.model_name}</Typography></Box>
                <Table size="small"><TableBody>{table.rows.map((row, rowIndex) => <TableRow key={rowIndex}>{row.map((cell, cellIndex) => <TableCell key={cellIndex}>{cell}</TableCell>)}</TableRow>)}</TableBody></Table>
              </Box>
            )) : <Typography color="text.secondary">No structural tables were detected.</Typography> : null}
            {tab === 2 ? document.risk_indicators.length ? document.risk_indicators.map((risk) => (
              <Accordion key={risk.risk_id} disableGutters elevation={0}>
                <AccordionSummary expandIcon={<ChevronDown size={17} />}><Box className="risk-heading"><StatusChip value={risk.severity} /><Typography fontWeight={700}>{risk.title}</Typography></Box></AccordionSummary>
                <AccordionDetails><Typography>{risk.description}</Typography><Typography color="text.secondary" sx={{ mt: 1 }}>{risk.recommended_action}</Typography>{risk.evidence.map((evidence) => evidence.bounding_box ? <Button key={evidence.evidence_id} size="small" startIcon={<ExternalLink size={14} />} onClick={() => { setPage(evidence.page_number); setFocus({ page: evidence.page_number, box: evidence.bounding_box as BoundingBox, id: `${risk.risk_id}-${evidence.evidence_id}` }); }}>Open evidence</Button> : null)}</AccordionDetails>
              </Accordion>
            )) : <Typography color="text.secondary">No risk-screening indicators were generated.</Typography> : null}
            {tab === 3 ? (
              <Table size="small"><TableHead><TableRow><TableCell>Stage</TableCell><TableCell>Model</TableCell><TableCell>Status</TableCell><TableCell>Confidence</TableCell></TableRow></TableHead><TableBody>{document.model_runs.map((run) => <TableRow key={String(run.run_id)}><TableCell>{String(run.stage)}</TableCell><TableCell>{String(run.model_name)} <Typography variant="caption">{String(run.model_version)}</Typography></TableCell><TableCell><StatusChip value={String(run.status)} /></TableCell><TableCell>{typeof run.confidence === "number" ? `${Math.round(run.confidence * 100)}%` : "Not reported"}</TableCell></TableRow>)}</TableBody></Table>
            ) : null}
          </Box>
        </Box>
      </Box>
      {document.quality.some((item) => Number(item.image_quality_score) < 0.5) ? <LinearProgress color="warning" /> : null}
    </Box>
  );
}

function masked(field: ExtractedField) {
  const value = field.reviewed_value || field.value;
  if (!field.name.toLowerCase().includes("bank") && !field.name.toLowerCase().includes("account")) return value;
  return `${"*".repeat(Math.max(4, value.length - 4))}${value.slice(-4)}`;
}
