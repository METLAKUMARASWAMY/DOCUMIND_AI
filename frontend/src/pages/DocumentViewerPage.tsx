import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Divider, LinearProgress, Tab, Tabs, Table, TableBody, TableCell, TableHead, TableRow, Typography } from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ChevronDown, ExternalLink, RefreshCw } from "lucide-react";
import { useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "../router";

import { api } from "../api/client";
import { DocumentCanvas, type FocusRegion } from "../components/DocumentCanvas";
import { ErrorState, LoadingRows } from "../components/Feedback";
import { FieldCompletenessSummary } from "../components/FieldCompletenessSummary";
import { GenericIntelligencePanel } from "../components/GenericIntelligencePanel";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";
import type { Citation, Evidence, ExtractedField, RiskIndicator, VisionDocument } from "../types/api";

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

  if (documentQuery.isLoading) return <Box className="page-stack"><PageHeader title="Document intelligence" /><LoadingRows rows={10} /></Box>;
  if (documentQuery.error || !document) return <Box className="page-stack"><PageHeader title="Document intelligence" /><ErrorState error={documentQuery.error || new Error("Document not found")} /></Box>;

  const focusField = (field: ExtractedField) => {
    if (!field.evidence.bounding_box) return;
    setPage(field.evidence.page_number);
    setFocus({ page: field.evidence.page_number, box: field.evidence.bounding_box, id: field.field_id });
  };
  const confidence = document.confidence_summary.mean_field_confidence;
  const confidenceLabel = confidence === null
    ? "field confidence not available"
    : `${Math.round(confidence * 100)}% across ${document.confidence_summary.field_count} fields`;
  const schemaLabel = document.schema_profile
    ? `schema mode ${document.schema_profile.mode} (${document.schema_profile.schema_status})`
    : "schema profile unavailable";

  return (
    <Box className="page-stack viewer-page">
      <PageHeader
        title={document.filename}
        subtitle={`${document.document_type.replaceAll("_", " ")} | ${document.page_count} page(s) | ${confidenceLabel} | ${schemaLabel}`}
        actions={<><StatusChip value={document.status} size="medium" />{document.review_task ? <Button component={Link} to={`/reviews/${document.review_task.review_id}`} variant="contained">Open review</Button> : null}<Button startIcon={<RefreshCw size={16} />} disabled={reprocess.isPending} onClick={() => reprocess.mutate(document.document_id)}>{reprocess.isPending ? "Reprocessing" : "Reprocess"}</Button></>}
      />
      {citation ? <Alert severity="info">Opened from answer citation on page {citation.page_number}.</Alert> : null}
      {Array.isArray(document.metadata.legacy_flags) && document.metadata.legacy_flags.includes("LEGACY_APPROVAL_REVIEW_REQUIRED") ? <Alert severity="warning">Approved under an earlier policy. Re-verification is recommended.</Alert> : null}
      {reprocess.error ? <Alert severity="error">{reprocess.error.message}</Alert> : null}
      <Box className="intelligence-workspace">
        <Box className="viewer-pane surface">
          <DocumentCanvas document={document} page={page} onPageChange={setPage} focus={focus} onRegionClick={setFocus} />
        </Box>
        <Box className="intelligence-panel surface">
          <Tabs value={tab} onChange={(_, value) => setTab(value)} variant="scrollable" scrollButtons="auto">
            <Tab label={`Fields ${document.extracted_fields.length}`} />
            <Tab label={`Discovered ${(document.generic_fields ?? []).length}`} />
            <Tab label={`Tables ${document.tables.length}`} />
            <Tab label={`Warnings ${document.processing_warnings.length}`} />
            <Tab label={`Risk ${document.risk_indicators.length}`} />
            <Tab label="Models" />
          </Tabs>
          <Divider />
          <Box className="panel-scroll">
            {tab === 0 ? (
              <Box className="field-list">
                <FieldCompletenessSummary completeness={document.field_completeness} />
                {!document.extracted_fields.length ? (
                  <Alert severity="info">
                    {document.field_completeness.schema_available
                      ? `No structured fields were extracted. ${document.field_completeness.missing_required_fields.length} required fields remain unresolved.`
                      : document.schema_profile?.mode === "C"
                        ? "No registered schema fields were extracted. Schema-independent discovery results are available in the Discovered tab."
                        : "No structured fields were extracted. Expected schema unavailable for this document type."}
                  </Alert>
                ) : null}
                {document.extracted_fields.map((field) => (
                  <button type="button" className={`field-row${focus?.id === field.field_id ? " selected" : ""}`} key={field.field_id} onClick={() => focusField(field)}>
                    <Box><Typography variant="caption" color="text.secondary">{field.name.replaceAll("_", " ")}</Typography><Typography fontWeight={650} className="field-value">{masked(field)}</Typography></Box>
                    <Box className="field-meta"><Typography variant="caption">{field.confidence === null ? "Not available" : `${Math.round(field.confidence * 100)}%`}</Typography><StatusChip value={field.reviewer_status} /></Box>
                  </button>
                ))}
              </Box>
            ) : null}
            {tab === 1 ? (
              <GenericIntelligencePanel
                document={document}
                onFocus={(focusPage, box, id) => {
                  setPage(focusPage);
                  setFocus({ page: focusPage, box, id });
                }}
              />
            ) : null}
            {tab === 2 ? document.tables.length ? document.tables.map((table) => (
              <Box key={table.table_id} className="table-result">
                <Box className="section-heading"><Typography variant="h4">Page {table.page_number}</Typography><Typography variant="caption">{Math.round(table.confidence * 100)}% | {table.model_name} | {table.structure_status.replaceAll("_", " ")}</Typography></Box>
                {table.structure_status === "table_uncertain" ? <Alert severity="warning">Table structure needs review: {table.uncertainty_reasons.map((reason) => reason.replaceAll("_", " ")).join(", ")}.</Alert> : null}
                <Table size="small"><TableBody>{table.rows.map((row, rowIndex) => <TableRow key={rowIndex}>{row.map((cell, cellIndex) => <TableCell key={cellIndex}>{cell}</TableCell>)}</TableRow>)}</TableBody></Table>
              </Box>
            )) : <Typography color="text.secondary">No structural tables were detected.</Typography> : null}
            {tab === 3 ? (
              <Box className="indicator-stack">
                <LanguageEvidence analysis={document.language_analysis} />
                <IndicatorList
                  indicators={document.processing_warnings}
                  empty="No processing warnings were generated."
                  onEvidence={(indicator, evidence) => {
                    setPage(evidence.page_number);
                    setFocus(evidence.bounding_box ? { page: evidence.page_number, box: evidence.bounding_box, id: `${indicator.risk_id}-${evidence.evidence_id}` } : null);
                  }}
                />
              </Box>
            ) : null}
            {tab === 4 ? (
              <IndicatorList
                indicators={document.risk_indicators}
                empty="No business or visual risk indicators were generated."
                onEvidence={(indicator, evidence) => {
                  setPage(evidence.page_number);
                  setFocus(evidence.bounding_box ? { page: evidence.page_number, box: evidence.bounding_box, id: `${indicator.risk_id}-${evidence.evidence_id}` } : null);
                }}
              />
            ) : null}
            {tab === 5 ? (
              <Table size="small"><TableHead><TableRow><TableCell>Stage</TableCell><TableCell>Model</TableCell><TableCell>Status</TableCell><TableCell>Confidence</TableCell></TableRow></TableHead><TableBody>{document.model_runs.map((run) => <TableRow key={String(run.run_id)}><TableCell>{String(run.stage)}</TableCell><TableCell>{String(run.model_name)} <Typography variant="caption">{String(run.model_version)}</Typography></TableCell><TableCell><StatusChip value={String(run.status)} /></TableCell><TableCell>{typeof run.confidence === "number" ? `${Math.round(run.confidence * 100)}%` : "Not reported"}</TableCell></TableRow>)}</TableBody></Table>
            ) : null}
          </Box>
        </Box>
      </Box>
      {document.quality.some((item) => Number(item.image_quality_score) < 0.5) ? <LinearProgress color="warning" /> : null}
    </Box>
  );
}

function LanguageEvidence({ analysis }: { analysis: VisionDocument["language_analysis"] }) {
  if (!analysis.valid_character_count) return null;
  return (
    <Box className="language-evidence">
      <Box className="section-heading">
        <Typography variant="h4">Language evidence</Typography>
        <StatusChip value={analysis.mixed_language_routing ? "mixed" : "single language"} />
      </Box>
      <Typography variant="body2">
        {analysis.detected_languages.map((item) => `${item.script} ${Math.round(item.ratio * 100)}% (${item.character_count})`).join(" | ")}
      </Typography>
      <Typography variant="caption" color="text.secondary">
        {analysis.valid_character_count} cleaned characters | secondary {Math.round(analysis.secondary_language_ratio * 100)}% | confidence {analysis.confidence === null ? "not available" : `${Math.round(analysis.confidence * 100)}%`} | {analysis.decision_reason.replaceAll("_", " ")}
      </Typography>
    </Box>
  );
}

function IndicatorList({
  indicators,
  empty,
  onEvidence,
}: {
  indicators: RiskIndicator[];
  empty: string;
  onEvidence: (indicator: RiskIndicator, evidence: Evidence) => void;
}) {
  if (!indicators.length) return <Typography color="text.secondary">{empty}</Typography>;
  return indicators.map((indicator) => (
    <Accordion key={indicator.risk_id} disableGutters elevation={0}>
      <AccordionSummary expandIcon={<ChevronDown size={17} />}>
        <Box className="risk-heading"><StatusChip value={indicator.severity} /><Box><Typography fontWeight={700}>{indicator.title}</Typography><Typography variant="caption" color="text.secondary">{indicator.category.replaceAll("_", " ")}</Typography></Box></Box>
      </AccordionSummary>
      <AccordionDetails className="indicator-details">
        <Typography>{indicator.description}</Typography>
        <Typography variant="caption" color="text.secondary">{indicator.model_or_rule || indicator.rule_id || `${indicator.model_name}:${indicator.model_version}`} | {indicator.confidence === null || indicator.confidence === undefined ? "Confidence not available" : `${Math.round(indicator.confidence * 100)}%`}</Typography>
        {indicator.affected_fields?.length ? <Typography variant="body2">Fields: {indicator.affected_fields.join(", ")}</Typography> : null}
        <Typography color="text.secondary">{indicator.recommended_action}</Typography>
        <Box className="indicator-evidence-actions">
          {indicator.evidence.map((evidence) => <Button key={evidence.evidence_id} size="small" startIcon={<ExternalLink size={14} />} onClick={() => onEvidence(indicator, evidence)}>Page {evidence.page_number}</Button>)}
        </Box>
      </AccordionDetails>
    </Accordion>
  ));
}

function masked(field: ExtractedField) {
  const value = field.reviewed_value || field.value;
  if (!field.name.toLowerCase().includes("bank") && !field.name.toLowerCase().includes("account")) return value;
  return `${"*".repeat(Math.max(4, value.length - 4))}${value.slice(-4)}`;
}
