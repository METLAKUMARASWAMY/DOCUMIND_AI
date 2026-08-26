import {
  Alert,
  Box,
  Button,
  MenuItem,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ExternalLink, FlaskConical } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "../router";

import { api } from "../api/client";
import { ErrorState, LoadingRows } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";

type ComparisonRow = Record<string, unknown>;

export function ModelLabPage() {
  const navigate = useNavigate();
  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const [documentId, setDocumentId] = useState("");
  const comparison = useMutation({ mutationFn: () => api.modelComparison(documentId) });
  const rows = (comparison.data?.field_output || []) as ComparisonRow[];
  const primary = (comparison.data?.primary_adapter || {}) as Record<string, unknown>;
  const fallback = (comparison.data?.fallback_adapter || {}) as Record<string, unknown>;
  const availability = (comparison.data?.availability || []) as Array<Record<string, unknown>>;
  const routingReasons = (comparison.data?.routing_reason || []) as string[];

  if (documents.isLoading) return <LoadingRows />;
  if (documents.error) return <ErrorState error={documents.error} />;

  return (
    <Box className="page-stack">
      <PageHeader
        title="Model comparison laboratory"
        subtitle="Observed field outputs, runtime provenance, and evidence for one processed document."
      />
      <Box className="lab-toolbar surface">
        <TextField
          select
          size="small"
          label="Document"
          value={documentId}
          onChange={(event) => setDocumentId(event.target.value)}
        >
          <MenuItem value="">Select document</MenuItem>
          {documents.data?.documents.map((document) => (
            <MenuItem key={document.document_id} value={document.document_id}>
              {document.filename}
            </MenuItem>
          ))}
        </TextField>
        <Button
          variant="contained"
          startIcon={<FlaskConical size={16} />}
          disabled={!documentId || comparison.isPending}
          onClick={() => comparison.mutate()}
        >
          {comparison.isPending ? "Comparing" : "Compare outputs"}
        </Button>
      </Box>
      {comparison.error ? <Alert severity="error">{comparison.error.message}</Alert> : null}
      {comparison.data ? (
        <>
          <Alert severity="info">{String(comparison.data.note)}</Alert>
          <Box className="lab-status-band">
            <AdapterSummary label="Observed primary" adapter={primary} />
            <AdapterSummary label="Deterministic fallback" adapter={fallback} />
            <Box className="lab-route-summary">
              <Typography variant="overline">Routing reason</Typography>
              <Typography variant="body2">
                {routingReasons.length
                  ? routingReasons.map(readable).join(", ")
                  : "Standard extraction route"}
              </Typography>
            </Box>
          </Box>
          <Box className="surface table-surface model-comparison-table">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Field</TableCell>
                  <TableCell>Primary result</TableCell>
                  <TableCell>Primary confidence</TableCell>
                  <TableCell>Fallback result</TableCell>
                  <TableCell>Fallback confidence</TableCell>
                  <TableCell>LayoutLMv3 result</TableCell>
                  <TableCell>Donut result</TableCell>
                  <TableCell>Validation result</TableCell>
                  <TableCell>Selected result</TableCell>
                  <TableCell>Review status</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {rows.length ? (
                  rows.map((row, index) => (
                    <TableRow
                      key={String(row.field_id || row.field_name || index)}
                      className={row.disagreement ? "comparison-disagreement" : undefined}
                    >
                      <TableCell>
                        <Typography fontWeight={700}>
                          {readable(String(row.field_name || "field"))}
                        </Typography>
                        {row.required ? (
                          <Typography variant="caption" color="error">
                            Required
                          </Typography>
                        ) : null}
                      </TableCell>
                      <ValueCell value={row.primary_result} model={row.primary_model} />
                      <ConfidenceCell value={row.primary_confidence} />
                      <ValueCell value={row.fallback_result} model={row.fallback_model} />
                      <ConfidenceCell value={row.fallback_confidence} />
                      <ValueCell value={row.layoutlmv3_result} />
                      <ValueCell value={row.donut_result} />
                      <TableCell>
                        <StatusChip value={String(row.validation_result || "not_available")} />
                        {row.disagreement ? (
                          <Typography variant="caption" color="warning.main" display="block">
                            Disagreement
                          </Typography>
                        ) : null}
                      </TableCell>
                      <TableCell>
                        <Stack spacing={0.5}>
                          <CellValue value={row.selected_result} />
                          {row.evidence ? (
                            <Button
                              size="small"
                              startIcon={<ExternalLink size={13} />}
                              onClick={() =>
                                openEvidence(
                                  navigate,
                                  documentId,
                                  row.evidence as Record<string, unknown>,
                                )
                              }
                            >
                              Evidence
                            </Button>
                          ) : null}
                        </Stack>
                      </TableCell>
                      <TableCell>
                        <StatusChip value={String(row.review_status || "not_available")} />
                      </TableCell>
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell colSpan={10}>
                      <Typography color="text.secondary">
                        No schema or observed field outputs are available for this document.
                      </Typography>
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </Box>
          <Box className="surface table-surface">
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Model availability</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Health</TableCell>
                  <TableCell>Smoke test</TableCell>
                  <TableCell>Reason</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {availability.map((model) => (
                  <TableRow key={String(model.model_id)}>
                    <TableCell>
                      <Typography fontWeight={700}>{String(model.model_id)}</Typography>
                    </TableCell>
                    <TableCell>
                      <StatusChip value={String(model.runtime_status || model.status)} />
                    </TableCell>
                    <TableCell>{String(model.health_status || "unknown")}</TableCell>
                    <TableCell>{String(model.smoke_test_status || "not_run")}</TableCell>
                    <TableCell>{String(model.reason || "Unavailable")}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </Box>
        </>
      ) : null}
    </Box>
  );
}

function AdapterSummary({ label, adapter }: { label: string; adapter: Record<string, unknown> }) {
  const status = String(adapter.status || "not_recorded");
  return (
    <Box className="lab-adapter-summary">
      <Typography variant="overline">{label}</Typography>
      <Typography fontWeight={800}>
        {adapter.model_id ? String(adapter.model_id) : "Unavailable"}
      </Typography>
      <Stack direction="row" spacing={1} alignItems="center">
        <StatusChip value={status} />
        <Typography variant="caption">
          {typeof adapter.runtime_ms === "number"
            ? `${Math.round(adapter.runtime_ms)} ms`
            : "Runtime not recorded"}
        </Typography>
      </Stack>
    </Box>
  );
}

function ValueCell({ value, model }: { value: unknown; model?: unknown }) {
  return (
    <TableCell>
      <CellValue value={value} />
      {model ? (
        <Typography variant="caption" color="text.secondary" display="block">
          {String(model)}
        </Typography>
      ) : null}
    </TableCell>
  );
}

function CellValue({ value }: { value: unknown }) {
  if (value === null || value === undefined || value === "")
    return (
      <Typography variant="caption" color="text.secondary">
        Unavailable
      </Typography>
    );
  return <Typography variant="body2">{String(value)}</Typography>;
}

function ConfidenceCell({ value }: { value: unknown }) {
  const numeric = typeof value === "number" ? value : Number.NaN;
  return (
    <TableCell>
      {Number.isFinite(numeric) ? (
        `${Math.round(numeric * 100)}%`
      ) : (
        <Typography variant="caption" color="text.secondary">
          Unavailable
        </Typography>
      )}
    </TableCell>
  );
}

function readable(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function openEvidence(
  navigate: ReturnType<typeof useNavigate>,
  documentId: string,
  evidence: Record<string, unknown>,
) {
  navigate(`/documents/${documentId}`, {
    state: {
      citation: {
        document_id: documentId,
        page_number: Number(evidence.page_number || 1),
        source_text: String(evidence.source_text || ""),
        bounding_boxes: evidence.bounding_box ? [evidence.bounding_box] : [],
        evidence_id: evidence.evidence_id,
        model_confidence: evidence.model_confidence,
        model_name: evidence.model_name,
        model_version: evidence.model_version,
        processing_timestamp: evidence.processing_timestamp,
        reviewer_status: evidence.reviewer_status,
      },
    },
  });
}
