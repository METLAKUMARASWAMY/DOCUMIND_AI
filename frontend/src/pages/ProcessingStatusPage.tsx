import {
  Alert,
  Box,
  Button,
  LinearProgress,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Typography,
} from "@mui/material";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Circle, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "../router";

import { api, streamDocumentEvents } from "../api/client";
import { ErrorState } from "../components/Feedback";
import { useNotify } from "../components/Notifications";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";

/**
 * The pipeline as the backend actually reports it.
 *
 * `status` is the raw VisionProcessingStatus the API returns; `label` and
 * `detail` are for the reader. The two are shown together rather than one
 * replacing the other, so the page stays understandable without hiding the
 * value an operator would quote in a support thread.
 */
const stages = [
  {
    status: "QUEUED",
    label: "Queued",
    detail: "The job is accepted and waiting for a worker.",
  },
  {
    status: "PREPROCESSING",
    label: "Preparing pages",
    detail: "Deskew, denoise and contrast normalisation before text is read.",
  },
  {
    status: "OCR_COMPLETED",
    label: "Reading text",
    detail: "OCR and layout analysis, with a confidence score for every word.",
  },
  {
    status: "EXTRACTION_COMPLETED",
    label: "Extracting fields",
    detail: "Structured fields and schema-independent observations, each tied to evidence.",
  },
  {
    status: "VALIDATION_COMPLETED",
    label: "Checking for risk",
    detail: "Validation rules, duplicate detection and manipulation screening.",
  },
  {
    status: "INDEXED",
    label: "Indexing and linking",
    detail: "Chunks written to vector search; entities and relationships written to the graph.",
  },
  {
    status: "COMPLETED",
    label: "Ready",
    detail: "The document can be queried, reviewed and cited.",
  },
] as const;

export function ProcessingStatusPage() {
  const { jobId = "" } = useParams();
  const location = useLocation();
  const queryClient = useQueryClient();
  const notify = useNotify();
  const [events, setEvents] = useState<Array<{ type: string; data: Record<string, unknown> }>>([]);
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId),
    refetchInterval: (query) => (isTerminal(String(query.state.data?.status || "")) ? false : 1200),
  });
  const documentIds = ((location.state as { documentIds?: string[] } | null)?.documentIds ||
    job.data?.document_ids ||
    []) as string[];
  const firstDocumentId = documentIds[0] || "";

  useEffect(() => {
    if (!firstDocumentId) return;
    const controller = new AbortController();
    void streamDocumentEvents(firstDocumentId, controller.signal, (event) => {
      setEvents((current) => [...current.slice(-39), event]);
      void queryClient.invalidateQueries({ queryKey: ["job", jobId] });
    }).catch(() => undefined);
    return () => controller.abort();
  }, [firstDocumentId, jobId, queryClient]);

  const status = String(job.data?.status || "QUEUED");
  const pipelineStatus = String(job.data?.pipeline_status || status);
  const decisionStatus = String(job.data?.decision_status || "PENDING");
  const currentIndex = Math.max(
    0,
    stages.findIndex((stage) => stage.status === pipelineStatus),
  );
  const extractionOutcomes = (job.data?.extraction_outcomes || []) as Array<
    Record<string, unknown>
  >;
  return (
    <Box className="page-stack">
      <PageHeader
        title="Processing status"
        subtitle={`${String(job.data?.status_summary || "Processing document")} | Job ${jobId}`}
        actions={
          <>
            <Box>
              <Typography variant="caption" display="block">
                Pipeline status
              </Typography>
              <StatusChip value={pipelineStatus} size="medium" />
            </Box>
            <Box>
              <Typography variant="caption" display="block">
                Decision status
              </Typography>
              <StatusChip value={decisionStatus} size="medium" />
            </Box>
          </>
        }
      />
      {job.error ? <ErrorState error={job.error} retry={() => void job.refetch()} /> : null}
      {decisionStatus === "REVIEW_REQUIRED" ? (
        <Alert severity="warning">Processing completed - awaiting human review.</Alert>
      ) : null}
      <Box className="processing-grid">
        <Box className="surface section-panel">
          <Typography variant="h3">Pipeline</Typography>
          {!isTerminal(status) ? <LinearProgress aria-label="Processing in progress" /> : null}
          <Typography className="visually-hidden" role="status">
            {`Stage ${currentIndex + 1} of ${stages.length}: ${stages[currentIndex]?.label ?? pipelineStatus}`}
          </Typography>
          <List aria-label="Processing pipeline">
            {stages.map((stage, index) => {
              const done = index <= currentIndex;
              const current = index === currentIndex && !isTerminal(status);
              return (
                <ListItem
                  key={stage.status}
                  disableGutters
                  className={current ? "pipeline-stage is-current" : "pipeline-stage"}
                  aria-current={current ? "step" : undefined}
                >
                  <ListItemIcon>
                    {done ? (
                      <CheckCircle2 size={19} color="#087f78" aria-hidden />
                    ) : (
                      <Circle size={19} color="#97a2ab" aria-hidden />
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary={
                      <>
                        {stage.label}
                        <Typography component="span" variant="caption" className="stage-status">
                          {stage.status}
                        </Typography>
                      </>
                    }
                    secondary={stage.detail}
                  />
                  <Typography variant="caption" color="text.secondary">
                    {done ? "Done" : current ? "In progress" : "Pending"}
                  </Typography>
                </ListItem>
              );
            })}
          </List>
          {extractionOutcomes.map((outcome) => (
            <Alert
              key={String(outcome.document_id)}
              severity={number(outcome.fields_extracted) ? "success" : "info"}
            >
              {number(outcome.fields_extracted)
                ? `Extraction completed: ${number(outcome.fields_extracted)} fields extracted using ${String(outcome.schema || "generic")} schema.`
                : "Extraction completed with no fields. Generic schema fallback was attempted."}
            </Alert>
          ))}
          <Box className="form-actions">
            {firstDocumentId && isTerminal(status) ? (
              <Button component={Link} to={`/documents/${firstDocumentId}`} variant="contained">
                Open document
              </Button>
            ) : null}
            {decisionStatus === "REVIEW_REQUIRED" ? (
              <Button component={Link} to="/reviews" variant="outlined">
                Open review queue
              </Button>
            ) : null}
            {(job.data?.errors as unknown[])?.length ? (
              <Button
                startIcon={<RefreshCw size={16} />}
                onClick={() =>
                  void api
                    .retryJob(jobId)
                    .then(() => {
                      notify("info", "The job was re-queued.");
                      return job.refetch();
                    })
                    .catch((reason: unknown) =>
                      notify(
                        "error",
                        reason instanceof Error ? reason.message : "The job could not be retried.",
                      ),
                    )
                }
              >
                Retry job
              </Button>
            ) : null}
          </Box>
        </Box>
        <Box className="surface section-panel live-events">
          <Typography variant="h3">Live events</Typography>
          {events.length ? (
            events.map((event, index) => (
              <Box className="event-row" key={`${event.type}-${index}`}>
                <Typography variant="caption" fontWeight={700}>
                  {event.type.replaceAll("_", " ")}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {String(event.data.message || event.data.status || "Updated")}
                </Typography>
              </Box>
            ))
          ) : (
            <Typography color="text.secondary">Waiting for the first processing event.</Typography>
          )}
        </Box>
      </Box>
    </Box>
  );
}

function number(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function isTerminal(status: string) {
  return [
    "REVIEW_REQUIRED",
    "COMPLETED",
    "APPROVED",
    "REJECTED",
    "CONFIRMED_ERROR",
    "CONFIRMED_FRAUD",
  ].includes(status);
}
