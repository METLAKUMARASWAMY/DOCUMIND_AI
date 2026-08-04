import { Alert, Box, Button, LinearProgress, List, ListItem, ListItemIcon, ListItemText, Typography } from "@mui/material";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CheckCircle2, Circle, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "../router";

import { api, streamDocumentEvents } from "../api/client";
import { ErrorState } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";

const stages = ["QUEUED", "PREPROCESSING", "OCR_COMPLETED", "EXTRACTION_COMPLETED", "VALIDATION_COMPLETED", "INDEXED", "REVIEW_REQUIRED", "COMPLETED"];

export function ProcessingStatusPage() {
  const { jobId = "" } = useParams();
  const location = useLocation();
  const queryClient = useQueryClient();
  const [events, setEvents] = useState<Array<{ type: string; data: Record<string, unknown> }>>([]);
  const job = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.job(jobId),
    refetchInterval: (query) => isTerminal(String(query.state.data?.status || "")) ? false : 1200,
  });
  const documentIds = ((location.state as { documentIds?: string[] } | null)?.documentIds || job.data?.document_ids || []) as string[];
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
  const currentIndex = Math.max(0, stages.indexOf(status));
  return (
    <Box className="page-stack">
      <PageHeader title="Processing status" subtitle={`Job ${jobId}`} actions={<StatusChip value={status} size="medium" />} />
      {job.error ? <ErrorState error={job.error} retry={() => void job.refetch()} /> : null}
      {status === "REVIEW_REQUIRED" ? <Alert severity="warning">Processing completed with evidence that requires reviewer verification.</Alert> : null}
      <Box className="processing-grid">
        <Box className="surface section-panel">
          <Typography variant="h3">Pipeline</Typography>
          {!isTerminal(status) ? <LinearProgress /> : null}
          <List>
            {stages.map((stage, index) => (
              <ListItem key={stage} disableGutters>
                <ListItemIcon>{index <= currentIndex ? <CheckCircle2 size={19} color="#087f78" /> : <Circle size={19} color="#97a2ab" />}</ListItemIcon>
                <ListItemText primary={stage.replaceAll("_", " ")} />
              </ListItem>
            ))}
          </List>
          <Box className="form-actions">
            {firstDocumentId && isTerminal(status) ? <Button component={Link} to={`/documents/${firstDocumentId}`} variant="contained">Open document</Button> : null}
            {status === "REVIEW_REQUIRED" ? <Button component={Link} to="/reviews" variant="outlined">Open review queue</Button> : null}
            {(job.data?.errors as unknown[])?.length ? <Button startIcon={<RefreshCw size={16} />} onClick={() => void api.retryJob(jobId).then(() => job.refetch())}>Retry job</Button> : null}
          </Box>
        </Box>
        <Box className="surface section-panel live-events">
          <Typography variant="h3">Live events</Typography>
          {events.length ? events.map((event, index) => (
            <Box className="event-row" key={`${event.type}-${index}`}>
              <Typography variant="caption" fontWeight={700}>{event.type.replaceAll("_", " ")}</Typography>
              <Typography variant="body2" color="text.secondary">{String(event.data.message || event.data.status || "Updated")}</Typography>
            </Box>
          )) : <Typography color="text.secondary">Waiting for the first processing event.</Typography>}
        </Box>
      </Box>
    </Box>
  );
}

function isTerminal(status: string) {
  return ["REVIEW_REQUIRED", "COMPLETED", "APPROVED", "REJECTED", "CONFIRMED_ERROR", "CONFIRMED_FRAUD"].includes(status);
}
