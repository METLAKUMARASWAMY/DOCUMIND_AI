import { Alert, Box, Button, ButtonGroup, Dialog, DialogActions, DialogContent, DialogTitle, Divider, MenuItem, TextField, Typography } from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, CheckCheck, CircleAlert, RefreshCw, ShieldAlert, X } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate, useParams } from "../router";

import { api } from "../api/client";
import { DocumentCanvas, type FocusRegion } from "../components/DocumentCanvas";
import { ErrorState, LoadingRows } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";
import type { ExtractedField } from "../types/api";

export function ReviewWorkspacePage() {
  const { reviewId = "" } = useParams();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const reviewQuery = useQuery({ queryKey: ["review", reviewId], queryFn: () => api.review(reviewId) });
  const [page, setPage] = useState(1);
  const [focus, setFocus] = useState<FocusRegion | null>(null);
  const [comment, setComment] = useState("");
  const [dialogAction, setDialogAction] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, { value: string; status: string }>>({});
  const document = reviewQuery.data?.document;
  const review = reviewQuery.data?.review;

  const patch = useMutation({
    mutationFn: ({ field, status }: { field: ExtractedField; status: string }) => api.patchField(reviewId, field.field_id, { reviewed_value: drafts[field.field_id]?.value, reviewer_status: status, comment }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["review", reviewId] }),
  });
  const decide = useMutation({
    mutationFn: (action: string) => api.decideReview(reviewId, action, comment),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["reviews"] });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      setDialogAction(null);
      navigate("/reviews");
    },
  });
  const unresolved = useMemo(
    () => document?.extracted_fields.filter((field) => !["accepted", "edited", "rejected"].includes(drafts[field.field_id]?.status || field.reviewer_status)).length || 0,
    [document, drafts],
  );

  if (reviewQuery.isLoading) return <Box className="page-stack"><PageHeader title="Review workspace" /><LoadingRows rows={10} /></Box>;
  if (reviewQuery.error || !document || !review) return <Box className="page-stack"><PageHeader title="Review workspace" /><ErrorState error={reviewQuery.error || new Error("Review not found")} /></Box>;

  return (
    <Box className="page-stack review-page">
      <PageHeader title={document.filename} subtitle={`${review.reasons.length} routing reason(s) | ${unresolved} unresolved field(s)`} actions={<StatusChip value={review.priority} size="medium" />} />
      {patch.error || decide.error ? <ErrorState error={patch.error || decide.error} /> : null}
      <Box className="review-workspace">
        <Box className="viewer-pane surface"><DocumentCanvas document={document} page={page} onPageChange={setPage} focus={focus} onRegionClick={setFocus} /></Box>
        <Box className="review-panel surface">
          <Box className="review-summary"><Box><Typography variant="h3">Extracted fields</Typography><Typography color="text.secondary">Original and reviewed values remain in the audit trail.</Typography></Box><Button startIcon={<CheckCheck size={16} />} onClick={() => void api.acceptHighConfidence(reviewId).then(() => reviewQuery.refetch())}>Accept high confidence</Button></Box>
          <Divider />
          <Box className="review-fields">
            {document.extracted_fields.map((field) => {
              const draft = drafts[field.field_id] || { value: field.reviewed_value || field.value, status: field.reviewer_status };
              return (
                <Box className="review-field" key={field.field_id} onClick={() => field.evidence.bounding_box && setFocus({ page: field.evidence.page_number, box: field.evidence.bounding_box, id: field.field_id })}>
                  <Box className="review-field-heading"><Typography fontWeight={700}>{field.name.replaceAll("_", " ")}</Typography><Typography variant="caption">{Math.round(field.confidence * 100)}% | {field.model_name}</Typography></Box>
                  <Typography variant="caption" color="text.secondary">Original: {mask(field.name, field.value)}</Typography>
                  <TextField size="small" value={draft.value} onChange={(event) => setDrafts((current) => ({ ...current, [field.field_id]: { ...draft, value: event.target.value, status: "edited" } }))} fullWidth />
                  <Box className="field-review-actions">
                    <ButtonGroup size="small" variant="outlined">
                      <Button color="success" startIcon={<Check size={14} />} onClick={() => { setDrafts((current) => ({ ...current, [field.field_id]: { ...draft, status: "accepted" } })); patch.mutate({ field, status: "accepted" }); }}>Accept</Button>
                      <Button color="error" startIcon={<X size={14} />} onClick={() => { setDrafts((current) => ({ ...current, [field.field_id]: { ...draft, status: "rejected" } })); patch.mutate({ field, status: "rejected" }); }}>Reject</Button>
                    </ButtonGroup>
                    {draft.status === "edited" ? <Button size="small" variant="contained" onClick={() => patch.mutate({ field, status: "edited" })}>Save edit</Button> : <StatusChip value={draft.status} />}
                  </Box>
                </Box>
              );
            })}
          </Box>
          <Box className="review-decision-bar">
            <TextField label="Reviewer comments" multiline minRows={2} value={comment} onChange={(event) => setComment(event.target.value)} fullWidth />
            <Box className="decision-actions">
              <Button variant="contained" color="success" startIcon={<Check size={16} />} onClick={() => setDialogAction("approve")}>Approve</Button>
              <Button variant="outlined" color="error" startIcon={<X size={16} />} onClick={() => setDialogAction("reject")}>Reject</Button>
              <Button variant="outlined" color="warning" startIcon={<CircleAlert size={16} />} onClick={() => setDialogAction("escalate")}>Escalate</Button>
              <Button variant="outlined" startIcon={<RefreshCw size={16} />} onClick={() => void api.reprocessDocument(document.document_id)}>Reprocess</Button>
              <Button variant="outlined" color="error" onClick={() => setDialogAction("confirm-error")}>Mark error</Button>
              <Button variant="contained" color="error" startIcon={<ShieldAlert size={16} />} onClick={() => setDialogAction("confirm-fraud")}>Confirm fraud</Button>
            </Box>
          </Box>
        </Box>
      </Box>
      <Dialog open={Boolean(dialogAction)} onClose={() => setDialogAction(null)} maxWidth="sm" fullWidth>
        <DialogTitle>{dialogAction?.replaceAll("-", " ")}</DialogTitle>
        <DialogContent>
          <Alert severity={dialogAction === "confirm-fraud" ? "error" : "warning"}>
            This authorised decision is immutable in the audit history. Confirm it only after reviewing source evidence and unresolved fields.
          </Alert>
          <TextField select label="Decision" value={dialogAction || ""} fullWidth sx={{ mt: 2 }} disabled>
            <MenuItem value={dialogAction || ""}>{dialogAction?.replaceAll("-", " ")}</MenuItem>
          </TextField>
        </DialogContent>
        <DialogActions><Button onClick={() => setDialogAction(null)}>Cancel</Button><Button variant="contained" color={dialogAction === "confirm-fraud" ? "error" : "primary"} disabled={decide.isPending} onClick={() => dialogAction && decide.mutate(dialogAction)}>Record decision</Button></DialogActions>
      </Dialog>
    </Box>
  );
}

function mask(name: string, value: string) {
  return name.includes("bank") || name.includes("account") ? `${"*".repeat(Math.max(4, value.length - 4))}${value.slice(-4)}` : value;
}
