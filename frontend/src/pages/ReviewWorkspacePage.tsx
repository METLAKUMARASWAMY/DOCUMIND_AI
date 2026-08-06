import { Alert, Box, Button, ButtonGroup, Checkbox, Dialog, DialogActions, DialogContent, DialogTitle, Divider, FormControlLabel, MenuItem, TextField, Typography } from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, CheckCheck, CircleAlert, RefreshCw, ShieldAlert, X } from "lucide-react";
import { useMemo, useState } from "react";
import { useNavigate, useParams } from "../router";

import { api } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { DocumentCanvas, type FocusRegion } from "../components/DocumentCanvas";
import { ErrorState, LoadingRows } from "../components/Feedback";
import { FieldCompletenessSummary } from "../components/FieldCompletenessSummary";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";
import type { ExtractedField } from "../types/api";
import { reviewApprovalControl, secondReviewerReady } from "../utils/reviewPolicy";

export function ReviewWorkspacePage() {
  const { reviewId = "" } = useParams();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const { user } = useAuth();
  const reviewQuery = useQuery({ queryKey: ["review", reviewId], queryFn: () => api.review(reviewId) });
  const [page, setPage] = useState(1);
  const [focus, setFocus] = useState<FocusRegion | null>(null);
  const [comment, setComment] = useState("");
  const [approvalReason, setApprovalReason] = useState("");
  const [manualVerification, setManualVerification] = useState(false);
  const [confirmationAcknowledged, setConfirmationAcknowledged] = useState(false);
  const [selectedEvidence, setSelectedEvidence] = useState<string[]>([]);
  const [confirmedFields, setConfirmedFields] = useState<Record<string, string>>({});
  const [fraudPassword, setFraudPassword] = useState("");
  const [fraudRationale, setFraudRationale] = useState("");
  const [automatedIndicatorsAcknowledged, setAutomatedIndicatorsAcknowledged] = useState(false);
  const [secondReviewerEmail, setSecondReviewerEmail] = useState("");
  const [secondReviewerApproved, setSecondReviewerApproved] = useState(false);
  const [dialogAction, setDialogAction] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, { value: string; status: string }>>({});
  const document = reviewQuery.data?.document;
  const review = reviewQuery.data?.review;
  const approvalPolicy = reviewQuery.data?.approval_policy;
  const canConfirmFraud = ["ADMIN", "SENIOR_REVIEWER", "COMPLIANCE_OFFICER"].includes(user?.role || "");
  const signedInRole = (user?.role || "VIEWER").replaceAll("_", " ").replace(/\b\w/g, (character) => character.toUpperCase());

  const patch = useMutation({
    mutationFn: ({ field, status }: { field: ExtractedField; status: string }) => api.patchField(reviewId, field.field_id, { reviewed_value: drafts[field.field_id]?.value, reviewer_status: status, comment }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["review", reviewId] }),
  });
  const decide = useMutation({
    mutationFn: async (action: string) => {
      if (action === "confirm-fraud") {
        const reauthentication = await api.reauthenticate(fraudPassword);
        return api.decideReview(reviewId, action, {
          comment,
          decision_rationale: fraudRationale,
          evidence_ids: selectedEvidence,
          automated_indicators_acknowledged: automatedIndicatorsAcknowledged,
          confirmation_acknowledged: confirmationAcknowledged,
          reauthentication_token: reauthentication.reauthentication_token,
          second_reviewer_email: secondReviewerEmail || undefined,
          second_reviewer_approved: secondReviewerApproved,
        });
      }
      if (action === "approve") {
        const requiredFields = approvalPolicy?.required_confirmed_fields || [];
        const values = Object.fromEntries(requiredFields.map((name) => {
          const field = document?.extracted_fields.find((item) => item.name === name);
          return [name, confirmedFields[name] ?? field?.reviewed_value ?? field?.value ?? ""];
        }));
        return api.decideReview(reviewId, action, {
          comment,
          approval_reason: approvalReason,
          manual_verification_completed: manualVerification,
          confirmed_fields: values,
          confirmation_acknowledged: confirmationAcknowledged,
          evidence_ids: selectedEvidence,
        });
      }
      return api.decideReview(reviewId, action, { comment });
    },
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["reviews"] });
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      setDialogAction(null);
      navigate("/reviews");
    },
  });
  const unresolved = useMemo(
    () => document?.field_completeness.unresolved_required_count ?? null,
    [document],
  );
  const requiredFieldValuesComplete = (approvalPolicy?.required_confirmed_fields || []).every((name) => {
    const field = document?.extracted_fields.find((item) => item.name === name);
    return Boolean((confirmedFields[name] ?? field?.reviewed_value ?? field?.value ?? "").trim());
  });
  const approvalControl = reviewApprovalControl({
    policy: approvalPolicy,
    comment,
    selectedEvidenceIds: selectedEvidence,
    manualVerification,
    approvalReason,
    requiredFieldValuesComplete,
  });
  const approvalReady = approvalControl.ready;
  const hardPolicyBlockers = approvalControl.hardBlockers;
  const fraudReady = Boolean(
    fraudPassword
    && fraudRationale.trim().length >= 20
    && comment.trim().length >= 10
    && selectedEvidence.length
    && automatedIndicatorsAcknowledged
    && confirmationAcknowledged
    && (
      !approvalPolicy?.fraud_second_reviewer_required
      || secondReviewerReady(secondReviewerEmail, user?.email, secondReviewerApproved)
    )
  );

  const toggleEvidence = (evidenceId: string) => {
    setSelectedEvidence((current) => current.includes(evidenceId)
      ? current.filter((item) => item !== evidenceId)
      : [...current, evidenceId]);
  };

  const openDecision = (action: string) => {
    setConfirmationAcknowledged(false);
    setDialogAction(action);
  };
  const closeDecision = () => {
    setDialogAction(null);
    setFraudPassword("");
    setFraudRationale("");
    setAutomatedIndicatorsAcknowledged(false);
    setSecondReviewerEmail("");
    setSecondReviewerApproved(false);
    setConfirmationAcknowledged(false);
  };

  if (reviewQuery.isLoading) return <Box className="page-stack"><PageHeader title="Review workspace" /><LoadingRows rows={10} /></Box>;
  if (reviewQuery.error || !document || !review) return <Box className="page-stack"><PageHeader title="Review workspace" /><ErrorState error={reviewQuery.error || new Error("Review not found")} /></Box>;

  return (
    <Box className="page-stack review-page">
      <PageHeader title={document.filename} subtitle={`Signed in as ${signedInRole} | ${review.reasons.length} routing reason(s) | ${unresolved === null ? "expected schema unavailable" : `${unresolved} unresolved required field(s)`}`} actions={<StatusChip value={review.priority} size="medium" />} />
      {patch.error || decide.error ? <ErrorState error={patch.error || decide.error} /> : null}
      {approvalPolicy?.manual_override_required ? <Alert severity="warning">{approvalPolicy.message}</Alert> : null}
      {hardPolicyBlockers.length ? <Alert severity="error">{hardPolicyBlockers.map((item) => item.message).join(" ")}</Alert> : null}
      <Box className="review-workspace">
        <Box className="viewer-pane surface"><DocumentCanvas document={document} page={page} onPageChange={setPage} focus={focus} onRegionClick={setFocus} /></Box>
        <Box className="review-panel surface">
          <Box className="review-summary"><Box><Typography variant="h3">Schema evidence</Typography><FieldCompletenessSummary completeness={document.field_completeness} /></Box><Button startIcon={<CheckCheck size={16} />} onClick={() => void api.acceptHighConfidence(reviewId).then(() => reviewQuery.refetch())}>Accept high confidence</Button></Box>
          <Divider />
          <Box className="review-fields">
            {!document.extracted_fields.length ? <Alert severity="info">No structured fields were extracted from this document.</Alert> : null}
            {document.extracted_fields.map((field) => {
              const draft = drafts[field.field_id] || { value: field.reviewed_value || field.value, status: field.reviewer_status };
              return (
                <Box className="review-field" key={field.field_id} onClick={() => field.evidence.bounding_box && setFocus({ page: field.evidence.page_number, box: field.evidence.bounding_box, id: field.field_id })}>
                  <Box className="review-field-heading"><Typography fontWeight={700}>{field.display_name || field.name.replaceAll("_", " ")}{field.required ? " *" : ""}</Typography><Box><Typography variant="caption">{field.confidence === null ? "Confidence not available" : `${Math.round(field.confidence * 100)}%`} | {field.model_name}</Typography><FormControlLabel onClick={(event) => event.stopPropagation()} control={<Checkbox size="small" checked={selectedEvidence.includes(field.evidence.evidence_id)} onChange={() => toggleEvidence(field.evidence.evidence_id)} />} label="Evidence reviewed" /></Box></Box>
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
            {approvalPolicy?.manual_override_required ? (
              <Box className="manual-verification-panel">
                <Typography variant="h4">Complete manual verification</Typography>
                {(approvalPolicy.required_confirmed_fields || []).map((name) => {
                  const field = document.extracted_fields.find((item) => item.name === name);
                  const value = confirmedFields[name] ?? field?.reviewed_value ?? field?.value ?? "";
                  return <TextField key={name} size="small" required label={name.replaceAll("_", " ")} value={value} onChange={(event) => setConfirmedFields((current) => ({ ...current, [name]: event.target.value }))} fullWidth />;
                })}
                <FormControlLabel control={<Checkbox checked={manualVerification} onChange={(event) => setManualVerification(event.target.checked)} />} label="Complete manual verification performed" />
                <TextField label="Approval reason" required value={approvalReason} onChange={(event) => setApprovalReason(event.target.value)} multiline minRows={2} fullWidth />
              </Box>
            ) : null}
          </Box>
          <Box className="review-decision-bar">
            <TextField label="Reviewer comments" multiline minRows={2} value={comment} onChange={(event) => setComment(event.target.value)} fullWidth />
            <Box className="decision-actions">
              <Button variant="contained" color="success" startIcon={<Check size={16} />} disabled={!approvalReady} onClick={() => openDecision("approve")}>Approve</Button>
              <Button variant="outlined" color="error" startIcon={<X size={16} />} onClick={() => openDecision("reject")}>Reject</Button>
              <Button variant="outlined" color="warning" startIcon={<CircleAlert size={16} />} onClick={() => openDecision("escalate")}>Escalate</Button>
              <Button variant="outlined" startIcon={<RefreshCw size={16} />} onClick={() => void api.reprocessDocument(document.document_id)}>Reprocess</Button>
              <Button variant="outlined" color="error" onClick={() => openDecision("confirm-error")}>Mark error</Button>
              {canConfirmFraud ? <Button variant="contained" color="error" startIcon={<ShieldAlert size={16} />} onClick={() => openDecision("confirm-fraud")}>Confirm fraud</Button> : null}
            </Box>
          </Box>
        </Box>
      </Box>
      <Dialog open={Boolean(dialogAction)} onClose={closeDecision} maxWidth="sm" fullWidth>
        <DialogTitle>{dialogAction?.replaceAll("-", " ")}</DialogTitle>
        <DialogContent>
          {dialogAction === "confirm-fraud" ? (
            <>
              <Alert severity="error">You are assigning CONFIRMED_FRAUD. Automated indicators do not independently prove fraud. Confirm that sufficient human-verified evidence has been reviewed.</Alert>
              <TextField type="password" label="Password" required value={fraudPassword} onChange={(event) => setFraudPassword(event.target.value)} fullWidth sx={{ mt: 2 }} />
              <TextField label="Decision rationale" required value={fraudRationale} onChange={(event) => setFraudRationale(event.target.value)} multiline minRows={3} fullWidth sx={{ mt: 2 }} />
              {approvalPolicy?.fraud_second_reviewer_required ? (
                <>
                  <TextField
                    type="email"
                    label="Second reviewer email"
                    required
                    value={secondReviewerEmail}
                    onChange={(event) => setSecondReviewerEmail(event.target.value)}
                    error={Boolean(secondReviewerEmail) && !secondReviewerReady(secondReviewerEmail, user?.email, true)}
                    helperText={secondReviewerEmail.trim().toLowerCase() === user?.email?.trim().toLowerCase() ? "The second reviewer must be a different user." : undefined}
                    fullWidth
                    sx={{ mt: 2 }}
                  />
                  <FormControlLabel control={<Checkbox checked={secondReviewerApproved} onChange={(event) => setSecondReviewerApproved(event.target.checked)} />} label="A distinct second reviewer approved this determination" />
                </>
              ) : null}
              <FormControlLabel control={<Checkbox checked={automatedIndicatorsAcknowledged} onChange={(event) => setAutomatedIndicatorsAcknowledged(event.target.checked)} />} label="I acknowledge that automated indicators alone are insufficient" />
              <FormControlLabel control={<Checkbox checked={confirmationAcknowledged} onChange={(event) => setConfirmationAcknowledged(event.target.checked)} />} label="I confirm sufficient human-verified evidence was reviewed" />
            </>
          ) : (
            <Alert severity="warning">This authorised decision is recorded in the audit history. Confirm it only after reviewing source evidence and unresolved fields.</Alert>
          )}
          {dialogAction === "approve" && approvalPolicy?.manual_override_required ? <FormControlLabel control={<Checkbox checked={confirmationAcknowledged} onChange={(event) => setConfirmationAcknowledged(event.target.checked)} />} label="I confirm complete manual verification and the recorded justification" /> : null}
          <TextField select label="Decision" value={dialogAction || ""} fullWidth sx={{ mt: 2 }} disabled>
            <MenuItem value={dialogAction || ""}>{dialogAction?.replaceAll("-", " ")}</MenuItem>
          </TextField>
        </DialogContent>
        <DialogActions><Button onClick={closeDecision}>Cancel</Button><Button variant="contained" color={dialogAction === "confirm-fraud" ? "error" : "primary"} disabled={decide.isPending || (dialogAction === "approve" && Boolean(approvalPolicy?.manual_override_required) && !confirmationAcknowledged) || (dialogAction === "confirm-fraud" && !fraudReady)} onClick={() => dialogAction && decide.mutate(dialogAction)}>Record decision</Button></DialogActions>
      </Dialog>
    </Box>
  );
}

function mask(name: string, value: string) {
  return name.includes("bank") || name.includes("account") ? `${"*".repeat(Math.max(4, value.length - 4))}${value.slice(-4)}` : value;
}
