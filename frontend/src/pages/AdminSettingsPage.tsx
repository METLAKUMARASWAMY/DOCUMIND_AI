import { Alert, Box, Button, Switch, Tab, Tabs, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Save } from "lucide-react";
import { useState } from "react";

import { api } from "../api/client";
import { ErrorState, LoadingRows } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";

export function AdminSettingsPage() {
  const [tab, setTab] = useState(0);
  const models = useQuery({ queryKey: ["admin", "models"], queryFn: api.models });
  const rules = useQuery({ queryKey: ["admin", "rules"], queryFn: api.rules });
  return (
    <Box className="page-stack">
      <PageHeader title="Admin settings" subtitle="Runtime model registry and organisation validation policy overrides." />
      <Tabs value={tab} onChange={(_, value) => setTab(value)}><Tab label="Models" /><Tab label="Rules" /></Tabs>
      {tab === 0 ? <ModelSettings query={models} /> : <RuleSettings query={rules} />}
    </Box>
  );
}

function ModelSettings({ query }: { query: ReturnType<typeof useQuery<Record<string, unknown>, Error>> }) {
  const queryClient = useQueryClient();
  const patch = useMutation({ mutationFn: ({ id, enabled }: { id: string; enabled: boolean }) => api.patchModel(id, { enabled }), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "models"] }) });
  if (query.isLoading) return <LoadingRows />;
  if (query.error) return <ErrorState error={query.error} />;
  const registry = (query.data?.registry || {}) as Record<string, unknown>;
  const backends = (registry.backends || []) as Array<Record<string, unknown>>;
  return <Box className="surface table-surface"><Alert severity="info">A recorded override does not fabricate model availability. Packages, checkpoints, flags, and adapter smoke tests remain required.</Alert><Table><TableHead><TableRow><TableCell>Stage</TableCell><TableCell>Model</TableCell><TableCell>Provider</TableCell><TableCell>Status</TableCell><TableCell>Requested</TableCell></TableRow></TableHead><TableBody>{backends.map((model) => <TableRow key={String(model.model_id)}><TableCell>{String(model.stage)}</TableCell><TableCell><ModelDescription model={model} /></TableCell><TableCell>{String(model.provider || "local")}</TableCell><TableCell><StatusChip value={String(model.status)} /></TableCell><TableCell><Switch checked={Boolean(model.enabled)} onChange={(event) => patch.mutate({ id: String(model.model_id), enabled: event.target.checked })} inputProps={{ "aria-label": `Request ${String(model.model_id)}` }} /></TableCell></TableRow>)}</TableBody></Table></Box>;
}

function ModelDescription({ model }: { model: Record<string, unknown> }) {
  const notes = String(model.notes || "");
  const reason = String(model.reason || "");
  return (
    <>
      <Typography fontWeight={700}>{String(model.model_id)}</Typography>
      {notes ? <Typography variant="caption" display="block">{notes}</Typography> : null}
      {reason && reason !== notes ? <Typography variant="caption" color="text.secondary">{reason}</Typography> : null}
    </>
  );
}

function RuleSettings({ query }: { query: ReturnType<typeof useQuery<Record<string, unknown>, Error>> }) {
  const queryClient = useQueryClient();
  const [thresholds, setThresholds] = useState<Record<string, string>>({});
  const patch = useMutation({ mutationFn: ({ id, threshold }: { id: string; threshold?: number }) => api.patchRule(id, { enabled: true, threshold }), onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "rules"] }) });
  if (query.isLoading) return <LoadingRows />;
  if (query.error) return <ErrorState error={query.error} />;
  const validationRules = (query.data?.validation_rules || []) as string[];
  return <Box className="surface table-surface"><Table><TableHead><TableRow><TableCell>Rule</TableCell><TableCell>Threshold override</TableCell><TableCell align="right">Action</TableCell></TableRow></TableHead><TableBody>{validationRules.map((rule) => <TableRow key={rule}><TableCell><Typography fontWeight={700}>{rule.replaceAll("_", " ")}</Typography></TableCell><TableCell><TextField size="small" type="number" value={thresholds[rule] || ""} onChange={(event) => setThresholds((current) => ({ ...current, [rule]: event.target.value }))} inputProps={{ step: "0.01" }} /></TableCell><TableCell align="right"><Button startIcon={<Save size={15} />} onClick={() => patch.mutate({ id: rule, threshold: thresholds[rule] ? Number(thresholds[rule]) : undefined })}>Record override</Button></TableCell></TableRow>)}</TableBody></Table></Box>;
}
