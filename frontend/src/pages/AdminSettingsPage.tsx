import {
  Alert,
  Box,
  Button,
  Stack,
  Switch,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Activity, Save } from "lucide-react";
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
      <PageHeader
        title="Admin settings"
        subtitle="Runtime model registry and organisation validation policy overrides."
      />
      <Tabs value={tab} onChange={(_, value) => setTab(value)}>
        <Tab label="Models" />
        <Tab label="Rules" />
      </Tabs>
      {tab === 0 ? <ModelSettings query={models} /> : <RuleSettings query={rules} />}
    </Box>
  );
}

function ModelSettings({
  query,
}: {
  query: ReturnType<typeof useQuery<Record<string, unknown>, Error>>;
}) {
  const queryClient = useQueryClient();
  const patch = useMutation({
    mutationFn: ({ id, requested }: { id: string; requested: boolean }) =>
      api.patchModel(id, { requested }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "models"] }),
  });
  const smoke = useMutation({
    mutationFn: (id: string) => api.smokeTestModel(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "models"] }),
  });
  if (query.isLoading) return <LoadingRows />;
  if (query.error) return <ErrorState error={query.error} />;
  const registry = (query.data?.registry || {}) as Record<string, unknown>;
  const backends = (registry.backends || []) as Array<Record<string, unknown>>;
  return (
    <Box className="surface table-surface">
      <Alert severity="info">
        Requested state is deployment intent. Runtime status changes to active only after
        initialization, inference, and output validation pass for the current configuration.
      </Alert>
      {smoke.error ? <Alert severity="error">{smoke.error.message}</Alert> : null}
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Stage</TableCell>
            <TableCell>Model</TableCell>
            <TableCell>Runtime</TableCell>
            <TableCell>Smoke evidence</TableCell>
            <TableCell>Requested</TableCell>
            <TableCell align="right">Action</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {backends.map((model) => {
            const id = String(model.model_id);
            return (
              <TableRow key={id}>
                <TableCell>{String(model.stage).replaceAll("_", " ")}</TableCell>
                <TableCell>
                  <ModelDescription model={model} />
                </TableCell>
                <TableCell>
                  <Stack spacing={0.75} alignItems="flex-start">
                    <StatusChip value={String(model.runtime_status || model.status)} />
                    <Typography variant="caption">
                      Health: {String(model.health_status || "unknown")}
                    </Typography>
                    <Typography variant="caption">
                      {String(model.device || "cpu")}
                      {model.package_version ? ` · ${String(model.package_version)}` : ""}
                    </Typography>
                  </Stack>
                </TableCell>
                <TableCell>
                  <SmokeEvidence model={model} />
                </TableCell>
                <TableCell>
                  <Switch
                    checked={Boolean(model.requested)}
                    disabled={patch.isPending}
                    onChange={(event) => patch.mutate({ id, requested: event.target.checked })}
                    inputProps={{ "aria-label": `Request ${id}` }}
                  />
                </TableCell>
                <TableCell align="right">
                  <Tooltip title="Run adapter initialization and one schema-validated inference">
                    <span>
                      <Button
                        size="small"
                        startIcon={<Activity size={15} />}
                        disabled={smoke.isPending}
                        onClick={() => smoke.mutate(id)}
                      >
                        {smoke.isPending && smoke.variables === id ? "Running" : "Smoke test"}
                      </Button>
                    </span>
                  </Tooltip>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </Box>
  );
}

function ModelDescription({ model }: { model: Record<string, unknown> }) {
  const notes = String(model.notes || "");
  const reason = String(model.reason || "");
  const requirements = (model.requirements || {}) as Record<string, unknown>;
  const missingPackages = (requirements.missing_packages || []) as string[];
  const variables = (requirements.environment_variables || []) as string[];
  return (
    <>
      <Typography fontWeight={700}>{String(model.model_id)}</Typography>
      <Typography variant="caption" display="block">
        {String(model.provider || "local")}
      </Typography>
      {notes ? (
        <Typography variant="caption" display="block">
          {notes}
        </Typography>
      ) : null}
      {reason && reason !== notes ? (
        <Typography variant="caption" color="text.secondary">
          {reason}
        </Typography>
      ) : null}
      {missingPackages.length ? (
        <Typography variant="caption" color="error" display="block">
          Missing package: {missingPackages.join(", ")}
        </Typography>
      ) : null}
      {requirements.missing_checkpoint ? (
        <Typography variant="caption" color="error" display="block">
          Missing checkpoint
        </Typography>
      ) : null}
      {variables.length && String(model.status) !== "active" ? (
        <Typography variant="caption" color="text.secondary" display="block">
          Environment: {variables.join(", ")}
        </Typography>
      ) : null}
      {requirements.preparation_command && String(model.status) !== "active" ? (
        <Typography className="model-command" variant="caption" display="block">
          {String(requirements.preparation_command)}
        </Typography>
      ) : null}
    </>
  );
}

function SmokeEvidence({ model }: { model: Record<string, unknown> }) {
  const smoke = model.smoke_test as Record<string, unknown> | null;
  const initialization = (smoke?.initialization || {}) as Record<string, unknown>;
  const inference = (smoke?.inference || {}) as Record<string, unknown>;
  const schema = (smoke?.output_schema || {}) as Record<string, unknown>;
  if (!smoke)
    return (
      <Typography variant="caption" color="text.secondary">
        Not run for this configuration
      </Typography>
    );
  const inferenceLabel =
    inference.status === "not_run" ? "not run" : duration(inference.duration_ms);
  const schemaLabel =
    schema.status === "not_evaluated" || schema.valid === null || schema.valid === undefined
      ? "not evaluated"
      : schema.valid
        ? String(schema.name || "valid")
        : "invalid";
  return (
    <Stack spacing={0.4}>
      <StatusChip value={String(smoke.status || "unknown")} />
      <Typography variant="caption">
        Init {duration(initialization.duration_ms)} · inference {inferenceLabel}
      </Typography>
      <Typography variant="caption">Schema: {schemaLabel}</Typography>
      <Typography variant="caption">{formatTimestamp(String(smoke.completed_at || ""))}</Typography>
      {smoke.error ? (
        <Typography variant="caption" color="error">
          {String(smoke.error)}
        </Typography>
      ) : null}
    </Stack>
  );
}

function duration(value: unknown) {
  return typeof value === "number" ? `${Math.round(value)} ms` : "not run";
}

function formatTimestamp(value: string) {
  if (!value) return "No timestamp";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function RuleSettings({
  query,
}: {
  query: ReturnType<typeof useQuery<Record<string, unknown>, Error>>;
}) {
  const queryClient = useQueryClient();
  const [thresholds, setThresholds] = useState<Record<string, string>>({});
  const patch = useMutation({
    mutationFn: ({ id, threshold }: { id: string; threshold?: number }) =>
      api.patchRule(id, { enabled: true, threshold }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["admin", "rules"] }),
  });
  if (query.isLoading) return <LoadingRows />;
  if (query.error) return <ErrorState error={query.error} />;
  const validationRules = (query.data?.validation_rules || []) as string[];
  return (
    <Box className="surface table-surface">
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Rule</TableCell>
            <TableCell>Threshold override</TableCell>
            <TableCell align="right">Action</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {validationRules.map((rule) => (
            <TableRow key={rule}>
              <TableCell>
                <Typography fontWeight={700}>{rule.replaceAll("_", " ")}</Typography>
              </TableCell>
              <TableCell>
                <TextField
                  size="small"
                  type="number"
                  value={thresholds[rule] || ""}
                  onChange={(event) =>
                    setThresholds((current) => ({ ...current, [rule]: event.target.value }))
                  }
                  inputProps={{ step: "0.01" }}
                />
              </TableCell>
              <TableCell align="right">
                <Button
                  startIcon={<Save size={15} />}
                  onClick={() =>
                    patch.mutate({
                      id: rule,
                      threshold: thresholds[rule] ? Number(thresholds[rule]) : undefined,
                    })
                  }
                >
                  Record override
                </Button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}
