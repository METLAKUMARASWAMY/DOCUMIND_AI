import { Alert, Box, Button, Card, CardContent, LinearProgress, Table, TableBody, TableCell, TableHead, TableRow, Typography } from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, ClipboardCheck, Database, FileText, Gauge, Upload } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Link } from "../router";

import { api } from "../api/client";
import { ErrorState, LoadingRows } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";

export function DashboardPage() {
  const queryClient = useQueryClient();
  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const overview = useQuery({ queryKey: ["analytics", "overview"], queryFn: api.analyticsOverview });
  const risk = useQuery({ queryKey: ["analytics", "risk"], queryFn: api.analyticsRisk });
  const review = useQuery({ queryKey: ["analytics", "review"], queryFn: api.analyticsReview });
  const seedDemo = useMutation({
    mutationFn: api.seedDemo,
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["documents"] });
      void queryClient.invalidateQueries({ queryKey: ["analytics"] });
      void queryClient.invalidateQueries({ queryKey: ["narratives"] });
      void queryClient.invalidateQueries({ queryKey: ["reports"] });
      void queryClient.invalidateQueries({ queryKey: ["reviews"] });
    },
  });
  const metrics = overview.data || {};
  const riskData = risk.data || {};
  const reviewData = review.data || {};
  const riskCounts = (metrics.risk_counts || {}) as Record<string, number>;
  const chartData = Object.entries(riskCounts).map(([name, value]) => ({ name, value }));

  return (
    <Box className="page-stack">
      <PageHeader
        title="Document operations"
        subtitle="Live processing, review, and evidence-grounded risk activity."
        actions={
          <>
            <Button
              variant="outlined"
              startIcon={<Database size={17} />}
              disabled={seedDemo.isPending}
              onClick={() => seedDemo.mutate()}
            >
              {seedDemo.isPending ? "Loading demo" : "Load demo data"}
            </Button>
            <Button component={Link} to="/upload" variant="contained" startIcon={<Upload size={17} />}>
              Upload documents
            </Button>
          </>
        }
      />
      {seedDemo.data ? <Alert severity="success">{seedDemo.data.message}</Alert> : null}
      {seedDemo.error ? <Alert severity="error">{seedDemo.error.message}</Alert> : null}
      {overview.error ? <ErrorState error={overview.error} retry={() => void overview.refetch()} /> : null}
      <Box className="metric-grid">
        <Metric label="Documents" value={number(metrics.documents)} icon={<FileText size={19} />} />
        <Metric label="Pending review" value={number(reviewData.pending)} icon={<ClipboardCheck size={19} />} tone="warning" />
        <Metric label="Risk indicators" value={number(riskData.total_indicators)} icon={<AlertTriangle size={19} />} tone="error" />
        <Metric
          label="Mean field confidence"
          value={`${Math.round(number(metrics.average_field_confidence) * 100)}%`}
          icon={<Gauge size={19} />}
          tone="secondary"
        />
      </Box>
      <Box className="dashboard-grid">
        <Box className="surface section-panel">
          <Box className="section-heading">
            <Box>
              <Typography variant="h3">Recent documents</Typography>
              <Typography color="text.secondary">Most recently updated in this organisation</Typography>
            </Box>
            <Button component={Link} to="/documents" size="small">View library</Button>
          </Box>
          {documents.isLoading ? <LoadingRows /> : documents.error ? <ErrorState error={documents.error} /> : (
            <Table size="small">
              <TableHead><TableRow><TableCell>Document</TableCell><TableCell>Type</TableCell><TableCell>Status</TableCell><TableCell>Updated</TableCell></TableRow></TableHead>
              <TableBody>
                {documents.data?.documents.slice(0, 7).map((document) => (
                  <TableRow key={document.document_id} hover>
                    <TableCell><Button component={Link} to={`/documents/${document.document_id}`} size="small">{document.filename}</Button></TableCell>
                    <TableCell>{document.document_type.replaceAll("_", " ")}</TableCell>
                    <TableCell><StatusChip value={document.status} /></TableCell>
                    <TableCell>{new Date(document.updated_at).toLocaleString()}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </Box>
        <Box className="surface section-panel chart-panel">
          <Typography variant="h3">Indicators by severity</Typography>
          {overview.isLoading ? <LinearProgress /> : chartData.length ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={chartData} margin={{ top: 15, right: 8, left: -24, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="value" fill="#087f78" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <Typography color="text.secondary">No risk indicators have been generated.</Typography>}
        </Box>
      </Box>
    </Box>
  );
}

function Metric({ label, value, icon, tone = "primary" }: { label: string; value: string | number; icon: React.ReactNode; tone?: string }) {
  return (
    <Card className={`metric-card metric-${tone}`}>
      <CardContent>
        <Box className="metric-icon">{icon}</Box>
        <Typography color="text.secondary">{label}</Typography>
        <Typography className="metric-value">{value}</Typography>
      </CardContent>
    </Card>
  );
}

function number(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}
