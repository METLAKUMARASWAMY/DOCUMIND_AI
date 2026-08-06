import { Box, Button, Card, CardContent, Tab, Tabs, TextField, Typography } from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Legend, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { api } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { EmptyState, ErrorState, LoadingRows } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { Link } from "../router";

const palette = ["#12324a", "#087f78", "#c28a13", "#b42318", "#7b5ca7", "#3568a8", "#5b7f3b"];

export function AnalyticsPage() {
  const { user } = useAuth();
  const [tab, setTab] = useState(0);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");
  const rangeKey = [dateFrom, dateTo];
  const overview = useQuery({ queryKey: ["analytics", "overview", ...rangeKey], queryFn: () => api.analyticsOverview(dateFrom, dateTo) });
  const vendors = useQuery({ queryKey: ["analytics", "vendors", ...rangeKey], queryFn: () => api.analyticsVendors(dateFrom, dateTo) });
  const risk = useQuery({ queryKey: ["analytics", "risk", ...rangeKey], queryFn: () => api.analyticsRisk(dateFrom, dateTo) });
  const review = useQuery({ queryKey: ["analytics", "review", ...rangeKey], queryFn: () => api.analyticsReview(dateFrom, dateTo) });
  const models = useQuery({ queryKey: ["analytics", "models", ...rangeKey], queryFn: () => api.analyticsModels(dateFrom, dateTo) });
  const selected = [overview, vendors, risk, review, models][tab];
  const documents = number(overview.data?.documents);
  const lastUpdated = String((selected.data?.last_updated || overview.data?.last_updated) ?? "");

  return (
    <Box className="page-stack">
      <PageHeader title="Analytics" />
      <Box className="analytics-filter-bar surface">
        <TextField size="small" label="From" type="date" value={dateFrom} onChange={(event) => setDateFrom(event.target.value)} slotProps={{ inputLabel: { shrink: true } }} />
        <TextField size="small" label="To" type="date" value={dateTo} onChange={(event) => setDateTo(event.target.value)} slotProps={{ inputLabel: { shrink: true } }} />
        <TextField size="small" label="Organisation" value={user?.tenant_id || ""} disabled />
        <Typography variant="caption" color="text.secondary">Last updated: {lastUpdated ? new Date(lastUpdated).toLocaleString() : "Not available"}</Typography>
      </Box>
      <Tabs value={tab} onChange={(_, value) => setTab(value)} variant="scrollable" scrollButtons="auto">{["Overview", "Vendors", "Risk", "Review", "Models"].map((label) => <Tab key={label} label={label} />)}</Tabs>
      {selected.isLoading ? <LoadingRows /> : selected.error ? <ErrorState error={selected.error} retry={() => void selected.refetch()} /> : null}
      {!selected.isLoading && !selected.error && tab === 0 ? <OverviewAnalytics data={overview.data || {}} sample={documents} /> : null}
      {!selected.isLoading && !selected.error && tab === 1 ? <VendorAnalytics data={vendors.data || {}} /> : null}
      {!selected.isLoading && !selected.error && tab === 2 ? <RiskAnalytics data={risk.data || {}} /> : null}
      {!selected.isLoading && !selected.error && tab === 3 ? <ReviewAnalytics data={review.data || {}} /> : null}
      {!selected.isLoading && !selected.error && tab === 4 ? <ModelAnalytics data={models.data || {}} /> : null}
    </Box>
  );
}

function OverviewAnalytics({ data, sample }: { data: Record<string, unknown>; sample: number }) {
  if (!sample) return <EmptyState title="No analytics data" detail="No documents fall within this date range." />;
  const status = entries(data.status_counts);
  const types = entries(data.document_type_counts);
  const processing = [
    { name: "Success", value: number(data.processing_success_count) },
    { name: "Failure", value: number(data.processing_failure_count) },
  ];
  const rag = entries(data.rag_answer_status);
  const citation = (data.citation_coverage || {}) as Record<string, unknown>;
  const timing = (data.processing_timing || {}) as Record<string, unknown>;
  const timingSample = number(timing.sample_size);
  const pipelineVersions = Object.entries((timing.pipeline_versions || {}) as Record<string, number>)
    .map(([version, count]) => `${version}: ${count}`)
    .join(" | ");
  return (
    <>
      <Box className="analytics-stat-band">
        <Stat label="Documents" value={sample} detail={`${number(data.documents_without_fields)} without extracted fields`} />
        <Stat label="Structured extraction coverage" value={percent(data.structured_extraction_coverage)} detail={`${number(data.documents_with_fields)} of ${sample} documents`} />
        <Stat label="Schema coverage" value={number(data.documents_using_exact_schema)} detail={`${number(data.documents_using_generic_schema)} generic | ${number(data.documents_schema_unavailable)} unavailable`} />
        <Stat label="Mean OCR confidence" value={percent(data.mean_ocr_confidence)} detail={`n=${number((data.sample_sizes as Record<string, unknown>)?.ocr_words)}`} />
        <Stat label="Mean field confidence" value={percent(data.mean_field_confidence)} detail={`n=${number((data.sample_sizes as Record<string, unknown>)?.fields)}`} />
        <Stat label="Mean processing" value={seconds(data.average_processing_time_seconds)} detail={`n=${timingSample}${pipelineVersions ? ` | ${pipelineVersions}` : ""}`} />
        <Stat label="Median processing" value={seconds(data.median_processing_time_seconds)} detail="Completed compatible pipeline runs" />
        <Stat label="P95 processing" value={seconds(data.p95_processing_time_seconds)} detail="Completed compatible pipeline runs" />
        <Stat label="Citation coverage" value={percent(citation.coverage)} detail={`${number(citation.supported_answers_with_citations)} of ${number(citation.supported_answer_count)} supported answers`} />
        <Stat label="Generated artefacts" value={number(data.narrative_count) + number(data.report_count)} detail={`${number(data.narrative_count)} narratives | ${number(data.report_count)} reports`} />
      </Box>
      <Box className="analytics-actions"><Button component={Link} to="/documents?status=REVIEW_REQUIRED" variant="outlined">Review Required</Button></Box>
      <Box className="analytics-grid">
        <BarPanel title="Documents by status" data={status} sample={sample} />
        <BarPanel title="Documents by type" data={types} sample={sample} />
        <PiePanel title="Processing outcome" data={processing} sample={sample} />
        <StageTimingPanel timing={timing} />
        <PiePanel title="RAG answer status" data={rag} sample={number((data.sample_sizes as Record<string, unknown>)?.rag_questions)} />
      </Box>
    </>
  );
}

function StageTimingPanel({ timing }: { timing: Record<string, unknown> }) {
  const perStage = (timing.per_stage || {}) as Record<string, Record<string, unknown>>;
  const rows = Object.entries(perStage).map(([name, summary]) => ({
    name: name.replaceAll("_", " "),
    value: number(summary.mean_ms) / 1000,
  }));
  return (
    <Chart title="Mean processing by stage" sample={rows.length ? number(timing.sample_size) : 0}>
      {rows.length ? (
        <ResponsiveContainer width="100%" height={290}>
          <BarChart data={rows} margin={{ bottom: 42 }}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis dataKey="name" angle={-25} textAnchor="end" interval={0} tick={{ fontSize: 10 }} />
            <YAxis unit="s" />
            <Tooltip formatter={(value) => [`${Number(value).toFixed(3)}s`, "Mean duration"]} />
            <Bar name="Mean seconds" dataKey="value" fill="#3568a8" />
          </BarChart>
        </ResponsiveContainer>
      ) : <EmptyState title="No stage timing" detail="Stage timing is available after processing with the current pipeline." />}
    </Chart>
  );
}

function VendorAnalytics({ data }: { data: Record<string, unknown> }) {
  const rows = ((data.vendors || []) as Array<Record<string, unknown>>).map((item) => ({ name: String(item.vendor_name), value: number(item.document_count), amount: number(item.total_amount) }));
  if (!rows.length) return <EmptyState title="No vendor data" detail="No vendor records fall within this date range." />;
  return <Chart title="Vendor document volume" sample={rows.reduce((total, item) => total + item.value, 0)}><ResponsiveContainer width="100%" height={420}><BarChart data={rows.slice(0, 20)} layout="vertical" margin={{ left: 80 }}><CartesianGrid horizontal={false} strokeDasharray="3 3" /><XAxis type="number" allowDecimals={false} /><YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 11 }} /><Tooltip formatter={(value, name, item) => [name === "value" ? `${value} documents` : value, name === "value" ? `Volume | total ${item.payload.amount}` : name]} /><Legend /><Bar name="Documents" dataKey="value" fill="#087f78" /></BarChart></ResponsiveContainer></Chart>;
}

function RiskAnalytics({ data }: { data: Record<string, unknown> }) {
  const severity = entries(data.by_severity);
  const category = entries(data.by_category);
  if (!severity.length && !category.length) return <EmptyState title="No risk indicators" detail="No business or visual indicators fall within this date range." />;
  return <Box className="analytics-grid"><PiePanel title="Risk severity" data={severity} sample={number(data.total_indicators)} /><BarPanel title="Risk category" data={category} sample={number(data.total_indicators)} /><StatPanel title="Processing warnings" value={number(data.processing_warning_count)} detail={`Across ${number(data.sample_size)} documents`} /></Box>;
}

function ReviewAnalytics({ data }: { data: Record<string, unknown> }) {
  const priorities = entries(data.by_priority);
  const outcomes = entries(data.by_status);
  if (!number(data.total)) return <EmptyState title="No review data" detail="No review tasks fall within this date range." />;
  return <><Box className="analytics-stat-band"><Stat label="Review tasks" value={number(data.total)} detail={`${number(data.pending)} pending`} /><Stat label="Unresolved required fields" value={number(data.schema_unresolved_total)} /><Stat label="Average review" value={seconds(data.average_review_time_seconds)} detail={`n=${number(data.review_duration_sample_size)}`} /></Box><Box className="analytics-grid"><PiePanel title="Review priority" data={priorities} sample={number(data.total)} /><BarPanel title="Review outcomes" data={outcomes} sample={number(data.total)} /></Box></>;
}

function ModelAnalytics({ data }: { data: Record<string, unknown> }) {
  const stages = entries(data.runs_by_stage);
  const statuses = entries(data.runs_by_status);
  if (!number(data.run_count)) return <EmptyState title="No model runs" detail="No model activity falls within this date range." />;
  return <><Box className="analytics-stat-band"><Stat label="Model runs" value={number(data.run_count)} /><Stat label="Failures" value={number(data.failure_count)} detail={percent(data.failure_rate)} /></Box><Box className="analytics-grid"><BarPanel title="Model usage by stage" data={stages} sample={number(data.run_count)} /><PiePanel title="Model run status" data={statuses} sample={number(data.run_count)} /></Box></>;
}

function BarPanel({ title, data, sample }: { title: string; data: Array<{ name: string; value: number }>; sample: number }) {
  return <Chart title={title} sample={sample}>{data.length ? <ResponsiveContainer width="100%" height={290}><BarChart data={data}><CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="name" tick={{ fontSize: 10 }} /><YAxis allowDecimals={false} /><Tooltip /><Legend /><Bar name="Documents" dataKey="value" fill="#12324a">{data.map((_, index) => <Cell key={index} fill={palette[index % palette.length]} />)}</Bar></BarChart></ResponsiveContainer> : <EmptyState title="No values" detail="No records are available for this metric." />}</Chart>;
}

function PiePanel({ title, data, sample }: { title: string; data: Array<{ name: string; value: number }>; sample: number }) {
  return <Chart title={title} sample={sample}>{data.length ? <ResponsiveContainer width="100%" height={290}><PieChart><Pie data={data} dataKey="value" nameKey="name" innerRadius={52} outerRadius={88} label={({ name, value }) => `${name}: ${value}`}>{data.map((_, index) => <Cell key={index} fill={palette[index % palette.length]} />)}</Pie><Tooltip /><Legend /></PieChart></ResponsiveContainer> : <EmptyState title="No values" detail="No records are available for this metric." />}</Chart>;
}

function StatPanel({ title, value, detail }: { title: string; value: number; detail: string }) { return <Chart title={title} sample={value}><Box className="analytics-large-stat"><Typography>{value}</Typography><Typography color="text.secondary">{detail}</Typography></Box></Chart>; }
function Chart({ title, sample, children }: { title: string; sample: number; children: React.ReactNode }) { return <Card><CardContent><Box className="section-heading"><Typography variant="h3">{title}</Typography><Typography variant="caption">n={sample}</Typography></Box><Box className="chart-box">{children}</Box></CardContent></Card>; }
function Stat({ label, value, detail }: { label: string; value: string | number; detail?: string }) { return <Box className="analytics-stat"><Typography variant="caption" color="text.secondary">{label}</Typography><Typography variant="h3">{value}</Typography>{detail ? <Typography variant="caption" color="text.secondary">{detail}</Typography> : null}</Box>; }
function entries(value: unknown) { return Object.entries((value || {}) as Record<string, number>).map(([name, count]) => ({ name: name.replaceAll("_", " "), value: count })); }
function number(value: unknown) { return typeof value === "number" && Number.isFinite(value) ? value : 0; }
function percent(value: unknown) { return typeof value === "number" && Number.isFinite(value) ? `${Math.round(value * 100)}%` : "Not available"; }
function seconds(value: unknown) { return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)}s` : "Not available"; }
