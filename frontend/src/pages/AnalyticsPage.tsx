import { Box, Card, CardContent, Tab, Tabs, Typography } from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { api } from "../api/client";
import { ErrorState, LoadingRows } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";

export function AnalyticsPage() {
  const [tab, setTab] = useState(0);
  const overview = useQuery({ queryKey: ["analytics", "overview"], queryFn: api.analyticsOverview });
  const vendors = useQuery({ queryKey: ["analytics", "vendors"], queryFn: api.analyticsVendors });
  const risk = useQuery({ queryKey: ["analytics", "risk"], queryFn: api.analyticsRisk });
  const review = useQuery({ queryKey: ["analytics", "review"], queryFn: api.analyticsReview });
  const models = useQuery({ queryKey: ["analytics", "models"], queryFn: api.analyticsModels });
  const selected = [overview, vendors, risk, review, models][tab];
  const statusCounts = entries(overview.data?.status_counts);
  const riskCounts = entries(risk.data?.by_severity);
  const vendorRows = ((vendors.data?.vendors || []) as Array<Record<string, unknown>>).slice(0, 12).map((item) => ({ name: String(item.vendor_name), documents: number(item.document_count), total: number(item.total_amount) }));

  return (
    <Box className="page-stack">
      <PageHeader title="Analytics" subtitle="Calculated organisation activity without synthetic accuracy claims." />
      <Tabs value={tab} onChange={(_, value) => setTab(value)}>{["Overview", "Vendors", "Risk", "Review", "Models"].map((label) => <Tab key={label} label={label} />)}</Tabs>
      {selected.isLoading ? <LoadingRows /> : selected.error ? <ErrorState error={selected.error} retry={() => void selected.refetch()} /> : null}
      {tab === 0 ? <Box className="analytics-grid"><Chart title="Documents by state"><ResponsiveContainer width="100%" height={280}><BarChart data={statusCounts}><CartesianGrid vertical={false} strokeDasharray="3 3" /><XAxis dataKey="name" tick={{ fontSize: 10 }} /><YAxis allowDecimals={false} /><Tooltip /><Bar dataKey="value" fill="#12324a" /></BarChart></ResponsiveContainer></Chart><Chart title="Risk severity"><ResponsiveContainer width="100%" height={280}><PieChart><Pie data={riskCounts} dataKey="value" nameKey="name" innerRadius={55} outerRadius={90}>{riskCounts.map((_, index) => <Cell key={index} fill={["#18794e", "#1769aa", "#b54708", "#b42318"][index % 4]} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer></Chart></Box> : null}
      {tab === 1 ? <Chart title="Vendor document volume"><ResponsiveContainer width="100%" height={360}><BarChart data={vendorRows} layout="vertical" margin={{ left: 60 }}><CartesianGrid horizontal={false} strokeDasharray="3 3" /><XAxis type="number" allowDecimals={false} /><YAxis type="category" dataKey="name" width={120} tick={{ fontSize: 11 }} /><Tooltip /><Bar dataKey="documents" fill="#087f78" /></BarChart></ResponsiveContainer></Chart> : null}
      {tab >= 2 ? <Box className="json-metrics surface"><Typography variant="h3">Calculated metrics</Typography><pre>{JSON.stringify(selected.data, null, 2)}</pre></Box> : null}
    </Box>
  );
}

function Chart({ title, children }: { title: string; children: React.ReactNode }) { return <Card><CardContent><Typography variant="h3">{title}</Typography><Box className="chart-box">{children}</Box></CardContent></Card>; }
function entries(value: unknown) { return Object.entries((value || {}) as Record<string, number>).map(([name, count]) => ({ name: name.replaceAll("_", " "), value: count })); }
function number(value: unknown) { return typeof value === "number" ? value : 0; }
