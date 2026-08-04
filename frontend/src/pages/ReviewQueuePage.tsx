import { Box, Button, MenuItem, Table, TableBody, TableCell, TableHead, TableRow, TextField, Typography } from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { ClipboardCheck } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "../router";

import { api } from "../api/client";
import { EmptyState, ErrorState, LoadingRows } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";

export function ReviewQueuePage() {
  const reviews = useQuery({ queryKey: ["reviews"], queryFn: api.reviews, refetchInterval: 5000 });
  const [priority, setPriority] = useState("all");
  const [status, setStatus] = useState("open");
  const rows = useMemo(() => (reviews.data?.reviews || []).filter((review) => {
    const priorityMatch = priority === "all" || review.priority === priority;
    const statusMatch = status === "all" || (status === "open" ? ["REVIEW_REQUIRED", "UNDER_REVIEW"].includes(review.status) : review.status === status);
    return priorityMatch && statusMatch;
  }), [reviews.data, priority, status]);

  return (
    <Box className="page-stack">
      <PageHeader title="Human review queue" subtitle="Evidence requiring authorised verification and a recorded decision." />
      <Box className="filter-bar surface">
        <TextField select size="small" label="Queue" value={status} onChange={(event) => setStatus(event.target.value)}>
          {["open", "all", "APPROVED", "REJECTED", "CONFIRMED_ERROR", "CONFIRMED_FRAUD"].map((value) => <MenuItem key={value} value={value}>{value.replaceAll("_", " ")}</MenuItem>)}
        </TextField>
        <TextField select size="small" label="Priority" value={priority} onChange={(event) => setPriority(event.target.value)}>
          {["all", "critical", "high", "medium", "low"].map((value) => <MenuItem key={value} value={value}>{value}</MenuItem>)}
        </TextField>
      </Box>
      <Box className="surface table-surface">
        {reviews.isLoading ? <LoadingRows rows={8} /> : reviews.error ? <ErrorState error={reviews.error} retry={() => void reviews.refetch()} /> : !rows.length ? (
          <EmptyState title="Review queue is clear" detail="No tasks match the current queue filters." />
        ) : (
          <Table><TableHead><TableRow><TableCell>Document</TableCell><TableCell>Priority</TableCell><TableCell>Reasons</TableCell><TableCell>Status</TableCell><TableCell>Created</TableCell><TableCell align="right">Action</TableCell></TableRow></TableHead>
            <TableBody>{rows.map((review) => <TableRow key={review.review_id} hover><TableCell><Typography fontWeight={700}>{review.document_id}</Typography><Typography variant="caption">{review.review_id}</Typography></TableCell><TableCell><StatusChip value={review.priority} /></TableCell><TableCell>{review.reasons.slice(0, 2).join("; ")}</TableCell><TableCell><StatusChip value={review.status} /></TableCell><TableCell>{new Date(review.created_at).toLocaleString()}</TableCell><TableCell align="right"><Button component={Link} to={`/reviews/${review.review_id}`} variant="outlined" startIcon={<ClipboardCheck size={16} />}>Review</Button></TableCell></TableRow>)}</TableBody>
          </Table>
        )}
      </Box>
    </Box>
  );
}
