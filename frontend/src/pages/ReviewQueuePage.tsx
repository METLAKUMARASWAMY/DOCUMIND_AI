import {
  Box,
  Chip,
  InputAdornment,
  MenuItem,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TablePagination,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { FileText, Search } from "lucide-react";
import { useEffect, useState } from "react";
import { Link } from "../router";

import { api, protectedBlob } from "../api/client";
import { EmptyState, ErrorState, LoadingRows } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";
import type { ReviewTask } from "../types/api";

export function ReviewQueuePage() {
  const [search, setSearch] = useState("");
  const [documentType, setDocumentType] = useState("all");
  const [priority, setPriority] = useState("all");
  const [reason, setReason] = useState("all");
  const [assignee, setAssignee] = useState("all");
  const [status, setStatus] = useState("open");
  const [sort, setSort] = useState("created_at:desc");
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(25);
  const [sortBy, sortDirection] = sort.split(":");
  const filters = {
    search,
    document_type: documentType,
    priority,
    reason,
    assignee,
    status,
    sort_by: sortBy,
    sort_direction: sortDirection,
    page: page + 1,
    page_size: pageSize,
  };
  const reviews = useQuery({
    queryKey: ["reviews", filters],
    queryFn: () => api.reviews(filters),
    refetchInterval: 5000,
  });
  const facets = reviews.data?.facets;
  const resetPage = (action: () => void) => {
    setPage(0);
    action();
  };

  return (
    <Box className="page-stack">
      <PageHeader title="Human review queue" />
      <Box className="review-queue-filters surface">
        <TextField
          size="small"
          placeholder="Search filename or ID"
          value={search}
          onChange={(event) => resetPage(() => setSearch(event.target.value))}
          slotProps={{ input: { startAdornment: <InputAdornment position="start"><Search size={16} /></InputAdornment> } }}
        />
        <TextField select size="small" label="Queue" value={status} onChange={(event) => resetPage(() => setStatus(event.target.value))}>
          <MenuItem value="open">Open</MenuItem>
          <MenuItem value="all">All</MenuItem>
          {facets?.statuses.map((value) => <MenuItem key={value} value={value}>{value.replaceAll("_", " ")}</MenuItem>)}
        </TextField>
        <TextField select size="small" label="Type" value={documentType} onChange={(event) => resetPage(() => setDocumentType(event.target.value))}>
          <MenuItem value="all">All types</MenuItem>
          {facets?.document_types.map((value) => <MenuItem key={value} value={value}>{value.replaceAll("_", " ")}</MenuItem>)}
        </TextField>
        <TextField select size="small" label="Priority" value={priority} onChange={(event) => resetPage(() => setPriority(event.target.value))}>
          <MenuItem value="all">All priorities</MenuItem>
          {facets?.priorities.map((value) => <MenuItem key={value} value={value}>{value}</MenuItem>)}
        </TextField>
        <TextField select size="small" label="Reason" value={reason} onChange={(event) => resetPage(() => setReason(event.target.value))}>
          <MenuItem value="all">All reasons</MenuItem>
          {facets?.reasons.map((item) => <MenuItem key={item.value} value={item.value}>{item.label}</MenuItem>)}
        </TextField>
        <TextField select size="small" label="Assignee" value={assignee} onChange={(event) => resetPage(() => setAssignee(event.target.value))}>
          <MenuItem value="all">All assignees</MenuItem>
          <MenuItem value="unassigned">Unassigned</MenuItem>
          {facets?.assignees.map((value) => <MenuItem key={value} value={value}>{value}</MenuItem>)}
        </TextField>
        <TextField select size="small" label="Sort" value={sort} onChange={(event) => resetPage(() => setSort(event.target.value))}>
          <MenuItem value="created_at:desc">Newest first</MenuItem>
          <MenuItem value="created_at:asc">Oldest first</MenuItem>
          <MenuItem value="priority:desc">Highest priority</MenuItem>
          <MenuItem value="filename:asc">Filename</MenuItem>
          <MenuItem value="confidence:asc">Lowest confidence</MenuItem>
        </TextField>
      </Box>
      <Box className="surface table-surface review-queue-table">
        {reviews.isLoading ? <LoadingRows rows={8} /> : reviews.error ? <ErrorState error={reviews.error} retry={() => void reviews.refetch()} /> : !reviews.data?.reviews.length ? (
          <EmptyState title="Review queue is clear" detail="No tasks match the current filters." />
        ) : (
          <>
            <Table>
              <TableHead><TableRow><TableCell>Document</TableCell><TableCell>Type</TableCell><TableCell>Priority and reasons</TableCell><TableCell>Confidence</TableCell><TableCell>Assignee</TableCell><TableCell>Status and created</TableCell></TableRow></TableHead>
              <TableBody>{reviews.data.reviews.map((review) => (
                <TableRow key={review.review_id} hover>
                  <TableCell><Box className="queue-document-cell"><ReviewThumbnail review={review} /><Box><Typography component={Link} to={`/reviews/${review.review_id}`} className="queue-document-name">{review.filename || "Untitled document"}</Typography><Typography variant="caption" color="text.secondary">{review.document_id}</Typography></Box></Box></TableCell>
                  <TableCell>{(review.document_type || "unknown").replaceAll("_", " ")}<Typography variant="caption" display="block" color="text.secondary">{review.page_count ?? 0} page(s)</Typography></TableCell>
                  <TableCell><StatusChip value={review.priority} /><Box className="queue-reasons">{review.reason_labels.slice(0, 3).map((label, index) => <Tooltip title={review.reasons[index] || label} key={`${label}-${index}`}><Chip size="small" label={label} /></Tooltip>)}</Box></TableCell>
                  <TableCell>{review.confidence_value === null || review.confidence_value === undefined ? "Not available" : `${Math.round(review.confidence_value * 100)}%`}<Typography variant="caption" display="block" color="text.secondary">{review.confidence_status.replaceAll("_", " ")}</Typography></TableCell>
                  <TableCell>{review.assigned_reviewer || "Unassigned"}<Typography variant="caption" display="block" color="text.secondary">{review.assigned_role}</Typography></TableCell>
                  <TableCell><StatusChip value={review.status} /><Typography variant="caption" display="block" color="text.secondary">{new Date(review.created_at).toLocaleString()}</Typography></TableCell>
                </TableRow>
              ))}</TableBody>
            </Table>
            <TablePagination
              component="div"
              count={reviews.data.total}
              page={page}
              rowsPerPage={pageSize}
              rowsPerPageOptions={[10, 25, 50, 100]}
              onPageChange={(_, nextPage) => setPage(nextPage)}
              onRowsPerPageChange={(event) => { setPageSize(Number(event.target.value)); setPage(0); }}
            />
          </>
        )}
      </Box>
    </Box>
  );
}

function ReviewThumbnail({ review }: { review: ReviewTask }) {
  const image = useQuery({
    queryKey: ["review-thumbnail", review.document_id],
    queryFn: async () => URL.createObjectURL(await protectedBlob(review.thumbnail_url || `/documents/${review.document_id}/pages/1/image?variant=original`)),
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: 0,
  });
  useEffect(() => {
    const source = image.data;
    return () => { if (source) URL.revokeObjectURL(source); };
  }, [image.data]);
  return (
    <Link to={`/reviews/${review.review_id}`} className="queue-thumbnail" aria-label={`Review ${review.filename || review.document_id}`}>
      {image.data ? <img src={image.data} alt="" /> : <FileText size={20} />}
    </Link>
  );
}
