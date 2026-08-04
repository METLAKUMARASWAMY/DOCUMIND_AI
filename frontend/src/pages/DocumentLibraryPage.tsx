import { Box, Button, IconButton, InputAdornment, MenuItem, Table, TableBody, TableCell, TableHead, TableRow, TextField, Tooltip, Typography } from "@mui/material";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Eye, Search, Trash2, Upload } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "../router";

import { api } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { EmptyState, ErrorState, LoadingRows } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";

export function DocumentLibraryPage() {
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("all");
  const remove = useMutation({
    mutationFn: api.deleteDocument,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["documents"] }),
  });
  const rows = useMemo(
    () => (documents.data?.documents || []).filter((document) => {
      const matchesText = `${document.filename} ${document.document_type} ${document.document_id}`.toLowerCase().includes(search.toLowerCase());
      return matchesText && (status === "all" || document.status === status);
    }),
    [documents.data, search, status],
  );

  return (
    <Box className="page-stack">
      <PageHeader title="Document library" subtitle="Tenant-isolated source documents and structured results." actions={<Button component={Link} to="/upload" variant="contained" startIcon={<Upload size={17} />}>Upload</Button>} />
      <Box className="filter-bar surface">
        <TextField
          size="small"
          placeholder="Search documents"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          InputProps={{ startAdornment: <InputAdornment position="start"><Search size={17} /></InputAdornment> }}
        />
        <TextField select size="small" label="Status" value={status} onChange={(event) => setStatus(event.target.value)}>
          {["all", "REVIEW_REQUIRED", "UNDER_REVIEW", "APPROVED", "COMPLETED", "REJECTED"].map((value) => <MenuItem key={value} value={value}>{value.replaceAll("_", " ")}</MenuItem>)}
        </TextField>
      </Box>
      <Box className="surface table-surface">
        {documents.isLoading ? <LoadingRows rows={8} /> : documents.error ? <ErrorState error={documents.error} retry={() => void documents.refetch()} /> : !rows.length ? (
          <EmptyState title="No matching documents" detail="Upload a document or change the current filters." />
        ) : (
          <Table>
            <TableHead><TableRow><TableCell>File</TableCell><TableCell>Type</TableCell><TableCell>Pages</TableCell><TableCell>Confidence</TableCell><TableCell>Status</TableCell><TableCell>Updated</TableCell><TableCell align="right">Actions</TableCell></TableRow></TableHead>
            <TableBody>{rows.map((document) => {
              const confidence = document.extracted_fields.length ? document.extracted_fields.reduce((sum, field) => sum + field.confidence, 0) / document.extracted_fields.length : 0;
              return (
                <TableRow key={document.document_id} hover>
                  <TableCell><Typography fontWeight={650}>{document.filename}</Typography><Typography variant="caption" color="text.secondary">{document.document_id}</Typography></TableCell>
                  <TableCell>{document.document_type.replaceAll("_", " ")}</TableCell>
                  <TableCell>{document.page_count}</TableCell>
                  <TableCell>{Math.round(confidence * 100)}%</TableCell>
                  <TableCell><StatusChip value={document.status} /></TableCell>
                  <TableCell>{new Date(document.updated_at).toLocaleString()}</TableCell>
                  <TableCell align="right">
                    <Tooltip title="Open document"><IconButton component={Link} to={`/documents/${document.document_id}`} aria-label={`Open ${document.filename}`}><Eye size={17} /></IconButton></Tooltip>
                    {user?.role === "ADMIN" || user?.role === "REVIEWER" ? <Tooltip title="Delete document"><span><IconButton color="error" disabled={remove.isPending} onClick={() => { if (window.confirm(`Delete ${document.filename}?`)) remove.mutate(document.document_id); }} aria-label={`Delete ${document.filename}`}><Trash2 size={17} /></IconButton></span></Tooltip> : null}
                  </TableCell>
                </TableRow>
              );
            })}</TableBody>
          </Table>
        )}
      </Box>
    </Box>
  );
}
