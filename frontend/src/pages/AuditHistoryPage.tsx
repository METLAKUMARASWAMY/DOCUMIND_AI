import {
  Box,
  InputAdornment,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Typography,
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { useMemo, useState } from "react";
import { Link } from "../router";

import { api } from "../api/client";
import { EmptyState, ErrorState, LoadingRows } from "../components/Feedback";
import { PageHeader } from "../components/PageHeader";

export function AuditHistoryPage() {
  const audit = useQuery({ queryKey: ["audit"], queryFn: api.audit });
  const [search, setSearch] = useState("");
  const rows = useMemo(
    () =>
      (audit.data?.events || []).filter((event) =>
        JSON.stringify(event).toLowerCase().includes(search.toLowerCase()),
      ),
    [audit.data, search],
  );
  return (
    <Box className="page-stack">
      <PageHeader
        title="Audit history"
        subtitle="Append-only processing and reviewer activity for this organisation."
      />
      <Box className="filter-bar surface">
        <TextField
          size="small"
          placeholder="Search actor, event, or document"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search size={16} />
              </InputAdornment>
            ),
          }}
        />
      </Box>
      <Box className="surface table-surface">
        {audit.isLoading ? (
          <LoadingRows rows={10} />
        ) : audit.error ? (
          <ErrorState error={audit.error} />
        ) : !rows.length ? (
          <EmptyState title="No audit events" detail="No events match the current search." />
        ) : (
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Time</TableCell>
                <TableCell>Actor</TableCell>
                <TableCell>Event</TableCell>
                <TableCell>Document</TableCell>
                <TableCell>Message</TableCell>
                <TableCell>Metadata</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {rows.map((event) => (
                <TableRow key={String(event.event_id)}>
                  <TableCell>{new Date(String(event.created_at)).toLocaleString()}</TableCell>
                  <TableCell>{String(event.actor)}</TableCell>
                  <TableCell>
                    <Typography fontWeight={700}>
                      {String(event.event_type).replaceAll("_", " ")}
                    </Typography>
                  </TableCell>
                  <TableCell>
                    <Typography
                      component={Link}
                      to={`/documents/${String(event.document_id)}`}
                      color="primary"
                    >
                      {String(event.document_id)}
                    </Typography>
                  </TableCell>
                  <TableCell>{String(event.message)}</TableCell>
                  <TableCell>
                    <code>{JSON.stringify(event.metadata)}</code>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Box>
    </Box>
  );
}
