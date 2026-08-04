import { Alert, Box, Button, CircularProgress, Skeleton, Typography } from "@mui/material";
import { RefreshCw } from "lucide-react";

export function LoadingRows({ rows = 5 }: { rows?: number }) {
  return (
    <Box aria-label="Loading" sx={{ display: "grid", gap: 1.2 }}>
      {Array.from({ length: rows }, (_, index) => (
        <Skeleton key={index} variant="rounded" height={44} />
      ))}
    </Box>
  );
}

export function LoadingState({ label = "Loading" }: { label?: string }) {
  return (
    <Box className="full-page-center" role="status" aria-label={label}>
      <CircularProgress size={30} />
      <Typography color="text.secondary">{label}</Typography>
    </Box>
  );
}

export function ErrorState({ error, retry }: { error: unknown; retry?: () => void }) {
  const message = error instanceof Error ? error.message : "The request could not be completed.";
  return (
    <Alert
      severity="error"
      action={
        retry ? (
          <Button color="inherit" size="small" startIcon={<RefreshCw size={15} />} onClick={retry}>
            Retry
          </Button>
        ) : undefined
      }
    >
      {message}
    </Alert>
  );
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <Box className="empty-state">
      <Typography variant="h3">{title}</Typography>
      <Typography color="text.secondary">{detail}</Typography>
    </Box>
  );
}
