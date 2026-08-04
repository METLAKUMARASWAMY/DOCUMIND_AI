import { Box, Typography } from "@mui/material";

export function PageHeader({
  title,
  subtitle,
  actions,
}: {
  title: string;
  subtitle?: string;
  actions?: React.ReactNode;
}) {
  return (
    <Box className="page-header">
      <Box>
        <Typography variant="h2" component="h1">
          {title}
        </Typography>
        {subtitle ? <Typography color="text.secondary">{subtitle}</Typography> : null}
      </Box>
      {actions ? <Box className="page-actions">{actions}</Box> : null}
    </Box>
  );
}
