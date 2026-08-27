import { Box, Typography } from "@mui/material";

/** Shared heading for the panel's sections: a title and the item count. */
export function SectionHeading({ title, count }: { title: string; count: number }) {
  return (
    <Box className="section-heading compact-heading">
      <Typography variant="h4">{title}</Typography>
      <Typography variant="caption">{count}</Typography>
    </Box>
  );
}
