import { Box, Typography } from "@mui/material";

import type { VisionDocument } from "../../types/api";
import { StatusChip } from "../StatusChip";
import { humanize } from "./values";

type SchemaProfile = NonNullable<VisionDocument["schema_profile"]>;

/** Which schema was applied to this document, and how it was chosen. */
export function SchemaProfileSummary({ profile }: { profile?: SchemaProfile | null }) {
  if (!profile) return null;

  return (
    <Box className="schema-profile-summary">
      <Box>
        <Typography variant="overline" color="text.secondary">
          Schema profile
        </Typography>
        <Typography variant="h4">
          Mode {profile.mode} · {humanize(profile.schema_status)}
        </Typography>
      </Box>
      <StatusChip value={profile.schema_status} />
      <Typography variant="caption" color="text.secondary">
        {profile.schema_id ? `${profile.schema_id} · ` : ""}
        {profile.schema_version} · selected by {humanize(profile.selection_source)}
      </Typography>
    </Box>
  );
}
