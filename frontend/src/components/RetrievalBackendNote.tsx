import { Tooltip, Typography } from "@mui/material";

import type { RetrievalTrace } from "../types/api";

/**
 * States which retrieval backend produced this answer.
 *
 * A lexical backend matches wording, not meaning, so "nothing was found" can
 * mean "nothing matched" rather than "nothing is there". Naming the backend is
 * what lets a reader tell those apart — and it is what stops the page implying
 * a semantic capability the system does not currently have.
 */
export function RetrievalBackendNote({ trace }: { trace?: RetrievalTrace | null }) {
  if (!trace?.embedding_backend) return null;

  const semantic = trace.semantic_matching === true;
  const label = semantic
    ? `Semantic matching · ${trace.embedding_model || trace.embedding_backend}`
    : `Keyword matching · ${trace.embedding_model || trace.embedding_backend}`;
  const explanation = semantic
    ? "Evidence was matched on meaning, so differently worded questions can still find it."
    : "Evidence was matched on the words used. A question phrased differently from the document may not reach evidence that exists.";

  return (
    <Tooltip title={explanation}>
      <Typography variant="caption" color="text.secondary" className="retrieval-backend-note">
        {label}
      </Typography>
    </Tooltip>
  );
}
