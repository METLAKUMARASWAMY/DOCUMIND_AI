import { Alert, AlertTitle, Box, Typography } from "@mui/material";

export interface AnswerDegradation {
  component: string;
  stage: string;
  detail: string;
  impact: string;
}

const STAGE_LABELS: Record<string, string> = {
  answer_generation: "Answer generation",
  grounding_validation: "Grounding validation",
  retrieval: "Retrieval",
  vector_search: "Vector search",
  graph_search: "Graph search",
};

/**
 * Surfaces backend-disclosed degradations on an answer.
 *
 * The backend records every fallback it took (a generator that failed, a
 * generated answer rejected for citing nothing resolvable) on
 * `VisionAskResponse.degradations`. Rendering it here is what makes the
 * disclosure real: a degradation the user never sees is still a silent fallback.
 *
 * A rejected answer is reported as reassurance, not alarm — the grounding
 * contract worked and the user is reading evidence-backed text instead.
 */
export function DegradationNotice({ degradations }: { degradations?: AnswerDegradation[] | null }) {
  if (!degradations?.length) return null;

  const rejected = degradations.some((item) => item.impact === "generated_answer_rejected");

  return (
    <Alert
      severity={rejected ? "info" : "warning"}
      variant="outlined"
      className="degradation-notice"
    >
      <AlertTitle>
        {rejected
          ? "An ungrounded answer was rejected"
          : "This answer was produced with reduced capability"}
      </AlertTitle>
      <Box component="ul" className="degradation-list">
        {degradations.map((item, index) => (
          <li key={`${item.component}-${item.stage}-${index}`}>
            <Typography variant="body2" component="span" fontWeight={650}>
              {STAGE_LABELS[item.stage] ?? item.stage.replaceAll("_", " ")}
            </Typography>
            <Typography variant="body2" component="span" color="text.secondary">
              {" — "}
              {item.detail}
            </Typography>
          </li>
        ))}
      </Box>
    </Alert>
  );
}
