import {
  Alert,
  Box,
  Button,
  Chip,
  IconButton,
  MenuItem,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ExternalLink, FileText, Send, ThumbsDown, ThumbsUp } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "../router";

import { api, protectedBlob } from "../api/client";
import { PageHeader } from "../components/PageHeader";
import { DegradationNotice } from "../components/DegradationNotice";
import { RetrievalBackendNote } from "../components/RetrievalBackendNote";
import { StatusChip } from "../components/StatusChip";
import type { AskResponse, Citation, VisionDocument } from "../types/api";

interface Turn {
  question: string;
  response: AskResponse;
}
interface AskSubmission {
  question: string;
  clarificationOption?: string;
}

export function AskDocumentsPage() {
  const navigate = useNavigate();
  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const [scope, setScope] = useState("all_accessible");
  const [documentId, setDocumentId] = useState("");
  const [documentIds, setDocumentIds] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [conversationId, setConversationId] = useState<string>();
  const ask = useMutation({
    mutationFn: (submission: AskSubmission) =>
      api.ask(submission.question, {
        scope,
        documentId: scope === "current_document" ? documentId || undefined : undefined,
        documentIds: scope === "selected_documents" ? documentIds : [],
        clarificationOption: submission.clarificationOption,
        conversationId,
      }),
    onSuccess: (response, submission) => {
      if (response.conversation_id) setConversationId(response.conversation_id);
      setTurns((current) => [...current, { question: submission.question, response }]);
      setQuestion("");
    },
  });
  const submit = (submittedQuestion: string, clarificationOption?: string) => {
    const normalized = submittedQuestion.trim();
    const scopeReady = !(
      (scope === "current_document" && !documentId) ||
      (scope === "selected_documents" && !documentIds.length)
    );
    if (normalized && scopeReady && !ask.isPending) {
      ask.mutate({ question: normalized, clarificationOption });
    }
  };

  const openCitation = (citation: Citation) => {
    navigate(`/documents/${citation.document_id}`, { state: { citation } });
  };
  const scopedDocuments = useMemo(() => {
    const all = documents.data?.documents || [];
    if (scope === "current_document")
      return all.filter((document) => document.document_id === documentId);
    if (scope === "selected_documents")
      return all.filter((document) => documentIds.includes(document.document_id));
    if (scope === "approved_documents")
      return all.filter((document) => document.status === "APPROVED");
    return all;
  }, [documentId, documentIds, documents.data?.documents, scope]);

  return (
    <Box className="page-stack ask-page">
      <PageHeader title="Ask documents" />
      <Box className="ask-layout">
        <Box className="surface conversation-panel">
          {!turns.length ? (
            <Box className="ask-empty">
              <Typography variant="h3">Ask a document question</Typography>
              <Box className="prompt-suggestions">
                {["Hi", "What is the fax number?", "Is this signed?", "Is this safe?"].map(
                  (prompt) => (
                    <Button key={prompt} variant="outlined" onClick={() => setQuestion(prompt)}>
                      {prompt}
                    </Button>
                  ),
                )}
              </Box>
            </Box>
          ) : (
            turns.map((turn, index) => {
              const activeClarification =
                index === turns.length - 1 &&
                turn.response.answer_status === "clarification_needed";
              return (
                <Box className="conversation-turn" key={`${turn.response.question_id}-${index}`}>
                  <Box className="user-message">
                    <Typography>{turn.question}</Typography>
                  </Box>
                  <Box className="answer-message">
                    <Box className="answer-meta">
                      <StatusChip value={turn.response.answer_status} />
                      <Typography variant="caption">
                        {turn.response.query_type.replaceAll("_", " ")}
                        {turn.response.citations.length
                          ? ` | ${Math.round(turn.response.confidence * 100)}%`
                          : ""}
                        {turn.response.retriever !== "not_used"
                          ? ` | ${turn.response.retriever}`
                          : ""}
                        {turn.response.retrieval_path !== "not_used"
                          ? ` | ${turn.response.retrieval_path.replaceAll("_", " ")}`
                          : ""}
                      </Typography>
                    </Box>
                    <Typography>{turn.response.answer}</Typography>
                    <Typography variant="body2" color="text.secondary">
                      {turn.response.reasoning_summary}
                    </Typography>
                    <DegradationNotice degradations={turn.response.degradations} />
                    {turn.response.calculation_trace ? (
                      <Box className="calculation-trace" aria-label="Calculation trace">
                        <Typography variant="caption" fontWeight={700}>
                          {turn.response.calculation_trace.operation.replaceAll("_", " ")}
                        </Typography>
                        <Typography variant="body2">
                          Inputs:{" "}
                          {turn.response.calculation_trace.inputs
                            .map(
                              (input) =>
                                `${input.role}=${input.normalized_value}${input.currency ? ` ${input.currency}` : ""}`,
                            )
                            .join(", ") || "none"}
                        </Typography>
                        <Typography variant="body2">
                          Result: {turn.response.calculation_trace.result ?? "not calculated"}
                          {turn.response.calculation_trace.unit
                            ? ` ${turn.response.calculation_trace.unit}`
                            : ""}
                        </Typography>
                      </Box>
                    ) : null}
                    {turn.response.clarification_options.length ? (
                      <Box className="clarification-options" aria-label="Clarification options">
                        {turn.response.clarification_options.map((option) => (
                          <Chip
                            clickable
                            disabled={ask.isPending || !activeClarification}
                            key={option.option_id}
                            label={option.label}
                            onClick={() => submit(option.label, option.option_id)}
                            variant="outlined"
                          />
                        ))}
                      </Box>
                    ) : null}
                    {turn.response.requires_review ? (
                      <Alert severity="warning">This answer requires human review.</Alert>
                    ) : null}
                    {turn.response.broader_scope_available ? (
                      <Button size="small" onClick={() => setScope("all_accessible")}>
                        Use all accessible documents
                      </Button>
                    ) : null}
                    <Box className="citation-list">
                      {turn.response.citations.map((citation, citationIndex) => (
                        <button
                          className="citation-row"
                          type="button"
                          key={citation.evidence_id || citationIndex}
                          onClick={() => openCitation(citation)}
                        >
                          <Box>
                            <Typography variant="caption" fontWeight={700}>
                              [{citationIndex + 1}] {citation.filename || "Document"}, page{" "}
                              {citation.page_number}
                            </Typography>
                            <Typography variant="body2">{citation.source_text}</Typography>
                          </Box>
                          <ExternalLink size={15} />
                        </button>
                      ))}
                    </Box>
                    {turn.response.retrieval_trace?.sources_checked.length ? (
                      <Typography variant="caption" color="text.secondary">
                        Checked {turn.response.retrieval_trace.sources_checked.length} evidence
                        sources in {Math.round(turn.response.retrieval_trace.retrieval_duration_ms)}{" "}
                        ms.
                      </Typography>
                    ) : null}
                    <RetrievalBackendNote trace={turn.response.retrieval_trace} />
                    <Box className="answer-feedback">
                      <Typography variant="caption">Was this supported?</Typography>
                      <Tooltip title="Helpful">
                        <IconButton
                          size="small"
                          onClick={() =>
                            void api.questionFeedback(turn.response.question_id, "helpful", "")
                          }
                        >
                          <ThumbsUp size={15} />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Not helpful">
                        <IconButton
                          size="small"
                          onClick={() =>
                            void api.questionFeedback(turn.response.question_id, "not_helpful", "")
                          }
                        >
                          <ThumbsDown size={15} />
                        </IconButton>
                      </Tooltip>
                    </Box>
                  </Box>
                </Box>
              );
            })
          )}
          {ask.error ? <Alert severity="error">{ask.error.message}</Alert> : null}
        </Box>
        <Box className="ask-composer surface">
          <Box className="ask-scope-row">
            <Box className="ask-scope-controls">
              <TextField
                select
                label="Scope"
                value={scope}
                onChange={(event) => setScope(event.target.value)}
                size="small"
              >
                <MenuItem value="current_document">Current document</MenuItem>
                <MenuItem value="selected_documents">Selected documents</MenuItem>
                <MenuItem value="approved_documents">All approved documents</MenuItem>
                <MenuItem value="all_accessible">All accessible documents</MenuItem>
              </TextField>
              {scope === "current_document" ? (
                <TextField
                  select
                  label="Document"
                  value={documentId}
                  onChange={(event) => setDocumentId(event.target.value)}
                  size="small"
                >
                  <MenuItem value="">Select document</MenuItem>
                  {documents.data?.documents.map((document) => (
                    <MenuItem value={document.document_id} key={document.document_id}>
                      {document.filename}
                    </MenuItem>
                  ))}
                </TextField>
              ) : null}
              {scope === "selected_documents" ? (
                <TextField
                  select
                  label="Documents"
                  value={documentIds}
                  onChange={(event) =>
                    setDocumentIds(
                      typeof event.target.value === "string"
                        ? event.target.value.split(",")
                        : event.target.value,
                    )
                  }
                  size="small"
                  slotProps={{ select: { multiple: true } }}
                >
                  {documents.data?.documents.map((document) => (
                    <MenuItem value={document.document_id} key={document.document_id}>
                      {document.filename}
                    </MenuItem>
                  ))}
                </TextField>
              ) : null}
            </Box>
            <Box className="ask-scope-documents">
              <Typography variant="caption" sx={{ minWidth: 92 }}>
                {scopedDocuments.length} document{scopedDocuments.length === 1 ? "" : "s"}
              </Typography>
              {scopedDocuments.length <= 6
                ? scopedDocuments.map((document) => (
                    <ScopeDocument key={document.document_id} document={document} />
                  ))
                : null}
            </Box>
          </Box>
          <Box className="ask-input-row">
            <TextField
              multiline
              minRows={2}
              maxRows={6}
              placeholder="Ask a question about your documents"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  submit(question);
                }
              }}
              fullWidth
            />
            <Button
              variant="contained"
              startIcon={<Send size={17} />}
              disabled={
                !question.trim() ||
                ask.isPending ||
                (scope === "current_document" && !documentId) ||
                (scope === "selected_documents" && !documentIds.length)
              }
              onClick={() => submit(question)}
            >
              {ask.isPending ? "Finding evidence" : "Ask"}
            </Button>
          </Box>
        </Box>
      </Box>
    </Box>
  );
}

function ScopeDocument({ document }: { document: VisionDocument }) {
  const thumbnail = useQuery({
    queryKey: ["ask-scope-thumbnail", document.document_id],
    queryFn: async () =>
      URL.createObjectURL(
        await protectedBlob(`/documents/${document.document_id}/pages/1/image?variant=original`),
      ),
    staleTime: Number.POSITIVE_INFINITY,
    gcTime: 0,
  });
  useEffect(() => {
    const source = thumbnail.data;
    return () => {
      if (source) URL.revokeObjectURL(source);
    };
  }, [thumbnail.data]);
  return (
    <Box
      sx={{
        display: "grid",
        gridTemplateColumns: "40px minmax(130px, 1fr)",
        gap: 1,
        alignItems: "center",
        minWidth: 210,
        border: "1px solid",
        borderColor: "divider",
        p: 0.75,
        borderRadius: 1,
      }}
    >
      <Box
        sx={{
          width: 40,
          height: 46,
          display: "grid",
          placeItems: "center",
          overflow: "hidden",
          bgcolor: "action.hover",
        }}
      >
        {thumbnail.data ? (
          <img
            src={thumbnail.data}
            alt=""
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <FileText size={18} />
        )}
      </Box>
      <Box sx={{ minWidth: 0 }}>
        <Typography variant="caption" fontWeight={700} noWrap>
          {document.filename}
        </Typography>
        <Typography variant="caption" color="text.secondary" display="block">
          {document.document_type.replaceAll("_", " ")} | {document.page_count}p |{" "}
          {document.extracted_fields.length} fields
        </Typography>
        <Typography variant="caption" color="text.secondary" display="block">
          {String(document.metadata.pipeline_version || "legacy")}
        </Typography>
      </Box>
    </Box>
  );
}
