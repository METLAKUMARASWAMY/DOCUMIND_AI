import { Alert, Box, Button, IconButton, MenuItem, TextField, Tooltip, Typography } from "@mui/material";
import { useMutation, useQuery } from "@tanstack/react-query";
import { ExternalLink, Send, ThumbsDown, ThumbsUp } from "lucide-react";
import { useState } from "react";
import { useNavigate } from "../router";

import { api } from "../api/client";
import { PageHeader } from "../components/PageHeader";
import { StatusChip } from "../components/StatusChip";
import type { AskResponse, Citation } from "../types/api";

interface Turn { question: string; response: AskResponse }

export function AskDocumentsPage() {
  const navigate = useNavigate();
  const documents = useQuery({ queryKey: ["documents"], queryFn: api.documents });
  const [documentId, setDocumentId] = useState("");
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const ask = useMutation({
    mutationFn: () => api.ask(question, documentId || undefined),
    onSuccess: (response) => {
      setTurns((current) => [...current, { question, response }]);
      setQuestion("");
    },
  });

  const openCitation = (citation: Citation) => {
    navigate(`/documents/${citation.document_id}`, { state: { citation } });
  };

  return (
    <Box className="page-stack ask-page">
      <PageHeader title="Ask documents" />
      <Box className="ask-layout">
        <Box className="surface conversation-panel">
          {!turns.length ? (
            <Box className="ask-empty">
              <Typography variant="h3">Ask a document question</Typography>
              <Box className="prompt-suggestions">
                {["Hi", "What is the fax number?", "Is this signed?", "Is this safe?"].map((prompt) => <Button key={prompt} variant="outlined" onClick={() => setQuestion(prompt)}>{prompt}</Button>)}
              </Box>
            </Box>
          ) : turns.map((turn, index) => (
            <Box className="conversation-turn" key={`${turn.response.question_id}-${index}`}>
              <Box className="user-message"><Typography>{turn.question}</Typography></Box>
              <Box className="answer-message">
                <Box className="answer-meta">
                  <StatusChip value={turn.response.answer_status} />
                  <Typography variant="caption">
                    {turn.response.query_type.replaceAll("_", " ")}
                    {turn.response.citations.length ? ` | ${Math.round(turn.response.confidence * 100)}%` : ""}
                    {turn.response.retriever !== "not_used" ? ` | ${turn.response.retriever}` : ""}
                  </Typography>
                </Box>
                <Typography>{turn.response.answer}</Typography>
                <Typography variant="body2" color="text.secondary">{turn.response.reasoning_summary}</Typography>
                {turn.response.requires_review ? <Alert severity="warning">This answer requires human review.</Alert> : null}
                <Box className="citation-list">
                  {turn.response.citations.map((citation, citationIndex) => (
                    <button className="citation-row" type="button" key={citation.evidence_id || citationIndex} onClick={() => openCitation(citation)}>
                      <Box>
                        <Typography variant="caption" fontWeight={700}>[{citationIndex + 1}] {citation.filename || "Document"}, page {citation.page_number}</Typography>
                        <Typography variant="body2">{citation.source_text}</Typography>
                      </Box>
                      <ExternalLink size={15} />
                    </button>
                  ))}
                </Box>
                <Box className="answer-feedback">
                  <Typography variant="caption">Was this supported?</Typography>
                  <Tooltip title="Helpful"><IconButton size="small" onClick={() => void api.questionFeedback(turn.response.question_id, "helpful", "") }><ThumbsUp size={15} /></IconButton></Tooltip>
                  <Tooltip title="Not helpful"><IconButton size="small" onClick={() => void api.questionFeedback(turn.response.question_id, "not_helpful", "") }><ThumbsDown size={15} /></IconButton></Tooltip>
                </Box>
              </Box>
            </Box>
          ))}
          {ask.error ? <Alert severity="error">{ask.error.message}</Alert> : null}
        </Box>
        <Box className="ask-composer surface">
          <TextField select label="Scope" value={documentId} onChange={(event) => setDocumentId(event.target.value)} size="small">
            <MenuItem value="">All accessible documents</MenuItem>
            {documents.data?.documents.map((document) => <MenuItem value={document.document_id} key={document.document_id}>{document.filename}</MenuItem>)}
          </TextField>
          <TextField
            multiline
            minRows={2}
            maxRows={6}
            placeholder="Ask a question about your documents"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); if (question.trim()) ask.mutate(); } }}
            fullWidth
          />
          <Button variant="contained" startIcon={<Send size={17} />} disabled={!question.trim() || ask.isPending} onClick={() => ask.mutate()}>
            {ask.isPending ? "Finding evidence" : "Ask"}
          </Button>
        </Box>
      </Box>
    </Box>
  );
}
