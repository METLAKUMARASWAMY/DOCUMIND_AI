import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { api } from "../api/client";
import { BrowserRouter } from "../router";
import type { AskResponse } from "../types/api";
import { AskDocumentsPage } from "./AskDocumentsPage";

vi.mock("../api/client", () => ({
  api: {
    ask: vi.fn(),
    documents: vi.fn(),
    questionFeedback: vi.fn(),
  },
  protectedBlob: vi.fn(),
}));

function response(overrides: Partial<AskResponse>): AskResponse {
  return {
    question_id: "question-1",
    answer: "Choose a safety check.",
    answer_status: "clarification_needed",
    query_type: "ambiguous_question",
    citations: [],
    confidence: 0,
    reasoning_summary: "Clarification is required.",
    requires_review: false,
    generator_name: "grounded-evidence-composer",
    generator_version: "v1",
    retriever: "not_used",
    retrieval_path: "not_used",
    scope: { scope: "all_accessible", document_count: 0, documents: [] },
    broader_scope_available: false,
    conversation_id: "conversation-from-server",
    clarification_options: [{
      option_id: "risk_indicators",
      label: "Check suspicious indicators",
      resolved_question: "What suspicious indicators are present?",
      query_type: "risk_question",
    }],
    reranker: "not_used",
    graph_paths: [],
    ...overrides,
  };
}

describe("AskDocumentsPage clarification state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(api.documents).mockResolvedValue({ documents: [], total: 0 });
  });

  it("posts the returned conversation id and selected option on the next turn", async () => {
    vi.mocked(api.ask)
      .mockResolvedValueOnce(response({}))
      .mockResolvedValueOnce(response({
        question_id: "question-2",
        answer: "No confirmed fraud determination is available.",
        answer_status: "supported",
        query_type: "risk_question",
        clarification_options: [],
      }));
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
    });
    const user = userEvent.setup();
    render(
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <AskDocumentsPage />
        </BrowserRouter>
      </QueryClientProvider>,
    );

    await user.type(
      screen.getByPlaceholderText("Ask a question about your documents"),
      "Is this safe?",
    );
    await user.click(screen.getByRole("button", { name: "Ask" }));
    await user.click(await screen.findByText("Check suspicious indicators"));

    await waitFor(() => expect(api.ask).toHaveBeenCalledTimes(2));
    expect(api.ask).toHaveBeenNthCalledWith(1, "Is this safe?", expect.objectContaining({
      conversationId: undefined,
    }));
    expect(api.ask).toHaveBeenNthCalledWith(2, "Check suspicious indicators", expect.objectContaining({
      clarificationOption: "risk_indicators",
      conversationId: "conversation-from-server",
    }));
    expect(screen.getByRole("button", { name: "Check suspicious indicators" })).toHaveAttribute("aria-disabled", "true");
  });
});
