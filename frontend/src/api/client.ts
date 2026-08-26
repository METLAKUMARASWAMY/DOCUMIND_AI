import type {
  ApiEnvelope,
  ApprovalPolicy,
  AskResponse,
  AuthUser,
  FinanceNarrative,
  GraphProjection,
  ReportRecord,
  ReviewDecisionPayload,
  ReviewQueueFilters,
  ReviewQueueResponse,
  ReviewTask,
  VisionDocument,
} from "../types/api";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api/v1";
const ACCESS_KEY = "documind.access";
const REFRESH_KEY = "documind.refresh";
const SESSION_KEY = "documind.session";

let refreshPromise: Promise<boolean> | null = null;

export class ApiError extends Error {
  status: number;
  details: unknown;

  constructor(message: string, status: number, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

export function accessToken(): string | null {
  return sessionStorage.getItem(ACCESS_KEY);
}

export function sessionId(): string {
  let value = sessionStorage.getItem(SESSION_KEY);
  if (!value) {
    value = crypto.randomUUID();
    sessionStorage.setItem(SESSION_KEY, value);
  }
  return value;
}

export function clearSession(): void {
  sessionStorage.removeItem(ACCESS_KEY);
  sessionStorage.removeItem(REFRESH_KEY);
}

function storeTokens(access: string, refresh: string): void {
  sessionStorage.setItem(ACCESS_KEY, access);
  sessionStorage.setItem(REFRESH_KEY, refresh);
}

function authHeaders(headers: HeadersInit = {}): Headers {
  const result = new Headers(headers);
  const token = accessToken();
  if (token) result.set("Authorization", `Bearer ${token}`);
  result.set("X-Session-ID", sessionId());
  return result;
}

async function parseResponse<T>(response: Response): Promise<T> {
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "object" && payload ? payload.detail : null;
    const message = typeof detail === "object" && detail
      ? String(detail.message || "Request failed")
      : typeof payload === "object" && payload
        ? String(detail || payload.message || "Request failed")
        : String(payload || `Request failed with ${response.status}`);
    throw new ApiError(message, response.status, payload);
  }
  if (typeof payload === "object" && payload && "success" in payload && "data" in payload) {
    const envelope = payload as ApiEnvelope<T>;
    if (!envelope.success) throw new ApiError("Request failed", response.status, envelope.errors);
    return envelope.data;
  }
  return payload as T;
}

async function refreshAccess(): Promise<boolean> {
  const token = sessionStorage.getItem(REFRESH_KEY);
  if (!token) return false;
  try {
    const response = await fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Session-ID": sessionId() },
      body: JSON.stringify({ refresh_token: token }),
    });
    const data = await parseResponse<{ access_token: string; refresh_token: string }>(response);
    storeTokens(data.access_token, data.refresh_token);
    return true;
  } catch {
    clearSession();
    return false;
  }
}

export async function apiRequest<T>(
  path: string,
  init: RequestInit = {},
  retry = true,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: authHeaders(init.headers),
  });
  if (response.status === 401 && retry && sessionStorage.getItem(REFRESH_KEY)) {
    refreshPromise ??= refreshAccess().finally(() => {
      refreshPromise = null;
    });
    if (await refreshPromise) return apiRequest<T>(path, init, false);
  }
  return parseResponse<T>(response);
}

export const api = {
  async login(email: string, password: string) {
    const data = await apiRequest<{
      access_token: string;
      refresh_token: string;
      user: AuthUser;
    }>("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    storeTokens(data.access_token, data.refresh_token);
    return data.user;
  },

  register(payload: {
    name: string;
    email: string;
    password: string;
    organisation_name: string;
  }) {
    return apiRequest<{ user: AuthUser }>("/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...payload, role: "VIEWER" }),
    });
  },

  forgotPassword(email: string) {
    return apiRequest<{ status: string; message: string }>("/auth/forgot-password", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email }),
    });
  },

  me: () => apiRequest<AuthUser>("/auth/me"),

  async logout() {
    const refresh = sessionStorage.getItem(REFRESH_KEY);
    try {
      await apiRequest("/auth/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refresh }),
      });
    } finally {
      clearSession();
    }
  },

  documents: () =>
    apiRequest<{ total: number; documents: VisionDocument[] }>("/documents"),
  document: (id: string) => apiRequest<VisionDocument>(`/documents/${id}`),
  deleteDocument: (id: string) => apiRequest(`/documents/${id}`, { method: "DELETE" }),
  documentStatus: (id: string) =>
    apiRequest<Record<string, unknown>>(`/documents/${id}/status`),
  reprocessDocument: (id: string) =>
    apiRequest<{
      case_id: string;
      job_id: string;
      document_ids: string[];
      status: string;
      dispatch: Record<string, unknown>;
    }>(`/documents/${id}/reprocess`, { method: "POST" }),
  seedDemo: () =>
    apiRequest<{
      message: string;
      document_ids: string[];
      primary_document_id: string;
      documents: Array<Record<string, unknown>>;
      narrative_id?: string | null;
      report_id?: string | null;
      ask_examples: string[];
      local_ocr_status: { status: string; reason: string };
    }>("/demo/vision/seed", { method: "POST" }),

  async uploadDocuments(files: File[], metadata: Record<string, unknown>) {
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    body.append("metadata", JSON.stringify(metadata));
    return apiRequest<{
      case_id: string;
      job_id: string;
      document_ids: string[];
      status: string;
      dispatch: Record<string, unknown>;
    }>("/documents/upload", { method: "POST", body });
  },

  job: (id: string) => apiRequest<Record<string, unknown>>(`/vision/jobs/${id}`),
  retryJob: (id: string) => apiRequest(`/vision/jobs/${id}/retry`, { method: "POST" }),
  reviews: (filters: ReviewQueueFilters = {}) => {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== "") params.set(key, String(value));
    });
    const query = params.toString();
    return apiRequest<ReviewQueueResponse>(`/reviews${query ? `?${query}` : ""}`);
  },
  review: (id: string) =>
    apiRequest<{ review: ReviewTask; document: VisionDocument; approval_policy: ApprovalPolicy }>(`/reviews/${id}`),
  patchField: (
    reviewId: string,
    fieldId: string,
    payload: {
      reviewed_value?: string | null;
      reviewed_field_name?: string | null;
      reviewer_status: string;
      comment: string;
    },
  ) =>
    apiRequest(`/reviews/${reviewId}/fields/${fieldId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  acceptHighConfidence: (reviewId: string) =>
    apiRequest(`/reviews/${reviewId}/accept-high-confidence`, { method: "POST" }),
  decideReview: (reviewId: string, action: string, payload: ReviewDecisionPayload) =>
    apiRequest(`/reviews/${reviewId}/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        decision: ({
          approve: "APPROVED",
          reject: "REJECTED",
          escalate: "UNDER_REVIEW",
          "confirm-error": "CONFIRMED_ERROR",
          "confirm-fraud": "CONFIRMED_FRAUD",
        } as Record<string, string>)[action],
        ...payload,
      }),
    }),
  reauthenticate: (password: string) =>
    apiRequest<{
      reauthentication_token: string;
      token_type: string;
      expires_in: number;
      authenticated_at: number;
    }>("/auth/reauthenticate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ password }),
    }),

  ask: (question: string, options: { scope: string; documentId?: string; documentIds?: string[]; clarificationOption?: string; conversationId?: string }) =>
    apiRequest<AskResponse>("/ai/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        conversation_id: options.conversationId || null,
        clarification_option: options.clarificationOption || null,
        scope: options.scope,
        document_id: options.documentId || null,
        document_ids: options.documentIds || [],
        top_k: 8,
      }),
    }),
  question: (id: string) => apiRequest<Record<string, unknown>>(`/ai/responses/${id}`),
  questionFeedback: (id: string, rating: string, comment: string) =>
    apiRequest(`/ai/responses/${id}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rating, comment }),
    }),

  graphDocument: (id: string) => apiRequest<GraphProjection>(`/graph/document/${id}`),
  graphEntity: (id: string) => apiRequest<Record<string, unknown>>(`/graph/entity/${encodeURIComponent(id)}`),
  graphQuery: (question: string, documentId?: string) =>
    apiRequest<{ query_type: string; answer: AskResponse }>("/graph/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, document_id: documentId || null }),
    }),
  verifyGraphRelationship: (id: string, comment: string, evidenceAcknowledged: boolean) =>
    apiRequest<Record<string, unknown>>(`/graph/relationships/${encodeURIComponent(id)}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comment, evidence_acknowledged: evidenceAcknowledged }),
    }),

  analyticsOverview: (dateFrom?: string, dateTo?: string) => apiRequest<Record<string, unknown>>(`/analytics/overview${dateRangeQuery(dateFrom, dateTo)}`),
  analyticsVendors: (dateFrom?: string, dateTo?: string) => apiRequest<Record<string, unknown>>(`/analytics/vendors${dateRangeQuery(dateFrom, dateTo)}`),
  analyticsRisk: (dateFrom?: string, dateTo?: string) => apiRequest<Record<string, unknown>>(`/analytics/risk${dateRangeQuery(dateFrom, dateTo)}`),
  analyticsReview: (dateFrom?: string, dateTo?: string) => apiRequest<Record<string, unknown>>(`/analytics/review${dateRangeQuery(dateFrom, dateTo)}`),
  analyticsModels: (dateFrom?: string, dateTo?: string) => apiRequest<Record<string, unknown>>(`/analytics/models${dateRangeQuery(dateFrom, dateTo)}`),

  narratives: () =>
    apiRequest<{ total: number; narratives: FinanceNarrative[] }>("/finance-narratives"),
  generateNarrative: (payload: Record<string, unknown>) =>
    apiRequest<FinanceNarrative>("/finance-narratives", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  reports: () => apiRequest<{ total: number; reports: ReportRecord[] }>("/reports"),
  generateReport: (payload: Record<string, unknown>) =>
    apiRequest<ReportRecord>("/reports/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  report: (id: string) => apiRequest<ReportRecord>(`/reports/${id}`),
  archiveReport: (id: string) => apiRequest<ReportRecord>(`/reports/${id}/archive`, { method: "POST" }),

  modelComparison: (id: string) =>
    apiRequest<Record<string, unknown>>(`/models/comparison/${id}`),
  models: () => apiRequest<Record<string, unknown>>("/admin/models"),
  patchModel: (id: string, payload: Record<string, unknown>) =>
    apiRequest(`/admin/models/${encodeURIComponent(id)}/requested-state`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  smokeTestModel: (id: string) =>
    apiRequest<Record<string, unknown>>(`/admin/models/${encodeURIComponent(id)}/smoke-test`, {
      method: "POST",
    }),
  rules: () => apiRequest<Record<string, unknown>>("/admin/rules"),
  patchRule: (id: string, payload: Record<string, unknown>) =>
    apiRequest(`/admin/rules/${encodeURIComponent(id)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),
  audit: () => apiRequest<{ total: number; events: Array<Record<string, unknown>> }>("/admin/audit"),
};

export async function protectedBlob(path: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}${path}`, { headers: authHeaders() });
  if (!response.ok) await parseResponse(response);
  return response.blob();
}

function dateRangeQuery(dateFrom?: string, dateTo?: string): string {
  const params = new URLSearchParams();
  if (dateFrom) params.set("date_from", dateFrom);
  if (dateTo) params.set("date_to", dateTo);
  const query = params.toString();
  return query ? `?${query}` : "";
}

export async function downloadReport(report: ReportRecord): Promise<void> {
  const blob = await protectedBlob(`/reports/${report.report_id}/download`);
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${report.report_id}.pdf`;
  anchor.click();
  URL.revokeObjectURL(url);
}

export async function streamDocumentEvents(
  documentId: string,
  signal: AbortSignal,
  onEvent: (event: { type: string; data: Record<string, unknown> }) => void,
): Promise<void> {
  const response = await fetch(`${API_BASE}/documents/${documentId}/events`, {
    headers: authHeaders({ Accept: "text/event-stream" }),
    signal,
  });
  if (!response.ok || !response.body) await parseResponse(response);
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (!signal.aborted) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const type = frame.match(/^event:\s*(.+)$/m)?.[1] || "message";
      const raw = frame.match(/^data:\s*(.+)$/m)?.[1];
      if (!raw) continue;
      try {
        onEvent({ type, data: JSON.parse(raw) as Record<string, unknown> });
      } catch {
        // A malformed event is ignored; the next status event remains authoritative.
      }
    }
  }
}

export function pdfRequest(path: string): { url: string; httpHeaders: Record<string, string> } {
  const token = accessToken();
  return {
    url: `${API_BASE}${path}`,
    httpHeaders: token ? { Authorization: `Bearer ${token}`, "X-Session-ID": sessionId() } : {},
  };
}
