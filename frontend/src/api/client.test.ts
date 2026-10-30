import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, apiRequest, clearSession } from "./client";

/**
 * The login screen renders `error.message` verbatim. Before this suite existed,
 * a non-JSON failure body (an nginx 502 page) was stringified straight into that
 * message, so the user was shown raw HTML -- including the MSIE padding comments
 * -- instead of being told the server was unreachable.
 */
function respond(body: string, init: { status: number; contentType?: string }): Response {
  return new Response(body, {
    status: init.status,
    headers: init.contentType ? { "content-type": init.contentType } : {},
  });
}

const NGINX_502 =
  "<html>\r\n<head><title>502 Bad Gateway</title></head>\r\n<body>\r\n" +
  "<center><h1>502 Bad Gateway</h1></center>\r\n<hr><center>nginx/1.27.5</center>\r\n" +
  "</body>\r\n</html>\r\n<!-- a padding to disable MSIE and Chrome friendly error page -->\r\n";

describe("apiRequest error messages", () => {
  beforeEach(() => {
    clearSession();
    vi.stubGlobal("fetch", vi.fn());
  });
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("never shows a proxy HTML error page to the user", async () => {
    vi.mocked(fetch).mockResolvedValue(
      respond(NGINX_502, { status: 502, contentType: "text/html" }),
    );
    const error = (await apiRequest("/auth/login", { method: "POST" }).catch(
      (e: unknown) => e,
    )) as ApiError;
    expect(error).toBeInstanceOf(ApiError);
    expect(error.message).not.toContain("<html>");
    expect(error.message).not.toContain("nginx");
    expect(error.message).not.toContain("MSIE");
    expect(error.message).toContain("server is unavailable");
    // The raw body is still retained for diagnostics, just not displayed.
    expect(String(error.details)).toContain("502 Bad Gateway");
  });

  it("uses the API's own message when the body is JSON", async () => {
    vi.mocked(fetch).mockResolvedValue(
      respond(
        JSON.stringify({
          success: false,
          errors: [{ code: "unauthenticated", message: "Invalid email or password" }],
          detail: "Invalid email or password",
        }),
        { status: 401, contentType: "application/json" },
      ),
    );
    const error = (await apiRequest("/auth/login", { method: "POST" }).catch(
      (e: unknown) => e,
    )) as ApiError;
    expect(error.message).toBe("Invalid email or password");
  });

  it("reads a structured detail object", async () => {
    vi.mocked(fetch).mockResolvedValue(
      respond(JSON.stringify({ detail: { message: "Password must include a symbol" } }), {
        status: 400,
        contentType: "application/json",
      }),
    );
    const error = (await apiRequest("/auth/register", { method: "POST" }).catch(
      (e: unknown) => e,
    )) as ApiError;
    expect(error.message).toBe("Password must include a symbol");
  });

  it("falls back to a status message when JSON carries no message", async () => {
    vi.mocked(fetch).mockResolvedValue(
      respond(JSON.stringify({ success: false }), { status: 500, contentType: "application/json" }),
    );
    const error = (await apiRequest("/documents").catch((e: unknown) => e)) as ApiError;
    expect(error.message).toContain("unexpected error");
  });

  it("keeps a short plain-text body, which is genuinely informative", async () => {
    vi.mocked(fetch).mockResolvedValue(
      respond("Upload exceeds the configured limit", { status: 413, contentType: "text/plain" }),
    );
    const error = (await apiRequest("/documents/upload", { method: "POST" }).catch(
      (e: unknown) => e,
    )) as ApiError;
    expect(error.message).toBe("Upload exceeds the configured limit");
  });

  it("does not mistake an empty body for a message", async () => {
    vi.mocked(fetch).mockResolvedValue(respond("", { status: 504, contentType: "text/html" }));
    const error = (await apiRequest("/documents").catch((e: unknown) => e)) as ApiError;
    expect(error.message).toContain("server is unavailable");
  });

  it("surfaces an error from a 200 envelope that reports failure", async () => {
    vi.mocked(fetch).mockResolvedValue(
      respond(
        JSON.stringify({ success: false, data: null, errors: [{ message: "Tenant mismatch" }] }),
        { status: 200, contentType: "application/json" },
      ),
    );
    const error = (await apiRequest("/documents").catch((e: unknown) => e)) as ApiError;
    expect(error.message).toBe("Tenant mismatch");
  });
});
