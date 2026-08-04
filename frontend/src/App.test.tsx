import { CssBaseline, ThemeProvider } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import App from "./App";
import { AuthProvider } from "./auth/AuthContext";
import { BrowserRouter } from "./router";
import { theme } from "./theme/theme";

function renderApp() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return render(
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <AuthProvider>
            <App />
          </AuthProvider>
        </BrowserRouter>
      </QueryClientProvider>
    </ThemeProvider>,
  );
}

describe("DocuMind app shell", () => {
  beforeEach(() => {
    sessionStorage.clear();
    window.history.pushState(null, "", "/dashboard");
  });

  it("routes unauthenticated reviewers to sign in", async () => {
    renderApp();

    expect(await screen.findByRole("heading", { name: "Sign in" })).toBeInTheDocument();
  });
});
