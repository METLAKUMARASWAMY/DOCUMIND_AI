import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  outputDir: "../var/playwright-results",
  use: {
    baseURL: process.env.DOCUMIND_E2E_BASE_URL || "http://127.0.0.1:5174",
    // Playwright API traces retain request bodies and authorization headers.
    // These E2E journeys perform real login, so keeping traces would persist
    // bootstrap credentials and bearer tokens in local artifacts.
    trace: "off",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
});
