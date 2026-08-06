import { expect, request as playwrightRequest, test, type Page } from "@playwright/test";
import { resolve } from "node:path";

const apiBase = process.env.DOCUMIND_E2E_API_URL || "http://127.0.0.1:8000/api/v1";
const screenshotDir = resolve(process.cwd(), "../docs/screenshots");
const email = process.env.DOCUMIND_E2E_EMAIL || "";
const password = process.env.DOCUMIND_E2E_PASSWORD || "";

let primaryDocumentId = "";
let reviewId = "";
let reviewFilename = "";

test.beforeAll(async () => {
  if (!email || !password) {
    throw new Error(
      "DOCUMIND_E2E_EMAIL and DOCUMIND_E2E_PASSWORD must contain a securely bootstrapped admin credential.",
    );
  }
  const api = await playwrightRequest.newContext();
  const login = await api.post(`${apiBase}/auth/login`, {
    data: { email, password },
  });
  expect(login.ok(), await login.text()).toBeTruthy();
  const loginPayload = await login.json();
  const headers = {
    Authorization: `Bearer ${loginPayload.data.access_token}`,
  };

  const seeded = await api.post(`${apiBase}/demo/vision/seed`, { headers });
  expect(seeded.ok(), await seeded.text()).toBeTruthy();
  primaryDocumentId = (await seeded.json()).data.primary_document_id;

  const suffix = Date.now().toString(36);
  reviewFilename = `acceptance-review-${suffix}.txt`;
  const analyzed = await api.post(`${apiBase}/vision/documents/analyze`, {
    headers,
    data: {
      filename: reviewFilename,
      text: [
        `Invoice INV-E2E-${suffix}`,
        "Vendor: Acceptance Review Services",
        "Invoice Date: 2026-08-04",
        "Due Date: 2026-08-20",
        "Bank Account: ACCT-99887766",
        "Subtotal: INR 95000.00",
        "Tax: INR 17100.00",
        "Total Amount: INR 112100.00",
        "Currency: INR",
      ].join("\n"),
      metadata: {
        document_type_hint: "invoice",
        approved_bank_identifier: "ACCT-11223344",
        required_signature: true,
      },
    },
  });
  expect(analyzed.ok(), await analyzed.text()).toBeTruthy();
  const analyzedPayload = await analyzed.json();
  reviewId = analyzedPayload.data.review_task.review_id;
  await api.dispose();
});

test("repaired workspaces are connected, readable, and responsive", async ({ page }) => {
  const browserErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });
  page.on("pageerror", (error) => browserErrors.push(error.message));

  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto("/login");
  await page.getByLabel("Work email").fill(email);
  await page.getByLabel("Password").fill(password);
  await page.getByRole("button", { name: "Sign in" }).click();

  await expect(page.getByRole("heading", { name: "Document operations" })).toBeVisible();
  await capture(page, "dashboard-desktop.png");

  await page.goto("/documents");
  await expect(page.getByRole("heading", { name: "Document library" })).toBeVisible();
  await expect(page.getByText("documind-demo-invoice.txt").first()).toBeVisible();
  await capture(page, "document-library.png");

  await page.goto(`/documents/${primaryDocumentId}`);
  await expect(page.getByRole("heading", { name: "documind-demo-invoice.txt" })).toBeVisible();
  await expect(page.locator(".page-stage img")).toHaveJSProperty("complete", true);
  await expect.poll(async () => page.locator(".page-stage img").evaluate((image: HTMLImageElement) => image.naturalWidth)).toBeGreaterThan(0);
  await expect(page.locator(".evidence-overlay")).not.toHaveCount(0);

  await page.getByRole("checkbox", { name: "Extracted fields" }).uncheck();
  await expect(page.locator(".evidence-overlay")).toHaveCount(0);
  await page.getByRole("checkbox", { name: "OCR" }).check();
  await expect(page.locator(".overlay-ocr")).not.toHaveCount(0);
  await page.getByRole("checkbox", { name: "OCR" }).uncheck();
  await expect(page.locator(".overlay-ocr")).toHaveCount(0);
  await capture(page, "document-viewer-readable.png");

  await page.getByRole("checkbox", { name: "Extracted fields" }).check();
  await page.getByRole("checkbox", { name: "Risk indicators" }).check();
  await capture(page, "document-viewer-overlays.png");
  await page.getByRole("button", { name: "Clear all overlays" }).click();
  await expect(page.locator(".evidence-overlay")).toHaveCount(0);

  await page.goto("/reviews");
  await expect(page.getByRole("heading", { name: "Human review queue" })).toBeVisible();
  await expect(page.getByText(reviewFilename).first()).toBeVisible();
  await capture(page, "review-queue.png");

  await page.goto(`/reviews/${reviewId}`);
  await expect(page.getByRole("heading", { name: reviewFilename })).toBeVisible();
  await expect(page.getByText(/unresolved required field/).first()).toBeVisible();
  await capture(page, "review-workspace.png");

  await page.goto("/ask");
  await expect(page.getByRole("heading", { name: "Ask documents" })).toBeVisible();
  await page.getByLabel("Scope").click();
  await page.getByRole("option", { name: "documind-demo-invoice.txt" }).click();
  await page.getByPlaceholder("Ask a question about your documents").fill("What is the total amount?");
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  await expect(page.locator(".citation-row").first()).toBeVisible();
  await expect(page.getByText("supported", { exact: true }).first()).toBeVisible();
  await capture(page, "ask-documents-citations.png");

  await page.locator(".citation-row").first().click();
  await expect(page.getByText(/Opened from answer citation/)).toBeVisible();
  await capture(page, "citation-source-navigation.png");

  await page.goto("/graph");
  await expect(page.getByRole("heading", { name: "Knowledge graph" })).toBeVisible();
  await page.getByLabel("Document graph").click();
  await page.getByRole("option", { name: "documind-demo-invoice.txt" }).click();
  await expect(page.locator(".graph-canvas canvas").first()).toBeVisible();
  await capture(page, "knowledge-graph.png");

  await page.goto("/analytics");
  await expect(page.getByRole("heading", { name: "Analytics" })).toBeVisible();
  await expect(page.getByText(/Citation coverage/).first()).toBeVisible();
  await capture(page, "analytics.png");

  await page.goto("/model-lab");
  await expect(page.getByRole("heading", { name: "Model comparison laboratory" })).toBeVisible();
  await page.getByLabel("Document").click();
  await page.getByRole("option", { name: "documind-demo-invoice.txt" }).click();
  await page.getByRole("button", { name: "Compare outputs" }).click();
  await expect(page.getByRole("columnheader", { name: "Primary result" })).toBeVisible();
  await expect(page.getByText("Observed primary")).toBeVisible();
  await capture(page, "model-laboratory.png");

  await page.goto("/admin");
  await expect(page.getByRole("heading", { name: "Admin settings" })).toBeVisible();
  await expect(page.getByText(/Requested state is deployment intent/)).toBeVisible();
  await expect(page.getByText("deterministic-schema-extractor-v3")).toBeVisible();
  await capture(page, "admin-model-runtime.png");

  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/dashboard");
  await expect(page.getByRole("heading", { name: "Document operations" })).toBeVisible();
  await capture(page, "dashboard-mobile.png");
  await page.goto(`/documents/${primaryDocumentId}`);
  await expect(page.getByRole("heading", { name: "documind-demo-invoice.txt" })).toBeVisible();
  await capture(page, "document-viewer-mobile.png");

  expect(browserErrors, browserErrors.join("\n")).toEqual([]);
});

async function capture(page: Page, filename: string) {
  await page.mouse.move(1, 1);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(250);
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= window.innerWidth + 2,
      ),
    )
    .toBeTruthy();
  await page.screenshot({
    path: resolve(screenshotDir, filename),
    fullPage: true,
  });
}
