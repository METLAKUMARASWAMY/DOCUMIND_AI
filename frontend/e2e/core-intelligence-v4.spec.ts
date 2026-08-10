import { expect, request as playwrightRequest, test, type APIRequestContext, type Page } from "@playwright/test";
import { readFileSync, readdirSync } from "node:fs";
import { resolve } from "node:path";

const apiBase = process.env.DOCUMIND_E2E_API_URL || "http://127.0.0.1:8000/api/v1";
const screenshotDir = resolve(process.cwd(), "../docs/screenshots");
const email = process.env.DOCUMIND_E2E_EMAIL || "";
const password = process.env.DOCUMIND_E2E_PASSWORD || "";

let documentId = "";
let headers: Record<string, string> = {};

test.beforeAll(async () => {
  if (!email || !password) {
    throw new Error("DOCUMIND_E2E_EMAIL and DOCUMIND_E2E_PASSWORD are required.");
  }
  const api = await playwrightRequest.newContext();
  const login = await api.post(`${apiBase}/auth/login`, { data: { email, password } });
  expect(login.ok(), await login.text()).toBeTruthy();
  headers = { Authorization: `Bearer ${(await login.json()).data.access_token}` };

  const uploadDir = resolve(process.cwd(), "../backend/var/uploads");
  const fixtureName = readdirSync(uploadDir)
    .filter((name) => name.endsWith("-00093726.png"))
    .sort()
    .at(-1);
  expect(fixtureName, "The required 00093726.png fixture is missing").toBeTruthy();
  const upload = await api.post(`${apiBase}/vision/documents/upload`, {
    headers: { ...headers, "Idempotency-Key": "core-intelligence-v4-00093726" },
    multipart: {
      files: {
        name: "00093726.png",
        mimeType: "image/png",
        buffer: readFileSync(resolve(uploadDir, fixtureName!)),
      },
      metadata: JSON.stringify({ acceptance_fixture: "00093726", requested_pipeline: "v4" }),
    },
  });
  expect(upload.ok(), await upload.text()).toBeTruthy();
  const uploaded = await upload.json();
  documentId = uploaded.data.document_ids[0];
  await waitForProcessedDocument(api, documentId);

  const reportRequest = {
    report_type: "document_intelligence_report",
    document_id: documentId,
  };
  const reportOne = await api.post(`${apiBase}/vision/reports/generate`, {
    headers,
    data: reportRequest,
  });
  const reportTwo = await api.post(`${apiBase}/vision/reports/generate`, {
    headers,
    data: reportRequest,
  });
  expect(reportOne.ok(), await reportOne.text()).toBeTruthy();
  expect(reportTwo.ok(), await reportTwo.text()).toBeTruthy();
  expect((await reportOne.json()).data.report_id).toBe((await reportTwo.json()).data.report_id);
  await api.dispose();
});

test("00093726 core-intelligence walkthrough is grounded and readable", async ({ page }) => {
  const browserErrors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });
  page.on("pageerror", (error) => browserErrors.push(error.message));

  await signIn(page);
  await page.goto(`/documents/${documentId}`);
  await expect(page.getByRole("heading", { name: "00093726.png" })).toBeVisible();
  await expect(page.getByLabel("Field completeness")).toBeVisible();
  await expectCount(page, "Required expected", "6");
  await expectCount(page, "Required matched", "6");
  await expectCount(page, "Required missing", "0");
  await expectCount(page, "Optional extracted", "0");
  await expectCount(page, "Total extracted", "6");
  await expectCount(page, "Duplicates rejected", "0");
  await expectCount(page, "Disputed", "0");
  for (const value of [
    "September 21, 1976",
    "6030",
    "85 mm Filter",
    "50 lbs.",
    "Dr. A. W. Spears",
    "W. E. Routh",
  ]) {
    await expect(page.getByText(value, { exact: true }).first()).toBeVisible();
  }
  await capture(page, "core-v4-00093726-fields.png");

  await page.goto("/ask");
  await selectDocument(page, "Scope", "00093726.png");
  await ask(page, "What is the sample number?");
  await expect(page.getByText("The sample number shown is 6030.", { exact: true })).toBeVisible();
  await expect(page.locator(".citation-row").last()).toContainText("Sample No. 6030");

  await ask(page, "Is this safe?");
  const options = page.locator(".clarification-options").last();
  await expect(options).toBeVisible();
  await expect(options.getByText("Check document quality", { exact: true })).toBeVisible();
  await options.getByText("Check document quality", { exact: true }).click();
  await expect(page.getByText("partially supported", { exact: true }).last()).toBeVisible();
  await expect(page.getByText(/Human review is required before relying on the document\./)).toBeVisible();
  await expect(page.locator(".citation-row").last()).toBeVisible();
  await capture(page, "core-v4-ask-clarification.png");

  await page.goto("/graph");
  await selectDocument(page, "Document graph", "00093726.png");
  await expect(page.locator(".graph-canvas canvas").first()).toBeVisible();
  await page.getByPlaceholder("Ask a relationship question").fill("Who made the original request?");
  await page.getByRole("button", { name: "Query graph" }).click();
  await expect(page.getByText(/Dr\. A\. W\. Spears is recorded as the person who requested/)).toBeVisible();
  await expect(page.getByText(/00093726\.png → REQUESTED BY → Dr\. A\. W\. Spears/)).toBeVisible();
  await expect(page.getByRole("button", { name: "Open citation 1" })).toBeVisible();
  await capture(page, "core-v4-graphrag-path.png");

  await page.goto("/analytics");
  await expect(page.getByRole("heading", { name: "Analytics" })).toBeVisible();
  await expect(page.getByText(/Median processing/).first()).toBeVisible();
  await capture(page, "core-v4-analytics.png");

  await page.goto("/admin");
  await expect(page.getByText("deterministic-schema-extractor-v3", { exact: true })).toBeVisible();
  await capture(page, "core-v4-admin-models.png");
  expect(browserErrors, browserErrors.join("\n")).toEqual([]);
});

async function waitForProcessedDocument(api: APIRequestContext, id: string) {
  await expect
    .poll(
      async () => {
        const response = await api.get(`${apiBase}/vision/documents/${id}`, { headers });
        if (!response.ok()) return "http_error";
        const document = (await response.json()).data;
        if (["FAILED", "REJECTED", "CONFIRMED_ERROR"].includes(document.status)) {
          throw new Error(`Fixture processing failed with ${document.status}`);
        }
        return document.extracted_fields?.length === 6 ? document.status : "processing";
      },
      { timeout: 60_000, intervals: [500, 1000, 2000] },
    )
    .toMatch(/^(COMPLETED|REVIEW_REQUIRED|APPROVED)$/);
}

async function signIn(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Work email").fill(email);
  await page.getByLabel("Password").fill(password);
  await page.getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByRole("heading", { name: "Document operations" })).toBeVisible();
}

async function selectDocument(page: Page, label: string, filename: string) {
  if (label === "Scope") {
    await page.getByLabel("Scope", { exact: true }).click();
    await page.getByRole("option", { name: "Current document", exact: true }).click();
    await page.getByLabel("Document", { exact: true }).click();
  } else {
    await page.getByLabel(label, { exact: true }).click();
  }
  await page.getByRole("option").filter({ hasText: filename }).first().click();
}

async function ask(page: Page, question: string) {
  await page.getByPlaceholder("Ask a question about your documents").fill(question);
  await page.getByRole("button", { name: "Ask", exact: true }).click();
}

async function expectCount(page: Page, label: string, value: string) {
  const item = page.locator(".field-completeness-item").filter({ hasText: label });
  await expect(item).toContainText(value);
}

async function capture(page: Page, filename: string) {
  await page.mouse.move(1, 1);
  await page.evaluate(() => document.fonts.ready);
  await expect
    .poll(() => page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 2))
    .toBeTruthy();
  await page.screenshot({ path: resolve(screenshotDir, filename), fullPage: true });
}
