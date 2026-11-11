// Drive the Documents page and the Qdrant dashboard, save screenshots (docs/screenshots/27-29).
//
//   node scripts/documents-screenshots.mjs                  # app on http://localhost:5173
//   BASE=http://localhost:8080 node scripts/documents-screenshots.mjs   # the docker compose app
//   QDRANT=http://localhost:6333 node scripts/documents-screenshots.mjs
//
// Signs in, opens Documents and checks the index card (Qdrant, points), the three document
// types, the filters by type and by active ingredient, and one document with its metadata and
// chunks (SHA-256 shown). Then the Qdrant dashboard: the collection tujya_chunks and its
// points with their payload. Fails if a step does not show what it should.
import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://localhost:5173";
const QDRANT = process.env.QDRANT ?? "http://localhost:6333";
const OUT = resolve(process.env.OUT ?? "../docs/screenshots");
const EMAIL = process.env.DEMO_EMAIL ?? "user@gmail.com";
const PASSWORD = process.env.DEMO_PASSWORD ?? "user";
const COLLECTION = "tujya_chunks";

function check(condition, message) {
  if (!condition) throw new Error(message);
}

async function shot(page, name, options = {}) {
  await page.waitForTimeout(900); // entrance animations
  await page.screenshot({ path: join(OUT, name), type: "jpeg", quality: 80, ...options });
  console.log(`  ${name}`);
}

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1366, height: 900 } });
page.on("pageerror", (error) => console.error("page error:", error.message));

await page.goto(`${BASE}/login`);
await page.getByLabel("Email").fill(EMAIL);
await page.getByLabel("Password", { exact: true }).fill(PASSWORD);
await page.getByRole("button", { name: /sign in/i }).click();
await page.waitForURL(/\/(home|admin)/);
await page.addStyleTag({ content: ".topbar { position: static !important; }" });

await page.getByRole("link", { name: "Documents" }).click();
await page.getByRole("heading", { name: "Documents" }).waitFor();
const index = page.getByRole("region", { name: "Vector index" });
await index.waitFor({ timeout: 30_000 });
const indexText = await index.innerText();
check(indexText.includes("Qdrant"), "the index card does not name Qdrant");
check(indexText.includes(COLLECTION), "the index card does not name the collection");
const rows = page.locator("table.table tbody tr");
await rows.first().waitFor();
const types = await page.getByLabel("Filter by document type").locator("option").allInnerTexts();
for (const type of ["Drug label", "PubMed abstract", "DDInter pair"]) {
  check(types.some((t) => t.startsWith(type)), `no ${type} in the type filter`);
}
await shot(page, "27-documents.jpg");

// Filters: labels about or mentioning warfarin (the API's rule, docs/api.md).
const unfiltered = (await (await page.request.get(`${BASE}/api/documents?limit=1`)).json()).total;
const filtered = page.waitForResponse(
  (r) => r.url().includes("/api/documents?") && r.url().includes("ingredient=warfarin") &&
    r.url().includes("doc_type=label"),
);
await page.getByLabel("Active ingredient").fill("warfarin");
const typeValue = await page
  .getByLabel("Filter by document type")
  .locator("option", { hasText: "Drug label" })
  .getAttribute("value");
await page.getByLabel("Filter by document type").selectOption(typeValue);
const body = await (await filtered).json();
check(body.total > 0 && body.total < unfiltered, "the filters do not narrow the list");
for (const doc of body.items) {
  check(doc.doc_type === "label", `${doc.doc_id} is not a label`);
  const names = [...doc.ingredients, ...doc.mentioned_ingredients];
  check(names.includes("warfarin"), `${doc.doc_id} neither about nor mentioning warfarin`);
}
await page.waitForFunction(
  (n) => document.querySelectorAll("table.table tbody tr").length === n,
  Math.min(body.total, 25),
);
console.log(`  filter type=label + ingredient=warfarin: ${body.total} of ${unfiltered} documents`);

// One document: metadata and chunks.
await page.locator("table.table tbody tr").first().getByRole("button").click();
const dialog = page.getByRole("dialog");
await dialog.getByText(/chunks? in the index/).waitFor({ timeout: 30_000 });
const dialogText = await dialog.innerText();
for (const field of ["Identifier", "Source", "Licence", "Language", "SHA-256"]) {
  check(dialogText.includes(field), `the document dialog has no ${field}`);
}
check(dialogText.includes("consult a pharmacist or a"), "the disclaimer is missing");
await shot(page, "28-document-chunks.jpg");
await page.keyboard.press("Escape");

// The Qdrant dashboard: collection and points with their payload.
const dashboard = await browser.newPage({ viewport: { width: 1366, height: 900 } });
await dashboard.goto(`${QDRANT}/dashboard#/collections/${COLLECTION}`);
await dashboard.getByText("doc_type").first().waitFor({ timeout: 30_000 });
const dashText = await dashboard.locator("body").innerText();
check(dashText.includes(COLLECTION), "the dashboard does not show the collection");
await shot(dashboard, "29-qdrant-dashboard.jpg");
await browser.close();
