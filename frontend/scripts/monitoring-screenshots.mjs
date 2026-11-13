// Drive the Monitoring page and the Scanner drift banner, save screenshots (docs/screenshots/30-31).
//
//   BASE=http://localhost:8080 node scripts/monitoring-screenshots.mjs   # the docker compose app
//
// Run it after a drift scenario was sent to the API (docs/module1/derive.md):
//   python -m src.data.drift_demo --api http://127.0.0.1:8000 --only dark
// Signs in, opens Monitoring and checks the drift status and the table of the eight
// statistics, then opens the Scanner and checks the drift banner. Fails if a step does not
// show what it should.
import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://localhost:5173";
const OUT = resolve(process.env.OUT ?? "../docs/screenshots");
const EMAIL = process.env.DEMO_EMAIL ?? "user@gmail.com";
const PASSWORD = process.env.DEMO_PASSWORD ?? "user";

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

const report = await (await page.request.get(`${BASE}/monitoring/drift`)).json();
check(report.status === "drift", `the API window is not in drift (${report.status}): send a scenario first`);

await page.goto(`${BASE}/login`);
await page.getByLabel("Email").fill(EMAIL);
await page.getByLabel("Password", { exact: true }).fill(PASSWORD);
await page.getByRole("button", { name: /sign in/i }).click();
await page.waitForURL(/\/(home|admin)/);
await page.addStyleTag({ content: ".topbar { position: static !important; }" });

await page.getByRole("link", { name: "Monitoring" }).click();
await page.getByRole("heading", { name: "Monitoring" }).waitFor();
const rows = page.locator("table.table tbody tr");
await rows.first().waitFor({ timeout: 30_000 });
check((await rows.count()) === 8, "the table does not show the eight statistics");
const summary = await page.getByRole("region", { name: "Drift summary" }).innerText();
check(summary.includes("Drift"), "the summary does not say Drift");
for (const feature of report.drifted) {
  check((await page.locator("table.table").innerText()).includes(feature), `${feature} missing`);
}
await shot(page, "30-monitoring-drift.jpg", { fullPage: true });

await page.getByRole("link", { name: "Scanner" }).click();
const banner = page.getByRole("alert").filter({ hasText: "Data drift" });
await banner.waitFor({ timeout: 30_000 });
await shot(page, "31-scan-drift-banner.jpg");

await browser.close();
