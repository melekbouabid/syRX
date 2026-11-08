// Medicine boxes in the Scanner (step 2.7): screenshots for docs/screenshots/ (22 to 25).
//
//   node scripts/box-screenshots.mjs           # app on http://localhost:5173 (npm run dev)
//   BOXES=../data/raw/medicine_boxes/italian_otc/out OFF_TOPIC=../data/raw/off_topic \
//     node scripts/box-screenshots.mjs
//
// Uploads real box photos of Italian-OTC-medicines (DVC data, Apache-2.0): a brand of the brand
// table (Efferalgan), a box whose brand is not in the table but prints its INN (Brufen:
// "Ibuprofene"), then an off-topic photo (scikit-image's coffee, CC0) that must be refused;
// finally the two boxes added to My pills and checked for interactions. Fails if a step does not show what it should.
import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://localhost:5173";
const OUT = resolve(process.env.OUT ?? "../docs/screenshots");
const BOXES = resolve(process.env.BOXES ?? "../data/raw/medicine_boxes/italian_otc/out");
const OFF_TOPIC = resolve(process.env.OFF_TOPIC ?? "../data/raw/off_topic");
const EMAIL = process.env.DEMO_EMAIL ?? "user@gmail.com";
const PASSWORD = process.env.DEMO_PASSWORD ?? "user";

async function shot(page, name, target) {
  await page.waitForTimeout(1000); // entrance animations and bars
  await (target ?? page).screenshot({ path: join(OUT, name), type: "jpeg", quality: 80 });
  console.log(`  ${name}`);
}

async function upload(page, file) {
  const name = file.split(/[\\/]/).pop();
  await page.locator('[data-testid="scan-file"]').setInputFiles(file);
  await page.locator(`img[alt="Photo sent for identification: ${name}"]`).waitFor({ timeout: 60_000 });
  await page.locator('[data-testid="scan-busy"]').waitFor({ state: "detached", timeout: 120_000 });
  await page.locator('[data-testid="scan-kind"]').waitFor({ timeout: 60_000 });
}

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch();
const page = await (await browser.newContext({ viewport: { width: 1366, height: 900 } })).newPage();
page.on("pageerror", (error) => console.error("page error:", error.message));

await page.goto(`${BASE}/login`);
await page.getByLabel("Email").fill(EMAIL);
await page.getByLabel("Password", { exact: true }).fill(PASSWORD);
await page.getByRole("button", { name: /sign in/i }).click();
await page.waitForURL(/\/(home|admin)/);
await page.addStyleTag({ content: ".topbar { position: static !important; }" });
await page.getByRole("link", { name: "Scanner" }).click();
await page.getByRole("heading", { name: "Scanner" }).waitFor();

// 22. A brand of the brand table: Efferalgan -> acetaminophen, with its BDPM source.
await upload(page, join(BOXES, "efferalgan16cpr-500mg.jpg"));
const card = page.locator('[data-testid="box-card"]');
await card.getByText("Efferalgan").first().waitFor();
await card.getByText("acetaminophen").first().waitFor();
await page.locator('[data-testid="scan-result"]').scrollIntoViewIfNeeded();
await shot(page, "22-box-brand.jpg", page.locator('[data-testid="scan-result"]'));
await card.getByRole("button", { name: "Add to my pills" }).click();

// 23. A brand absent from the table, found by the INN printed on it: Brufen, "Ibuprofene".
await upload(page, join(BOXES, "brufen-analges12cpr-riv-200mg.jpg"));
await card.getByText("ibuprofen").first().waitFor();
await card.getByText("name printed on the box", { exact: false }).first().waitFor();
await shot(page, "23-box-printed-inn.jpg", page.locator('[data-testid="scan-result"]'));
await card.getByRole("button", { name: "Add to my pills" }).click();

// 24. Off topic: neither a pill nor a box, nothing identified.
await upload(page, join(OFF_TOPIC, "coffee.png"));
await page.getByText("Neither a pill nor a medicine box").waitFor();
if ((await page.locator(".pill-card").count()) !== 0) throw new Error("a pill was guessed");
await shot(page, "24-off-topic-refused.jpg", page.locator('[data-testid="scan-result"]'));

// 25. Two boxes in My pills, checked by their active ingredients.
await page.getByRole("button", { name: "Check interactions" }).click();
await page.locator(".report").waitFor({ timeout: 30_000 });
await page.getByText("consultez un pharmacien ou un médecin").last().waitFor();
await shot(page, "25-box-interactions.jpg", page.getByRole("region", { name: "My pills" }));

await browser.close();
