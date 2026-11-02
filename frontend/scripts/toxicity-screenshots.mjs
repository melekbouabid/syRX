// Drive the Toxicity page (bonus) and save a screenshot (docs/screenshots/16-toxicity.jpg).
//
//   BASE=http://localhost:8080 node scripts/toxicity-screenshots.mjs
//
// Signs in, predicts warfarin and fluconazole, and checks that each molecule has its own
// card with the 12 Tox21 assays, that the in vitro wording and the disclaimer are shown,
// and that nothing calls a molecule safe or non-toxic. Then a SMILES far from the training
// molecules must get the "far from the training molecules" badge.
import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://localhost:8080";
const OUT = resolve(process.env.OUT ?? "../docs/screenshots");
const EMAIL = process.env.DEMO_EMAIL ?? "user@gmail.com";
const PASSWORD = process.env.DEMO_PASSWORD ?? "user";
const WORDING = "pas une évaluation clinique";

function check(condition, message) {
  if (!condition) throw new Error(message);
}

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1366, height: 1000 } });
page.on("pageerror", (error) => console.error("page error:", error.message));

await page.goto(`${BASE}/login`);
await page.getByLabel("Email").fill(EMAIL);
await page.getByLabel("Password", { exact: true }).fill(PASSWORD);
await page.getByRole("button", { name: /sign in/i }).click();
await page.waitForURL(/\/(home|admin)/);
await page.addStyleTag({ content: ".topbar { position: static !important; }" });

await page.getByRole("link", { name: "Toxicity" }).click();
await page.getByRole("heading", { name: /Toxicity/ }).waitFor();
await page.getByLabel("Molecule 1").fill("warfarin");
await page.getByLabel("Molecule 2 (optional)").fill("fluconazole");
const started = Date.now();
await page.getByRole("button", { name: "Predict" }).click();
await page.locator(".tox-card").nth(1).waitFor({ timeout: 30_000 });
const seconds = (Date.now() - started) / 1000;
await page.waitForTimeout(900); // entrance animations

const cards = page.locator(".tox-card");
check((await cards.count()) === 2, "expected one card per molecule");
for (const card of await cards.all()) {
  check((await card.locator("tbody tr").count()) === 12, "expected the 12 Tox21 assays");
}
const text = (await page.locator("main").innerText()).toLowerCase();
check(text.includes(WORDING), "the in vitro wording is missing");
check(text.includes("consultez un pharmacien ou un médecin"), "the disclaimer is missing");
check(text.includes("ne dit rien de leur association"), "the mixture note is missing");
for (const word of ["safe", " sûr", "non toxique", "not toxic", "harmless"]) {
  check(!text.includes(word), `forbidden wording: ${word}`);
}
await page.screenshot({ path: join(OUT, "16-toxicity.jpg"), type: "jpeg", quality: 82, fullPage: true });
console.log(`two molecules predicted in ${seconds.toFixed(1)} s; 16-toxicity.jpg`);

await page.getByLabel("Molecule 1").fill("citalopram");
await page.getByLabel("Molecule 2 (optional)").fill("");
await page.getByRole("button", { name: "Predict" }).click();
await page.getByText("far from the training molecules").waitFor({ timeout: 30_000 });
check((await cards.count()) === 1, "expected a single card");
console.log("citalopram: flagged as far from the training molecules");
await browser.close();
