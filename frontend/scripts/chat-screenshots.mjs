// Ask the Tujya agent real questions from the Scanner page and save screenshots
// (docs/screenshots/08-*.jpg ...). Needs the whole stack, Ollama included:
//
//   BASE=http://localhost:8080 node scripts/chat-screenshots.mjs
//
// Signs in, confirms two demo pills (fluconazole and citalopram), asks a French question about
// them in the chat panel, waits for the streamed answer and its numbered sources, opens a
// citation and the tool calls; then asks a dose question, which must be refused. Fails if a
// step does not show what it should.
import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://localhost:8080";
const OUT = resolve(process.env.OUT ?? "../docs/screenshots");
const EMAIL = process.env.DEMO_EMAIL ?? "user@gmail.com";
const PASSWORD = process.env.DEMO_PASSWORD ?? "user";
const DEMOS = [2, 1]; // config/demo_gallery.csv: fluconazole, citalopram
const QUESTION = "Le fluconazole et le citalopram interagissent-ils ? Que disent les notices ?";
const DOSE_QUESTION = "Quelle dose de citalopram dois-je prendre ?";

async function shot(page, name, target) {
  await page.waitForTimeout(700);
  await (target ?? page).screenshot({ path: join(OUT, name), type: "jpeg", quality: 82 });
}

async function waitForResult(page) {
  await page.locator(".scan-busy").waitFor({ state: "detached", timeout: 60_000 });
  await page.locator(".pill-card").first().waitFor({ timeout: 60_000 });
}

async function ask(page, question) {
  const before = await page.locator(".chat-turn .chat-meta").count();
  await page.getByLabel("Question").fill(question);
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  const started = Date.now();
  // The newest turn is first; its meta line appears with the final, checked answer.
  await page.waitForFunction(
    (count) => document.querySelectorAll(".chat-turn .chat-meta").length > count,
    before,
    { timeout: 240_000 },
  );
  return (Date.now() - started) / 1000;
}

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
const page = await context.newPage();
page.on("pageerror", (error) => console.error("page error:", error.message));

await page.goto(`${BASE}/login`);
await page.getByLabel("Email").fill(EMAIL);
await page.getByLabel("Password", { exact: true }).fill(PASSWORD);
await page.getByRole("button", { name: /sign in/i }).click();
await page.waitForURL(/\/(home|admin)/);
await page.addStyleTag({ content: ".topbar { position: static !important; }" });

await page.getByRole("link", { name: "Scanner" }).click();
await page.getByRole("heading", { name: "Scanner" }).waitFor();
await page.getByRole("tab", { name: /demo photos/i }).click();
for (const id of DEMOS) {
  await page.locator(`[data-testid="demo-${id}"]`).click();
  await waitForResult(page);
  await page.getByRole("button", { name: "Confirm this pill" }).first().click();
}

// The chat panel before any question: disclaimer, suggestions about the confirmed pills.
const panel = page.locator(".chat-panel");
await panel.scrollIntoViewIfNeeded();
await shot(page, "08-chat-empty.jpg", panel);

// A real question, streamed, with numbered citations and the tool calls.
const seconds = await ask(page, QUESTION);
const turn = page.locator(".chat-turn").first();
const answer = await turn.locator(".chat-answer").innerText();
if (answer.trim().length < 40) throw new Error("no answer");
if ((await turn.locator("a.cite").count()) === 0) throw new Error("no numbered citation");
if (/\bsûr|sans danger/i.test(answer)) throw new Error("the answer calls something safe");
await turn.locator(".chat-tools summary").click();
await shot(page, "09-chat-answer.jpg", panel);
console.log(`answer in ${seconds.toFixed(1)} s:\n${answer}`);

// Clicking [1] jumps to its source in the list.
await turn.locator("a.cite").first().click();
await page.waitForTimeout(500);
await shot(page, "10-chat-citation.jpg", turn.locator(".chat-sources"));

// A dose question is refused before any model call.
await ask(page, DOSE_QUESTION);
const refused = await page.locator(".chat-turn").first().innerText();
if (!/jamais de dose/i.test(refused)) throw new Error("the dose question was not refused");
await shot(page, "11-chat-dose-refused.jpg", page.locator(".chat-turn").first());

await browser.close();
console.log(`screenshots written to ${OUT}`);
