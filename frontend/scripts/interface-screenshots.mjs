// Before / after screenshots of the interface step (docs/screenshots/interface/): the start
// page, the sign-in page, the scanner with a demo result and the chat panel, in light and
// dark mode.
//
//   BASE=http://localhost:8080 TAG=avant node scripts/interface-screenshots.mjs
//   BASE=http://localhost:8080 TAG=apres LANG_UI=fr node scripts/interface-screenshots.mjs
//
// The theme follows the browser's colour scheme until someone presses the toggle, so the
// two modes are taken with Playwright's `colorScheme`. LANG_UI sets the browser language
// (the interface follows it until a language is picked); the selectors below work in both.
import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://localhost:5173";
const OUT = resolve(process.env.OUT ?? "../docs/screenshots/interface");
const TAG = process.env.TAG ?? "apres";
const LANG_UI = process.env.LANG_UI ?? "en";
const EMAIL = process.env.DEMO_EMAIL ?? "user@gmail.com";
const PASSWORD = process.env.DEMO_PASSWORD ?? "user";
const DEMO = 2; // config/demo_gallery.csv: fluconazole, identified above the threshold
const suffix = LANG_UI === "en" ? "" : `-${LANG_UI}`;

async function shot(page, name, target) {
  await page.waitForTimeout(3500); // entrance animations, bars, the 3D scene's first frames
  // Software WebGL slows the frames down: wait until every visible counter shows its final
  // figure (data-value), so a screenshot never catches a number half-way.
  await page
    .waitForFunction(
      () => [...document.querySelectorAll(".count-up")].every((el) => el.textContent === el.getAttribute("data-value")),
      undefined,
      { timeout: 15_000 },
    )
    .catch(() => undefined);
  const path = join(OUT, `${TAG}-${name}${suffix}.jpg`);
  await (target ?? page).screenshot({ path, type: "jpeg", quality: 80 });
  console.log(`  ${path}`);
}

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch({ args: ["--enable-unsafe-swiftshader"] });
for (const scheme of ["light", "dark"]) {
  const theme = scheme === "light" ? "clair" : "sombre";
  const context = await browser.newContext({
    viewport: { width: 1366, height: 900 },
    colorScheme: scheme,
    locale: LANG_UI === "fr" ? "fr-FR" : "en-US",
  });
  const page = await context.newPage();
  page.on("pageerror", (error) => console.error("page error:", error.message));

  await page.goto(`${BASE}/`);
  await page.locator("h1").first().waitFor();
  await shot(page, `accueil-${theme}`);

  await page.goto(`${BASE}/login`);
  await page.locator("form").waitFor();
  await shot(page, `connexion-${theme}`);

  await page.locator('input[type="email"]').fill(EMAIL);
  await page.locator('input[autocomplete="current-password"]').fill(PASSWORD);
  await page.locator('form button[type="submit"]').click();
  await page.waitForURL(/\/(home|admin)/);
  await page.addStyleTag({ content: ".topbar { position: static !important; }" });

  await page.goto(`${BASE}/scan`);
  await page.getByRole("tab", { name: /gallery|galerie/i }).click();
  await page.locator(`[data-testid="demo-${DEMO}"]`).click();
  await page.locator('[data-testid="scan-busy"]').waitFor({ state: "detached", timeout: 60_000 });
  await page.locator(".pill-card").first().waitFor({ timeout: 60_000 });
  await page.locator('[data-testid="scan-result"]').scrollIntoViewIfNeeded();
  await shot(page, `scan-${theme}`);

  const panel = page.locator(".chat-panel");
  await panel.scrollIntoViewIfNeeded();
  await shot(page, `chat-${theme}`, panel);
  await context.close();
}
await browser.close();
