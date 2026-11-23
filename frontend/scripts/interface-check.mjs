// Check every page in a real browser, light and dark, English and French: no page error, no
// horizontal scroll, readable contrast, remembered theme and language, reduced motion.
//   BASE=http://localhost:8080 [ADMIN_EMAIL=... ADMIN_PASSWORD=...] node scripts/interface-check.mjs
import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://localhost:5173";
const USER = { email: process.env.DEMO_EMAIL ?? "user@gmail.com", password: process.env.DEMO_PASSWORD ?? "user" };
const ADMIN = { email: process.env.ADMIN_EMAIL ?? "admin@gmail.com", password: process.env.ADMIN_PASSWORD ?? "admin" };
const PUBLIC = ["/", "/login", "/signup", "/about", "/privacy", "/help"];
const USER_PAGES = ["/home", "/scan", "/reports", "/toxicity", "/documents", "/monitoring", "/profile"];
const ADMIN_PAGES = ["/admin", "/admin/users", "/admin/activity", "/admin/usage", "/admin/account"];
const WIDTHS = [1366, 390];
const failures = [];

function fail(message) {
  failures.push(message);
  console.log(`  FAIL ${message}`);
}

/** WCAG contrast ratio of two #rrggbb or rgb() colours. */
function contrast(a, b) {
  const lum = (colour) => {
    const [r, g, bl] = colour.match(/\d+(\.\d+)?/g).slice(0, 3).map(Number).map((v) => {
      const c = v / 255;
      return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    });
    return 0.2126 * r + 0.7152 * g + 0.0722 * bl;
  };
  const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p);
  return (x + 0.05) / (y + 0.05);
}

async function signIn(page, account) {
  await page.goto(`${BASE}/login`);
  await page.locator('input[type="email"]').fill(account.email);
  await page.locator('input[autocomplete="current-password"]').fill(account.password);
  await page.locator('form button[type="submit"]').click();
  try {
    await page.waitForURL(/\/(home|admin)/, { timeout: 15_000 });
    return true;
  } catch {
    return false;
  }
}

async function checkPages(context, label, paths) {
  const page = await context.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  for (const width of WIDTHS) {
    await page.setViewportSize({ width, height: 860 });
    for (const path of paths) {
      errors.length = 0;
      await page.goto(`${BASE}${path}`);
      await page.locator("h1, h2").first().waitFor({ timeout: 20_000 });
      await page.waitForTimeout(600);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      if (overflow > 1) fail(`${label} ${path} @${width}px: ${overflow}px of horizontal scrolling`);
      if (errors.length) fail(`${label} ${path}: page error ${errors[0]}`);
      const theme = await page.evaluate(() => document.documentElement.dataset.theme);
      console.log(`  ok ${label} ${path} @${width}px (theme ${theme})`);
    }
  }
  return page;
}

const browser = await chromium.launch({ args: ["--enable-unsafe-swiftshader"] });

for (const scheme of ["light", "dark"]) {
  for (const locale of ["en-US", "fr-FR"]) {
    const label = `${scheme}/${locale.slice(0, 2)}`;
    console.log(`\n${label}`);
    const visitor = await browser.newContext({ colorScheme: scheme, locale });
    const page = await checkPages(visitor, label, PUBLIC);

    // Colour tokens of the page: body text, muted text and links must stay readable.
    const tokens = await page.evaluate(() => {
      const css = getComputedStyle(document.documentElement);
      const read = (name) => {
        const probe = document.createElement("span");
        probe.style.color = css.getPropertyValue(name);
        document.body.append(probe);
        const value = getComputedStyle(probe).color;
        probe.remove();
        return value;
      };
      return Object.fromEntries(
        ["--text", "--muted", "--bg", "--surface", "--primary-text", "--heading"].map((n) => [n, read(n)]),
      );
    });
    for (const [fg, bg, min] of [
      ["--text", "--bg", 7],
      ["--muted", "--surface", 4.5],
      ["--muted", "--bg", 4.5],
      ["--primary-text", "--surface", 4.5],
      ["--heading", "--surface", 7],
    ]) {
      const ratio = contrast(tokens[fg], tokens[bg]);
      if (ratio < min) fail(`${label}: contrast ${fg} on ${bg} is ${ratio.toFixed(2)} (< ${min})`);
    }
    await visitor.close();

    const user = await browser.newContext({ colorScheme: scheme, locale });
    const userPage = await user.newPage();
    if (await signIn(userPage, USER)) await checkPages(user, `${label} user`, USER_PAGES);
    else fail(`${label}: cannot sign in as ${USER.email}`);
    await user.close();

    const admin = await browser.newContext({ colorScheme: scheme, locale });
    const adminPage = await admin.newPage();
    if (await signIn(adminPage, ADMIN)) await checkPages(admin, `${label} admin`, ADMIN_PAGES);
    else console.log(`  skip admin pages: cannot sign in as ${ADMIN.email} (password changed?)`);
    await admin.close();
  }
}

// The buttons: the choice is applied at once and kept after a reload.
console.log("\ntoggles");
const context = await browser.newContext({ colorScheme: "light", locale: "en-US" });
const page = await context.newPage();
await page.goto(`${BASE}/login`);
await page.locator("[data-theme-toggle]").first().click();
await page.locator("[data-lang-toggle]").first().click();
await page.waitForTimeout(400);
await page.reload();
// React renders after the load event: wait for the card before reading its title.
await page.locator(".auth-card h2").waitFor({ timeout: 10_000 }).catch(() => undefined);
const state = await page.evaluate(() => ({
  theme: document.documentElement.dataset.theme,
  lang: document.documentElement.lang,
  title: document.querySelector(".auth-card h2")?.textContent,
}));
if (state.theme !== "dark" || state.lang !== "fr" || state.title !== "Bon retour") {
  fail(`toggles not kept after reload: ${JSON.stringify(state)}`);
} else console.log(`  ok theme and language kept: ${JSON.stringify(state)}`);
await context.close();

// Reduced motion: no WebGL canvas, the still emblem instead, and three.js never downloaded.
console.log("\nreduced motion");
const calm = await browser.newContext({ reducedMotion: "reduce" });
const calmPage = await calm.newPage();
const scripts = [];
calmPage.on("request", (request) => scripts.push(request.url()));
await calmPage.goto(`${BASE}/`);
await calmPage.waitForTimeout(1500);
const scene = await calmPage.evaluate(() => ({
  motion: document.querySelector(".scene3d")?.getAttribute("data-motion"),
  canvas: document.querySelectorAll(".scene3d canvas").length,
}));
const threeLoaded = scripts.some((url) => url.includes("pillScene"));
if (scene.motion !== "off" || scene.canvas !== 0 || threeLoaded) {
  fail(`reduced motion still animates: ${JSON.stringify({ ...scene, threeLoaded })}`);
} else console.log("  ok no canvas, three.js not downloaded");
await calm.close();

const normal = await browser.newContext();
const normalPage = await normal.newPage();
await normalPage.goto(`${BASE}/`);
await normalPage.locator(".scene3d canvas").waitFor({ timeout: 15_000 }).catch(() => undefined);
console.log(`  ok with motion: ${await normalPage.locator(".scene3d canvas").count()} canvas`);
await normal.close();

await browser.close();
console.log(failures.length ? `\n${failures.length} failure(s)` : "\nall checks passed");
process.exit(failures.length ? 1 : 0);
