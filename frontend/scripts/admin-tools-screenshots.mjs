// Check the Tools section of the admin dashboard and save screenshots (light, dark, mobile),
// against the running stack (`docker compose up -d`):
//   BASE=http://localhost:8080 node scripts/admin-tools-screenshots.mjs
// Also checks that a plain user does not see the section.
import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://localhost:5173";
const OUT = resolve(process.env.OUT ?? "../docs/screenshots");
const ADMIN = [process.env.ADMIN_EMAIL ?? "admin@gmail.com", process.env.ADMIN_PASSWORD ?? "admin"];
const USER = [process.env.DEMO_EMAIL ?? "user@gmail.com", process.env.DEMO_PASSWORD ?? "user"];
const HOST = new URL(BASE).hostname;
const EXPECTED = [
  `http://${HOST}:5000/`,
  `http://${HOST}:6333/dashboard`,
  `http://${HOST}:8081/`,
  `${new URL(BASE).origin}/docs`,
  `http://${HOST}:8090/`,
];

function check(condition, message) {
  if (!condition) throw new Error(message);
}

async function signIn(browser, [email, password], { theme = "light", viewport } = {}) {
  const context = await browser.newContext({ viewport: viewport ?? { width: 1366, height: 900 } });
  await context.addInitScript(
    ([key, value]) => window.localStorage.setItem(key, value),
    ["tujya-theme", theme],
  );
  await context.addInitScript(() => window.localStorage.setItem("tujya-lang", "en"));
  const page = await context.newPage();
  page.on("pageerror", (error) => console.error("page error:", error.message));
  await page.goto(`${BASE}/login`);
  await page.getByLabel("Email").fill(email);
  await page.getByLabel("Password", { exact: true }).fill(password);
  await page.getByRole("button", { name: /sign in/i }).click();
  await page.waitForURL(/\/(home|admin)/);
  return { context, page };
}

async function shotSection(page, name) {
  // The sticky top bar would cover the cards in a stitched element screenshot.
  await page.addStyleTag({ content: ".topbar { position: static !important; }" });
  const section = page.getByRole("region", { name: "Tools" });
  await section.scrollIntoViewIfNeeded();
  await page.waitForTimeout(900); // entrance animations
  await section.screenshot({ path: join(OUT, name), type: "jpeg", quality: 85 });
  console.log(`  ${name}`);
}

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch();

const admin = await signIn(browser, ADMIN);
const section = admin.page.getByRole("region", { name: "Tools" });
await section.waitFor();
const links = section.getByRole("link");
check((await links.count()) === 5, "the section does not have five tool links");
for (let i = 0; i < 5; i += 1) {
  const link = links.nth(i);
  const href = await link.evaluate((a) => a.href);
  check(href === EXPECTED[i], `link ${i}: ${href} instead of ${EXPECTED[i]}`);
  check((await link.getAttribute("target")) === "_blank", `link ${i} does not open a new tab`);
  check((await link.getAttribute("rel")) === "noopener noreferrer", `link ${i} has no rel`);
}
const text = await section.innerText();
check(!/password/i.test(text), "the section mentions a password");
await shotSection(admin.page, "39-admin-tools.jpg");
await admin.context.close();

const dark = await signIn(browser, ADMIN, { theme: "dark" });
await shotSection(dark.page, "40-admin-tools-dark.jpg");
await dark.context.close();

const mobile = await signIn(browser, ADMIN, { viewport: { width: 390, height: 844 } });
const overflow = await mobile.page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth);
check(!overflow, "the page scrolls sideways at mobile width");
await shotSection(mobile.page, "41-admin-tools-mobile.jpg");
await mobile.context.close();

const user = await signIn(browser, USER);
check(user.page.url().endsWith("/home"), "the user did not land on /home");
await user.page.goto(`${BASE}/admin`);
await user.page.waitForTimeout(1500);
check((await user.page.getByRole("region", { name: "Tools" }).count()) === 0, "a user sees the Tools section");
console.log("  a user does not see the section");
await user.context.close();

await browser.close();
