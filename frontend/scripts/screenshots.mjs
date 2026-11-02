// Drive the Scanner page in a real browser and save screenshots (docs/screenshots/).
//
//   npx playwright install chromium            # once
//   node scripts/screenshots.mjs                # app on http://localhost:5173 (npm run dev)
//   BASE=http://localhost:8080 node scripts/screenshots.mjs   # the docker compose app
//
// Signs in with the demo account, uploads a C3PI test photo from the demo gallery, adds two
// more demo photos, checks the interactions, opens the webcam (Chromium's fake camera) and
// Swagger. Fails if any step does not show what it should.
import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://localhost:5173";
const OUT = resolve(process.env.OUT ?? "../docs/screenshots");
const EMAIL = process.env.DEMO_EMAIL ?? "user@gmail.com";
const PASSWORD = process.env.DEMO_PASSWORD ?? "user";
// Gallery positions (config/demo_gallery.csv): 2 fluconazole, 1 citalopram, 0 tramadol,
// 6 trazodone (below the confirmation threshold).
const UPLOAD = 2;
const DEMOS = [1, 0];
const UNSURE = 6;

async function shot(page, name, target) {
  await page.waitForTimeout(900); // let the entrance animations finish
  await (target ?? page).screenshot({ path: join(OUT, name), type: "jpeg", quality: 82 });
}

async function waitForResult(page) {
  await page.locator(".scan-busy").waitFor({ state: "detached", timeout: 60_000 });
  await page.locator(".pill-card").first().waitFor({ timeout: 60_000 });
}

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch({
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
});
const context = await browser.newContext({ viewport: { width: 1366, height: 900 } });
await context.grantPermissions(["camera"], { origin: BASE });
const page = await context.newPage();
page.on("pageerror", (error) => console.error("page error:", error.message));

// Sign in.
await page.goto(`${BASE}/login`);
await page.getByLabel("Email").fill(EMAIL);
await page.getByLabel("Password", { exact: true }).fill(PASSWORD);
await page.getByRole("button", { name: /sign in/i }).click();
await page.waitForURL(/\/(home|admin)/);
// The sticky top bar would cover the top of element screenshots.
await page.addStyleTag({ content: ".topbar { position: static !important; }" });

// 1. Scanner page, upload of a test photo saved to a temporary file.
await page.getByRole("link", { name: "Scanner" }).click();
await page.getByRole("heading", { name: "Scanner" }).waitFor();
await shot(page, "01-scanner.jpg");
const gallery = await (await page.request.get(`${BASE}/api/demo/images`)).json();
if (gallery.items.length < 7) throw new Error("the demo gallery is not available");
const photo = await page.request.get(`${BASE}${gallery.items[UPLOAD].url}`);
const file = join(tmpdir(), "tujya-upload-test.jpg");
await writeFile(file, await photo.body());
await page.locator('[data-testid="scan-file"]').setInputFiles(file);
await waitForResult(page);
await page.locator(".scan-box").first().waitFor();
await shot(page, "02-upload-result.jpg");
await page.getByRole("button", { name: "Confirm this pill" }).first().click();

// 2. Two demo photos, each confirmed.
await page.getByRole("tab", { name: /demo photos/i }).click();
for (const id of DEMOS) {
  await page.locator(`[data-testid="demo-${id}"]`).click();
  await waitForResult(page);
  await page.getByRole("button", { name: "Confirm this pill" }).first().click();
}
await page.locator(".scan-result").scrollIntoViewIfNeeded();
await shot(page, "03-demo-result.jpg");

// 3. Interactions between the three confirmed pills.
await page.getByRole("button", { name: "Check interactions" }).click();
await page.locator(".report").waitFor({ timeout: 30_000 });
await shot(page, "04-interactions.jpg", page.locator(".my-pills"));

// 4. A photo the model is not sure about: the confirmation step.
await page.locator(".scan-input").scrollIntoViewIfNeeded();
await page.locator(`[data-testid="demo-${UNSURE}"]`).click();
await waitForResult(page);
await page.getByText("Please confirm").first().waitFor();
await shot(page, "05-needs-confirmation.jpg", page.locator(".scan-result"));

// 5. Webcam (Chromium's fake camera: a test pattern, so usually no pill) with live predictions.
await page.getByRole("tab", { name: /webcam/i }).click();
await page.getByRole("button", { name: "Start the camera" }).click();
await page.locator(".webcam video").waitFor();
await page.waitForTimeout(3000);
await shot(page, "06-webcam.jpg", page.locator(".scan-input"));

// 6. Swagger UI at /docs.
await page.goto(`${BASE}/docs`);
await page.locator(".swagger-ui .opblock").first().waitFor({ timeout: 30_000 });
await shot(page, "07-swagger.jpg");

await browser.close();
console.log(`screenshots written to ${OUT}`);
