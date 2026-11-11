// Drive the Scanner page in a real browser and save screenshots (docs/screenshots/).
//
//   npx playwright install chromium            # once
//   node scripts/screenshots.mjs                # app on http://localhost:5173 (npm run dev)
//   BASE=http://localhost:8080 node scripts/screenshots.mjs   # the docker compose app
//   FAKE_VIDEO=/tmp/pills.y4m node scripts/screenshots.mjs    # webcam fed by a video file
//
// Signs in with the demo account and shows the three input modes of the course guide: a photo
// dropped on the drop zone and one chosen with the file dialog, the demo gallery in one click
// and the webcam in continuous inference (Chromium's fake camera, fed by FAKE_VIDEO when given:
// a .y4m or .mjpeg file made from C3PI test photos, kept out of git). Then the Grad-CAM of a
// gallery photo, the champion models with the client and server latency, two more demo photos,
// the interaction check, the confirmation step and Swagger. Fails if a step does not show what
// it should.
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const BASE = process.env.BASE ?? "http://localhost:5173";
const OUT = resolve(process.env.OUT ?? "../docs/screenshots");
const EMAIL = process.env.DEMO_EMAIL ?? "user@gmail.com";
const PASSWORD = process.env.DEMO_PASSWORD ?? "user";
const FAKE_VIDEO = process.env.FAKE_VIDEO;
// Gallery positions (config/demo_gallery.csv): 0 haloperidol, 1 citalopram, 2 fluconazole,
// 4 nortriptyline, 5 two cyclobenzaprine capsules, 6 trazodone (below the threshold).
const UPLOAD = 2;
const DROPPED = 4;
const DEMOS = [1, 0];
const GRADCAM = 5;
const UNSURE = 6;

async function shot(page, name, target) {
  await page.waitForTimeout(1000); // let the entrance animations and the bars finish
  await (target ?? page).screenshot({ path: join(OUT, name), type: "jpeg", quality: 80 });
  console.log(`  ${name}`);
}

/** Wait for the scan of the photo sent as `name` (the previous result may still be shown). */
async function waitForResult(page, name) {
  await page.locator(`img[alt="Photo sent for identification: ${name}"]`).waitFor({ timeout: 60_000 });
  await page.locator('[data-testid="scan-busy"]').waitFor({ state: "detached", timeout: 60_000 });
  await page.locator(".pill-card").first().waitFor({ timeout: 60_000 });
  await page.locator('[data-testid="scan-result"] canvas[data-boxes]').first().waitFor();
}

async function photoFile(page, gallery, index) {
  const photo = await page.request.get(`${BASE}${gallery.items[index].url}`);
  const file = join(tmpdir(), `tujya-test-photo-${index}.jpg`);
  await writeFile(file, await photo.body());
  return file;
}

await mkdir(OUT, { recursive: true });
const args = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"];
if (FAKE_VIDEO) args.push(`--use-file-for-fake-video-capture=${resolve(FAKE_VIDEO)}`);
const browser = await chromium.launch({ args });
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

// 1. Scanner page: champion models (GET /model/metadata) and the drop zone.
await page.getByRole("link", { name: "Scanner" }).click();
await page.getByRole("heading", { name: "Scanner" }).waitFor();
await page.getByText("tujya-pill-detector", { exact: false }).first().waitFor({ timeout: 30_000 });
await shot(page, "01-scanner.jpg");
const gallery = await (await page.request.get(`${BASE}/api/demo/images`)).json();
if (gallery.items.length < 8) throw new Error("the demo gallery is not available");

// 2. Drag and drop: the drop zone while a photo is dragged over it, then the drop.
const dropped = await readFile(await photoFile(page, gallery, DROPPED));
const transfer = await page.evaluateHandle(async (base64) => {
  const blob = await (await fetch(`data:image/jpeg;base64,${base64}`)).blob();
  const data = new DataTransfer();
  data.items.add(new File([blob], "nortriptyline-test.jpg", { type: "image/jpeg" }));
  return data;
}, dropped.toString("base64"));
const zone = page.locator('[data-testid="dropzone"]');
await zone.dispatchEvent("dragover", { dataTransfer: transfer });
await page.locator('[data-testid="dropzone"][data-dragging="true"]').waitFor();
await shot(page, "18-drag-and-drop.jpg", page.getByRole("region", { name: "Photo" }));
await zone.dispatchEvent("drop", { dataTransfer: transfer });
await waitForResult(page, "nortriptyline-test.jpg");

// 3. Upload with the file dialog: boxes by class, top-5 bars, latency client and server.
await page.locator('[data-testid="scan-file"]').setInputFiles(await photoFile(page, gallery, UPLOAD));
await waitForResult(page, `tujya-test-photo-${UPLOAD}.jpg`);
await page.locator('[data-testid="gradcam-note"]').waitFor();
await page.getByText("Client").first().waitFor();
await shot(page, "02-upload-result.jpg");
await page.getByRole("button", { name: "Confirm this pill" }).first().click();

// 4. Demo gallery, one click per photo.
await page.getByRole("tab", { name: /demo gallery/i }).click();
await page.locator(`[data-testid="demo-${GRADCAM}"]`).waitFor();
await shot(page, "19-demo-gallery.jpg", page.getByRole("region", { name: "Photo" }));

// 5. Grad-CAM of a gallery photo (two capsules), with the boxes of the scan.
await page.locator(`[data-testid="demo-${GRADCAM}"]`).click();
await waitForResult(page, `demo-${GRADCAM}.jpg`);
await page.getByRole("tab", { name: "Grad-CAM" }).click();
await page.getByText("Grad-CAM of the served classifier").waitFor();
await page.waitForTimeout(500);
await shot(page, "20-gradcam.jpg", page.locator('[data-testid="result-stage"]'));

// 6. Champion models served (GET /model/metadata), at the top of the page.
await page.getByRole("tab", { name: "Photo and boxes" }).click();
await page.evaluate(() => window.scrollTo(0, 0));
await shot(page, "21-champion-models.jpg", page.locator('[data-testid="model-card"]'));

// 7. Two more demo photos, each confirmed, then their interactions.
for (const id of DEMOS) {
  await page.locator(`[data-testid="demo-${id}"]`).click();
  await waitForResult(page, `demo-${id}.jpg`);
  await page.getByRole("button", { name: "Confirm this pill" }).first().click();
}
await page.locator('[data-testid="scan-result"]').scrollIntoViewIfNeeded();
await shot(page, "03-demo-result.jpg");
await page.getByRole("button", { name: "Check interactions" }).click();
await page.locator(".report").waitFor({ timeout: 30_000 });
await shot(page, "04-interactions.jpg", page.locator(".my-pills"));

// 8. A photo the model is not sure about: the confirmation step.
await page.getByRole("region", { name: "Photo" }).scrollIntoViewIfNeeded();
await page.locator(`[data-testid="demo-${UNSURE}"]`).click();
await waitForResult(page, `demo-${UNSURE}.jpg`);
await page.getByText("Please confirm").first().waitFor();
await shot(page, "05-needs-confirmation.jpg", page.locator('[data-testid="scan-result"]'));

// 9. Webcam in continuous inference: boxes and top-5 bars of each frame, latency.
await page.getByRole("tab", { name: /webcam/i }).click();
await page.getByRole("button", { name: "Start the camera" }).click();
await page.locator("video").waitFor();
await page.getByText(/^Frame [3-9]|^Frame \d\d/).waitFor({ timeout: 60_000 });
if (FAKE_VIDEO) await page.locator('canvas[aria-label="Live predictions"]:not([data-boxes="0"])').waitFor();
await page.getByRole("region", { name: "Photo" }).scrollIntoViewIfNeeded();
await shot(page, "06-webcam.jpg", page.getByRole("region", { name: "Photo" }));

// 10. Swagger UI at /docs.
await page.goto(`${BASE}/docs`);
await page.locator(".swagger-ui .opblock").first().waitFor({ timeout: 30_000 });
await shot(page, "07-swagger.jpg");

await browser.close();
console.log(`screenshots written to ${OUT}`);
