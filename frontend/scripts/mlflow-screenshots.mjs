// Save screenshots of the MLflow UI for the demo script and the reports (docs/screenshots/).
//
//   MLFLOW=http://127.0.0.1:5000 node scripts/mlflow-screenshots.mjs
//
// Opens the vision and RAG experiments (run tables, "Model training" view) and the Model
// Registry (both registered models, then the classifier), closing MLflow 3's guidance
// pop-ups and assistant panel. Fails if a page does not show what it should.
import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const MLFLOW = process.env.MLFLOW ?? "http://127.0.0.1:5000";
const OUT = resolve(process.env.OUT ?? "../docs/screenshots");

async function experimentId(name) {
  const response = await fetch(
    `${MLFLOW}/api/2.0/mlflow/experiments/get-by-name?experiment_name=${name}`,
  );
  if (!response.ok) throw new Error(`experiment ${name} not found`);
  return (await response.json()).experiment.experiment_id;
}

async function dismissOverlays(page) {
  for (const button of [
    page.getByRole("button", { name: "Got it" }),
    page.getByRole("button", { name: "Close guidance" }),
    page.getByRole("dialog").getByRole("button", { name: /close/i }),
    page.getByRole("button", { name: "Close", exact: true }),
  ]) {
    for (const one of await button.all()) {
      if (await one.isVisible()) await one.click().catch(() => {});
    }
  }
  await page.keyboard.press("Escape");
}

async function open(page, path, text, name) {
  await page.goto(`${MLFLOW}/#${path}`);
  await page.getByText(text).first().waitFor({ timeout: 30_000 });
  await page.waitForTimeout(2000); // tables and pop-ups appear after the first paint
  await dismissOverlays(page);
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(OUT, name), type: "jpeg", quality: 82 });
  console.log(name);
}

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1366, height: 860 } });
await open(page, `/experiments/${await experimentId("tujya-vision")}/runs?workflowType=machine_learning`, "tujya-vision", "12-mlflow-vision.jpg");
await open(page, `/experiments/${await experimentId("tujya-rag")}/runs?workflowType=machine_learning`, "tujya-rag", "13-mlflow-rag.jpg");
await open(page, "/models", "tujya-pill-classifier", "14-mlflow-registry.jpg");
await open(page, "/models/tujya-pill-classifier", "Version 2", "15-mlflow-classifier.jpg");
await browser.close();
