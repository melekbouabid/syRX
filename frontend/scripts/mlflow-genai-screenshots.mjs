// Screenshots of the MLflow UI: agent traces, generation evaluation runs, Prompt Registry, RAG
// models and the vision model comparisons (docs/screenshots/). Fails on a missing page.
//   MLFLOW=http://127.0.0.1:5000 node scripts/mlflow-genai-screenshots.mjs
import { mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import { chromium } from "playwright";

const MLFLOW = process.env.MLFLOW ?? "http://127.0.0.1:5000";
const OUT = resolve(process.env.OUT ?? "../docs/screenshots");

async function api(path) {
  const response = await fetch(`${MLFLOW}/api/2.0/mlflow/${path}`);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
}

async function experimentId(name) {
  const body = await api(`experiments/get-by-name?experiment_name=${name}`);
  return body.experiment.experiment_id;
}

// Run ids of the registered versions of `model` whose tag `status` is one of `statuses` and
// whose run has `metric` (the candidates of a comparison, measured on the same test split:
// the baselines of v1 have no v2 test metric, the quantised variants another status).
async function candidateRuns(model, statuses, metric) {
  const filter = encodeURIComponent(`name='${model}'`);
  const body = await api(`model-versions/search?filter=${filter}&max_results=100`);
  const runs = [];
  for (const version of body.model_versions) {
    const status = (version.tags ?? []).find((t) => t.key === "status")?.value;
    if (!statuses.includes(status)) continue;
    const run = await api(`runs/get?run_id=${version.run_id}`);
    if ((run.run.data.metrics ?? []).some((m) => m.key === metric)) runs.push(version.run_id);
  }
  return runs;
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

// Parallel coordinates of the comparison view: the parameters and the metric that decided
// (MLflow picks an arbitrary metric otherwise). A parameter missing from every run is skipped.
async function plotSettings(page, params, metric) {
  await page.getByRole("button", { name: "Clear All" }).click();
  for (const [index, value] of [[1, metric], ...params.map((v) => [0, v])]) {
    const box = page.getByRole("combobox").nth(index);
    await box.click();
    await box.fill(value);
    await page.waitForTimeout(1000);
    const offered = (await page.getByRole("option").allInnerTexts()).map((t) => t.trim());
    // The first option of the filtered list (shortest match first) is the exact name; a click on
    // it is intercepted by the list's own scroll container, the keyboard is not.
    if (offered[0] === value) await page.keyboard.press("Enter");
    else if (index === 1) throw new Error(`metric ${metric} not offered (${offered})`);
    await page.keyboard.press("Escape");
  }
  await page.waitForTimeout(1500);
}

// A new page for each screenshot: the MLflow app does not re-render when only the hash route
// changes between two pages of the same kind (two registered models).
async function open(browser, path, text, name, settings = null) {
  const page = await browser.newPage({ viewport: { width: 1366, height: 860 } });
  await page.goto(`${MLFLOW}/#${path}`);
  await page.getByText(text).first().waitFor({ timeout: 45_000 });
  await page.waitForTimeout(2500); // tables and pop-ups appear after the first paint
  await dismissOverlays(page);
  if (settings) await plotSettings(page, settings.params, settings.metric);
  await page.waitForTimeout(800);
  await page.screenshot({ path: join(OUT, name), type: "jpeg", quality: 82 });
  await page.close();
  console.log(name);
}

const generation = await experimentId("tujya-rag-generation");
const vision = await experimentId("tujya-vision");
const statuses = ["champion", "challenger", "archived"];
const detectors = await candidateRuns("tujya-pill-detector", statuses, "test_v2_mAP50_95");
const classifiers = await candidateRuns("tujya-pill-classifier", statuses, "test_v2_top1");
const compare = (runs) =>
  `/compare-runs?runs=${encodeURIComponent(JSON.stringify(runs))}` +
  `&experiments=${encodeURIComponent(JSON.stringify([vision]))}`;

await mkdir(OUT, { recursive: true });
const browser = await chromium.launch();
await open(browser, `/experiments/${generation}/traces`, "question", "32-mlflow-traces.jpg");
await open(browser, `/experiments/${generation}/evaluation-runs`, "generation-", "33-mlflow-evaluations.jpg");
await open(browser, "/prompts", "tujya-answer-fr", "34-mlflow-prompts.jpg");
await open(browser, "/prompts/tujya-answer-fr", "champion", "35-mlflow-prompt-versions.jpg");
await open(browser, "/models/tujya-rag-retrieval", "Version 4", "36-mlflow-rag-retrieval-model.jpg");
await open(browser, "/models/tujya-rag-generation", "qwen2.5-3b__v1", "37-mlflow-rag-generation-model.jpg");
await open(browser, compare(detectors), "detector-yolo11s", "38-mlflow-compare-detectors.jpg", {
  params: ["arch", "imgsz", "lr0", "batch"],
  metric: "test_v2_mAP50_95",
});
await open(browser, compare(classifiers), "classifier-efficientnetv2s", "39-mlflow-compare-classifiers.jpg", {
  params: ["arch", "imgsz", "lr", "weight_decay"],
  metric: "test_v2_top1",
});
await browser.close();
