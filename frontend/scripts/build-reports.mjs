// Build the technical reports (docs/rapports/): Mermaid diagrams -> PNG, Markdown -> PDF.
//
//   cd frontend
//   node scripts/build-reports.mjs            # all diagrams and reports
//
// No pandoc or LaTeX on this machine: Chromium (Playwright) renders everything. Mermaid and
// marked are loaded from jsDelivr at pinned versions, so the build needs Internet access.
// Diagrams: docs/rapports/diagrams/*.mmd -> docs/rapports/img/*.png.
// Reports: docs/rapports/rapport_*.md -> docs/rapports/rapport_*.pdf (A4, page numbers).
// Fails if a diagram does not render or an image of a report does not load.
import { readdir, readFile, rm, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { chromium } from "playwright";

const ROOT = resolve(process.env.REPORTS ?? "../docs/rapports");
const MERMAID = "https://cdn.jsdelivr.net/npm/mermaid@11.4.1/dist/mermaid.min.js";
const MARKED = "https://cdn.jsdelivr.net/npm/marked@12.0.2/marked.min.js";

const CSS = `
  @page { size: A4; margin: 18mm 16mm 20mm 16mm; }
  body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45;
         color: #1d1d1f; }
  h1 { font-size: 20pt; color: #0b4f6c; border-bottom: 2px solid #0b4f6c; padding-bottom: 4px; }
  h2 { font-size: 14pt; color: #0b4f6c; margin-top: 22px; break-after: avoid; }
  h3 { font-size: 11.5pt; margin-top: 14px; break-after: avoid; }
  table { border-collapse: collapse; margin: 8px 0; font-size: 9pt; break-inside: avoid; }
  th, td { border: 1px solid #c8ccd0; padding: 3px 6px; vertical-align: top; }
  th { background: #eef3f6; }
  code { font-family: Consolas, monospace; font-size: 9pt; background: #f3f4f6; padding: 0 2px; }
  pre { background: #f3f4f6; padding: 8px; font-size: 8.5pt; white-space: pre-wrap;
        break-inside: avoid; }
  pre code { background: none; }
  img { max-width: 100%; max-height: 120mm; display: block; margin: 8px auto;
        border: 1px solid #d0d4d8; break-inside: avoid; }
  blockquote { border-left: 4px solid #e0a100; margin: 8px 0; padding: 2px 10px;
               background: #fff8e6; }
  p:has(> img) { break-inside: avoid; text-align: center; }
  .caption { display: block; font-size: 8.5pt; font-style: italic; color: #555; }
`;

async function renderDiagrams(browser) {
  const dir = join(ROOT, "diagrams");
  const page = await browser.newPage({ deviceScaleFactor: 2, viewport: { width: 1400, height: 900 } });
  await page.setContent(`<html><body style="margin:0;background:white"><div id="out"></div></body></html>`);
  await page.addScriptTag({ url: MERMAID });
  await page.evaluate(() =>
    window.mermaid.initialize({
      startOnLoad: false,
      theme: "neutral",
      themeVariables: { fontFamily: "Arial", fontSize: "18px" },
      flowchart: { wrappingWidth: 260 },
    }),
  );
  for (const file of (await readdir(dir)).filter((name) => name.endsWith(".mmd"))) {
    const source = await readFile(join(dir, file), "utf8");
    await page.evaluate(async (text) => {
      const { svg } = await window.mermaid.render("diagram", text);
      const out = document.getElementById("out");
      out.innerHTML = svg;
      out.style.display = "inline-block";
      out.style.padding = "16px";
    }, source);
    const png = join(ROOT, "img", `${basename(file, ".mmd")}.png`);
    await page.locator("#out").screenshot({ path: png });
    console.log("diagram", png);
  }
  await page.close();
}

async function renderReport(browser, file) {
  const markdown = await readFile(join(ROOT, file), "utf8");
  const html = join(ROOT, `.${basename(file, ".md")}.html`);
  await writeFile(
    html,
    `<!doctype html><html lang="fr"><head><meta charset="utf-8"><style>${CSS}</style>
     <script src="${MARKED}"></script></head><body><main id="doc"></main>
     <script id="md" type="text/plain">${markdown.replaceAll("</script", "<\\/script")}</script>
     <script>document.getElementById("doc").innerHTML =
       marked.parse(document.getElementById("md").textContent);
     for (const img of document.querySelectorAll("#doc img")) {
       const caption = document.createElement("span");
       caption.className = "caption";
       caption.textContent = img.alt;
       img.after(caption);
     }</script></body></html>`,
  );
  const page = await browser.newPage();
  try {
    await page.goto(pathToFileURL(html).href, { waitUntil: "networkidle" });
    const broken = await page.evaluate(() =>
      [...document.images].filter((img) => !img.complete || img.naturalWidth === 0).map((img) => img.src),
    );
    if (broken.length) throw new Error(`${file}: images not loaded: ${broken.join(", ")}`);
    const pdf = join(ROOT, `${basename(file, ".md")}.pdf`);
    await page.pdf({
      path: pdf,
      format: "A4",
      printBackground: true,
      displayHeaderFooter: true,
      headerTemplate: "<span></span>",
      footerTemplate: `<div style="font-size:8pt;width:100%;text-align:center;color:#777">
        Tujya · ${basename(file, ".md")} · <span class="pageNumber"></span> / <span class="totalPages"></span></div>`,
      margin: { top: "16mm", bottom: "18mm", left: "15mm", right: "15mm" },
    });
    const images = await page.evaluate(() => document.images.length);
    console.log("report", pdf, `${images} images`);
  } finally {
    await page.close();
    await rm(html, { force: true });
  }
}

const browser = await chromium.launch();
try {
  await renderDiagrams(browser);
  for (const file of (await readdir(ROOT)).filter((name) => /^rapport_.*\.md$/.test(name))) {
    await renderReport(browser, file);
  }
} finally {
  await browser.close();
}
