import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { ModelMetadata } from "../services/api";
import { LatencyReadout, ModelCardView, ProbabilityBars, ScanCanvas } from "./vision";

const CANDIDATES = [
  { class_id: 120, name: "Citalopram", probability: 0.952 },
  { class_id: 9, name: "Fluconazole", probability: 0.03 },
  { class_id: 22, name: "Haloperidol", probability: 0.01 },
  { class_id: 0, name: "Amrix", probability: 0.005 },
  { class_id: 98, name: "Trazodone", probability: 0.002 },
  { class_id: 104, name: "Formula B", probability: 0.001 },
];

describe("ProbabilityBars", () => {
  it("shows the top-5 with their probabilities, the first in its class colour", () => {
    const html = renderToStaticMarkup(<ProbabilityBars candidates={CANDIDATES} />);
    expect(html.match(/<li/g)).toHaveLength(5); // the sixth is left out
    expect(html).toContain("Citalopram");
    expect(html).toContain("95%");
    expect(html).not.toContain("Formula B");
    expect(html).toContain("width:95.19999999999999%;background:#ff3838"); // class 120
    expect(html).toContain("animate-grow"); // the bar fills when it appears
  });
});

describe("LatencyReadout", () => {
  it("shows client and server latency in ms", () => {
    const html = renderToStaticMarkup(
      <LatencyReadout latency={{ clientMs: 412.4, serverMs: 380, inferenceMs: 210.6 }} />,
    );
    expect(html).toContain("Client");
    expect(html).toContain("412 ms");
    expect(html).toContain("Server");
    expect(html).toContain("380 ms");
    expect(html).toContain("211 ms");
    expect(html).toContain("network and browser 32 ms");
  });

  it("says when the server did not give its time", () => {
    const html = renderToStaticMarkup(
      <LatencyReadout latency={{ clientMs: 90, serverMs: null, inferenceMs: null }} />,
    );
    expect(html).toContain("90 ms");
    expect(html).toContain("–");
    expect(html).not.toContain("network and browser");
  });
});

function model(overrides: Partial<ModelMetadata>): ModelMetadata {
  return {
    task: "detect",
    name: "tujya-pill-detector",
    architecture: "yolo11s",
    version: 9,
    aliases: ["champion", "served"],
    precision: "fp32",
    variant_of: null,
    run_id: "894871cd",
    input_size: 640,
    metrics: { test_v2_mAP50_95: 0.88846, onnx_cpu_latency_ms: 63 },
    classes: ["pill"],
    num_classes: 1,
    ...overrides,
  };
}

describe("ModelCardView", () => {
  it("names the champion of each task with its Registry version and scores", () => {
    const html = renderToStaticMarkup(
      <ModelCardView
        detector={model({})}
        classifier={model({
          task: "classify",
          name: "tujya-pill-classifier",
          architecture: "yolo11s-cls",
          version: 1,
          input_size: 224,
          metrics: { test_top1: 0.937 },
          num_classes: 150,
        })}
      />,
    );
    expect(html).toContain("yolo11s");
    expect(html).toContain("tujya-pill-detector v9");
    expect(html).toContain("champion");
    expect(html).toContain("0.888");
    expect(html).toContain("tujya-pill-classifier v1");
    expect(html).toContain("93.7%");
    expect(html).toContain("150 classes");
    expect(html).toContain("640 px · 1 class<");
  });
});

describe("ScanCanvas", () => {
  it("renders an accessible canvas that says how many boxes it draws", () => {
    const html = renderToStaticMarkup(
      <ScanCanvas pills={[]} width={640} height={480} label="Detected pills" />,
    );
    expect(html).toContain('role="img"');
    expect(html).toContain('aria-label="Detected pills"');
    expect(html).toContain('data-boxes="0"');
  });
});
