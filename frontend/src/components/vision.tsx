// Vision components of the Scanner page (course guide, step 2.6), styled with Tailwind CSS
// (D106): boxes coloured by class on a canvas, animated probability bars, client and server
// latency, and the champion models served by the API.
import { useEffect, useRef } from "react";
import type { CSSProperties } from "react";
import { api, type ModelMetadata, type ScannedPill } from "../services/api";
import { useQuery } from "../hooks";
import {
  classColour,
  containRect,
  formatMs,
  headlineMetrics,
  percent,
  scaleBox,
  textOn,
  transferMs,
  type Latency,
} from "../lib/vision";
import { Icon } from "./icons";

const LABEL_FONT = "600 13px Inter, 'Segoe UI', system-ui, sans-serif";

/**
 * Boxes of the detected pills on a canvas laid over the photo or the video. Each box has the
 * colour of its most probable class (the same colours as POST /predict/visualize) and a label
 * with the class and its probability; a dashed box asks the user to confirm. The canvas
 * follows the size of its parent (the stage), so it stays aligned when the page is resized.
 */
export function ScanCanvas({
  pills,
  width,
  height,
  label,
}: {
  pills: ScannedPill[];
  /** Size of the image the boxes were computed on (`image_width` / `image_height`). */
  width: number;
  height: number;
  label: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const element = canvas.current;
    const stage = element?.parentElement;
    if (!element || !stage) return;

    const draw = () => {
      const ratio = window.devicePixelRatio || 1;
      const view = { width: stage.clientWidth, height: stage.clientHeight };
      element.width = Math.round(view.width * ratio);
      element.height = Math.round(view.height * ratio);
      const context = element.getContext("2d");
      if (!context) return;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, view.width, view.height);
      const frame = containRect({ width, height }, view);
      context.font = LABEL_FONT;
      context.textBaseline = "middle";
      pills.forEach((pill, index) => {
        const top = pill.candidates[0];
        const colour = classColour(top?.class_id ?? -1);
        const box = scaleBox(pill.box_normalized, frame);
        context.setLineDash(pill.needs_confirmation ? [8, 5] : []);
        context.lineWidth = 3;
        context.strokeStyle = colour;
        context.fillStyle = `${colour}22`;
        context.beginPath();
        context.roundRect(box.x, box.y, box.width, box.height, 6);
        context.fill();
        context.stroke();
        context.setLineDash([]);

        const text = `${index + 1}. ${top ? `${top.name} ${percent(top.probability)}` : ""}`;
        const textWidth = Math.min(context.measureText(text).width + 12, view.width);
        const above = box.y - 24 >= 0;
        const x = Math.min(Math.max(0, box.x), view.width - textWidth);
        const y = above ? box.y - 24 : box.y + 2;
        context.fillStyle = colour;
        context.beginPath();
        context.roundRect(x, y, textWidth, 22, 5);
        context.fill();
        context.fillStyle = textOn(colour);
        context.fillText(text, x + 6, y + 11, textWidth - 12);
      });
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(stage);
    return () => observer.disconnect();
  }, [pills, width, height]);

  return (
    <canvas
      ref={canvas}
      role="img"
      aria-label={label}
      data-boxes={pills.length}
      className="pointer-events-none absolute inset-0 h-full w-full"
    />
  );
}

/** One probability as an animated bar: it fills from zero, then follows the new value. */
export function ProbBar({ value, colour }: { value: number; colour?: string }) {
  const fill: CSSProperties = { width: `${Math.max(0, Math.min(1, value)) * 100}%` };
  if (colour) fill.background = colour;
  return (
    <span className="block h-2 w-full overflow-hidden rounded-full bg-surface-2" aria-hidden="true">
      <span
        className="block h-full origin-left animate-grow rounded-full bg-brand-line transition-[width] duration-500 ease-brand"
        style={fill}
      />
    </span>
  );
}

/** Top-5 classes of one pill with their probabilities; the first one in its box colour. */
export function ProbabilityBars({
  candidates,
  limit = 5,
}: {
  candidates: { class_id: number; name: string; probability: number }[];
  limit?: number;
}) {
  return (
    <ol className="m-0 grid list-none gap-1.5 p-0" aria-label="Most probable pills">
      {candidates.slice(0, limit).map((candidate, rank) => (
        <li
          // By rank, so a live update animates the bar from its previous width.
          key={rank}
          className="grid grid-cols-[minmax(0,1fr)_minmax(64px,40%)_3.2rem] items-center gap-2 text-[0.86rem]"
        >
          <span className="flex min-w-0 items-center gap-1.5">
            <span
              className="size-2.5 flex-none rounded-full"
              style={{ background: classColour(candidate.class_id) }}
              aria-hidden="true"
            />
            <span className={rank === 0 ? "truncate font-semibold text-heading" : "truncate text-ink"}>
              {candidate.name}
            </span>
          </span>
          <ProbBar
            value={candidate.probability}
            colour={rank === 0 ? classColour(candidate.class_id) : undefined}
          />
          <span className="text-right font-semibold tabular-nums text-ink">
            {percent(candidate.probability)}
          </span>
        </li>
      ))}
    </ol>
  );
}

function Figure({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <span
      className="inline-flex items-baseline gap-1.5 rounded-lg border border-line bg-surface px-2.5 py-1"
      title={hint}
    >
      <span className="text-[0.78rem] font-semibold uppercase tracking-wide text-muted">{label}</span>
      <span className="font-semibold tabular-nums text-heading">{value}</span>
    </span>
  );
}

/** Client latency (measured in the browser) next to the server's own figures, in ms. */
export function LatencyReadout({ latency }: { latency: Latency }) {
  const transfer = transferMs(latency);
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm" aria-label="Latency">
      <Figure
        label="Client"
        value={formatMs(latency.clientMs)}
        hint="Measured in the browser: from sending the photo to having the answer"
      />
      <Figure
        label="Server"
        value={formatMs(latency.serverMs)}
        hint="Returned by the API: the whole request on the server"
      />
      <Figure
        label="Model"
        value={formatMs(latency.inferenceMs)}
        hint="ONNX Runtime only: detection plus identification"
      />
      {transfer !== null && (
        <span className="text-muted" title="Client minus server: upload, download, browser">
          network and browser {formatMs(transfer)}
        </span>
      )}
    </div>
  );
}

function ModelLine({ title, model }: { title: string; model: ModelMetadata }) {
  return (
    <div className="grid gap-1.5 rounded-xl border border-line bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[0.78rem] font-semibold uppercase tracking-wide text-muted">{title}</span>
        {model.aliases.map((alias) => (
          <span
            key={alias}
            className={
              alias === "champion"
                ? "rounded-full bg-accent-soft px-2 py-0.5 text-xs font-semibold text-accent"
                : "rounded-full bg-primary-soft px-2 py-0.5 text-xs font-semibold text-primary-text"
            }
          >
            {alias}
          </span>
        ))}
      </div>
      <div className="font-semibold text-heading">
        {model.architecture}{" "}
        <span className="font-normal text-muted">
          · {model.name} v{model.version ?? "?"} · {model.precision.toUpperCase()} ·{" "}
          {model.input_size} px · {model.num_classes} {model.num_classes === 1 ? "class" : "classes"}
        </span>
      </div>
      <div className="flex flex-wrap gap-2">
        {headlineMetrics(model).map((metric) => (
          <Figure key={metric.label} label={metric.label} value={metric.value} hint="MLflow Model Registry" />
        ))}
      </div>
    </div>
  );
}

/** What the API serves, from GET /model/metadata (MLflow Model Registry, alias champion). */
export function ModelCardView({ detector, classifier }: { detector: ModelMetadata; classifier: ModelMetadata }) {
  return (
    <div className="grid gap-3 md:grid-cols-2">
      <ModelLine title="Detector" model={detector} />
      <ModelLine title="Classifier" model={classifier} />
    </div>
  );
}

export function ModelCard() {
  const metadata = useQuery((signal) => api.modelMetadata(signal), []);
  return (
    <section
      className="card mb-5 grid gap-3"
      aria-label="Models served"
      data-testid="model-card"
    >
      <header className="flex flex-wrap items-center gap-2">
        <Icon name="spark" size={18} />
        <h2 className="m-0 text-base">Models served (MLflow Registry)</h2>
        <span className="text-sm text-muted">ONNX Runtime on the API, GET /model/metadata</span>
      </header>
      {metadata.error && (
        <p className="m-0 text-sm text-warn">The model metadata is not available: {metadata.error}</p>
      )}
      {!metadata.data && !metadata.error && <p className="m-0 text-sm text-muted">Loading the models…</p>}
      {metadata.data && (
        <ModelCardView detector={metadata.data.detector} classifier={metadata.data.classifier} />
      )}
    </section>
  );
}
