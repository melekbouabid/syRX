// Pure helpers of the vision part of the Scanner page (course guide, step 2.6): class
// colours, box geometry on a canvas, latency figures and the live webcam loop. No React and
// no DOM here, so src/lib/vision.test.ts checks them in Node (Vitest).

/** A box with corners in 0-1 of the image width and height (`box_normalized` of the API). */
export interface NormBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** A rectangle in canvas pixels. */
export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** The 20 colours of Ultralytics' own plots, as in backend/app/routers/vision.py PALETTE. */
export const PALETTE = [
  "#ff3838", "#ff9d97", "#ff701f", "#ffb21d", "#cfd231", "#48f90a", "#92cc17",
  "#3ddb86", "#1a9334", "#00d4bb", "#2c99a8", "#00c2ff", "#344593", "#6473ff",
  "#0018ec", "#8438ff", "#520085", "#cb38ff", "#ff95c8", "#ff37c7",
];

/**
 * A stable colour per class, the same as the boxes of POST /predict/visualize
 * (backend/app/routers/vision.py `class_colour`): colour class_id mod 20 of the palette.
 */
export function classColour(classId: number): string {
  const n = PALETTE.length;
  return PALETTE[((Math.trunc(classId) % n) + n) % n];
}

/** Black or white, whichever reads better on `background` (#rrggbb). */
export function textOn(background: string): "#000000" | "#ffffff" {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(background.slice(i, i + 2), 16) / 255);
  const luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  return luminance > 0.5 ? "#000000" : "#ffffff";
}

/**
 * Where an image of `natural` size is drawn inside a `view` box with `object-fit: contain`
 * (centred, scaled to fit, letterboxed). A video element always draws its frames this way.
 */
export function containRect(
  natural: { width: number; height: number },
  view: { width: number; height: number },
): Rect {
  if (natural.width <= 0 || natural.height <= 0) return { x: 0, y: 0, ...view };
  const scale = Math.min(view.width / natural.width, view.height / natural.height);
  const width = natural.width * scale;
  const height = natural.height * scale;
  return { x: (view.width - width) / 2, y: (view.height - height) / 2, width, height };
}

/** A normalised box in the pixels of the drawn image (`frame`, from containRect). */
export function scaleBox(box: NormBox, frame: Rect): Rect {
  const clamp = (v: number) => Math.min(1, Math.max(0, v));
  const x1 = clamp(box.x1);
  const y1 = clamp(box.y1);
  return {
    x: frame.x + x1 * frame.width,
    y: frame.y + y1 * frame.height,
    width: (clamp(box.x2) - x1) * frame.width,
    height: (clamp(box.y2) - y1) * frame.height,
  };
}

export function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** Milliseconds, rounded ("212 ms"), as the course guide asks; "–" when unknown. */
export function formatMs(ms: number | null | undefined): string {
  if (ms === null || ms === undefined || !Number.isFinite(ms)) return "–";
  return `${Math.round(ms)} ms`;
}

export interface Latency {
  /** Measured in the browser: from sending the request to having the parsed answer. */
  clientMs: number;
  /** Returned by the API: the whole request on the server (`server_ms`). */
  serverMs: number | null;
  /** ONNX Runtime only (detection + identification), part of the server time. */
  inferenceMs: number | null;
}

/** Time spent outside the server (upload, download, browser): client minus server, >= 0. */
export function transferMs(latency: Latency): number | null {
  if (latency.serverMs === null) return null;
  return Math.max(0, latency.clientMs - latency.serverMs);
}

/** Run `call` and measure it with `now` (performance.now in the browser). */
export async function timed<T>(
  call: () => Promise<T>,
  now: () => number = () => performance.now(),
): Promise<{ value: T; clientMs: number }> {
  const started = now();
  const value = await call();
  return { value, clientMs: now() - started };
}

export interface LiveLoopOptions<T> {
  /** One step: grab a frame and send it; resolves with the answer. */
  step: (signal: AbortSignal) => Promise<T | null>;
  onResult: (result: T) => void;
  onError: (error: unknown) => void;
  /** Pause after each answer before the next frame, in ms. */
  pauseMs: number;
  setTimer?: (callback: () => void, ms: number) => unknown;
  clearTimer?: (handle: unknown) => void;
}

/**
 * Continuous inference for the webcam, one request at a time: the next frame is taken only
 * after the previous answer arrived (plus `pauseMs`), so requests never pile up when the
 * server is slower than the camera. Returns `stop`, which aborts the request in flight.
 * An error stops the loop (the page shows it and lets the user restart).
 */
export function startLiveLoop<T>(options: LiveLoopOptions<T>): () => void {
  const setTimer = options.setTimer ?? ((cb, ms) => globalThis.setTimeout(cb, ms));
  const clearTimer =
    options.clearTimer ??
    ((handle) => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>));
  const controller = new AbortController();
  let timer: unknown = null;

  const tick = async () => {
    timer = null;
    if (controller.signal.aborted) return;
    try {
      const result = await options.step(controller.signal);
      if (controller.signal.aborted) return;
      if (result !== null) options.onResult(result);
    } catch (error) {
      if (!controller.signal.aborted) options.onError(error);
      return;
    }
    timer = setTimer(() => void tick(), options.pauseMs);
  };
  void tick();
  return () => {
    controller.abort();
    if (timer !== null) clearTimer(timer);
  };
}

export interface Headline {
  label: string;
  value: string;
}

/**
 * The scores worth showing for a served model (GET /model/metadata `metrics`): the test
 * metric its promotion used, then the CPU latency of ONNX Runtime when it was measured.
 */
export function headlineMetrics(model: {
  task: "detect" | "classify";
  metrics: Record<string, number>;
}): Headline[] {
  const m = model.metrics;
  const out: Headline[] = [];
  const score = (label: string, key: string, asPercent: boolean) => {
    const value = m[key];
    if (value === undefined || !Number.isFinite(value)) return false;
    out.push({ label, value: asPercent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3) });
    return true;
  };
  if (model.task === "detect") {
    // Promotion metric of step 2.2 (D93), else the v1 test split, else validation.
    const found =
      score("test mAP50-95", "test_v2_mAP50_95", false) ||
      score("test mAP50-95", "test_mAP50_95", false);
    if (!found) score("val mAP50-95", "val_mAP50_95", false);
    if (!score("test mAP50", "test_v2_mAP50", false)) score("test mAP50", "test_mAP50", false);
  } else {
    // Promotion metric of step 2.3 (comparison split of C3PI v2, D100), else the v1 test split.
    if (!score("test top-1", "test_v2_top1", true)) score("test top-1", "test_top1", true);
    if (!score("test top-5", "test_v2_top5", true)) score("test top-5", "test_top5", true);
  }
  if (m.onnx_cpu_latency_ms !== undefined) {
    out.push({ label: "CPU latency", value: formatMs(m.onnx_cpu_latency_ms) });
  }
  return out;
}
