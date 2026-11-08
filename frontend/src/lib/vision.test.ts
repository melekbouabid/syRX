import { describe, expect, it, vi } from "vitest";
import {
  classColour,
  containRect,
  formatMs,
  headlineMetrics,
  percent,
  scaleBox,
  startLiveLoop,
  textOn,
  timed,
  transferMs,
} from "./vision";

describe("classColour", () => {
  it("gives the colours of POST /predict/visualize (backend class_colour)", () => {
    // Same values as backend/tests/test_vision.py.
    expect(classColour(0)).toBe("#ff3838");
    expect(classColour(9)).toBe("#00d4bb");
    expect(classColour(120)).toBe(classColour(0)); // 120 mod 20
    expect(classColour(-1)).toBe(classColour(19));
  });

  it("is stable and differs between neighbouring classes", () => {
    expect(classColour(42)).toBe(classColour(42));
    expect(classColour(42)).not.toBe(classColour(43));
    expect(new Set(Array.from({ length: 20 }, (_, i) => classColour(i))).size).toBe(20);
  });

  it("picks a readable label colour", () => {
    expect(textOn(classColour(4))).toBe("#000000"); // light yellow-green
    expect(textOn(classColour(16))).toBe("#ffffff"); // dark violet
  });
});

describe("box geometry", () => {
  it("letterboxes like object-fit: contain", () => {
    // A 4:3 photo in a 16:9 view: full height, centred horizontally.
    expect(containRect({ width: 400, height: 300 }, { width: 1600, height: 900 })).toEqual({
      x: 200,
      y: 0,
      width: 1200,
      height: 900,
    });
    // Same proportions: fills the view.
    expect(containRect({ width: 640, height: 480 }, { width: 320, height: 240 })).toEqual({
      x: 0,
      y: 0,
      width: 320,
      height: 240,
    });
  });

  it("scales a normalised box onto the drawn image and clamps it", () => {
    const frame = { x: 200, y: 0, width: 1200, height: 900 };
    expect(scaleBox({ x1: 0.25, y1: 0.5, x2: 0.75, y2: 1 }, frame)).toEqual({
      x: 500,
      y: 450,
      width: 600,
      height: 450,
    });
    expect(scaleBox({ x1: -0.1, y1: 0, x2: 1.2, y2: 0.1 }, frame)).toEqual({
      x: 200,
      y: 0,
      width: 1200,
      height: 90,
    });
  });

  it("falls back to the whole view before the image size is known", () => {
    expect(containRect({ width: 0, height: 0 }, { width: 50, height: 40 })).toEqual({
      x: 0,
      y: 0,
      width: 50,
      height: 40,
    });
  });
});

describe("latency", () => {
  it("measures the client time around the call", async () => {
    const clock = [100, 342.6];
    const result = await timed(async () => "answer", () => clock.shift() ?? 0);
    expect(result.value).toBe("answer");
    expect(result.clientMs).toBeCloseTo(242.6, 6);
  });

  it("splits client time into server and transfer", () => {
    expect(transferMs({ clientMs: 250, serverMs: 180, inferenceMs: 120 })).toBe(70);
    expect(transferMs({ clientMs: 100, serverMs: 120, inferenceMs: 90 })).toBe(0); // clocks
    expect(transferMs({ clientMs: 100, serverMs: null, inferenceMs: null })).toBeNull();
  });

  it("formats milliseconds and percentages", () => {
    expect(formatMs(212.4)).toBe("212 ms");
    expect(formatMs(null)).toBe("–");
    expect(formatMs(Number.NaN)).toBe("–");
    expect(percent(0.8549)).toBe("85%");
  });
});

describe("live webcam loop", () => {
  it("sends one frame at a time and waits for each answer", async () => {
    vi.useFakeTimers();
    try {
      let inFlight = 0;
      let maxInFlight = 0;
      const answers: number[] = [];
      const resolvers: ((value: number) => void)[] = [];
      const stop = startLiveLoop<number>({
        step: () => {
          inFlight += 1;
          maxInFlight = Math.max(maxInFlight, inFlight);
          return new Promise((resolve) => resolvers.push(resolve));
        },
        onResult: (value) => {
          inFlight -= 1;
          answers.push(value);
        },
        onError: () => undefined,
        pauseMs: 400,
      });
      await vi.advanceTimersByTimeAsync(5_000); // a slow server: no answer for 5 s
      expect(resolvers).toHaveLength(1); // still the first request, nothing queued
      resolvers[0](1);
      await vi.advanceTimersByTimeAsync(399);
      expect(resolvers).toHaveLength(1); // the pause after the answer
      await vi.advanceTimersByTimeAsync(1);
      expect(resolvers).toHaveLength(2);
      resolvers[1](2);
      await vi.advanceTimersByTimeAsync(400);
      stop();
      await vi.advanceTimersByTimeAsync(5_000);
      expect(answers).toEqual([1, 2]);
      expect(maxInFlight).toBe(1);
      expect(resolvers).toHaveLength(3); // the third was aborted by stop
    } finally {
      vi.useRealTimers();
    }
  });

  it("stops on an error and ignores answers after stop", async () => {
    vi.useFakeTimers();
    try {
      const errors: unknown[] = [];
      const results: string[] = [];
      let calls = 0;
      startLiveLoop<string>({
        step: async () => {
          calls += 1;
          throw new Error("503");
        },
        onResult: (r) => results.push(r),
        onError: (e) => errors.push(e),
        pauseMs: 10,
      });
      await vi.advanceTimersByTimeAsync(1_000);
      expect(calls).toBe(1);
      expect(errors).toHaveLength(1);

      let release: (value: string) => void = () => undefined;
      const stop = startLiveLoop<string>({
        step: (signal) =>
          new Promise((resolve) => {
            release = resolve;
            expect(signal.aborted).toBe(false);
          }),
        onResult: (r) => results.push(r),
        onError: (e) => errors.push(e),
        pauseMs: 10,
      });
      stop();
      release("late");
      await vi.advanceTimersByTimeAsync(100);
      expect(results).toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });
});

describe("headlineMetrics", () => {
  it("shows the promotion metric of each task and the latency", () => {
    expect(
      headlineMetrics({
        task: "detect",
        metrics: { test_v2_mAP50_95: 0.88846, test_v2_mAP50: 0.97394, onnx_cpu_latency_ms: 62.97 },
      }),
    ).toEqual([
      { label: "test mAP50-95", value: "0.888" },
      { label: "test mAP50", value: "0.974" },
      { label: "CPU latency", value: "63 ms" },
    ]);
    expect(headlineMetrics({ task: "classify", metrics: { test_top1: 0.93707, test_top5: 0.98469 } })).toEqual([
      { label: "test top-1", value: "93.7%" },
      { label: "test top-5", value: "98.5%" },
    ]);
  });

  it("falls back to older split names and skips what is missing", () => {
    expect(headlineMetrics({ task: "detect", metrics: { val_mAP50_95: 0.8677 } })).toEqual([
      { label: "val mAP50-95", value: "0.868" },
    ]);
    expect(headlineMetrics({ task: "classify", metrics: {} })).toEqual([]);
  });
});
