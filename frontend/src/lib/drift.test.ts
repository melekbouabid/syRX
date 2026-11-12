import { describe, expect, it } from "vitest";
import { driftStatusLabel, driftTone, formatStat, showDriftBanner } from "./drift";

describe("drift display", () => {
  it("labels and colours every status", () => {
    expect(driftStatusLabel("drift")).toBe("Drift");
    expect(driftTone("drift")).toBe("danger");
    expect(driftTone("ok")).toBe("active");
    expect(driftTone("something else")).toBe("neutral");
  });

  it("formats the statistics", () => {
    expect(formatStat(null)).toBe("n/a");
    expect(formatStat(0.123456)).toBe("0.123");
    expect(formatStat(142.7)).toBe("143");
  });

  it("shows the banner only on drift", () => {
    expect(showDriftBanner("drift")).toBe(true);
    expect(showDriftBanner("insufficient")).toBe(false);
    expect(showDriftBanner(undefined)).toBe(false);
  });
});
