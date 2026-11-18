import { describe, expect, it } from "vitest";
import type { ActivityEvent } from "../services/api";
import { niceTicks } from "../components/charts";
import { actionLabel, browserName, eventDetail } from "./activity";

function event(action: string, details: Record<string, unknown>): ActivityEvent {
  return {
    id: 1,
    created_at: "2026-11-18T08:00:00Z",
    user_id: 2,
    user_name: "Demo User",
    user_email: "user@gmail.com",
    action,
    group: "usage",
    target_id: null,
    target_email: null,
    details,
    ip: "127.0.0.1",
    user_agent: null,
  };
}

describe("activity labels", () => {
  it("names every action and falls back on unknown ones", () => {
    expect(actionLabel("scan")).toBe("Scanned a photo");
    expect(actionLabel("some_new_action")).toBe("some new action");
  });

  it("describes the counts of an event, never more", () => {
    expect(eventDetail(event("scan", { kind: "pill", pills: 2 }))).toBe("2 pills");
    expect(eventDetail(event("scan", { kind: "box", pills: 0 }))).toBe("medicine box");
    expect(eventDetail(event("interactions", { items: 3, pairs_with_interaction: 1 }))).toBe(
      "3 items · 1 pair with an interaction",
    );
    expect(eventDetail(event("interactions", { items: 2, pairs_with_interaction: 0 }))).toBe(
      "2 items · no known interaction",
    );
    expect(eventDetail(event("report_share", { days: 7 }))).toBe("link valid 7 days");
    expect(eventDetail(event("user_update", { fields: ["is_active"], is_active: false }))).toBe(
      "status (deactivated)",
    );
  });

  it("shortens a User-Agent", () => {
    const chrome =
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36";
    expect(browserName(chrome)).toBe("Chrome · Windows");
    expect(browserName(null)).toBe("");
  });
});

describe("chart ticks", () => {
  it("starts at zero and ends on a clean value above the maximum", () => {
    expect(niceTicks(0)).toEqual([0, 1]);
    expect(niceTicks(7)).toEqual([0, 2, 4, 6, 8]);
    expect(niceTicks(130)).toEqual([0, 50, 100, 150]);
  });
});
