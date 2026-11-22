// Display helpers of the Monitoring page and the drift banner (GET /monitoring/drift, step 2.8).
import { messages } from "../i18n";

export type DriftStatus = "unavailable" | "insufficient" | "ok" | "warning" | "drift";

const TONES: Record<DriftStatus, string> = {
  unavailable: "neutral",
  insufficient: "neutral",
  ok: "active",
  warning: "warn",
  drift: "danger",
};

export function driftStatusLabel(status: string): string {
  return messages().drift[status] ?? status;
}

/** Badge tone (styles.css .badge-*) of a status. */
export function driftTone(status: string): string {
  return TONES[status as DriftStatus] ?? "neutral";
}

/** A statistic with 3 significant digits, "n/a" when the test did not run. */
export function formatStat(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return messages().common.na;
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toPrecision(3);
}

/** The banner of the Scanner page shows only a real drift, never a lack of data. */
export function showDriftBanner(status: string | undefined): boolean {
  return status === "drift";
}
