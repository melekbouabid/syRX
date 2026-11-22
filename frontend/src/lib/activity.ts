// Labels of the activity log (GET /api/admin/activity): one sentence per event, built from
// its action and its few counts. The log never holds health data, so neither does this text.

import { messages } from "../i18n";
import type { ActivityEvent, ActivityGroup } from "../services/api";

// Texts in src/i18n (activity.*): read when called, so they follow the interface language.
export const GROUPS: { group: ActivityGroup; tone: string }[] = [
  { group: "access", tone: "active" },
  { group: "security", tone: "danger" },
  { group: "usage", tone: "info" },
  { group: "reports", tone: "admin" },
  { group: "account", tone: "neutral" },
  { group: "admin", tone: "warn" },
];

export function groupLabel(group: ActivityGroup): string {
  return messages().activity.groups[group] ?? group;
}

export const GROUP_TONE = Object.fromEntries(GROUPS.map((g) => [g.group, g.tone])) as Record<
  ActivityGroup,
  string
>;

export function actionOptions(): { value: string; label: string }[] {
  return Object.entries(messages().activity.actions).map(([value, label]) => ({ value, label }));
}

export function actionLabel(action: string): string {
  return messages().activity.actions[action] ?? action.replaceAll("_", " ");
}

/** The counts of an event, in words: "2 pills", "for 7 days", "fields: role". */
export function eventDetail(event: ActivityEvent): string {
  const d = event.details;
  const w = messages().activity;
  const n = (key: string) => (typeof d[key] === "number" ? (d[key] as number) : null);
  switch (event.action) {
    case "scan":
      return d.kind === "box" ? w.box : d.kind === "unknown" ? w.notAPill : w.pills(n("pills") ?? 0);
    case "interactions": {
      const found = n("pairs_with_interaction") ?? 0;
      return `${w.items(n("items") ?? 0)} · ${found ? w.pairsWith(found) : w.noKnown}`;
    }
    case "chat":
      return n("pills") ? w.about(w.pills(n("pills") ?? 0)) : "";
    case "toxicity":
      return w.molecules(n("molecules") ?? 0);
    case "medicine_save":
      return d.from_scan ? w.fromScan : w.fromList;
    case "report_create":
      return `${w.items(n("items") ?? 0)}${n("chat_answers") ? ` · ${w.answers(n("chat_answers") ?? 0)}` : ""}`;
    case "report_share":
      return w.linkValid(n("days") ?? 0);
    case "profile_update":
    case "user_update": {
      const fields = Array.isArray(d.fields)
        ? (d.fields as string[]).map((f) => w.fields[f] ?? f)
        : [];
      const extra =
        event.action === "user_update" && (d.fields as string[] | undefined)?.includes("is_active")
          ? d.is_active
            ? w.activated
            : w.deactivated
          : "";
      return fields.length ? `${fields.join(", ")}${extra}` : "";
    }
    case "user_create":
      return d.role === "admin" ? w.administrator : "";
    case "photo_update":
    case "signature_update":
      return d.removed ? w.removed : "";
    case "data_export":
      return `${String(d.table ?? "")} · ${w.rows(n("rows") ?? 0)}`;
    default:
      return "";
  }
}

/** A short browser name from the User-Agent, for the log table. */
export function browserName(agent: string | null): string {
  if (!agent) return "";
  const os = /Windows/.test(agent)
    ? "Windows"
    : /Android/.test(agent)
      ? "Android"
      : /iPhone|iPad/.test(agent)
        ? "iOS"
        : /Mac OS X/.test(agent)
          ? "macOS"
          : /Linux/.test(agent)
            ? "Linux"
            : "";
  const browser = /Edg\//.test(agent)
    ? "Edge"
    : /Firefox\//.test(agent)
      ? "Firefox"
      : /Chrome\//.test(agent)
        ? "Chrome"
        : /Safari\//.test(agent)
          ? "Safari"
          : /python|httpx|curl|testclient/i.test(agent)
            ? messages().activity.script
            : messages().activity.browser;
  return os ? `${browser} · ${os}` : browser;
}
