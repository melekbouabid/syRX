// Labels of the activity log (GET /api/admin/activity): one sentence per event, built from
// its action and its few counts. The log never holds health data, so neither does this text.

import type { ActivityEvent, ActivityGroup } from "../services/api";

export const GROUPS: { group: ActivityGroup; label: string; tone: string }[] = [
  { group: "access", label: "Sign-ins", tone: "active" },
  { group: "security", label: "Security", tone: "danger" },
  { group: "usage", label: "Use of the app", tone: "info" },
  { group: "reports", label: "Reports", tone: "admin" },
  { group: "account", label: "Profile changes", tone: "neutral" },
  { group: "admin", label: "Administration", tone: "warn" },
];

export const GROUP_LABEL = Object.fromEntries(GROUPS.map((g) => [g.group, g.label])) as Record<
  ActivityGroup,
  string
>;
export const GROUP_TONE = Object.fromEntries(GROUPS.map((g) => [g.group, g.tone])) as Record<
  ActivityGroup,
  string
>;

const ACTIONS: Record<string, string> = {
  sign_up: "Created an account",
  sign_in: "Signed in",
  sign_in_failed: "Failed sign-in (wrong password)",
  sign_in_blocked: "Sign-in refused (account deactivated)",
  sign_out: "Signed out",
  profile_update: "Updated their details",
  photo_update: "Changed their photo",
  signature_update: "Changed their signature",
  password_change: "Changed their password",
  scan: "Scanned a photo",
  interactions: "Checked interactions",
  chat: "Asked the assistant",
  toxicity: "Ran a toxicity prediction",
  medicine_save: "Saved a medicine",
  report_create: "Saved a report",
  report_share: "Shared a report",
  report_unshare: "Stopped sharing a report",
  report_delete: "Deleted a report",
  user_create: "Created an account",
  user_update: "Edited an account",
  password_reset: "Reset a password",
  user_delete: "Deleted an account",
  drift_reset: "Reset the drift window",
  data_export: "Exported data (CSV)",
};

export const ACTION_OPTIONS = Object.entries(ACTIONS).map(([value, label]) => ({ value, label }));

export function actionLabel(action: string): string {
  return ACTIONS[action] ?? action.replaceAll("_", " ");
}

const FIELD: Record<string, string> = {
  full_name: "name",
  email: "email",
  role: "role",
  is_active: "status",
  phone: "phone",
};

function plural(count: number, word: string): string {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/** The counts of an event, in words: "2 pills", "for 7 days", "fields: role". */
export function eventDetail(event: ActivityEvent): string {
  const d = event.details;
  const n = (key: string) => (typeof d[key] === "number" ? (d[key] as number) : null);
  switch (event.action) {
    case "scan":
      return d.kind === "box"
        ? "medicine box"
        : d.kind === "unknown"
          ? "not a pill (refused)"
          : plural(n("pills") ?? 0, "pill");
    case "interactions": {
      const found = n("pairs_with_interaction") ?? 0;
      return `${plural(n("items") ?? 0, "item")} · ${found ? plural(found, "pair") + " with an interaction" : "no known interaction"}`;
    }
    case "chat":
      return n("pills") ? `about ${plural(n("pills") ?? 0, "pill")}` : "";
    case "toxicity":
      return plural(n("molecules") ?? 0, "molecule");
    case "medicine_save":
      return d.from_scan ? "from a scan" : "from the list";
    case "report_create":
      return `${plural(n("items") ?? 0, "item")}${n("chat_answers") ? ` · ${plural(n("chat_answers") ?? 0, "answer")}` : ""}`;
    case "report_share":
      return `link valid ${plural(n("days") ?? 0, "day")}`;
    case "profile_update":
    case "user_update": {
      const fields = Array.isArray(d.fields) ? (d.fields as string[]).map((f) => FIELD[f] ?? f) : [];
      const extra =
        event.action === "user_update" && (d.fields as string[] | undefined)?.includes("is_active")
          ? d.is_active
            ? " (activated)"
            : " (deactivated)"
          : "";
      return fields.length ? `${fields.join(", ")}${extra}` : "";
    }
    case "user_create":
      return d.role === "admin" ? "administrator" : "";
    case "photo_update":
    case "signature_update":
      return d.removed ? "removed" : "";
    case "data_export":
      return `${String(d.table ?? "")} · ${plural(n("rows") ?? 0, "row")}`;
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
            ? "Script"
            : "Browser";
  return os ? `${browser} · ${os}` : browser;
}
