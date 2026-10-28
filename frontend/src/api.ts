// Typed client for the FastAPI backend. Every call goes to /api on the same origin; the
// session lives in an HttpOnly cookie, so there is no token to store or attach here.

export type Role = "admin" | "user";

export interface User {
  id: number;
  email: string;
  full_name: string;
  role: Role;
  is_active: boolean;
  created_at: string;
  last_login_at: string | null;
}

export interface Page<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface AdminStats {
  users_total: number;
  users_active: number;
  users_inactive: number;
  admins: number;
  signups_last_7_days: number;
  active_sessions: number;
}

export interface CatalogSummary {
  drugs: number | null;
  cell_lines: number | null;
  synergy_measurements: number | null;
  abstracts: number | null;
}

export interface Drug {
  drug_id: string;
  name: string;
  synonyms: string[];
  pubchem_cid: number | null;
  chembl_id: string | null;
  smiles: string | null;
  sources: string[];
  n_measurements: number;
}

export interface CellLine {
  cell_line_id: string;
  name: string;
  tissue: string | null;
  disease: string | null;
  depmap_id: string | null;
  sources: string[];
  n_measurements: number;
}

export interface UserFilters {
  q?: string;
  role?: Role | "";
  status?: "active" | "inactive" | "";
  page?: number;
  page_size?: number;
}

export interface NewUser {
  email: string;
  full_name: string;
  password: string;
  role: Role;
  is_active: boolean;
}

export type UserChanges = Partial<Pick<User, "email" | "full_name" | "role" | "is_active">>;

// --- Tujya: scan, pills, interactions (docs/API.md) ------------------------------------------

export interface Source {
  name: string;
  title: string;
  url: string;
  license: string;
  citation: string;
}

export interface PillCandidate {
  class_id: number;
  name: string;
  generic_name: string;
  active_ingredients: string[];
  probability: number;
}

export interface Box {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ScannedPill {
  box: Box;
  box_normalized: Box;
  detection_confidence: number;
  candidates: PillCandidate[];
  needs_confirmation: boolean;
}

export interface ScanResult {
  image_width: number;
  image_height: number;
  inference_ms: number;
  timings: { detection_ms: number; classification_ms: number; total_ms: number };
  confirm_threshold: number;
  pills: ScannedPill[];
  message: string | null;
  models: Record<string, { registered_model?: string; version?: number } | string>;
  sources: Source[];
  disclaimer: string;
}

export interface Pill {
  class_id: number;
  name: string;
  generic_name: string;
  labeled_by: string;
  imprint: string;
  color: string;
  shape: string;
  active_ingredients: string[];
}

export interface PillList {
  items: Pill[];
  total: number;
  sources: Source[];
}

export type Severity = "Major" | "Moderate" | "Minor" | "Unknown";

export interface CheckedItem {
  kind: "pill" | "ingredient";
  label: string;
  class_id: number | null;
  found: boolean;
  active_ingredients: string[];
}

export interface InteractionFound {
  ingredient_a: string;
  ingredient_b: string;
  severity: Severity;
  description: string;
  source: Source;
}

export interface InteractionPair {
  item_a: CheckedItem;
  item_b: CheckedItem;
  highest_severity: Severity | null;
  interactions: InteractionFound[];
  shared_ingredients: string[];
  message: string | null;
}

export interface InteractionsResult {
  items: CheckedItem[];
  pairs: InteractionPair[];
  highest_severity: Severity | null;
  message: string;
  sources: Source[];
  disclaimer: string;
}

export interface DemoImage {
  id: number;
  class_id: number;
  expected_name: string;
  why: string;
  url: string;
  thumbnail_url: string;
}

export interface DemoGallery {
  items: DemoImage[];
  note: string;
}

export interface ChatCitation {
  n: number;
  title: string;
  url: string;
  section: string;
  snippet: string;
  source_name: string;
  doc_type: "label" | "abstract" | "interaction" | string;
  date: string;
  cited: boolean;
}

export interface ChatToolCall {
  name: string;
  args: Record<string, unknown>;
  chosen_by: "llm" | "rule" | string;
  ok: boolean;
  summary: string;
}

export interface ChatResult {
  question: string;
  answer: string;
  language: string;
  ingredients: string[];
  citations: ChatCitation[];
  tool_calls: ChatToolCall[];
  safety: { action?: "ok" | "rewritten" | "refused"; violations?: string[]; cited?: number[] };
  model: string;
  timings: Record<string, number>;
  disclaimer: string;
}

/** One Server-Sent Event of POST /api/chat with `stream: true`. */
export type ChatEvent =
  | { type: "tool"; tool: ChatToolCall }
  | { type: "token"; text: string }
  | { type: "final"; result: ChatResult }
  | { type: "error"; detail: string };

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : "Something went wrong.";
}

// Called when a request made as a signed-in user comes back 401: the session has ended.
let onSessionEnded: (() => void) | null = null;

export function setSessionEndedHandler(handler: (() => void) | null): void {
  onSessionEnded = handler;
}

type QueryParams = Record<string, string | number | undefined>;

interface RequestOptions {
  body?: unknown;
  /** Multipart body (file upload); the browser sets its Content-Type and boundary. */
  form?: FormData;
  query?: QueryParams;
  signal?: AbortSignal;
  /** A 401 is an expected answer here (sign-in form, first load), not a lost session. */
  anonymous?: boolean;
}

const FIELD_LABELS: Record<string, string> = {
  email: "Email",
  full_name: "Name",
  password: "Password",
  new_password: "New password",
  current_password: "Current password",
};

/** Turn a FastAPI error body into one readable sentence. */
function describeFailure(status: number, data: unknown): string {
  const detail = (data as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((issue: { loc?: unknown[]; msg?: string }) => {
        const field = String(issue.loc?.at(-1) ?? "");
        const message = (issue.msg ?? "is invalid").replace(/^Value error, /, "");
        return `${FIELD_LABELS[field] ?? field}: ${message}`;
      })
      .join(" ");
  }
  if (status >= 500) return "The server is not responding. Check that the API is running.";
  return `Request failed (${status}).`;
}

async function request<T>(method: string, path: string, options: RequestOptions = {}): Promise<T> {
  const url = new URL(`/api${path}`, window.location.origin);
  for (const [key, value] of Object.entries(options.query ?? {})) {
    if (value !== undefined && value !== "") url.searchParams.set(key, String(value));
  }

  let response: Response;
  try {
    response = await fetch(url, {
      method,
      credentials: "same-origin",
      headers: options.body === undefined ? undefined : { "Content-Type": "application/json" },
      body: options.form ?? (options.body === undefined ? undefined : JSON.stringify(options.body)),
      signal: options.signal,
    });
  } catch (error) {
    if (options.signal?.aborted) throw error;
    throw new ApiError(0, "Cannot reach the server. Check that the API is running.");
  }

  if (response.status === 204) return undefined as T;
  const data: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    if (response.status === 401 && !options.anonymous) onSessionEnded?.();
    throw new ApiError(response.status, describeFailure(response.status, data));
  }
  return data as T;
}

/**
 * Ask the Tujya agent; the answer streams as Server-Sent Events (tool calls, then the
 * checked sentences, then the final result, which is the one to display).
 */
async function chatStream(
  question: string,
  pills: number[],
  onEvent: (event: ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  let response: Response;
  try {
    response = await fetch(new URL("/api/chat", window.location.origin), {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify({ question, pills, stream: true }),
      signal,
    });
  } catch (error) {
    if (signal?.aborted) throw error;
    throw new ApiError(0, "Cannot reach the server. Check that the API is running.");
  }
  if (!response.ok || !response.body) {
    const data: unknown = await response.json().catch(() => null);
    if (response.status === 401) onSessionEnded?.();
    throw new ApiError(response.status, describeFailure(response.status, data));
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end = buffer.indexOf("\n\n");
    while (end !== -1) {
      const block = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const name = /^event: (.+)$/m.exec(block)?.[1];
      const data = /^data: (.+)$/m.exec(block)?.[1];
      if (name && data) {
        const payload = JSON.parse(data);
        if (name === "final") onEvent({ type: "final", result: payload as ChatResult });
        else if (name === "tool") onEvent({ type: "tool", tool: payload.tool });
        else if (name === "token") onEvent({ type: "token", text: payload.text });
        else if (name === "error") onEvent({ type: "error", detail: payload.detail });
      }
      end = buffer.indexOf("\n\n");
    }
  }
}

export const api = {
  me: () => request<User>("GET", "/auth/me", { anonymous: true }),
  signIn: (email: string, password: string) =>
    request<User>("POST", "/auth/login", { body: { email, password }, anonymous: true }),
  signUp: (account: { email: string; full_name: string; password: string }) =>
    request<User>("POST", "/auth/signup", { body: account, anonymous: true }),
  signOut: () => request<void>("POST", "/auth/logout", { anonymous: true }),
  updateProfile: (full_name: string) =>
    request<User>("PATCH", "/auth/me", { body: { full_name } }),
  changePassword: (current_password: string, new_password: string) =>
    request<void>("POST", "/auth/me/password", { body: { current_password, new_password } }),

  // Public: the introduction page shows these counts before anyone has signed in.
  catalogSummary: (signal?: AbortSignal) =>
    request<CatalogSummary>("GET", "/public/summary", { signal, anonymous: true }),
  drugs: (query: QueryParams, signal?: AbortSignal) =>
    request<Page<Drug>>("GET", "/catalog/drugs", { query, signal }),
  cellLines: (query: QueryParams, signal?: AbortSignal) =>
    request<Page<CellLine>>("GET", "/catalog/cell-lines", { query, signal }),

  scan: (image: Blob, filename: string, signal?: AbortSignal) => {
    const form = new FormData();
    form.append("image", image, filename);
    return request<ScanResult>("POST", "/scan", { form, signal });
  },
  pills: (q: string, signal?: AbortSignal) =>
    request<PillList>("GET", "/pills", { query: { q, limit: 12 }, signal }),
  interactions: (pills: number[], ingredients: string[] = []) =>
    request<InteractionsResult>("POST", "/interactions", { body: { pills, ingredients } }),
  demoImages: (signal?: AbortSignal) => request<DemoGallery>("GET", "/demo/images", { signal }),
  chat: (question: string, pills: number[]) =>
    request<ChatResult>("POST", "/chat", { body: { question, pills, stream: false } }),
  chatStream,
  /** A demo photo as a file, to send it to the scan like an upload. */
  demoImage: async (image: DemoImage): Promise<Blob> => {
    const response = await fetch(image.url, { credentials: "same-origin" });
    if (!response.ok) throw new ApiError(response.status, "Cannot load this demo photo.");
    return response.blob();
  },

  admin: {
    stats: (signal?: AbortSignal) => request<AdminStats>("GET", "/admin/stats", { signal }),
    users: (filters: UserFilters, signal?: AbortSignal) =>
      request<Page<User>>("GET", "/admin/users", { query: { ...filters }, signal }),
    createUser: (account: NewUser) => request<User>("POST", "/admin/users", { body: account }),
    updateUser: (id: number, changes: UserChanges) =>
      request<User>("PATCH", `/admin/users/${id}`, { body: changes }),
    resetPassword: (id: number, new_password: string) =>
      request<void>("POST", `/admin/users/${id}/password`, { body: { new_password } }),
    deleteUser: (id: number) => request<void>("DELETE", `/admin/users/${id}`),
  },
};
