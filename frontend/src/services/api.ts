// Typed client for the FastAPI backend: every call of the app goes through this file. Calls
// go to /api on the same origin, except the model routes of the course guide (/health,
// /model/metadata), which sit at the root; the session lives in an HttpOnly cookie, so there
// is no token to store or attach here.

import { timed } from "../lib/vision";

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

// --- Tujya: scan, pills, interactions (docs/api.md) ------------------------------------------

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

/** An active ingredient read on a medicine box (step 2.7), with where the link comes from. */
export interface BoxIngredient {
  name: string;
  /** brand table, ingredient name printed on the box, or RxNorm (RxNav). */
  method: "brand" | "printed_name" | "rxnav";
  matched_text: string;
  confidence: number;
  source: string;
  reference: string;
  url: string;
  /** False: a known ingredient that the interaction table does not list. */
  in_interaction_table: boolean;
}

export interface BoxReading {
  /** Lines read on the photo. */
  text: string[];
  brand: string | null;
  brand_country: string | null;
  brand_text: string | null;
  ingredients: BoxIngredient[];
  recognised: boolean;
  confidence: number;
  rxnav: string;
}

export interface ScanResult {
  /** pill: pills below; box: a medicine box read by OCR; unknown: neither (refused). */
  kind?: "pill" | "box" | "unknown";
  box?: BoxReading | null;
  image_width: number;
  image_height: number;
  inference_ms: number;
  timings: {
    detection_ms: number;
    classification_ms: number;
    /** ONNX Runtime only: detection plus identification. */
    total_ms: number;
    /** The whole request on the server (upload, decoding, database included). */
    server_ms?: number | null;
    /** Reading of the printed text (boxes). */
    ocr_ms?: number | null;
  };
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
  /** Grad-CAM precomputed for this photo (D104), null when there is none. */
  gradcam_url: string | null;
}

export interface DemoGallery {
  items: DemoImage[];
  note: string;
}

/** One served model, as GET /model/metadata describes it (MLflow Model Registry). */
export interface ModelMetadata {
  task: "detect" | "classify";
  name: string;
  architecture: string;
  version: number | null;
  aliases: string[];
  precision: string;
  variant_of: number | null;
  run_id: string | null;
  input_size: number;
  metrics: Record<string, number>;
  classes: string[];
  num_classes: number;
}

export interface MetadataOut {
  detector: ModelMetadata;
  classifier: ModelMetadata;
  published_at: string | null;
  disclaimer: string;
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
  doc_id?: string;
  license?: string;
  /** Cross-encoder score when the search was reranked (D117). */
  rerank_score?: number | null;
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

// Bonus toxicity tool (POST /api/toxicity, docs/bonus/toxicity.md): each molecule on its own.
export interface ToxPrediction {
  task: string;
  assay: string;
  probability: number;
  test_roc_auc: number | null;
}

export interface ToxMolecule {
  query: string;
  resolved_as: "ingredient" | "smiles";
  ingredient: string | null;
  smiles: string;
  predictions: ToxPrediction[];
  max_similarity_to_training: number;
  in_domain: boolean;
}

export interface ToxicityResult {
  wording: string;
  molecules: ToxMolecule[];
  errors: { query: string; error: string }[];
  mixture_note: string;
  low_probability_note: string;
  model: { name: string; version: string; tasks: number };
  source: { title: string; url: string; citation: string };
  disclaimer: string;
}

// Indexed documents of the RAG (GET /api/documents, docs/api.md, D114).
export interface IndexInfo {
  store: string;
  collection: string;
  points: number;
  embedding_model: string;
  dim: number;
  indexed_at: string;
  documents: number;
  chunks: number;
  payload_indexes: string[];
  dashboard_url: string | null;
}

export interface IndexedDocument {
  doc_id: string;
  doc_type: string;
  title: string;
  source_name: string;
  url: string;
  date: string;
  version: string;
  license: string;
  language: string;
  article_type: string | null;
  is_retracted: boolean;
  ingredients: string[];
  mentioned_ingredients: string[];
  sections: string[];
  chunk_count: number;
}

export interface DocumentsPage {
  total: number;
  offset: number;
  limit: number;
  items: IndexedDocument[];
  facets: Record<"doc_type" | "language" | "source", Record<string, number>>;
  index: IndexInfo;
}

export interface DocumentChunk {
  chunk_id: string;
  chunk_index: number;
  section: string;
  title: string;
  text: string;
  content_hash: string;
}

export interface DocumentDetail extends IndexedDocument {
  chunks: DocumentChunk[];
}

export interface DocumentFilters {
  q?: string;
  doc_type?: string;
  ingredient?: string;
  source?: string;
  language?: string;
  limit: number;
  offset: number;
}

// Data drift of the scanned photos (GET /monitoring/drift, docs/module1/derive.md, step 2.8).
export interface DriftFeature {
  feature: string;
  description: string;
  reference_n: number;
  current_n: number;
  reference_mean: number | null;
  current_mean: number | null;
  ks_statistic: number | null;
  ks_pvalue: number | null;
  psi: number | null;
  status: "insufficient" | "ok" | "warning" | "drift";
}

export interface DriftReport {
  status: "unavailable" | "insufficient" | "ok" | "warning" | "drift";
  window_size: number;
  window_max: number;
  photos_seen: number;
  drifted: string[];
  features: DriftFeature[];
  thresholds: Record<string, number>;
  reference: Record<string, unknown>;
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
  /** The path is at the root of the API (course guide routes), not under /api. */
  root?: boolean;
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
  const url = new URL(options.root ? path : `/api${path}`, window.location.origin);
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

  scan: (image: Blob, filename: string, signal?: AbortSignal) => {
    const form = new FormData();
    form.append("image", image, filename);
    return request<ScanResult>("POST", "/scan", { form, signal });
  },
  /** The scan plus the client latency measured around it in the browser (ms). */
  scanTimed: (image: Blob, filename: string, signal?: AbortSignal) =>
    timed(() => api.scan(image, filename, signal)),
  /** Champion models served by the API (course guide route, no sign-in). */
  modelMetadata: (signal?: AbortSignal) =>
    request<MetadataOut>("GET", "/model/metadata", { signal, root: true, anonymous: true }),
  pills: (q: string, signal?: AbortSignal) =>
    request<PillList>("GET", "/pills", { query: { q, limit: 12 }, signal }),
  interactions: (pills: number[], ingredients: string[] = []) =>
    request<InteractionsResult>("POST", "/interactions", { body: { pills, ingredients } }),
  demoImages: (signal?: AbortSignal) => request<DemoGallery>("GET", "/demo/images", { signal }),
  toxicity: (molecules: string[]) =>
    request<ToxicityResult>("POST", "/toxicity", { body: { molecules } }),
  chat: (question: string, pills: number[]) =>
    request<ChatResult>("POST", "/chat", { body: { question, pills, stream: false } }),
  chatStream,
  documents: (filters: DocumentFilters, signal?: AbortSignal) =>
    request<DocumentsPage>("GET", "/documents", { query: { ...filters }, signal }),
  document: (docId: string, signal?: AbortSignal) =>
    request<DocumentDetail>("GET", `/documents/${encodeURIComponent(docId)}`, { signal }),
  /** Data drift of the recent scans against the reference photos (no sign-in needed). */
  drift: (signal?: AbortSignal) =>
    request<DriftReport>("GET", "/monitoring/drift", { signal, root: true, anonymous: true }),
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
