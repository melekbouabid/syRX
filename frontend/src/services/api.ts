// Typed client for the FastAPI backend: every call of the app goes through this file. Calls
// go to /api on the same origin, except the model routes of the course guide (/health,
// /model/metadata), which sit at the root; the session lives in an HttpOnly cookie, so there
// is no token to store or attach here.

import { messages } from "../i18n";
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
  phone?: string | null;
  /** Address of the profile photo, null when none. */
  photo_url?: string | null;
  /** Own account only: administrators never receive it. */
  signature_url?: string | null;
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
  scans_last_7_days: number;
  questions_last_7_days: number;
  reports_total: number;
  failed_sign_ins_last_7_days: number;
}

// --- administration: activity log and daily usage ---------------------------------------------

export type ActivityGroup = "access" | "security" | "usage" | "reports" | "account" | "admin";

export interface ActivityEvent {
  id: number;
  created_at: string;
  user_id: number;
  user_name: string;
  user_email: string;
  action: string;
  group: ActivityGroup;
  target_id: number | null;
  target_email: string | null;
  details: Record<string, unknown>;
  ip: string | null;
  user_agent: string | null;
}

export interface ActivityPage extends Page<ActivityEvent> {
  counts: Record<ActivityGroup, number>;
}

export interface ActivityFilters {
  user_id?: number;
  group?: ActivityGroup | "";
  action?: string;
  q?: string;
  days?: number;
  page?: number;
  page_size?: number;
}

export interface UsageDay {
  day: string;
  sign_ins: number;
  sign_ups: number;
  scans: number;
  interactions: number;
  questions: number;
  toxicity: number;
  reports: number;
  active_users: number;
}

export interface Usage {
  days: UsageDay[];
  totals: Record<string, number>;
  top_users: { user_id: number; name: string; email: string; actions: number }[];
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

// --- my medicines and reports -----------------------------------------------------------------

export interface SavedMedicine {
  class_id: number;
  name: string;
  generic_name: string;
  active_ingredients: string[];
  found: boolean;
  probability: number | null;
  added_at: string;
}

export interface SavedMedicines {
  items: SavedMedicine[];
  limit: number;
}

export interface ChatExcerpt {
  question: string;
  answer: string;
  citations: { n: number; title: string; url: string }[];
}

export interface ShareState {
  active: boolean;
  expires_at: string;
  views: number;
}

export interface ReportSummary {
  id: number;
  created_at: string;
  title: string;
  pill_names: string[];
  ingredients: string[];
  highest_severity: Severity | null;
  pairs_with_interaction: number;
  chat_answers: number;
  share: ShareState | null;
}

export interface ReportBody {
  id: number;
  created_at: string;
  title: string;
  pills: { class_id: number; name: string; generic_name: string; active_ingredients: string[] }[];
  ingredients: string[];
  interactions: InteractionsResult | null;
  chat: ChatExcerpt[];
  highest_severity: Severity | null;
  author: string;
  signature: string | null;
  disclaimer: string;
}

export interface Report extends ReportBody {
  share: ShareState | null;
}

export interface NewReport {
  title: string;
  pills: number[];
  ingredients: string[];
  chat: (ChatExcerpt & { seal: string })[];
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

/** GET /api/catalog/summary: public figures and one real scan of a test photo (D128). */
export interface CatalogSummary {
  counts: { pills: number; ingredients: number; interaction_pairs: number } | null;
  confirm_threshold: number;
  disclaimer: string;
}

/** GET /api/catalog/showcase: the served models' top-1 on one C3PI test photo. */
export interface Showcase {
  photo_id: number;
  pill_name: string;
  active_ingredients: string[];
  probability: number;
  model: string;
  model_version: number | null;
  interaction: { ingredient_a: string; ingredient_b: string; severity: Severity; source: string } | null;
}

export interface ShowcaseOut {
  showcase: Showcase | null;
  confirm_threshold: number;
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
  /** Server seal: needed to add this exact answer to a saved report. */
  seal?: string;
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
  return error instanceof Error ? error.message : messages().api.failed;
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


/** Turn a FastAPI error body into one readable sentence. */
function describeFailure(status: number, data: unknown): string {
  const detail = (data as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((issue: { loc?: unknown[]; msg?: string }) => {
        const field = String(issue.loc?.at(-1) ?? "");
        const message = (issue.msg ?? messages().api.invalid).replace(/^Value error, /, "");
        return `${messages().api.fields[field] ?? field}: ${message}`;
      })
      .join(" ");
  }
  if (status >= 500) return messages().api.serverDown;
  return messages().api.requestFailed(status);
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
    throw new ApiError(0, messages().api.unreachable);
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
    throw new ApiError(0, messages().api.unreachable);
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
  /** `phone` undefined keeps it, "" removes it. */
  updateProfile: (full_name: string, phone?: string) =>
    request<User>("PATCH", "/auth/me", { body: { full_name, phone } }),
  setImage: (kind: "photo" | "signature", image: Blob, filename = "image.png") => {
    const form = new FormData();
    form.append("image", image, filename);
    return request<User>("PUT", `/auth/me/${kind}`, { form });
  },
  deleteImage: (kind: "photo" | "signature") => request<User>("DELETE", `/auth/me/${kind}`),
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
  /** Public: catalogue figures and one real scan of a demo photo, for the start page. */
  catalogSummary: (signal?: AbortSignal) =>
    request<CatalogSummary>("GET", "/catalog/summary", { signal, anonymous: true }),
  /** Public: the served models' result on one demo photo (null when not available). */
  catalogShowcase: (signal?: AbortSignal) =>
    request<ShowcaseOut>("GET", "/catalog/showcase", { signal, anonymous: true }),
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
    if (!response.ok) throw new ApiError(response.status, messages().api.demoPhoto);
    return response.blob();
  },

  my: {
    medicines: (signal?: AbortSignal) =>
      request<SavedMedicines>("GET", "/my/medicines", { signal }),
    saveMedicine: (class_id: number, probability: number | null) =>
      request<SavedMedicines>("POST", "/my/medicines", { body: { class_id, probability } }),
    removeMedicine: (class_id: number) => request<void>("DELETE", `/my/medicines/${class_id}`),
    clearMedicines: () => request<void>("DELETE", "/my/medicines"),
    reports: (signal?: AbortSignal) =>
      request<{ items: ReportSummary[]; limit: number }>("GET", "/my/reports", { signal }),
    report: (id: number, signal?: AbortSignal) =>
      request<Report>("GET", `/my/reports/${id}`, { signal }),
    createReport: (report: NewReport) => request<Report>("POST", "/my/reports", { body: report }),
    deleteReport: (id: number) => request<void>("DELETE", `/my/reports/${id}`),
    share: (id: number, days: number) =>
      request<{ token: string; share: ShareState }>("POST", `/my/reports/${id}/share`, {
        body: { days },
      }),
    unshare: (id: number) => request<Report>("DELETE", `/my/reports/${id}/share`),
  },
  /** A report shared by link: no sign-in; the token is the part after # in /shared#… */
  shared: (token: string, signal?: AbortSignal) =>
    request<ReportBody>("POST", "/shared", { body: { token }, signal, anonymous: true }),

  admin: {
    stats: (signal?: AbortSignal) => request<AdminStats>("GET", "/admin/stats", { signal }),
    users: (filters: UserFilters, signal?: AbortSignal) =>
      request<Page<User>>("GET", "/admin/users", { query: { ...filters }, signal }),
    user: (id: number, signal?: AbortSignal) =>
      request<User>("GET", `/admin/users/${id}`, { signal }),
    createUser: (account: NewUser) => request<User>("POST", "/admin/users", { body: account }),
    updateUser: (id: number, changes: UserChanges) =>
      request<User>("PATCH", `/admin/users/${id}`, { body: changes }),
    resetPassword: (id: number, new_password: string) =>
      request<void>("POST", `/admin/users/${id}/password`, { body: { new_password } }),
    deleteUser: (id: number) => request<void>("DELETE", `/admin/users/${id}`),
    activity: (filters: ActivityFilters, signal?: AbortSignal) =>
      request<ActivityPage>("GET", "/admin/activity", { query: { ...filters }, signal }),
    usage: (days: number, signal?: AbortSignal) =>
      request<Usage>("GET", "/admin/usage", { query: { days }, signal }),
  },
};

/** Address of a CSV export (downloaded by the browser with the session cookie). */
export function exportUrl(table: "users" | "activity", filters: ActivityFilters = {}): string {
  const url = new URL(`/api/admin/export/${table}.csv`, window.location.origin);
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== "") url.searchParams.set(key, String(value));
  }
  return url.pathname + url.search;
}
