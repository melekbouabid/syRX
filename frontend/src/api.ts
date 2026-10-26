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
      body: options.body === undefined ? undefined : JSON.stringify(options.body),
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
