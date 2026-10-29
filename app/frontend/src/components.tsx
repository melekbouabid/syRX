import {
  useEffect,
  useId,
  useRef,
  type CSSProperties,
  type InputHTMLAttributes,
  type MouseEvent,
  type ReactNode,
} from "react";
import logoDarkUrl from "./assets/logo-dark.png";
import logoMarkUrl from "./assets/logo-mark.png";
import logoUrl from "./assets/logo.png";
import { APP_NAME } from "./brand";
import { Icon, type IconName } from "./icons";
import { CountUp } from "./motion";

const numberFormat = new Intl.NumberFormat("en");

export function formatNumber(value: number): string {
  return numberFormat.format(value);
}

export function formatDate(iso: string | null, withTime = false): string {
  if (!iso) return "Never";
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: withTime ? "short" : undefined,
  });
}

export function initials(name: string): string {
  const words = name.trim().split(/\s+/).filter(Boolean);
  const letters = words.length > 1 ? [words[0][0], words[words.length - 1][0]] : [words[0]?.[0]];
  return letters.join("").toUpperCase() || "?";
}

/** "Good morning", "Good afternoon" or "Good evening", by the visitor's own clock. */
export function greeting(now = new Date()): string {
  const hour = now.getHours();
  if (hour >= 5 && hour < 12) return "Good morning";
  if (hour >= 12 && hour < 18) return "Good afternoon";
  return "Good evening";
}

const LOGO_RATIO = 490 / 240; // width / height of the logo files

interface LogoProps {
  height?: number;
  /**
   * "auto" follows the page: the original logo on a light page, and on a dark page the
   * version whose wordmark is lightened so it stays readable. "dark" forces that version,
   * for the blue-to-green brand background. Neither has a background of its own.
   */
  tone?: "auto" | "dark";
}

/** The full logo: emblem and wordmark, transparent, sitting directly on the page. */
export function Logo({ height = 64, tone = "auto" }: LogoProps) {
  const size = { height, width: Math.round(height * LOGO_RATIO) };
  if (tone === "dark") return <img className="logo" src={logoDarkUrl} alt={APP_NAME} {...size} />;
  return (
    <picture className="logo">
      <source srcSet={logoDarkUrl} media="(prefers-color-scheme: dark)" />
      <img src={logoUrl} alt={APP_NAME} {...size} />
    </picture>
  );
}

/** The emblem alone, for decorative spots next to text that already names the site. */
export function LogoMark({ size = 28, className = "" }: { size?: number; className?: string }) {
  return (
    <img className={`logo-mark ${className}`} src={logoMarkUrl} alt="" width={size} height={size} />
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div className="spinner-wrap" role="status">
      <span className="spinner" aria-hidden="true" />
      <span className="sr-only">{label}</span>
    </div>
  );
}

/** A grey shimmering block standing in for content that is still loading. */
export function Skeleton({ width = "100%", height = "1em" }: { width?: string; height?: string }) {
  return <span className="skeleton" style={{ width, height }} aria-hidden="true" />;
}

export function TableSkeleton({ rows = 8 }: { rows?: number }) {
  return (
    <div className="card table-card" role="status" aria-label="Loading">
      {Array.from({ length: rows }, (_, row) => (
        <div className="skeleton-row" key={row}>
          <Skeleton width="26%" />
          <Skeleton width="14%" />
          <Skeleton width="20%" />
          <Skeleton width="10%" />
        </div>
      ))}
    </div>
  );
}

type AlertKind = "error" | "success" | "info";

export function Alert({ kind = "error", children }: { kind?: AlertKind; children: ReactNode }) {
  return (
    <div className={`alert alert-${kind}`} role={kind === "error" ? "alert" : "status"}>
      {children}
    </div>
  );
}

export function Badge({ tone, children }: { tone: string; children: ReactNode }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

/** Initials on one of four brand gradients; the same name always gets the same one. */
export function Avatar({ name, size = "md" }: { name: string; size?: "sm" | "md" | "lg" }) {
  const variant = [...name].reduce((sum, letter) => sum + letter.charCodeAt(0), 0) % 4;
  return (
    <span className={`avatar avatar-${size} avatar-v${variant}`} aria-hidden="true">
      {initials(name)}
    </span>
  );
}

interface StatTileProps {
  icon: IconName;
  label: string;
  /** null or undefined means "not available"; the tile then shows a dash. */
  value: number | null | undefined;
  note: string;
  loading?: boolean;
}

export function StatTile({ icon, label, value, note, loading = false }: StatTileProps) {
  return (
    <div className="card stat">
      <span className="stat-icon">
        <Icon name={icon} size={20} />
      </span>
      <span className="stat-label">{label}</span>
      <span className="stat-value">
        {loading ? <Skeleton width="4ch" /> : value == null ? "—" : <CountUp value={value} />}
      </span>
      <span className="stat-note">{note}</span>
    </div>
  );
}

const STRENGTH_LABELS = ["Too short", "Weak", "Fair", "Good", "Strong"];

/** A rough 0 to 4 guide from length and variety. It cannot know if a password is reused. */
export function passwordScore(password: string, minLength: number): number {
  if (password.length < minLength) return 0;
  const kinds = [/[a-z]/, /[A-Z]/, /\d/, /[^A-Za-z0-9]/].filter((kind) => kind.test(password));
  let score = 1;
  if (password.length >= 12) score += 1;
  if (kinds.length >= 2) score += 1;
  if (kinds.length >= 3 && password.length >= 10) score += 1;
  return score;
}

export function PasswordStrength({ password, minLength }: { password: string; minLength: number }) {
  if (!password) return null;
  const score = passwordScore(password, minLength);
  return (
    <div className={`strength strength-${score}`}>
      <div className="strength-bars" aria-hidden="true">
        {[1, 2, 3, 4].map((step) => (
          <span key={step} className={step <= score ? "on" : undefined} />
        ))}
      </div>
      <span role="status">{STRENGTH_LABELS[score]}</span>
    </div>
  );
}

type TextFieldProps = Omit<InputHTMLAttributes<HTMLInputElement>, "onChange" | "id"> & {
  label: string;
  hint?: string;
  onChange: (value: string) => void;
};

export function TextField({ label, hint, onChange, ...input }: TextFieldProps) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        {...input}
        onChange={(event) => onChange(event.target.value)}
        aria-describedby={hint ? `${id}-hint` : undefined}
      />
      {hint && (
        <p className="hint" id={`${id}-hint`}>
          {hint}
        </p>
      )}
    </div>
  );
}

interface SubmitButtonProps {
  busy: boolean;
  children: ReactNode;
  className?: string;
  disabled?: boolean;
}

/** A submit button that shows a spinner and blocks double clicks while its request runs. */
export function SubmitButton({ busy, children, className = "btn btn-primary", disabled }: SubmitButtonProps) {
  return (
    <button type="submit" className={className} disabled={busy || disabled}>
      {busy && <span className="btn-spinner" aria-hidden="true" />}
      {busy ? "Please wait…" : children}
    </button>
  );
}

interface ModalProps {
  title: string;
  onClose: () => void;
  children: ReactNode;
}

/** A native <dialog>: the browser handles Escape, focus trapping and the backdrop. */
export function Modal({ title, onClose, children }: ModalProps) {
  const ref = useRef<HTMLDialogElement>(null);
  const titleId = useId();

  useEffect(() => {
    const dialog = ref.current;
    if (!dialog || dialog.open) return;
    dialog.showModal();
    // The browser focuses the first control, which is the close button. Start in the first
    // field instead; a dialog without fields (a confirmation) keeps the safe default.
    dialog.querySelector<HTMLElement>("input:enabled, select:enabled, textarea:enabled")?.focus();
  }, []);

  // The backdrop belongs to the dialog element, so a press on it targets the dialog but
  // lands outside its box. (A press on the dialog's own scrollbar lands inside: ignored.)
  const closeOnBackdrop = (event: MouseEvent<HTMLDialogElement>) => {
    if (event.target !== ref.current) return;
    const box = ref.current.getBoundingClientRect();
    const inside =
      event.clientX >= box.left &&
      event.clientX <= box.right &&
      event.clientY >= box.top &&
      event.clientY <= box.bottom;
    if (!inside) onClose();
  };

  return (
    <dialog
      ref={ref}
      className="modal"
      aria-labelledby={titleId}
      onClose={onClose}
      onMouseDown={closeOnBackdrop}
    >
      <div className="modal-body">
        <header className="modal-header">
          <h2 id={titleId}>{title}</h2>
          <button type="button" className="icon-btn" aria-label="Close" onClick={onClose}>
            ×
          </button>
        </header>
        {children}
      </div>
    </dialog>
  );
}

interface PaginationProps {
  page: number;
  pageSize: number;
  total: number;
  onPage: (page: number) => void;
}

export function Pagination({ page, pageSize, total, onPage }: PaginationProps) {
  const pages = Math.max(1, Math.ceil(total / pageSize));
  const first = total === 0 ? 0 : (page - 1) * pageSize + 1;
  const last = Math.min(page * pageSize, total);
  return (
    <nav className="pagination" aria-label="Pages">
      <span className="muted">
        {formatNumber(first)}–{formatNumber(last)} of {formatNumber(total)}
      </span>
      <div className="pagination-buttons">
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          disabled={page <= 1}
          onClick={() => onPage(page - 1)}
        >
          Previous
        </button>
        <span className="muted">
          Page {formatNumber(page)} of {formatNumber(pages)}
        </span>
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          disabled={page >= pages}
          onClick={() => onPage(page + 1)}
        >
          Next
        </button>
      </div>
    </nav>
  );
}

/** Position of a row or card in its list, for staggered entrance animations (see --i in CSS). */
export function staggerIndex(index: number): CSSProperties {
  return { "--i": Math.min(index, 12) } as CSSProperties;
}
