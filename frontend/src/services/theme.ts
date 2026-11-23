import { useSyncExternalStore } from "react";

/**
 * Light or dark mode (`data-theme` on <html>): the system setting until the toggle is pressed,
 * then kept in localStorage when available. index.html applies it before the first paint.
 */
export type Theme = "light" | "dark";

export const THEME_KEY = "tujya-theme"; // also read by the inline script of index.html

/** The saved choice, or null when there is none or storage cannot be read. */
export function readSavedTheme(storage: Pick<Storage, "getItem"> | undefined): Theme | null {
  try {
    const value = storage?.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : null;
  } catch {
    return null;
  }
}

/** Remember a choice; false when storage refused it (the theme still applies for this visit). */
export function saveTheme(storage: Pick<Storage, "setItem"> | undefined, theme: Theme): boolean {
  try {
    if (!storage) return false;
    storage.setItem(THEME_KEY, theme);
    return true;
  } catch {
    return false;
  }
}

/** The saved choice wins; without one, the system setting decides. */
export function resolveTheme(saved: Theme | null, systemDark: boolean): Theme {
  return saved ?? (systemDark ? "dark" : "light");
}

const hasDom = typeof document !== "undefined" && typeof window !== "undefined";
const listeners = new Set<() => void>();

function storage(): Storage | undefined {
  try {
    return window.localStorage;
  } catch {
    return undefined;
  }
}

function systemQuery(): MediaQueryList | null {
  return hasDom && typeof window.matchMedia === "function"
    ? window.matchMedia("(prefers-color-scheme: dark)")
    : null;
}

export function currentTheme(): Theme {
  return hasDom && document.documentElement.dataset.theme === "dark" ? "dark" : "light";
}

function apply(theme: Theme) {
  if (!hasDom || theme === currentTheme()) return;
  const update = () => {
    document.documentElement.dataset.theme = theme;
    listeners.forEach((listener) => listener());
  };
  // A short cross-fade where the browser has it, unless the visitor asked for less motion.
  const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  if (typeof document.startViewTransition === "function" && !reduce) {
    document.startViewTransition(update);
  } else {
    update();
  }
}

export function setTheme(theme: Theme) {
  saveTheme(hasDom ? storage() : undefined, theme);
  apply(theme);
}

if (hasDom) {
  const query = systemQuery();
  // In case the script of index.html did not run.
  if (!document.documentElement.dataset.theme) {
    document.documentElement.dataset.theme = resolveTheme(
      readSavedTheme(storage()),
      query?.matches ?? false,
    );
  }
  // Follow the system while no choice is saved, and keep other open tabs in step.
  query?.addEventListener("change", () => {
    if (!readSavedTheme(storage())) apply(resolveTheme(null, query.matches));
  });
  window.addEventListener("storage", (event) => {
    if (event.key === THEME_KEY) {
      apply(resolveTheme(readSavedTheme(storage()), query?.matches ?? false));
    }
  });
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function useTheme() {
  const theme = useSyncExternalStore(subscribe, currentTheme, () => "light" as Theme);
  const toggle = () => setTheme(theme === "dark" ? "light" : "dark");
  return { theme, setTheme, toggle };
}
