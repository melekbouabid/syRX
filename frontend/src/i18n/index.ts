import { useSyncExternalStore } from "react";
import en, { type Messages } from "./en";
import fr from "./fr";

/**
 * The interface language: English or French. Until someone picks one with the language
 * button, the browser's first language decides (French when it is French, English
 * otherwise); the choice is then kept in this browser, like the theme (services/theme.ts),
 * inside try/catch. Components read their texts with `useT()`, which re-renders them when
 * the language changes; code outside React calls `messages()`.
 *
 * What the API writes (error details, agent answers, source excerpts) is shown as it comes:
 * only the interface itself is translated (docs/decisions.md D128).
 */
export type Lang = "en" | "fr";
export type { Messages };

export const LANGUAGES: { code: Lang; name: string; short: string }[] = [
  { code: "en", name: "English", short: "EN" },
  { code: "fr", name: "Français", short: "FR" },
];

export const MESSAGES: Record<Lang, Messages> = { en, fr };
export const LANG_KEY = "tujya-lang"; // also read by the inline script of index.html

export function isLang(value: unknown): value is Lang {
  return value === "en" || value === "fr";
}

/** The saved choice, or null when there is none or storage cannot be read. */
export function readSavedLang(storage: Pick<Storage, "getItem"> | undefined): Lang | null {
  try {
    const value = storage?.getItem(LANG_KEY);
    return isLang(value) ? value : null;
  } catch {
    return null;
  }
}

/** French when the browser's first language is French, English otherwise. */
export function browserLang(languages: readonly string[] | undefined): Lang {
  const first = languages?.[0]?.slice(0, 2).toLowerCase();
  return first === "fr" ? "fr" : "en";
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

function initialLang(): Lang {
  // Outside a browser (unit tests in Node) the interface is English, whatever the machine.
  if (!hasDom) return "en";
  const fromPage = document.documentElement.lang;
  if (isLang(fromPage) && readSavedLang(storage()) === null) {
    // index.html already chose from the browser language; keep the same answer.
    return fromPage;
  }
  return readSavedLang(storage()) ?? browserLang(navigator.languages ?? [navigator.language]);
}

let current: Lang = initialLang();
if (hasDom) document.documentElement.lang = current;

function apply(lang: Lang) {
  if (lang === current) return;
  current = lang;
  if (hasDom) document.documentElement.lang = lang; // screen readers, hyphenation
  listeners.forEach((listener) => listener());
}

export function setLang(lang: Lang) {
  if (hasDom) {
    try {
      storage()?.setItem(LANG_KEY, lang);
    } catch {
      // Not remembered, but still applied for this visit.
    }
  }
  apply(lang);
}

if (hasDom) {
  window.addEventListener("storage", (event) => {
    if (event.key === LANG_KEY && isLang(event.newValue)) apply(event.newValue);
  });
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

export function getLang(): Lang {
  return current;
}

/** The texts of the current language, for code outside React components. */
export function messages(): Messages {
  return MESSAGES[current];
}

export function useLang() {
  const lang = useSyncExternalStore(subscribe, getLang, getLang);
  return { lang, setLang };
}

/** The texts of the current language; the component re-renders when it changes. */
export function useT(): Messages {
  return MESSAGES[useSyncExternalStore(subscribe, getLang, getLang)];
}

/** Locale tag for Intl: numbers and dates follow the interface language. */
export function locale(lang: Lang = current): string {
  return lang === "fr" ? "fr-FR" : "en-GB";
}
