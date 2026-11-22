// The two buttons every page shows: light / dark mode and the interface language.
import { LANGUAGES, useLang, useT, type Lang } from "../i18n";
import { useTheme, type Theme } from "../services/theme";
import { Icon } from "./icons";

export function ThemeButton({
  theme,
  onToggle,
  labels,
  className = "",
}: {
  theme: Theme;
  onToggle: () => void;
  labels: { toDark: string; toLight: string };
  className?: string;
}) {
  const label = theme === "dark" ? labels.toLight : labels.toDark;
  return (
    <button
      type="button"
      className={`icon-btn pref-btn ${className}`}
      onClick={onToggle}
      aria-label={label}
      title={label}
      data-theme-toggle={theme}
    >
      <Icon name={theme === "dark" ? "sun" : "moon"} size={18} />
    </button>
  );
}

export function ThemeToggle({ className = "" }: { className?: string }) {
  const { theme, toggle } = useTheme();
  const t = useT();
  return <ThemeButton theme={theme} onToggle={toggle} labels={t.prefs} className={className} />;
}

/** Shows the other language: one click switches, the choice is remembered. */
export function LanguageButton({
  lang,
  onChange,
  switchTo,
  className = "",
}: {
  lang: Lang;
  onChange: (lang: Lang) => void;
  switchTo: (name: string) => string;
  className?: string;
}) {
  const other = LANGUAGES.find((item) => item.code !== lang) ?? LANGUAGES[0];
  return (
    <button
      type="button"
      className={`icon-btn pref-btn pref-lang ${className}`}
      onClick={() => onChange(other.code)}
      aria-label={switchTo(other.name)}
      title={switchTo(other.name)}
      lang={other.code}
      data-lang-toggle={lang}
    >
      <Icon name="globe" size={16} />
      <span>{other.short}</span>
    </button>
  );
}

export function LanguageToggle({ className = "" }: { className?: string }) {
  const { lang, setLang } = useLang();
  const t = useT();
  return <LanguageButton lang={lang} onChange={setLang} switchTo={t.prefs.switchTo} className={className} />;
}

/** Both buttons side by side. */
export function Preferences({ className = "" }: { className?: string }) {
  return (
    <div className={`prefs ${className}`}>
      <LanguageToggle />
      <ThemeToggle />
    </div>
  );
}
