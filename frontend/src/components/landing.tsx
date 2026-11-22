// Pieces shared by the start page and the sign-in page: the catalogue figures and the
// floating cards over the 3D scene. Every figure comes from GET /api/catalog/summary (the
// medicine tables and one real scan of a demo gallery photo with the served models); when the
// API cannot give one, nothing is invented: the figure is left out and the cards say "example".
import type { CSSProperties, PointerEvent as ReactPointerEvent } from "react";
import { useQuery } from "../hooks";
import { locale, useLang, useT, type Lang, type Messages } from "../i18n";
import { api, type CatalogSummary, type ShowcaseOut } from "../services/api";
import { formatNumber } from "./index";
import { Icon } from "./icons";
import { SEVERITY_TONE } from "./interactions";
import { CountUp } from "./motion";

export function useCatalogSummary() {
  return useQuery((signal) => api.catalogSummary(signal), []);
}

export function useShowcase() {
  return useQuery((signal) => api.catalogShowcase(signal), []);
}

export interface Fact {
  key: "pills" | "ingredients" | "pairs";
  value: number;
  label: string;
}

/** The three figures, or none when the medicine tables are not loaded. */
export function catalogFacts(summary: CatalogSummary | null | undefined, t: Messages): Fact[] {
  const counts = summary?.counts;
  if (!counts) return [];
  return [
    { key: "pills", value: counts.pills, label: t.facts.pills },
    { key: "ingredients", value: counts.ingredients, label: t.facts.ingredients },
    { key: "pairs", value: counts.interaction_pairs, label: t.facts.pairs },
  ];
}

export function FactList({
  summary,
  loading,
  t,
  className = "",
  animate = true,
}: {
  summary: CatalogSummary | null | undefined;
  loading: boolean;
  t: Messages;
  className?: string;
  animate?: boolean;
}) {
  const facts = catalogFacts(summary, t);
  if (loading && !summary) {
    return <p className={`facts-note ${className}`}>{t.facts.loading}…</p>;
  }
  if (facts.length === 0) {
    return <p className={`facts-note ${className}`}>{t.facts.unavailable}</p>;
  }
  return (
    <ul className={`facts ${className}`} aria-label={t.facts.label}>
      {facts.map((fact) => (
        <li key={fact.key} data-fact={fact.key}>
          <strong>{animate ? <CountUp value={fact.value} duration={1400} /> : formatNumber(fact.value)}</strong>
          <span>{fact.label}</span>
        </li>
      ))}
    </ul>
  );
}

const depth = (value: number) => ({ "--depth": value }) as CSSProperties;

/**
 * Glass cards floating over the 3D scene. With a showcase, they show the real top-1 of a
 * C3PI test photo and the most severe DDInter interaction of its ingredient; without one,
 * they are generic and marked as an example. Decoration only: hidden from screen readers
 * (the same facts are in the page text).
 */
/** 0.9512 -> "95.1%" in English, "95,1 %" in French. */
export function formatShare(value: number, lang: Lang): string {
  return new Intl.NumberFormat(locale(lang), { style: "percent", maximumFractionDigits: 1 }).format(value);
}

export function FloatCardsView({
  result,
  t,
  lang = "en",
  variant = "hero",
}: {
  /** The answer of GET /api/catalog/showcase; null when it failed. */
  result: ShowcaseOut | null;
  t: Messages;
  lang?: Lang;
  variant?: "hero" | "auth";
}) {
  const showcase = result?.showcase ?? null;
  const pair = showcase?.interaction ?? null;
  return (
    <div className={`float-cards float-cards-${variant}`} aria-hidden="true" data-real={showcase ? "yes" : "no"}>
      <div className="float-card fc-1" style={depth(1.3)}>
        <span className="fc-icon fc-blue">
          <Icon name="scan" size={16} />
        </span>
        <span className="fc-text">
          <small>{showcase ? t.floatCards.topMatch : t.floatCards.exampleMatch}</small>
          <strong>{showcase ? showcase.pill_name : t.floatCards.examplePill}</strong>
          {showcase && (
            <small className="fc-sub">
              {t.floatCards.modelLine(showcase.model, String(showcase.model_version ?? "?"))}
            </small>
          )}
        </span>
        {showcase ? (
          <b className="fc-value">{formatShare(showcase.probability, lang)}</b>
        ) : (
          <span className="fc-example">{t.common.example}</span>
        )}
      </div>
      <div className="float-card fc-2" style={depth(0.8)}>
        <span className="fc-icon fc-red">
          <Icon name="alert" size={16} />
        </span>
        <span className="fc-text">
          <small>{pair ? pair.source : t.floatCards.exampleCheck}</small>
          <strong>
            {pair ? `${pair.ingredient_a} + ${pair.ingredient_b}` : t.floatCards.examplePair}
          </strong>
        </span>
        {pair ? (
          <span className={`badge badge-${SEVERITY_TONE[pair.severity] ?? "neutral"}`}>
            {t.severity[pair.severity] ?? pair.severity}
          </span>
        ) : (
          <span className="fc-example">{t.common.example}</span>
        )}
      </div>
      {variant === "hero" && (
        <div className="float-card fc-3" style={depth(1.6)}>
          <span className="fc-icon fc-green">
            <Icon name="book" size={16} />
          </span>
          <span className="fc-text">
            <small>{t.floatCards.answer}</small>
            <strong>{t.floatCards.everyCited}</strong>
          </span>
          <span className="fc-cites">
            <span>[1]</span>
            <span>[2]</span>
          </span>
        </div>
      )}
    </div>
  );
}

/** The cards appear once the showcase has answered: real, or marked as an example. */
export function FloatCards({ variant = "hero" }: { variant?: "hero" | "auth" }) {
  const t = useT();
  const { lang } = useLang();
  const showcase = useShowcase();
  if (showcase.loading && !showcase.data) return null;
  return <FloatCardsView result={showcase.data} t={t} lang={lang} variant={variant} />;
}

/** Pointer position over an element as --px / --py in -1..1, for the cards' parallax. */
export function parallax(event: ReactPointerEvent<HTMLElement>) {
  const box = event.currentTarget.getBoundingClientRect();
  const style = event.currentTarget.style;
  style.setProperty("--px", (((event.clientX - box.left) / box.width) * 2 - 1).toFixed(3));
  style.setProperty("--py", (((event.clientY - box.top) / box.height) * 2 - 1).toFixed(3));
}
