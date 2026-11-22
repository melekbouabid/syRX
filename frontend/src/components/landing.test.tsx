import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import en from "../i18n/en";
import fr from "../i18n/fr";
import type { CatalogSummary, ShowcaseOut } from "../services/api";
import { catalogFacts, FactList, FloatCardsView } from "./landing";
import { LanguageButton, ThemeButton } from "./prefs";
import { SceneFrame } from "./Scene3D";

// What GET /api/catalog/summary answers with the medicine tables of the project loaded.
const SUMMARY: CatalogSummary = {
  counts: { pills: 380, ingredients: 1793, interaction_pairs: 75653 },
  confirm_threshold: 0.85,
  disclaimer: "Information uniquement, consultez un pharmacien ou un médecin.",
};

const SHOWCASE: ShowcaseOut = {
  confirm_threshold: 0.85,
  showcase: {
    photo_id: 2,
    pill_name: "FLUCONAZOLE 100MG TABLETS",
    active_ingredients: ["fluconazole"],
    probability: 0.9512,
    model: "tujya-pill-classifier",
    model_version: 8,
    interaction: { ingredient_a: "fluconazole", ingredient_b: "warfarin", severity: "Major", source: "DDInter" },
  },
};

describe("catalogue figures", () => {
  it("come from the API answer, not from the code", () => {
    expect(catalogFacts(SUMMARY, en).map((fact) => fact.value)).toEqual([380, 1793, 75653]);
    const other = { ...SUMMARY, counts: { pills: 7, ingredients: 8, interaction_pairs: 9 } };
    expect(catalogFacts(other, en).map((fact) => fact.value)).toEqual([7, 8, 9]);
  });

  it("are shown in the interface language", () => {
    const english = renderToStaticMarkup(<FactList summary={SUMMARY} loading={false} t={en} animate={false} />);
    expect(english).toContain("380");
    expect(english).toContain("75,653");
    expect(english).toContain(en.facts.pairs);
    const french = renderToStaticMarkup(<FactList summary={SUMMARY} loading={false} t={fr} animate={false} />);
    expect(french).toContain(fr.facts.pills);
  });

  it("are left out, never invented, when the tables are not loaded", () => {
    const html = renderToStaticMarkup(
      <FactList summary={{ ...SUMMARY, counts: null }} loading={false} t={en} animate={false} />,
    );
    expect(html).toContain(en.facts.unavailable);
    expect(html).not.toMatch(/\d/);
    expect(catalogFacts(null, en)).toEqual([]);
  });
});

describe("floating cards", () => {
  it("show the real scan of a test photo and its DDInter interaction", () => {
    const html = renderToStaticMarkup(<FloatCardsView result={SHOWCASE} t={en} />);
    expect(html).toContain('data-real="yes"');
    expect(html).toContain("FLUCONAZOLE 100MG TABLETS");
    expect(html).toContain("95.1%");
    const french = renderToStaticMarkup(<FloatCardsView result={SHOWCASE} t={fr} lang="fr" />);
    expect(french).toMatch(/95,1\s%/); // French puts a non-breaking space before %
    expect(html).toContain("fluconazole + warfarin");
    expect(html).toContain("tujya-pill-classifier v8");
    expect(html).not.toContain(en.common.example);
  });

  it("say “example” and show no figure when the API has no showcase", () => {
    for (const result of [null, { showcase: null, confirm_threshold: 0.85 }]) {
      const html = renderToStaticMarkup(<FloatCardsView result={result} t={fr} />);
      expect(html).toContain('data-real="no"');
      expect(html).toContain(fr.common.example);
      expect(html).not.toMatch(/\d+([.,]\d)?\s?%/);
    }
  });
});

describe("3D scene", () => {
  it("is replaced by a still emblem when the visitor asks for reduced motion", () => {
    const html = renderToStaticMarkup(<SceneFrame variant="hero" reduced state="loading" />);
    expect(html).toContain('data-motion="off"');
    expect(html).not.toContain("<canvas");
  });

  it("draws on a canvas otherwise, and falls back when WebGL fails", () => {
    expect(renderToStaticMarkup(<SceneFrame variant="compact" reduced={false} state="ready" />)).toContain(
      "<canvas",
    );
    expect(renderToStaticMarkup(<SceneFrame variant="compact" reduced={false} state="failed" />)).not.toContain(
      "<canvas",
    );
  });
});

describe("preference buttons", () => {
  it("name the theme they switch to", () => {
    const light = renderToStaticMarkup(<ThemeButton theme="light" onToggle={() => {}} labels={fr.prefs} />);
    expect(light).toContain(fr.prefs.toDark);
    const dark = renderToStaticMarkup(<ThemeButton theme="dark" onToggle={() => {}} labels={en.prefs} />);
    expect(dark).toContain(en.prefs.toLight);
  });

  it("offer the other language, English or French only", () => {
    const html = renderToStaticMarkup(<LanguageButton lang="en" onChange={() => {}} switchTo={en.prefs.switchTo} />);
    expect(html).toContain(">FR<");
    expect(html).toContain("Show the interface in Français");
    expect(renderToStaticMarkup(<LanguageButton lang="fr" onChange={() => {}} switchTo={fr.prefs.switchTo} />)).toContain(
      ">EN<",
    );
  });
});
