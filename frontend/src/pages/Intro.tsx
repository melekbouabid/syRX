import { useRef, type CSSProperties } from "react";
import { Link } from "react-router-dom";
import { api, type CatalogSummary } from "../api";
import { APP_NAME } from "../brand";
import { Logo, LogoMark } from "../components";
import { useQuery } from "../hooks";
import { Icon, type IconName } from "../icons";
import { CountUp, Reveal, useInView } from "../motion";

const STATS: { key: keyof CatalogSummary; label: string; icon: IconName }[] = [
  { key: "synergy_measurements", label: "synergy measurements", icon: "flask" },
  { key: "drugs", label: "drugs", icon: "pill" },
  { key: "cell_lines", label: "cancer cell lines", icon: "cells" },
  { key: "abstracts", label: "PubMed abstracts", icon: "book" },
];

// Each score compares the measured effect of a pair with a different idea of "no interaction".
const SCORES = [
  { name: "Bliss", expects: "the two drugs acting independently of each other." },
  { name: "Loewe", expects: "simply giving more of a single drug (dose additivity)." },
  { name: "HSA", expects: "the better of the two drugs on its own (highest single agent)." },
  { name: "ZIP", expects: "neither drug changing how potent the other one is." },
];

const FEATURES: { title: string; text: string; icon: IconName }[] = [
  {
    icon: "search",
    title: "Search every drug",
    text: "Find a compound by name, synonym or ChEMBL ID, then jump to its PubChem and ChEMBL records.",
  },
  {
    icon: "cells",
    title: "Browse cell lines",
    text: "See the tissue and disease of each cancer cell line, with its DepMap identifier.",
  },
  {
    icon: "check",
    title: "Know where it came from",
    text: "Every drug and cell line shows which screens it was measured in, and how many times.",
  },
];

const SOURCES = [
  { name: "DrugComb", gives: "Combination screens with Bliss, Loewe, HSA and ZIP scores" },
  { name: "Therapeutics Data Commons", gives: "A curated drug-synergy benchmark" },
  { name: "O'Neil et al.", gives: "The combination screen used to train DeepSynergy" },
  { name: "DepMap", gives: "Cell line names, tissues and diseases" },
  { name: "PubChem and ChEMBL", gives: "Compound identifiers, structures and synonyms" },
  { name: "PubMed", gives: "Abstracts about drug combinations" },
];

// Growth inhibition in percent. "Expected" follows Bliss independence:
// 35 + 45 - (35 x 45) / 100 = 64. An illustration of the idea, not measured data.
const FIGURE = { a: 35, b: 45, expected: 64, measured: 85 };

/** A bar that grows to `--w` once its card is on screen (see .figure-card in the CSS). */
function barWidth(percent: number, left = 0): CSSProperties {
  return { "--w": `${percent}%`, left: `${left}%` } as CSSProperties;
}

function SynergyFigure() {
  const ref = useRef<HTMLElement>(null);
  const inView = useInView(ref);
  const rows = [
    { label: "Drug A alone", width: FIGURE.a, kind: "single" },
    { label: "Drug B alone", width: FIGURE.b, kind: "single" },
    { label: "Expected together", width: FIGURE.expected, kind: "expected" },
  ];
  return (
    <figure
      ref={ref}
      className={inView ? "figure-card is-visible" : "figure-card"}
      role="img"
      aria-label="Illustration: each drug alone slows the cancer cells a little. Together they are expected to do more, but the measured effect is higher still. The extra part is the synergy."
    >
      <p className="figure-title">How synergy is read</p>
      <p className="figure-sub">Growth inhibition of one cancer cell line</p>
      {rows.map((row) => (
        <div className="bar-row" key={row.label}>
          <span>{row.label}</span>
          <div className="bar-track">
            <span className={`bar-fill bar-${row.kind}`} style={barWidth(row.width)} />
          </div>
        </div>
      ))}
      <div className="bar-row">
        <span>Measured together</span>
        <div className="bar-track">
          <span className="bar-fill bar-expected" style={barWidth(FIGURE.measured)} />
          <span
            className="bar-fill bar-excess"
            style={barWidth(FIGURE.measured - FIGURE.expected, FIGURE.expected)}
          />
        </div>
      </div>
      <figcaption>
        <span>
          <span className="bar-swatch" /> Synergy: the effect beyond what was expected.
        </span>
        <span className="figure-note">An illustration of the idea, not measured data.</span>
      </figcaption>
    </figure>
  );
}

/** The public start page: what the site is, shown before anyone signs in. */
export default function Intro() {
  const summary = useQuery((signal) => api.catalogSummary(signal), []);
  // Only tables that exist are shown; if the API is down the band is simply left out.
  const stats = STATS.flatMap((stat) => {
    const value = summary.data?.[stat.key];
    return value == null ? [] : [{ ...stat, value }];
  });

  return (
    <div className="intro">
      <header className="intro-bar">
        <div className="intro-inner intro-bar-inner">
          <Logo height={68} />
          <nav className="intro-bar-actions" aria-label="Account">
            <Link className="btn btn-ghost" to="/login">
              Sign in
            </Link>
            <Link className="btn btn-primary" to="/signup">
              Create account
            </Link>
          </nav>
        </div>
      </header>

      <main>
        <section className="intro-hero">
          <div className="orbs" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div className="intro-inner intro-hero-grid">
            <div className="hero-copy">
              <p className="eyebrow">Drug-combination synergy in cancer cell lines</p>
              <h1>Find drug combinations that work better together.</h1>
              <p className="intro-lead">
                Two drugs can do more together than either does alone. {APP_NAME} gathers the
                published experiments that measured this into one place, so you can see which
                pairs were tested, on which cancer cells, and how strongly they interact.
              </p>
              <div className="intro-cta">
                <Link className="btn btn-light btn-lg" to="/signup">
                  Create an account
                  <Icon name="arrow" />
                </Link>
                <Link className="btn btn-on-dark btn-lg" to="/login">
                  I already have one
                </Link>
              </div>
            </div>
            <div className="hero-figure">
              <SynergyFigure />
            </div>
          </div>
        </section>

        {stats.length > 0 && (
          <section className="intro-stats" aria-label="What the collection contains">
            <Reveal stagger className="intro-inner intro-stats-grid">
              {stats.map((stat) => (
                <div className="intro-stat" key={stat.key}>
                  <span className="stat-icon">
                    <Icon name={stat.icon} size={20} />
                  </span>
                  <span className="intro-stat-value">
                    <CountUp value={stat.value} duration={1400} />
                  </span>
                  <span className="intro-stat-label">{stat.label}</span>
                </div>
              ))}
            </Reveal>
          </section>
        )}

        <section className="intro-section">
          <div className="intro-inner">
            <Reveal>
              <h2>What is synergy?</h2>
              <p className="intro-text">
                A combination is <strong>synergistic</strong> when it stops more cancer cells
                than expected from each drug alone, and <strong>antagonistic</strong> when it
                stops fewer. "Expected" can be defined in several ways, so every experiment
                here keeps four scores. Above zero means synergy; below zero, antagonism.
              </p>
            </Reveal>
            <Reveal stagger className="intro-grid intro-grid-4">
              {SCORES.map((score) => (
                <div className="card score-card lift" key={score.name}>
                  <h3>{score.name}</h3>
                  <p className="muted">Compares the pair with {score.expects}</p>
                </div>
              ))}
            </Reveal>
          </div>
        </section>

        <section className="intro-section intro-section-alt">
          <div className="intro-inner">
            <Reveal>
              <h2>What you can do here</h2>
            </Reveal>
            <Reveal stagger className="intro-grid intro-grid-3">
              {FEATURES.map((feature) => (
                <div className="feature" key={feature.title}>
                  <span className="feature-icon" aria-hidden="true">
                    <Icon name={feature.icon} size={22} />
                  </span>
                  <h3>{feature.title}</h3>
                  <p className="muted">{feature.text}</p>
                </div>
              ))}
            </Reveal>
            <Reveal>
              <p className="intro-next">
                <span className="badge badge-admin badge-neutral">In development</span>{" "}
                Predicted synergy for pairs nobody has tested yet, with supporting evidence
                from PubMed.
              </p>
            </Reveal>
          </div>
        </section>

        <section className="intro-section">
          <div className="intro-inner">
            <Reveal>
              <h2>Where the data comes from</h2>
              <p className="intro-text">
                Only public scientific sources, collected through their official APIs and
                files. Nothing is invented or estimated: every number is a published
                measurement.
              </p>
            </Reveal>
            <Reveal stagger className="source-list">
              {SOURCES.map((source) => (
                <div className="source lift" key={source.name}>
                  <strong>{source.name}</strong>
                  <span className="muted">{source.gives}</span>
                </div>
              ))}
            </Reveal>
          </div>
        </section>

        <section className="intro-closing">
          <Reveal className="intro-inner">
            <div className="closing-card">
              <div className="orbs" aria-hidden="true">
                <span />
                <span />
              </div>
              <LogoMark size={96} className="emblem-float" />
              <h2>Ready to explore?</h2>
              <p>An account takes less than a minute to create.</p>
              <div className="intro-cta">
                <Link className="btn btn-light btn-lg" to="/signup">
                  Create an account
                  <Icon name="arrow" />
                </Link>
                <Link className="btn btn-on-dark btn-lg" to="/login">
                  Sign in
                </Link>
              </div>
            </div>
          </Reveal>
        </section>
      </main>

      <footer className="intro-footer">
        <div className="intro-inner intro-footer-inner">
          <LogoMark size={28} />
          <span>{APP_NAME}. Research use only. Not medical advice.</span>
        </div>
      </footer>
    </div>
  );
}
