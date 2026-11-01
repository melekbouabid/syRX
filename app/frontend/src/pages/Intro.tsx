import { useRef, type CSSProperties } from "react";
import { Link } from "react-router-dom";
import { APP_NAME } from "../brand";
import { Logo, LogoMark } from "../components";
import { Icon, type IconName } from "../icons";
import { Reveal, useInView } from "../motion";

// The three things the app does, in the order a visit goes through them.
const STEPS = [
  {
    name: "Scan",
    text: "A photo or the webcam: each pill is detected, then identified among the known pills, with a confidence score.",
  },
  {
    name: "Check",
    text: "The active ingredients of every pair are looked up in DDInter: severity and explanation, as the source gives them.",
  },
  {
    name: "Ask",
    text: "An assistant answers follow-up questions from official drug labels, PubMed and DDInter, and cites each source.",
  },
];

// The safety rules of the project (docs/project_rules.md), as a visitor sees them.
const RULES: { title: string; text: string; icon: IconName }[] = [
  {
    icon: "ban",
    title: "Never a dose",
    text: "Questions about how much to take are refused: that is for a pharmacist or doctor.",
  },
  {
    icon: "alert",
    title: "Never called safe",
    text: "A pair missing from DDInter is reported as “no known interaction in this database”.",
  },
  {
    icon: "book",
    title: "Always a source",
    text: "Every answer cites the label, article or database entry it comes from.",
  },
  {
    icon: "eye",
    title: "A suggestion, not a certainty",
    text: "Each identified pill shows its confidence; when it is low, you confirm or pick from the top 5.",
  },
];

const SOURCES = [
  { name: "NLM C3PI", gives: "Pill photographs used to train and test the vision models" },
  { name: "RxNorm (RxNav)", gives: "From a pill to its active ingredients" },
  { name: "DDInter 2", gives: "Interactions between active ingredients, with their severity" },
  { name: "openFDA drug labels", gives: "Official labels the assistant answers from" },
  { name: "PubMed", gives: "Abstracts of articles on drug interactions" },
  { name: "Tox21", gives: "In vitro toxicity data for the bonus model" },
];

// Confidence of the three best suggestions for one detected pill, in percent.
// An illustration of the idea, not a real prediction.
const FIGURE = [
  { label: "Top 1 · 81%", width: 81, kind: "top" },
  { label: "Top 2 · 12%", width: 12, kind: "other" },
  { label: "Top 3 · 4%", width: 4, kind: "other" },
];

/** A bar that grows to `--w` once its card is on screen (see .figure-card in the CSS). */
function barWidth(percent: number): CSSProperties {
  return { "--w": `${percent}%` } as CSSProperties;
}

function ScanFigure() {
  const ref = useRef<HTMLElement>(null);
  const inView = useInView(ref);
  return (
    <figure
      ref={ref}
      className={inView ? "figure-card is-visible" : "figure-card"}
      role="img"
      aria-label="Illustration: for one detected pill, the best suggestion has a confidence of 81 percent, the next two 12 and 4 percent."
    >
      <p className="figure-title">How a scan reads</p>
      <p className="figure-sub">The three best suggestions for one detected pill</p>
      {FIGURE.map((row) => (
        <div className="bar-row" key={row.label}>
          <span>{row.label}</span>
          <div className="bar-track">
            <span className={`bar-fill bar-${row.kind}`} style={barWidth(row.width)} />
          </div>
        </div>
      ))}
      <figcaption>
        <span>A suggestion, never a certainty: when the confidence is low, you confirm it.</span>
        <span className="figure-note">An illustration of the idea, not a real prediction.</span>
      </figcaption>
    </figure>
  );
}

/** The public start page: what the site is, shown before anyone signs in. */
export default function Intro() {
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
              <p className="eyebrow">Pill identification and interaction checks</p>
              <h1>Know which pills you have and whether they interact.</h1>
              <p className="intro-lead">
                Take a photo of one or more pills. {APP_NAME} detects each one, suggests what it
                is with a confidence score, checks its active ingredients for known interactions
                and answers your questions, citing every source.
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
              <ScanFigure />
            </div>
          </div>
        </section>

        <section className="intro-section">
          <div className="intro-inner">
            <Reveal>
              <h2>How it works</h2>
              <p className="intro-text">
                Three steps, each built on public data, each answer traced back to where it
                comes from.
              </p>
            </Reveal>
            <Reveal stagger className="intro-grid intro-grid-3">
              {STEPS.map((step) => (
                <div className="card step-card lift" key={step.name}>
                  <h3>{step.name}</h3>
                  <p className="muted">{step.text}</p>
                </div>
              ))}
            </Reveal>
          </div>
        </section>

        <section className="intro-section intro-section-alt">
          <div className="intro-inner">
            <Reveal>
              <h2>Rules it always follows</h2>
            </Reveal>
            <Reveal stagger className="intro-grid intro-grid-4">
              {RULES.map((rule) => (
                <div className="feature" key={rule.title}>
                  <span className="feature-icon" aria-hidden="true">
                    <Icon name={rule.icon} size={22} />
                  </span>
                  <h3>{rule.title}</h3>
                  <p className="muted">{rule.text}</p>
                </div>
              ))}
            </Reveal>
            <Reveal>
              <p className="intro-next">
                <span className="badge badge-admin badge-neutral">Information only</span>{" "}
                {APP_NAME} does not replace a pharmacist or doctor: consult one before changing
                any treatment.
              </p>
            </Reveal>
          </div>
        </section>

        <section className="intro-section">
          <div className="intro-inner">
            <Reveal>
              <h2>Where the data comes from</h2>
              <p className="intro-text">
                Only public sources, collected through their official APIs and files, each with
                its licence. Nothing is invented: what the sources do not say is not said.
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
              <h2>Ready to scan?</h2>
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
          <span>{APP_NAME}. Information only: consult a pharmacist or doctor.</span>
        </div>
      </footer>
    </div>
  );
}
