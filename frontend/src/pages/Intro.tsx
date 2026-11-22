import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { APP_NAME } from "../brand";
import { Logo, LogoMark } from "../components";
import { Icon, type IconName } from "../components/icons";
import { FactList, FloatCards, parallax, useCatalogSummary } from "../components/landing";
import { Reveal } from "../components/motion";
import { Preferences } from "../components/prefs";
import { Scene3D } from "../components/Scene3D";
import { useT } from "../i18n";

// Icons of the steps and the rules; their texts are in src/i18n, in the same order.
const STEP_ICONS: IconName[] = ["camera", "shield", "book"];
const RULE_ICONS: IconName[] = ["ban", "alert", "book", "eye"];

/** The public start page: what the site is, shown before anyone signs in. Always dark. */
export default function Intro() {
  const t = useT();
  const summary = useCatalogSummary();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="lp">
      <header className={scrolled ? "lp-nav is-scrolled" : "lp-nav"}>
        <div className="lp-inner lp-nav-inner">
          <Link to="/" className="lp-brand" aria-label={APP_NAME}>
            <Logo height={54} tone="dark" />
          </Link>
          <nav className="lp-links" aria-label={t.intro.sections}>
            <a href="#how">{t.intro.linkHow}</a>
            <a href="#safety">{t.intro.linkSafety}</a>
            <a href="#sources">{t.intro.linkData}</a>
          </nav>
          <div className="lp-nav-actions">
            <Preferences className="prefs-glass" />
            <Link className="btn btn-glass" to="/login">
              {t.common.signIn}
            </Link>
            <Link className="btn btn-glow lp-hide-sm" to="/signup">
              {t.intro.createAccount}
            </Link>
          </div>
        </div>
      </header>

      <main>
        <section className="lp-hero" onPointerMove={parallax}>
          <div className="lp-hero-bg" aria-hidden="true">
            <span className="aurora aurora-1" />
            <span className="aurora aurora-2" />
            <span className="lp-grid-lines" />
          </div>
          <Scene3D variant="hero" className="lp-scene" />
          <FloatCards />
          <div className="lp-inner lp-hero-inner">
            <div className="lp-hero-copy">
              <p className="lp-chip">
                <span className="lp-chip-dot" />
                {t.intro.chip}
              </p>
              <h1 className="lp-title">
                <span className="line">{t.intro.title1}</span>
                <span className="line text-gradient">{t.intro.title2}</span>
              </h1>
              <p className="lp-lead">{t.intro.lead(APP_NAME)}</p>
              <div className="lp-cta">
                <Link className="btn btn-glow btn-lg" to="/signup">
                  {t.intro.startScanning}
                  <Icon name="arrow" />
                </Link>
                <Link className="btn btn-glass btn-lg" to="/login">
                  {t.intro.haveOne}
                </Link>
              </div>
              <p className="lp-disclaimer" data-testid="disclaimer">
                <Icon name="shield" size={15} />
                {t.common.disclaimer}
              </p>
            </div>
          </div>
        </section>

        <section className="lp-stats">
          <div className="lp-inner">
            <FactList summary={summary.data} loading={summary.loading} t={t} className="facts-lp" />
          </div>
        </section>

        <section id="how" className="lp-section">
          <div className="lp-inner">
            <Reveal className="lp-head">
              <h2>{t.intro.howTitle}</h2>
              <p className="lp-sub">{t.intro.howSub}</p>
            </Reveal>
            <Reveal stagger className="lp-steps">
              {t.intro.steps.map((step, index) => (
                <article className="lp-step glass" key={step.title}>
                  <span className="lp-step-num">0{index + 1}</span>
                  <span className="lp-step-icon">
                    <Icon name={STEP_ICONS[index]} size={22} />
                  </span>
                  <h3>{step.title}</h3>
                  <p>{step.text}</p>
                </article>
              ))}
            </Reveal>
          </div>
        </section>

        <section id="safety" className="lp-section">
          <div className="lp-inner">
            <Reveal className="lp-head">
              <h2>{t.intro.rulesTitle}</h2>
            </Reveal>
            <Reveal stagger className="lp-rules">
              {t.intro.rules.map((rule, index) => (
                <div className="lp-rule glass" key={rule.title}>
                  <span className="lp-rule-icon">
                    <Icon name={RULE_ICONS[index]} size={20} />
                  </span>
                  <div>
                    <h3>{rule.title}</h3>
                    <p>{rule.text}</p>
                  </div>
                </div>
              ))}
            </Reveal>
            <Reveal>
              <p className="lp-note">
                <span className="badge badge-neutral">{t.intro.informationOnly}</span>{" "}
                {t.intro.notReplace(APP_NAME)}
              </p>
            </Reveal>
          </div>
        </section>

        <section id="sources" className="lp-section">
          <div className="lp-inner">
            <Reveal className="lp-head">
              <h2>{t.intro.dataTitle}</h2>
              <p className="lp-sub">{t.intro.dataSub}</p>
            </Reveal>
            <Reveal stagger className="lp-sources">
              {t.intro.sources.map((source) => (
                <div className="lp-source glass" key={source.name}>
                  <span className="lp-source-icon">
                    <Icon name="book" size={18} />
                  </span>
                  <strong>{source.name}</strong>
                  <span>{source.gives}</span>
                </div>
              ))}
            </Reveal>
          </div>
        </section>

        <section className="lp-section">
          <Reveal className="lp-inner">
            <div className="lp-final">
              <span className="aurora aurora-1" aria-hidden="true" />
              <LogoMark size={92} className="emblem-float" />
              <h2>{t.intro.readyTitle}</h2>
              <p>{t.intro.readyText}</p>
              <div className="lp-cta lp-cta-center">
                <Link className="btn btn-glow btn-lg" to="/signup">
                  {t.intro.startScanning}
                  <Icon name="arrow" />
                </Link>
                <Link className="btn btn-glass btn-lg" to="/login">
                  {t.common.signIn}
                </Link>
              </div>
            </div>
          </Reveal>
        </section>
      </main>

      <footer className="lp-footer">
        <div className="lp-inner lp-footer-inner">
          <LogoMark size={28} />
          <span>{t.intro.footer(APP_NAME)}</span>
          <nav aria-label={t.shell.information}>
            <Link to="/about">{t.shell.about}</Link>
            <Link to="/privacy">{t.shell.privacy}</Link>
            <Link to="/help">{t.shell.help}</Link>
          </nav>
        </div>
      </footer>
    </div>
  );
}
