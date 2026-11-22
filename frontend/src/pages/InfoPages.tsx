import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { APP_NAME } from "../brand";
import { Icon, type IconName } from "../components/icons";
import { Disclaimer } from "../components/interactions";
import { useT } from "../i18n";

// About, Privacy and Help: public, in the interface language (texts in src/i18n, info.*).

function InfoPage({ icon, title, lead, children }: { icon: IconName; title: string; lead: string; children: ReactNode }) {
  return (
    <article className="info-page">
      <header className="page-header">
        <div className="page-title">
          <span className="page-icon">
            <Icon name={icon} size={24} />
          </span>
          <div>
            <h1>{title}</h1>
            <p className="muted">{lead}</p>
          </div>
        </div>
      </header>
      <Disclaimer />
      <div className="info-body">{children}</div>
    </article>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="card info-section">
      <h2>{title}</h2>
      {children}
    </section>
  );
}

export function AboutPage() {
  const about = useT().info.about;
  return (
    <InfoPage icon="pill" title={about.title(APP_NAME)} lead={about.lead}>
      <Section title={about.doesTitle}>
        <ul>
          {about.does.map((item) => (
            <li key={item.title}>
              <strong>{item.title}</strong> {item.text}
            </li>
          ))}
        </ul>
      </Section>
      <Section title={about.neverTitle}>
        <ul>
          {about.never.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </Section>
      <Section title={about.projectTitle}>
        <p>{about.project}</p>
      </Section>
    </InfoPage>
  );
}

export function PrivacyPage() {
  const privacy = useT().info.privacy;
  return (
    <InfoPage icon="shield" title={privacy.title} lead={privacy.lead}>
      {privacy.sections.map((section) => (
        <Section key={section.title} title={section.title}>
          <p>{section.text}</p>
        </Section>
      ))}
    </InfoPage>
  );
}

export function HelpPage() {
  const help = useT().info.help;
  return (
    <InfoPage icon="help" title={help.title} lead={help.lead}>
      <Section title={help.stepsTitle}>
        <ol className="help-steps">
          {help.steps.map((step) => (
            <li key={step.title}>
              <strong>{step.title}</strong>
              <p>{step.text}</p>
            </li>
          ))}
          <li>
            <strong>{help.saveTitle}</strong>
            <p>
              {help.saveBefore}
              <Link to="/reports">{help.saveLink}</Link>
              {help.saveAfter}
            </p>
          </li>
        </ol>
      </Section>
      <Section title={help.questionsTitle}>
        {help.questions.map((item) => (
          <details key={item.q}>
            <summary>{item.q}</summary>
            <p>{item.a}</p>
          </details>
        ))}
      </Section>
    </InfoPage>
  );
}
