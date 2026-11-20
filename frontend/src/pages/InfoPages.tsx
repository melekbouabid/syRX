import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { APP_NAME } from "../brand";
import { Icon, type IconName } from "../components/icons";
import { DISCLAIMER } from "../components/interactions";

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
      <p className="disclaimer">
        <Icon name="shield" size={16} />
        {DISCLAIMER}
      </p>
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
  return (
    <InfoPage
      icon="pill"
      title={`About ${APP_NAME}`}
      lead="A medicine-safety assistant: identify a pill from a photo, check interactions, ask questions answered from cited sources."
    >
      <Section title="What it does">
        <ul>
          <li>
            <strong>Identify.</strong> A YOLO model finds each pill in the photo; a classifier
            trained by transfer learning suggests which of the 380 NLM C3PI pills it is, with its
            confidence. A medicine box is read by OCR instead.
          </li>
          <li>
            <strong>Check.</strong> The active ingredients of the confirmed pills are compared pair
            by pair with the DDInter 2.0 interaction table, each finding with its source.
          </li>
          <li>
            <strong>Ask.</strong> An agent (LangGraph) answers follow-up questions from openFDA
            labels, PubMed abstracts and DDInter, and cites them.
          </li>
          <li>
            <strong>Keep.</strong> Save a check as a report, print it, or share a read-only link
            with a pharmacist or a doctor.
          </li>
        </ul>
      </Section>
      <Section title="What it never does">
        <ul>
          <li>It never gives a dose or a schedule.</li>
          <li>
            It never says a combination is “safe”: at most, “aucune interaction connue dans cette
            base de données”.
          </li>
          <li>An identification is a suggestion, never a certainty: the confidence is always shown.</li>
        </ul>
      </Section>
      <Section title="The project">
        <p>
          A student project graded in two modules, <em>Industrialisation Vision &amp; MLOps</em> and{" "}
          <em>AI Engineering</em>. It runs only on the author's computer; the code is published
          for assessment. Data sources keep their own licences (C3PI, DDInter, openFDA, PubMed,
          Tox21).
        </p>
      </Section>
    </InfoPage>
  );
}

export function PrivacyPage() {
  return (
    <InfoPage
      icon="shield"
      title="Privacy"
      lead="What Tujya keeps about you, who can see it, and for how long."
    >
      <Section title="Your account">
        <p>
          Your name, email, optional phone number and photo, and your password stored as an
          Argon2 hash (it cannot be read back, only replaced). Administrators can see your name,
          email, phone and photo, never your password or your signature.
        </p>
      </Section>
      <Section title="Photos you scan">
        <p>
          A scanned photo is analysed and then forgotten: it is never stored. To watch for data
          drift, eight numbers describing the photo (brightness, contrast…) are kept in memory
          only, with nothing that links them to you.
        </p>
      </Section>
      <Section title="My medicines and reports">
        <p>
          The medicines you save and the reports you create are yours: only you can open them.
          Administrators only see how many reports exist. A shared link lets anyone who has it
          read one report, read-only, until it expires or you stop sharing it.
        </p>
      </Section>
      <Section title="The activity log">
        <p>
          To keep the service safe, administrators see a log of what accounts did and when
          (signed in, scanned a photo, asked a question, saved a report), with the address and
          browser used. It holds counts only: never a photo, a medicine name or the text of a
          question. Events older than 180 days are deleted.
        </p>
      </Section>
      <Section title="Deleting your data">
        <p>
          Deleting an account deletes its profile, saved medicines, reports and activity. Ask an
          administrator to delete yours.
        </p>
      </Section>
    </InfoPage>
  );
}

const STEPS: { title: string; text: ReactNode }[] = [
  {
    title: "Take or choose a photo",
    text: "In the Scanner, upload a photo, use the webcam, or pick one from the demo gallery. One pill per photo works best: whole, sharp, on a plain background.",
  },
  {
    title: "Confirm each pill",
    text: "Check the suggestion against the pill in your hand (imprint, colour, shape). Below the confirmation threshold, Tujya asks you to choose among the five best guesses or to search the list.",
  },
  {
    title: "Check the interactions",
    text: "With two or more confirmed pills (or medicine boxes), choose “Check interactions”. Each pair is shown with its DDInter severity and source.",
  },
  {
    title: "Ask a question",
    text: "Ask about your pills in French or English. Each answer cites its sources; mark the ones you want to keep with “Add to report”.",
  },
  {
    title: "Save, print or share",
    text: (
      <>
        “Save as report” keeps the check as it is today. Open it in <Link to="/reports">Reports</Link>{" "}
        to print it (or save it as PDF) or to create a read-only link.
      </>
    ),
  },
];

export function HelpPage() {
  return (
    <InfoPage icon="help" title="Help" lead="How to use Tujya, step by step, and answers to common questions.">
      <Section title="Step by step">
        <ol className="help-steps">
          {STEPS.map((step) => (
            <li key={step.title}>
              <strong>{step.title}</strong>
              <p>{step.text}</p>
            </li>
          ))}
        </ol>
      </Section>
      <Section title="Questions">
        <details>
          <summary>The pill is not recognised.</summary>
          <p>
            Tujya knows the 380 pills of the NLM C3PI collection; others cannot be identified. Search
            the list by name or ingredient, or photograph the face of the box where the name is
            printed.
          </p>
        </details>
        <details>
          <summary>“Aucune interaction connue” — is the combination safe?</summary>
          <p>
            No: it only means the DDInter table lists nothing for that pair. Ask a pharmacist or a
            doctor.
          </p>
        </details>
        <details>
          <summary>Who can read a shared report?</summary>
          <p>
            Anyone with the link, until it expires (1 to 30 days). A new link replaces the old one;
            “Stop sharing” ends it at once.
          </p>
        </details>
        <details>
          <summary>I forgot my password.</summary>
          <p>An administrator can set a new one; you can then change it in your profile.</p>
        </details>
      </Section>
    </InfoPage>
  );
}
