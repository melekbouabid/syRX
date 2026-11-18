// Interaction check results, shared by the Scanner and the saved reports (owner and shared
// views). Safety texts are shown exactly as the project rules write them (docs/project_rules.md).
import { type InteractionPair, type InteractionsResult, type Severity, type Source } from "../services/api";
import { Badge, staggerIndex } from "./index";
import { Icon } from "./icons";

export const DISCLAIMER = "Information uniquement, consultez un pharmacien ou un médecin.";

export const SEVERITY_TONE: Record<Severity, string> = {
  Major: "danger",
  Moderate: "warn",
  Minor: "info",
  Unknown: "neutral",
};

const SEVERITY_TEXT: Record<Severity, string> = {
  Major: "Major",
  Moderate: "Moderate",
  Minor: "Minor",
  Unknown: "Unknown severity (DDInter lists the interaction without grading it)",
};

export function Disclaimer({ text = DISCLAIMER }: { text?: string }) {
  return (
    <p className="disclaimer">
      <Icon name="shield" size={16} />
      {text}
    </p>
  );
}

export function SourceList({ sources }: { sources: Source[] }) {
  if (sources.length === 0) return null;
  return (
    <div className="cited-sources">
      <h4>Sources</h4>
      <ul>
        {sources.map((source) => (
          <li key={source.name}>
            <a href={source.url} target="_blank" rel="noreferrer">
              {source.title}
            </a>{" "}
            <span className="muted">({source.license})</span>
            {source.citation && <span className="cell-sub">{source.citation}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}

function PairCard({ pair, index }: { pair: InteractionPair; index: number }) {
  const tone = pair.highest_severity ? SEVERITY_TONE[pair.highest_severity] : "none";
  return (
    <article className={`card pair-card pair-${tone}`} style={staggerIndex(index)}>
      <header className="pair-head">
        <h3>
          {pair.item_a.label} <span className="muted">+</span> {pair.item_b.label}
        </h3>
        {pair.highest_severity && (
          <Badge tone={SEVERITY_TONE[pair.highest_severity]}>{pair.highest_severity}</Badge>
        )}
      </header>
      {pair.message && <p className="pair-message">{pair.message}</p>}
      {pair.interactions.length > 0 && (
        <ul className="interaction-list">
          {pair.interactions.map((found) => (
            <li key={`${found.ingredient_a}-${found.ingredient_b}`}>
              <strong>
                {found.ingredient_a} + {found.ingredient_b}
              </strong>
              : {SEVERITY_TEXT[found.severity]}. {found.description}{" "}
              <a href={found.source.url} target="_blank" rel="noreferrer">
                Source: {found.source.title}
              </a>
            </li>
          ))}
        </ul>
      )}
      {pair.shared_ingredients.length > 0 && (
        <p className="pill-note">
          <Icon name="alert" size={16} /> Same active ingredient in both:{" "}
          {pair.shared_ingredients.join(", ")}.
        </p>
      )}
    </article>
  );
}

export function InteractionReport({ report }: { report: InteractionsResult }) {
  const counts = report.pairs.reduce<Record<string, number>>((total, pair) => {
    const key = pair.highest_severity ?? "none";
    total[key] = (total[key] ?? 0) + 1;
    return total;
  }, {});
  return (
    <section className="report" aria-live="polite">
      <div className={`report-summary report-${report.highest_severity ? SEVERITY_TONE[report.highest_severity] : "none"}`}>
        <p className="report-message">{report.message}</p>
        <span className="badge-row">
          {(["Major", "Moderate", "Minor", "Unknown"] as Severity[]).map(
            (severity) =>
              counts[severity] && (
                <Badge key={severity} tone={SEVERITY_TONE[severity]}>
                  {counts[severity]} {severity}
                </Badge>
              ),
          )}
        </span>
      </div>
      {report.pairs.map((pair, index) => (
        <PairCard key={`${pair.item_a.label}|${pair.item_b.label}`} pair={pair} index={index} />
      ))}
      <SourceList sources={report.sources} />
      <Disclaimer text={report.disclaimer} />
    </section>
  );
}
