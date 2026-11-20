import type { ReactNode } from "react";
import type { ReportBody } from "../services/api";
import { Badge, formatDate, Logo } from "./index";
import { Disclaimer, InteractionReport, SEVERITY_TONE } from "./interactions";

/** Each [n] of a saved answer as a superscript reference to its numbered source. */
function AnswerText({ text }: { text: string }) {
  return (
    <>
      {text
        .split("\n")
        .filter((line) => line.trim() !== "")
        .map((line, index) => {
          const parts: ReactNode[] = [];
          let last = 0;
          for (const match of line.matchAll(/\[(\d+)\]/g)) {
            parts.push(line.slice(last, match.index));
            parts.push(<sup key={`${index}-${match.index}`}>[{match[1]}]</sup>);
            last = (match.index ?? 0) + match[0].length;
          }
          parts.push(line.slice(last));
          return <p key={index}>{parts}</p>;
        })}
    </>
  );
}

/**
 * A saved report, as its owner sees it and as a shared link shows it; also the printed page
 * (styles.css, @media print). It shows what the check found on the day it was saved.
 */
export function ReportDocument({ report, actions }: { report: ReportBody; actions?: ReactNode }) {
  const severity = report.highest_severity;
  return (
    <article className="report-doc">
      <header className="report-doc-head">
        <Logo height={44} />
        <div className="report-doc-title">
          <p className="eyebrow">Interaction report</p>
          <h1>{report.title}</h1>
          <p className="muted">
            Saved on {formatDate(report.created_at, true)} by {report.author}
          </p>
        </div>
        {severity ? (
          <Badge tone={SEVERITY_TONE[severity]}>Highest: {severity}</Badge>
        ) : (
          <Badge tone="neutral">{report.interactions ? "No known interaction" : "Single item"}</Badge>
        )}
      </header>
      {actions && <div className="report-doc-actions no-print">{actions}</div>}

      <section className="report-doc-section">
        <h2>Medicines checked</h2>
        <table className="table report-table">
          <thead>
            <tr>
              <th>Medicine</th>
              <th>Active ingredients</th>
            </tr>
          </thead>
          <tbody>
            {report.pills.map((pill) => (
              <tr key={pill.class_id}>
                <td>
                  <span className="cell-title">{pill.name}</span>
                  {pill.generic_name && <span className="cell-sub">{pill.generic_name}</span>}
                </td>
                <td>{pill.active_ingredients.join(", ") || "none known in this database"}</td>
              </tr>
            ))}
            {report.ingredients.map((name) => (
              <tr key={name}>
                <td>
                  <span className="cell-title">{name}</span>
                  <span className="cell-sub">active ingredient (medicine box or typed)</span>
                </td>
                <td>{name}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="report-doc-section">
        <h2>Interactions</h2>
        {report.interactions ? (
          <InteractionReport report={report.interactions} />
        ) : (
          <p className="muted">Only one medicine: there is nothing to compare.</p>
        )}
      </section>

      {report.chat.length > 0 && (
        <section className="report-doc-section">
          <h2>Questions to the assistant</h2>
          {report.chat.map((excerpt, index) => (
            <div key={index} className="report-chat">
              <p className="report-chat-q">{excerpt.question}</p>
              <div className="report-chat-a">
                <AnswerText text={excerpt.answer} />
              </div>
              {excerpt.citations.length > 0 && (
                <ol className="report-citations">
                  {excerpt.citations.map((citation) => (
                    <li key={citation.n} value={citation.n}>
                      <a href={citation.url} target="_blank" rel="noreferrer">
                        {citation.title}
                      </a>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          ))}
        </section>
      )}

      <footer className="report-doc-foot">
        {/* With a check, its own block already ends with the disclaimer. */}
        {report.interactions ? <span /> : <Disclaimer text={report.disclaimer} />}
        <div className="report-signature">
          {report.signature ? (
            <img src={report.signature} alt={`Signature of ${report.author}`} />
          ) : (
            <span className="report-signature-line" aria-hidden="true" />
          )}
          <span>
            {report.author} · {formatDate(report.created_at)}
          </span>
        </div>
      </footer>
    </article>
  );
}
