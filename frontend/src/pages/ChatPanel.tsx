import { Fragment, useRef, useState, type FormEvent, type ReactNode } from "react";
import { api, errorMessage, type ChatCitation, type ChatResult, type ChatToolCall } from "../services/api";
import { Alert, Badge } from "../components";
import { Icon } from "../components/icons";
import { useT, type Messages } from "../i18n";

// The notice the agent appends to every answer (src/agent/safety.py): removed from the
// answer text because the panel always shows the notice once, in the interface language.
const AGENT_DISCLAIMER = "Information uniquement, consultez un pharmacien ou un médecin.";

export interface ChatPill {
  class_id: number | null; // null for a medicine box (its ingredients are used instead)
  name: string;
  active_ingredients: string[];
}

interface Turn {
  id: number;
  question: string;
  streamed: string; // checked sentences received so far
  tools: ChatToolCall[];
  result: ChatResult | null;
  error: string | null;
}

/** Questions to start with, in the interface language (the agent answers in both). */
function suggestions(pills: ChatPill[], t: Messages): string[] {
  const names = [...new Set(pills.flatMap((pill) => pill.active_ingredients))];
  if (names.length >= 2) return t.chat.suggestTwo(names[0], names[1]);
  if (names.length === 1) return t.chat.suggestOne(names[0]);
  return t.chat.suggestNone;
}

/** The answer text with each [n] turned into a link to its numbered source below. */
function AnswerText({ text, turn }: { text: string; turn: number }) {
  const lines = text.split("\n").filter((line) => line.trim() !== "");
  return (
    <>
      {lines.map((line, index) => {
        const parts: ReactNode[] = [];
        let last = 0;
        for (const match of line.matchAll(/\[(\d+)\]/g)) {
          parts.push(line.slice(last, match.index));
          const n = match[1];
          parts.push(
            <a key={`${index}-${match.index}`} className="cite" href={`#source-${turn}-${n}`}>
              {n}
            </a>,
          );
          last = (match.index ?? 0) + match[0].length;
        }
        parts.push(line.slice(last));
        if (line.trim() === AGENT_DISCLAIMER) return null; // shown once, above the answers
        return <p key={index}>{parts.map((part, i) => <Fragment key={i}>{part}</Fragment>)}</p>;
      })}
    </>
  );
}

function Citations({ citations, turn }: { citations: ChatCitation[]; turn: number }) {
  const t = useT();
  if (citations.length === 0) return null;
  const cited = citations.filter((c) => c.cited);
  const others = citations.filter((c) => !c.cited);
  const item = (c: ChatCitation) => (
    <li key={c.n} id={`source-${turn}-${c.n}`} className={c.cited ? "citation" : "citation citation-other"}>
      <span className="citation-n">{c.n}</span>
      <div>
        <a href={c.url} target="_blank" rel="noreferrer">
          {c.title}
        </a>
        <span className="cell-sub">
          {t.chat.docLabels[c.doc_type] ?? c.doc_type}
          {c.section && c.section !== "interaction" && c.section !== "brand" ? ` · ${c.section.replaceAll("_", " ")}` : ""}
          {c.date ? ` · ${c.date}` : ""}
        </span>
        <span className="cell-sub">
          {c.source_name}
          {c.license ? ` · ${c.license}` : ""}
          {c.rerank_score != null ? t.chat.reranker(c.rerank_score.toFixed(2)) : ""}
        </span>
        <p className="citation-snippet">{c.snippet}</p>
      </div>
    </li>
  );
  return (
    <div className="chat-sources">
      <h3>{t.common.sources}</h3>
      <ol className="citation-list">{cited.map(item)}</ol>
      {others.length > 0 && (
        <details>
          <summary>{t.chat.others(others.length)}</summary>
          <ol className="citation-list">{others.map(item)}</ol>
        </details>
      )}
    </div>
  );
}

function ToolCalls({ tools }: { tools: ChatToolCall[] }) {
  const t = useT();
  if (tools.length === 0) return null;
  return (
    <details className="chat-tools">
      <summary>
        <Icon name="spark" size={14} /> {t.chat.tools(tools.length)}
      </summary>
      <ul>
        {tools.map((tool, index) => (
          <li key={index}>
            <code>{tool.name}</code>
            <span className="muted"> ({tool.chosen_by === "llm" ? t.chat.byModel : t.chat.byRule})</span>
            {" — "}
            {tool.ok ? tool.summary : <span className="error-text">{tool.summary}</span>}
          </li>
        ))}
      </ul>
    </details>
  );
}

interface KeepProps {
  /** Answers chosen for the next saved report (Scanner page). */
  kept?: ChatResult[];
  onKeep?: (result: ChatResult) => void;
}

function TurnView({ turn, kept = [], onKeep }: { turn: Turn } & KeepProps) {
  const t = useT();
  const result = turn.result;
  const busy = !result && !turn.error;
  const action = result?.safety.action;
  return (
    <article className="chat-turn" aria-busy={busy}>
      <p className="chat-question">
        <Icon name="user" size={16} />
        {turn.question}
      </p>
      <div className="chat-answer" aria-live="polite">
        {result ? (
          <AnswerText text={result.answer} turn={turn.id} />
        ) : turn.streamed ? (
          <AnswerText text={turn.streamed} turn={turn.id} />
        ) : busy ? (
          <p className="muted chat-thinking">
            <span className="btn-spinner" aria-hidden="true" />
            {turn.tools.length === 0 ? t.chat.reading : t.chat.searching}
          </p>
        ) : null}
        {turn.error && <Alert>{turn.error}</Alert>}
        {result && (
          <div className="chat-meta">
            {action === "rewritten" && <Badge tone="warn">{t.chat.rewritten}</Badge>}
            {action === "refused" && <Badge tone="info">{t.chat.refused}</Badge>}
            <span className="muted">
              {result.model} · {Math.round((result.timings.total_ms ?? 0) / 100) / 10} s
            </span>
            {onKeep && result.seal && (
              <button
                type="button"
                className={`chip-btn ${kept.includes(result) ? "chip-btn-on" : ""}`}
                aria-pressed={kept.includes(result)}
                onClick={() => onKeep(result)}
              >
                <Icon name="bookmark" size={14} />{" "}
                {kept.includes(result) ? t.chat.inReport : t.chat.addToReport}
              </button>
            )}
          </div>
        )}
      </div>
      <ToolCalls tools={result?.tool_calls ?? turn.tools} />
      {result && <Citations citations={result.citations} turn={turn.id} />}
    </article>
  );
}

export default function ChatPanel({ pills, kept, onKeep }: { pills: ChatPill[] } & KeepProps) {
  const t = useT();
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const counter = useRef(0);
  const classIds = [
    ...new Set(pills.flatMap((pill) => (pill.class_id === null ? [] : [pill.class_id]))),
  ];

  const update = (id: number, change: (turn: Turn) => Turn) =>
    setTurns((current) => current.map((turn) => (turn.id === id ? change(turn) : turn)));

  const ask = async (text: string) => {
    const asked = text.trim();
    if (asked.length < 2 || busy) return;
    const id = ++counter.current;
    setTurns((current) => [
      { id, question: asked, streamed: "", tools: [], result: null, error: null },
      ...current,
    ]);
    setQuestion("");
    setBusy(true);
    try {
      await api.chatStream(asked, classIds, (event) => {
        if (event.type === "tool") update(id, (t) => ({ ...t, tools: [...t.tools, event.tool] }));
        else if (event.type === "token") update(id, (t) => ({ ...t, streamed: t.streamed + event.text }));
        else if (event.type === "final") update(id, (t) => ({ ...t, result: event.result }));
        else update(id, (t) => ({ ...t, error: event.detail }));
      });
    } catch (failure) {
      update(id, (t) => ({ ...t, error: errorMessage(failure) }));
    } finally {
      setBusy(false);
    }
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    void ask(question);
  };

  return (
    <section className="card chat-panel" aria-label={t.chat.title}>
      <header className="pill-card-head">
        <h2>
          <Icon name="book" size={18} /> {t.chat.title}
        </h2>
        <span className="muted">
          {pills.length > 0
            ? t.chat.about(
                [...new Set(pills.flatMap((p) => p.active_ingredients))].join(", ") ||
                  t.chat.noIngredient,
              )
            : t.chat.fromSources}
        </span>
      </header>
      <p className="disclaimer chat-disclaimer">
        <Icon name="shield" size={16} />
        {t.common.disclaimer} {t.chat.rules}
      </p>
      <form className="chat-form" onSubmit={submit}>
        <input
          type="text"
          value={question}
          maxLength={1000}
          placeholder={t.chat.placeholder}
          aria-label={t.chat.question}
          onChange={(event) => setQuestion(event.target.value)}
        />
        <button type="submit" className="btn btn-primary" disabled={busy || question.trim().length < 2}>
          {busy && <span className="btn-spinner" aria-hidden="true" />}
          {t.chat.ask}
        </button>
      </form>
      <div className="chat-suggestions">
        {suggestions(pills, t).map((text) => (
          <button key={text} type="button" className="chip-btn" disabled={busy} onClick={() => void ask(text)}>
            {text}
          </button>
        ))}
      </div>
      {turns.map((turn) => (
        <TurnView key={turn.id} turn={turn} kept={kept} onKeep={onKeep} />
      ))}
    </section>
  );
}
