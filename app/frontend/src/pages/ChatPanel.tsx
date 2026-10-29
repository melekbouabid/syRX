import { Fragment, useRef, useState, type FormEvent, type ReactNode } from "react";
import { api, errorMessage, type ChatCitation, type ChatResult, type ChatToolCall } from "../api";
import { Alert, Badge } from "../components";
import { Icon } from "../icons";

// Shown exactly as the project rules write it (CLAUDE.md, D53); always visible.
const DISCLAIMER = "Information uniquement, consultez un pharmacien ou un médecin.";

export interface ChatPill {
  class_id: number;
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

const DOC_LABELS: Record<string, string> = {
  label: "openFDA label",
  abstract: "PubMed",
  interaction: "DDInter",
};

function suggestions(pills: ChatPill[]): string[] {
  const names = [...new Set(pills.flatMap((pill) => pill.active_ingredients))];
  if (names.length >= 2) {
    return [
      `Le ${names[0]} et le ${names[1]} interagissent-ils ?`,
      `What are the contraindications of ${names[0]}?`,
      `Quels effets indésirables sont décrits pour le ${names[1]} ?`,
    ];
  }
  if (names.length === 1) {
    return [
      `Quelles sont les contre-indications du ${names[0]} ?`,
      `What is the mechanism of action of ${names[0]}?`,
    ];
  }
  return [
    "La warfarine interagit-elle avec le fluconazole ?",
    "What are the contraindications of metformin?",
  ];
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
        if (line.trim() === DISCLAIMER) return null; // shown once, below every answer
        return <p key={index}>{parts.map((part, i) => <Fragment key={i}>{part}</Fragment>)}</p>;
      })}
    </>
  );
}

function Citations({ citations, turn }: { citations: ChatCitation[]; turn: number }) {
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
          {DOC_LABELS[c.doc_type] ?? c.doc_type}
          {c.section && c.section !== "interaction" ? ` · ${c.section.replaceAll("_", " ")}` : ""}
          {c.date ? ` · ${c.date}` : ""}
        </span>
        <p className="citation-snippet">{c.snippet}</p>
      </div>
    </li>
  );
  return (
    <div className="chat-sources">
      <h3>Sources</h3>
      <ol className="citation-list">{cited.map(item)}</ol>
      {others.length > 0 && (
        <details>
          <summary>{others.length} other source(s) consulted, not cited</summary>
          <ol className="citation-list">{others.map(item)}</ol>
        </details>
      )}
    </div>
  );
}

function ToolCalls({ tools }: { tools: ChatToolCall[] }) {
  if (tools.length === 0) return null;
  return (
    <details className="chat-tools">
      <summary>
        <Icon name="spark" size={14} /> {tools.length} tool call(s)
      </summary>
      <ul>
        {tools.map((tool, index) => (
          <li key={index}>
            <code>{tool.name}</code>
            <span className="muted"> ({tool.chosen_by === "llm" ? "chosen by the model" : "added by rule"})</span>
            {" — "}
            {tool.ok ? tool.summary : <span className="error-text">{tool.summary}</span>}
          </li>
        ))}
      </ul>
    </details>
  );
}

function TurnView({ turn }: { turn: Turn }) {
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
            {turn.tools.length === 0 ? "Reading the question…" : "Searching the sources…"}
          </p>
        ) : null}
        {turn.error && <Alert>{turn.error}</Alert>}
        {result && (
          <div className="chat-meta">
            {action === "rewritten" && <Badge tone="warn">checked: sentence(s) removed</Badge>}
            {action === "refused" && <Badge tone="info">answer refused by the safety check</Badge>}
            <span className="muted">
              {result.model} · {Math.round((result.timings.total_ms ?? 0) / 100) / 10} s
            </span>
          </div>
        )}
      </div>
      <ToolCalls tools={result?.tool_calls ?? turn.tools} />
      {result && <Citations citations={result.citations} turn={turn.id} />}
    </article>
  );
}

export default function ChatPanel({ pills }: { pills: ChatPill[] }) {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const counter = useRef(0);
  const classIds = [...new Set(pills.map((pill) => pill.class_id))];

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
    <section className="card chat-panel" aria-label="Ask about your pills">
      <header className="pill-card-head">
        <h2>
          <Icon name="book" size={18} /> Ask about your pills
        </h2>
        <span className="muted">
          {pills.length > 0
            ? `about: ${[...new Set(pills.flatMap((p) => p.active_ingredients))].join(", ") || "no known ingredient"}`
            : "answers from openFDA labels, PubMed and DDInter"}
        </span>
      </header>
      <p className="disclaimer chat-disclaimer">
        <Icon name="shield" size={16} />
        {DISCLAIMER} Tujya ne donne jamais de dose et ne dit jamais qu'une association est « sûre ».
      </p>
      <form className="chat-form" onSubmit={submit}>
        <input
          type="text"
          value={question}
          maxLength={1000}
          placeholder="Une question sur ces médicaments… (français ou anglais)"
          aria-label="Question"
          onChange={(event) => setQuestion(event.target.value)}
        />
        <button type="submit" className="btn btn-primary" disabled={busy || question.trim().length < 2}>
          {busy && <span className="btn-spinner" aria-hidden="true" />}
          Ask
        </button>
      </form>
      <div className="chat-suggestions">
        {suggestions(pills).map((text) => (
          <button key={text} type="button" className="chip-btn" disabled={busy} onClick={() => void ask(text)}>
            {text}
          </button>
        ))}
      </div>
      {turns.map((turn) => (
        <TurnView key={turn.id} turn={turn} />
      ))}
    </section>
  );
}
