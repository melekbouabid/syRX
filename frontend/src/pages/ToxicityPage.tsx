import { useState, type FormEvent } from "react";
import { api, errorMessage, type ToxicityResult, type ToxMolecule } from "../services/api";
import { Alert, Badge, SubmitButton, TextField, staggerIndex } from "../components";
import { Icon } from "../components/icons";
import { useT } from "../i18n";

// Bonus tool (docs/module2/README.md). Safety texts in src/i18n (toxicity.wording, the notice).
const EXAMPLES: [string, string][] = [
  ["warfarin", "fluconazole"],
  ["citalopram", ""],
  ["CC(=O)OC1=CC=CC=C1C(=O)O", ""],
];

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** One molecule, on its own: never anything about the combination. */
function MoleculeCard({ molecule, index }: { molecule: ToxMolecule; index: number }) {
  const t = useT();
  const x = t.toxicity;
  return (
    <article className="card tox-card" style={staggerIndex(index)}>
      <header className="tox-card-head">
        <div>
          <h3>{molecule.ingredient ?? molecule.query}</h3>
          <p className="cell-sub tox-smiles" title={x.structure}>
            {molecule.resolved_as === "ingredient" ? x.ingredient : x.smiles}
            {molecule.smiles}
          </p>
        </div>
        {molecule.in_domain ? (
          <Badge tone="neutral">{x.similarity(molecule.max_similarity_to_training.toFixed(2))}</Badge>
        ) : (
          <Badge tone="warn">{x.far(molecule.max_similarity_to_training.toFixed(2))}</Badge>
        )}
      </header>
      <table className="tox-table">
        <thead>
          <tr>
            <th scope="col">{x.assay}</th>
            <th scope="col">{x.probability}</th>
            <th scope="col" title={x.aucHint}>
              {x.auc}
            </th>
          </tr>
        </thead>
        <tbody>
          {molecule.predictions.map((p) => (
            <tr key={p.task}>
              <td>
                <span className="cell-title">{p.task}</span>
                <span className="cell-sub">{p.assay}</span>
              </td>
              <td>
                <span className="prob" title={x.modelProbability}>
                  <span className="prob-bar">
                    <span style={{ width: `${p.probability * 100}%` }} />
                  </span>
                  {percent(p.probability)}
                </span>
              </td>
              <td className="tox-auc">{p.test_roc_auc?.toFixed(2) ?? t.common.na}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </article>
  );
}

export default function ToxicityPage() {
  const t = useT();
  const x = t.toxicity;
  const [first, setFirst] = useState("warfarin");
  const [second, setSecond] = useState("");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  const [result, setResult] = useState<ToxicityResult | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    const molecules = [first, second].map((m) => m.trim()).filter(Boolean);
    if (!molecules.length) return;
    setBusy(true);
    setProblem(null);
    try {
      setResult(await api.toxicity(molecules));
    } catch (error) {
      setResult(null);
      setProblem(errorMessage(error));
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <header className="page-header">
        <div className="page-title">
          <span className="page-icon">
            <Icon name="flask" size={24} />
          </span>
          <div>
            <h1>{x.title}</h1>
            <p className="muted">{x.lead}</p>
          </div>
        </div>
      </header>

      <p className="disclaimer tox-wording">
        <Icon name="alert" size={16} />
        {x.wording}. {t.common.disclaimer}
      </p>

      <section className="card tox-form" aria-label={x.molecules}>
        <form onSubmit={submit}>
          <div className="tox-fields">
            <TextField
              label={x.molecule1}
              value={first}
              onChange={setFirst}
              hint={x.molecule1Hint}
              required
            />
            <TextField
              label={x.molecule2}
              value={second}
              onChange={setSecond}
              hint={x.molecule2Hint}
            />
          </div>
          <div className="tox-actions">
            <SubmitButton busy={busy}>{x.predict}</SubmitButton>
            <span className="muted">{x.examples}</span>
            {EXAMPLES.map(([a, b]) => (
              <button
                type="button"
                key={`${a}+${b}`}
                className="btn btn-ghost btn-sm"
                onClick={() => {
                  setFirst(a);
                  setSecond(b);
                }}
              >
                {b ? `${a} / ${b}` : a.length > 20 ? x.aspirin : a}
              </button>
            ))}
          </div>
        </form>
      </section>

      {problem && <Alert>{problem}</Alert>}

      {result && (
        <section className="tox-results" aria-label={x.predictions}>
          {result.errors.map((e) => (
            <Alert key={e.query} kind="info">
              {e.query}: {e.error}
            </Alert>
          ))}
          {result.molecules.map((molecule, index) => (
            <MoleculeCard key={molecule.query} molecule={molecule} index={index} />
          ))}
          <div className="card tox-notes">
            <p>
              <strong>{result.wording.charAt(0).toUpperCase() + result.wording.slice(1)}.</strong>{" "}
              {result.low_probability_note}
            </p>
            {result.molecules.length > 1 && <p>{result.mixture_note}</p>}
            <p className="cell-sub">
              {x.modelLine(result.model.name, String(result.model.version))}{" "}
              <a href={result.source.url} target="_blank" rel="noreferrer">
                {result.source.title}
              </a>{" "}
              ({result.source.citation})
            </p>
            <p className="disclaimer">
              <Icon name="shield" size={16} />
              {result.disclaimer}
            </p>
          </div>
        </section>
      )}
    </>
  );
}
