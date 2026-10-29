import { useState, type FormEvent } from "react";
import { api, errorMessage, type ToxicityResult, type ToxMolecule } from "../api";
import { Alert, Badge, SubmitButton, TextField, staggerIndex } from "../components";
import { Icon } from "../icons";

// Bonus tool (docs/TOXICITY.md). Safety texts exactly as the project rules write them.
const DISCLAIMER = "Information uniquement, consultez un pharmacien ou un médecin.";
const WORDING = "prédiction d'un modèle sur des tests in vitro, pas une évaluation clinique";
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
  return (
    <article className="card tox-card" style={staggerIndex(index)}>
      <header className="tox-card-head">
        <div>
          <h3>{molecule.ingredient ?? molecule.query}</h3>
          <p className="cell-sub tox-smiles" title="Structure used by the model">
            {molecule.resolved_as === "ingredient" ? "active ingredient · " : "SMILES · "}
            {molecule.smiles}
          </p>
        </div>
        {molecule.in_domain ? (
          <Badge tone="neutral">
            similarity to training molecules {molecule.max_similarity_to_training.toFixed(2)}
          </Badge>
        ) : (
          <Badge tone="warn">
            far from the training molecules ({molecule.max_similarity_to_training.toFixed(2)}):
            even less reliable
          </Badge>
        )}
      </header>
      <table className="tox-table">
        <thead>
          <tr>
            <th scope="col">Tox21 in vitro assay</th>
            <th scope="col">Probability of activity</th>
            <th scope="col" title="How well the model separates actives on held-out scaffolds">
              Model test ROC-AUC
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
                <span className="prob" title="Model probability">
                  <span className="prob-bar">
                    <span style={{ width: `${p.probability * 100}%` }} />
                  </span>
                  {percent(p.probability)}
                </span>
              </td>
              <td className="tox-auc">{p.test_roc_auc?.toFixed(2) ?? "n/a"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </article>
  );
}

export default function ToxicityPage() {
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
            <h1>Toxicity (in vitro model)</h1>
            <p className="muted">
              Bonus: probability that a molecule is active in each of the 12 Tox21 in vitro
              assays, from a model trained on public data. One or two molecules, each predicted
              on its own.
            </p>
          </div>
        </div>
      </header>

      <p className="disclaimer tox-wording">
        <Icon name="alert" size={16} />
        {WORDING.charAt(0).toUpperCase() + WORDING.slice(1)}. {DISCLAIMER}
      </p>

      <section className="card tox-form" aria-label="Molecules">
        <form onSubmit={submit}>
          <div className="tox-fields">
            <TextField
              label="Molecule 1"
              value={first}
              onChange={setFirst}
              hint="Active ingredient name (e.g. warfarin, warfarine) or SMILES"
              required
            />
            <TextField
              label="Molecule 2 (optional)"
              value={second}
              onChange={setSecond}
              hint="Predicted separately: nothing is said about the combination"
            />
          </div>
          <div className="tox-actions">
            <SubmitButton busy={busy}>Predict</SubmitButton>
            <span className="muted">Examples:</span>
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
                {b ? `${a} / ${b}` : a.length > 20 ? "aspirin (SMILES)" : a}
              </button>
            ))}
          </div>
        </form>
      </section>

      {problem && <Alert>{problem}</Alert>}

      {result && (
        <section className="tox-results" aria-label="Predictions">
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
              Model {result.model.name} v{String(result.model.version)} · data:{" "}
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
