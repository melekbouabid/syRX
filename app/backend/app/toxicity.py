"""Single-molecule toxicity: one LightGBM model per Tox21 assay on RDKit Morgan fingerprints.

Bonus tool of the agent and the MCP server (docs/TOXICITY.md). The model predicts the
probability that a molecule is *active* in each of the 12 Tox21 in vitro assays; it is
"prédiction d'un modèle sur des tests in vitro, pas une évaluation clinique". Each molecule
is predicted on its own: nothing is ever said about a mixture.

Training: `python scripts/train_tox21.py train` (scaffold split, MLflow `tujya-tox`,
registered model `tujya-tox21`); `python scripts/train_tox21.py publish` copies the
registered bundle to `models/tox21/`, which this module loads on first use:

    models/tox21/model.json        tasks, fingerprint settings, test ROC-AUC per task, version
    models/tox21/<task>.txt        one LightGBM booster per task
    models/tox21/train_fps.npy     packed training fingerprints (applicability domain)

Ingredient names are resolved to structures with `config/ingredient_smiles.csv` (PubChem).
"""

from __future__ import annotations

import csv
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

WORDING = "prédiction d'un modèle sur des tests in vitro, pas une évaluation clinique"
MIXTURE_NOTE = "Chaque molécule est prédite séparément ; ce modèle ne dit rien de leur association."
LOW_NOTE = (
    "Une probabilité basse n'est pas une preuve d'absence de toxicité : le modèle ne couvre "
    "que 12 tests in vitro."
)
SOURCE = {
    "title": "Tox21 (MoleculeNet / DeepChem), 12 in vitro assays",
    "url": "https://tripod.nih.gov/tox21/challenge/",
    "citation": "Huang R. et al., Front. Environ. Sci. 2016; Wu Z. et al., Chem. Sci. 2018 "
    "(MoleculeNet)",
}

# The 12 Tox21 assays (column names of the MoleculeNet file) and what they measure.
TASKS: dict[str, str] = {
    "NR-AR": "androgen receptor, full length (nuclear receptor)",
    "NR-AR-LBD": "androgen receptor, ligand-binding domain",
    "NR-AhR": "aryl hydrocarbon receptor",
    "NR-Aromatase": "aromatase inhibition",
    "NR-ER": "estrogen receptor alpha, full length",
    "NR-ER-LBD": "estrogen receptor alpha, ligand-binding domain",
    "NR-PPAR-gamma": "PPAR-gamma receptor",
    "SR-ARE": "antioxidant response element (oxidative stress)",
    "SR-ATAD5": "ATAD5 (genotoxic stress)",
    "SR-HSE": "heat shock response",
    "SR-MMP": "mitochondrial membrane potential disruption",
    "SR-p53": "p53 response (DNA damage)",
}
FP_RADIUS = 2
FP_BITS = 2048
# Heuristic applicability domain: below this nearest-neighbour Tanimoto similarity to the
# training molecules, the prediction is flagged as outside the model's chemical space.
DOMAIN_SIMILARITY = 0.3
MAX_MOLECULES = 2


class ToxicityUnavailable(RuntimeError):
    """No published model (run `python scripts/train_tox21.py publish`)."""


def standardise(smiles: str) -> Any | None:
    """RDKit molecule of the largest fragment (salts and solvents removed), or None."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles.strip()) if smiles and smiles.strip() else None
    if mol is None:
        return None
    mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    return mol if mol.GetNumAtoms() > 0 else None


def canonical(mol: Any) -> str:
    from rdkit import Chem

    return str(Chem.MolToSmiles(mol))


def fingerprint(mol: Any) -> np.ndarray:
    """Morgan (ECFP4-like) bit vector, radius 2, 2,048 bits, as uint8."""
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    return np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=np.uint8)


def scaffold(mol: Any) -> str:
    """Bemis-Murcko scaffold SMILES (empty for acyclic molecules)."""
    from rdkit.Chem.Scaffolds import MurckoScaffold

    return str(MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False))


def tanimoto_max(query: np.ndarray, packed: np.ndarray) -> float:
    """Highest Tanimoto similarity between one bit vector and packed training vectors."""
    if packed.size == 0:
        return 0.0
    q = np.packbits(query.astype(bool))
    common = np.unpackbits(packed & q, axis=1).sum(axis=1)
    union = np.unpackbits(packed | q, axis=1).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(union > 0, common / union, 0.0)
    return float(sims.max())


Predictor = Callable[[np.ndarray], np.ndarray]


@dataclass
class ToxModel:
    """The 12 per-assay predictors plus what is needed to explain their output."""

    predictors: dict[str, Predictor]
    metrics: dict[str, float]  # test ROC-AUC per task
    version: str = "unregistered"
    train_fps: np.ndarray = field(default_factory=lambda: np.zeros((0, FP_BITS // 8), np.uint8))

    @classmethod
    def from_dir(cls, directory: Path) -> ToxModel:
        meta_file = directory / "model.json"
        if not meta_file.is_file():
            raise ToxicityUnavailable(
                f"no toxicity model in {directory}: run "
                "`python scripts/train_tox21.py publish` (backend venv, MLflow running)"
            )
        import lightgbm as lgb

        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        predictors: dict[str, Predictor] = {}
        for task in meta["tasks"]:
            booster = lgb.Booster(model_file=str(directory / f"{task}.txt"))
            predictors[task] = lambda x, b=booster: np.asarray(b.predict(x), dtype=float)
        fps_file = directory / "train_fps.npy"
        train_fps = np.load(fps_file) if fps_file.is_file() else None
        return cls(
            predictors=predictors,
            metrics={k: float(v) for k, v in meta.get("test_roc_auc", {}).items()},
            version=str(meta.get("version", "unregistered")),
            **({"train_fps": train_fps} if train_fps is not None else {}),
        )

    def predict(self, mol: Any) -> dict[str, Any]:
        fp = fingerprint(mol)
        x = fp.reshape(1, -1).astype(np.float32)
        rows = [
            {
                "task": task,
                "assay": TASKS.get(task, task),
                "probability": round(float(predict(x)[0]), 3),
                "test_roc_auc": round(self.metrics[task], 3) if task in self.metrics else None,
            }
            for task, predict in self.predictors.items()
        ]
        similarity = tanimoto_max(fp, self.train_fps)
        return {
            "predictions": rows,
            "max_similarity_to_training": round(similarity, 3),
            "in_domain": similarity >= DOMAIN_SIMILARITY,
        }


def load_smiles_table(path: Path) -> dict[str, str]:
    """ingredient (lower case) -> SMILES, from config/ingredient_smiles.csv."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as f:
        return {
            row["ingredient"].strip().lower(): row["smiles"].strip()
            for row in csv.DictReader(f)
            if row.get("smiles", "").strip()
        }


class ToxHolder:
    """Loads the model on first use, so the API starts (and the tool says why) without it."""

    def __init__(
        self,
        directory: Path,
        smiles_file: Path,
        *,
        model: ToxModel | None = None,
        smiles: dict[str, str] | None = None,
    ) -> None:
        self.directory, self.smiles_file = directory, smiles_file
        self._model, self._smiles = model, smiles
        self._lock = threading.Lock()

    def model(self) -> ToxModel:
        with self._lock:
            if self._model is None:
                self._model = ToxModel.from_dir(self.directory)
            return self._model

    def smiles(self) -> dict[str, str]:
        if self._smiles is None:
            self._smiles = load_smiles_table(self.smiles_file)
        return self._smiles


def predict_molecules(
    holder: ToxHolder,
    molecules: list[str],
    resolve_name: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Predict each molecule (an ingredient name or a SMILES) separately; at most two."""
    from app.medicines import DISCLAIMER

    wanted = [m.strip() for m in molecules if m and m.strip()][:MAX_MOLECULES]
    model = holder.model()
    table = holder.smiles()
    results, errors = [], []
    for query in wanted:
        name = (resolve_name(query) if resolve_name else None) or query.lower()
        if name in table:
            smiles, kind, ingredient = table[name], "ingredient", name
        else:
            smiles, kind, ingredient = query, "smiles", None
        mol = standardise(smiles)
        if mol is None:
            errors.append(
                {
                    "query": query,
                    "error": "neither a known active ingredient nor a valid SMILES "
                    "(or an ingredient without a single structure, e.g. a multivitamin)",
                }
            )
            continue
        results.append(
            {
                "query": query,
                "resolved_as": kind,
                "ingredient": ingredient,
                "smiles": canonical(mol),
                **model.predict(mol),
            }
        )
    return {
        "wording": WORDING,
        "molecules": results,
        "errors": errors,
        "mixture_note": MIXTURE_NOTE,
        "low_probability_note": LOW_NOTE,
        "model": {"name": "tujya-tox21", "version": model.version, "tasks": len(TASKS)},
        "source": SOURCE,
        "disclaimer": DISCLAIMER,
    }
