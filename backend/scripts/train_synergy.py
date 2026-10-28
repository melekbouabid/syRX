"""Bonus: can drug-combination synergy (DrugComb, ZIP) be predicted for unseen drugs?

cd backend                       # backend venv (rdkit, lightgbm, mlflow)
python scripts/train_synergy.py  # MLflow experiment tujya-synergy; docs/SYNERGY.md

Data: the SynergyRx pipeline tables in data/parquet (synergy_pairs, drugs, expression).
Decisions in docs/DECISIONS.md (D75): DrugComb is the single source per experiment (TDC is
a copy of it, O'Neil disagrees), target ZIP, only drugs with a valid structure,
leave-drug-out split (15 % of the drugs for test, 10 % for validation). Features: the sum
of the two drugs' Morgan fingerprints (order-free, 1,024 bits), 30 principal components of
the cell line's expression (missing for cell lines without expression) and its tissue.
Compared with the mean baseline (the training mean) and, for context, the per-cell-line
mean. The rule for adding a tool is fixed in D75: RMSE at least 10 % below the mean
baseline and Pearson r >= 0.3 on the test set.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.toxicity import standardise

ROOT = Path(__file__).resolve().parents[2]
PARQUET = ROOT / "data" / "parquet"
EXPERIMENT = "tujya-synergy"
TARGET = "synergy_zip"
SEED = 7
FP_BITS = 1024
PCS = 30
TEST_DRUGS, VALID_DRUGS = 0.15, 0.10
PARAMS: dict[str, Any] = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
}
MAX_ROUNDS, EARLY_STOP = 3000, 100


def morgan(smiles: str) -> np.ndarray | None:
    from rdkit.Chem import rdFingerprintGenerator

    mol = standardise(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    return np.asarray(generator.GetFingerprintAsNumPy(mol), dtype=np.uint8)


def load() -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    pairs = pd.read_parquet(PARQUET / "synergy_pairs.parquet")
    for column in ("drug_a_id", "drug_b_id", "cell_line_id", "source", "tissue"):
        pairs[column] = pairs[column].astype(str)
    pairs["a"] = np.minimum(pairs.drug_a_id.to_numpy(), pairs.drug_b_id.to_numpy())
    pairs["b"] = np.maximum(pairs.drug_a_id.to_numpy(), pairs.drug_b_id.to_numpy())
    stats: dict[str, Any] = {"rows": len(pairs)}
    for source, n in pairs["source"].value_counts().items():
        stats[f"rows_{source}"] = int(n)
    # One source per experiment: DrugComb (every experiment is there; TDC copies it).
    dc = pairs[pairs["source"] == "drugcomb"].copy()
    keys = ["a", "b", "cell_line_id"]
    stats["experiments"] = int(pairs.drop_duplicates(keys).shape[0])
    dc = dc.groupby(keys, as_index=False).agg(
        {TARGET: "mean", "tissue": "first", "depmap_id": "first"}
    )
    stats["drugcomb_experiments"] = len(dc)

    drugs = pd.read_parquet(PARQUET / "drugs.parquet")
    drugs = drugs[drugs["smiles_valid"].fillna(False).astype(bool)]
    fps = {}
    for row in drugs.itertuples(index=False):
        fp = morgan(str(row.smiles))
        if fp is not None:
            fps[str(row.drug_id)] = fp
    dc = dc[dc["a"].isin(fps) & dc["b"].isin(fps) & dc[TARGET].notna()].reset_index(drop=True)
    stats.update(
        drugs_with_structure=len(fps),
        modelled_experiments=len(dc),
        modelled_drugs=len(set(dc.a) | set(dc.b)),
        modelled_cell_lines=int(dc.cell_line_id.nunique()),
    )
    return dc, fps, stats


def cell_features(dc: pd.DataFrame) -> pd.DataFrame:
    """30 expression PCs (log values standardised; PCA on the cell lines, no labels) + tissue."""
    expr = pd.read_parquet(PARQUET / "expression.parquet").set_index("depmap_id")
    expr = expr.select_dtypes("number")  # drop the provenance columns
    x = expr.to_numpy(dtype=np.float64)
    x = x[:, x.std(axis=0) > 0]
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    pcs = pd.DataFrame(
        u[:, :PCS] * s[:PCS], index=expr.index, columns=[f"pc{i}" for i in range(PCS)]
    )
    cells = dc[["cell_line_id", "depmap_id", "tissue"]].drop_duplicates("cell_line_id")
    cells = cells.join(pcs, on="depmap_id")
    cells["tissue_code"] = cells["tissue"].astype("category").cat.codes
    return cells.set_index("cell_line_id")


def split_by_drug(dc: pd.DataFrame) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    drugs = np.array(sorted(set(dc.a) | set(dc.b)))
    rng.shuffle(drugs)
    n_test, n_valid = round(TEST_DRUGS * len(drugs)), round(VALID_DRUGS * len(drugs))
    test, valid = set(drugs[:n_test]), set(drugs[n_test : n_test + n_valid])
    in_test = dc.a.isin(test) | dc.b.isin(test)
    in_valid = ~in_test & (dc.a.isin(valid) | dc.b.isin(valid))
    split = np.where(in_test, "test", np.where(in_valid, "valid", "train"))
    unseen = dc.a.isin(test).astype(int) + dc.b.isin(test).astype(int)
    return np.where(split == "test", np.where(unseen == 2, "test2", "test1"), split)


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    pearson = float(np.corrcoef(y, p)[0, 1]) if np.std(p) > 0 else 0.0
    return {"rmse": round(rmse, 4), "pearson": round(pearson, 4), "n": len(y)}


def main() -> None:
    import lightgbm as lgb
    import mlflow

    started = time.time()
    dc, fps, stats = load()
    cells = cell_features(dc)
    split = split_by_drug(dc)
    cell_cols = [f"pc{i}" for i in range(PCS)] + ["tissue_code"]
    fa = np.stack([fps[d] for d in dc.a])
    fb = np.stack([fps[d] for d in dc.b])
    x = np.hstack(
        [(fa + fb).astype(np.float32), cells.loc[dc.cell_line_id, cell_cols].to_numpy(np.float32)]
    )
    y = dc[TARGET].to_numpy(dtype=np.float64)
    del fa, fb
    masks = {s: split == s for s in ("train", "valid", "test1", "test2")}
    masks["test"] = masks["test1"] | masks["test2"]
    sizes = {s: int(m.sum()) for s, m in masks.items()}
    print(json.dumps(stats), sizes)

    tissue_col = x.shape[1] - 1
    dtrain = lgb.Dataset(x[masks["train"]], y[masks["train"]], categorical_feature=[tissue_col])
    dvalid = lgb.Dataset(x[masks["valid"]], y[masks["valid"]], reference=dtrain)
    booster = lgb.train(
        {**PARAMS, "metric": "rmse"},
        dtrain,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    train_mean = float(y[masks["train"]].mean())
    cell_mean = (
        pd.Series(y[masks["train"]]).groupby(dc.cell_line_id[masks["train"]].to_numpy()).mean()
    )
    results: dict[str, dict[str, dict[str, float]]] = {}
    for part in ("valid", "test", "test1", "test2"):
        m = masks[part]
        yt = y[m]
        per_cell = dc.cell_line_id[m].map(cell_mean).fillna(train_mean).to_numpy()
        results[part] = {
            "mean_baseline": metrics(yt, np.full(len(yt), train_mean)),
            "cell_line_mean": metrics(yt, per_cell),
            "lightgbm": metrics(yt, booster.predict(x[m], num_iteration=booster.best_iteration)),
        }
    test = results["test"]
    gain = 1 - test["lightgbm"]["rmse"] / test["mean_baseline"]["rmse"]
    clearly_better = gain >= 0.10 and test["lightgbm"]["pearson"] >= 0.3
    print(json.dumps(results, indent=1))
    print(f"RMSE gain over the mean baseline: {gain:.1%}; clearly better (D75): {clearly_better}")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="lgbm-leave-drug-out-zip"):
        mlflow.log_params(
            {
                **{f"lgbm_{k}": v for k, v in PARAMS.items()},
                "target": TARGET,
                "source": "drugcomb only (TDC = copy, O'Neil disagrees), D75",
                "features": f"sum of Morgan r2 {FP_BITS}-bit fps + {PCS} expression PCs + tissue",
                "split": f"leave-drug-out, {TEST_DRUGS:.0%} test / {VALID_DRUGS:.0%} valid drugs",
                "best_iteration": booster.best_iteration,
                **{f"n_{k}": v for k, v in sizes.items()},
                **stats,
            }
        )
        flat = {
            f"{part}_{model}_{name}": value
            for part, models in results.items()
            for model, values in models.items()
            for name, value in values.items()
            if name != "n"
        }
        mlflow.log_metrics({**flat, "test_rmse_gain_vs_mean": gain})
        mlflow.set_tag("clearly_better_than_mean_baseline", str(clearly_better))
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "results.json"
            out.write_text(
                json.dumps(
                    {"stats": stats, "sizes": sizes, "results": results, "gain": gain}, indent=2
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(out))
    print(f"done in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
