"""Train, log and publish the Tox21 toxicity model (bonus tool, docs/TOXICITY.md).

cd backend                                    # backend venv (rdkit, lightgbm, mlflow)
python scripts/train_tox21.py download        # MoleculeNet tox21.csv.gz -> data/tox21/
python scripts/train_tox21.py train           # scaffold split, 12 LightGBM models, MLflow tujya-tox
python scripts/train_tox21.py publish         # registered version -> models/tox21/ for the API

Featurisation is the API's own (app/toxicity.py): largest fragment, Morgan radius 2, 2,048
bits. Split: Bemis-Murcko scaffolds, largest groups first into train until 80 %, then
validation (10 %) and test (10 %), as in MoleculeNet / DeepChem's scaffold splitter, so the
test molecules have scaffolds never seen in training. Early stopping on validation AUC.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.toxicity import FP_BITS, FP_RADIUS, TASKS, fingerprint, scaffold, standardise

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "tox21"
RAW = DATA / "tox21.csv.gz"
URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
MODELS = ROOT / "models" / "tox21"
EXPERIMENT = "tujya-tox"
REGISTERED = "tujya-tox21"
SEED = 42
PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 10,
    "feature_fraction": 0.3,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
    "num_threads": 0,
}
MAX_ROUNDS = 2000
EARLY_STOP = 100


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """ROC-AUC by the rank formula (ties averaged); NaN if one class is missing."""
    y = np.asarray(y, dtype=int)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = pd.Series(p).rank(method="average").to_numpy()
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def download() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(URL, headers={"User-Agent": "tujya-student-project"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 fixed https URL
        RAW.write_bytes(response.read())
    digest = hashlib.sha256(RAW.read_bytes()).hexdigest()
    (DATA / "SOURCE.json").write_text(
        json.dumps(
            {
                "url": URL,
                "sha256": digest,
                "bytes": RAW.stat().st_size,
                "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "licence": "see docs/TOXICITY.md (Tox21: US federal data; MoleculeNet copy)",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{RAW} {RAW.stat().st_size} bytes sha256 {digest}")


def featurise() -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(RAW)
    rows, fps = [], []
    for record in frame.itertuples(index=False):
        mol = standardise(str(record.smiles))
        if mol is None:
            continue
        rows.append({"mol_id": record.mol_id, "scaffold": scaffold(mol)})
        fps.append(fingerprint(mol))
    table = pd.DataFrame(rows)
    labels = frame.set_index("mol_id").loc[table["mol_id"], list(TASKS)].reset_index(drop=True)
    return pd.concat([table, labels], axis=1), np.vstack(fps)


def scaffold_split(scaffolds: pd.Series) -> np.ndarray:
    """'train' / 'valid' / 'test' per molecule: scaffold groups, largest first (80/10/10)."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(scaffolds):
        groups[s].append(i)
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(scaffolds)
    split = np.empty(n, dtype=object)
    train_cut, valid_cut = 0.8 * n, 0.9 * n
    count = 0
    for group in ordered:
        if count + len(group) <= train_cut:
            name = "train"
        elif count + len(group) <= valid_cut:
            name = "valid"
        else:
            name = "test"
        split[group] = name
        count += len(group)
    return split


def train(args: argparse.Namespace) -> None:
    import lightgbm as lgb
    import mlflow

    started = time.time()
    table, fps = featurise()
    split = scaffold_split(table["scaffold"])
    x = fps.astype(np.float32)
    sizes = {s: int((split == s).sum()) for s in ("train", "valid", "test")}
    print(f"{len(table)} molecules, {table['scaffold'].nunique()} scaffolds, split {sizes}")

    out = Path(tempfile.mkdtemp(prefix="tox21-"))
    metrics: dict[str, float] = {}
    per_task: list[dict[str, Any]] = []
    for task in TASKS:
        y = table[task]
        masks = {s: (split == s) & y.notna().to_numpy() for s in ("train", "valid", "test")}
        ytr, yva, yte = (y[masks[s]].astype(int).to_numpy() for s in ("train", "valid", "test"))
        dtrain = lgb.Dataset(x[masks["train"]], ytr)
        dvalid = lgb.Dataset(x[masks["valid"]], yva, reference=dtrain)
        booster = lgb.train(
            {**PARAMS, "metric": "auc"},
            dtrain,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
        )
        booster.save_model(str(out / f"{task}.txt"), num_iteration=booster.best_iteration)
        p_va = booster.predict(x[masks["valid"]], num_iteration=booster.best_iteration)
        p_te = booster.predict(x[masks["test"]], num_iteration=booster.best_iteration)
        p_tr = booster.predict(x[masks["train"]], num_iteration=booster.best_iteration)
        row = {
            "task": task,
            "train_n": len(ytr),
            "train_pos": int(ytr.sum()),
            "valid_n": len(yva),
            "valid_pos": int(yva.sum()),
            "test_n": len(yte),
            "test_pos": int(yte.sum()),
            "rounds": booster.best_iteration,
            "train_roc_auc": round(roc_auc(ytr, p_tr), 4),
            "valid_roc_auc": round(roc_auc(yva, p_va), 4),
            "test_roc_auc": round(roc_auc(yte, p_te), 4),
        }
        per_task.append(row)
        metrics[task] = row["test_roc_auc"]
        print(
            f"{task:14s} rounds {row['rounds']:4d}  valid {row['valid_roc_auc']:.3f}  "
            f"test {row['test_roc_auc']:.3f}  (test {row['test_n']} mols, {row['test_pos']} active)"
        )

    train_mask = split == "train"
    np.save(out / "train_fps.npy", np.packbits(fps[train_mask].astype(bool), axis=1))
    mean_test = float(np.nanmean(list(metrics.values())))
    mean_valid = float(np.nanmean([r["valid_roc_auc"] for r in per_task]))
    print(f"mean ROC-AUC: valid {mean_valid:.3f}, test {mean_test:.3f}")
    pd.DataFrame(per_task).to_csv(out / "per_task.csv", index=False)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=f"lgbm-morgan{FP_BITS}-scaffold") as run:
        mlflow.log_params(
            {
                **{f"lgbm_{k}": v for k, v in PARAMS.items()},
                "max_rounds": MAX_ROUNDS,
                "early_stopping": EARLY_STOP,
                "fingerprint": f"morgan r{FP_RADIUS} {FP_BITS} bits, largest fragment",
                "split": "scaffold 80/10/10 (largest scaffold groups first)",
                "molecules": len(table),
                "scaffolds": int(table["scaffold"].nunique()),
                **{f"n_{k}": v for k, v in sizes.items()},
                "data_url": URL,
                "data_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
            }
        )
        mlflow.log_metrics(
            {
                "mean_test_roc_auc": mean_test,
                "mean_valid_roc_auc": mean_valid,
                **{f"test_roc_auc_{t}": v for t, v in metrics.items()},
                **{f"valid_roc_auc_{r['task']}": r["valid_roc_auc"] for r in per_task},
            }
        )
        meta = {
            "name": REGISTERED,
            "tasks": list(TASKS),
            "fingerprint": {"type": "morgan", "radius": FP_RADIUS, "bits": FP_BITS},
            "test_roc_auc": metrics,
            "mean_test_roc_auc": round(mean_test, 4),
            "split": "scaffold 80/10/10",
            "run_id": run.info.run_id,
        }
        (out / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        mlflow.log_artifacts(str(out), artifact_path="bundle")

        class Tox21Bundle(mlflow.pyfunc.PythonModel):  # type: ignore[misc]
            """pyfunc wrapper: SMILES column -> one probability column per Tox21 task."""

            def load_context(self, context: Any) -> None:
                from app.toxicity import ToxModel

                self.model = ToxModel.from_dir(Path(context.artifacts["bundle"]))

            def predict(self, context: Any, model_input: pd.DataFrame) -> pd.DataFrame:
                rows = []
                for smiles in model_input["smiles"]:
                    mol = standardise(str(smiles))
                    out = self.model.predict(mol) if mol is not None else {"predictions": []}
                    rows.append({p["task"]: p["probability"] for p in out["predictions"]})
                return pd.DataFrame(rows)

        info = mlflow.pyfunc.log_model(
            name="model",
            python_model=Tox21Bundle(),
            artifacts={"bundle": str(out)},
            code_paths=[str(Path(__file__).resolve().parent.parent / "app")],
            registered_model_name=REGISTERED,
        )
        version = getattr(info, "registered_model_version", None)
        print(f"MLflow run {run.info.run_id}, registered {REGISTERED} version {version}")
    shutil.rmtree(out, ignore_errors=True)
    print(f"done in {time.time() - started:.0f} s")


def publish(args: argparse.Namespace) -> None:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    client = MlflowClient()
    version = args.version or max(
        int(v.version) for v in client.search_model_versions(f"name='{REGISTERED}'")
    )
    mv = client.get_model_version(REGISTERED, str(version))
    tmp = Path(tempfile.mkdtemp(prefix="tox21-publish-"))
    local = Path(
        mlflow.artifacts.download_artifacts(
            run_id=mv.run_id, artifact_path="bundle", dst_path=str(tmp)
        )
    )
    if MODELS.exists():
        shutil.rmtree(MODELS)
    shutil.copytree(local, MODELS)
    meta = json.loads((MODELS / "model.json").read_text(encoding="utf-8"))
    meta["version"] = str(version)
    (MODELS / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"{REGISTERED} version {version} (run {mv.run_id}) -> {MODELS}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download")
    sub.add_parser("train")
    p = sub.add_parser("publish")
    p.add_argument("--version", type=int, default=None, help="default: latest version")
    args = parser.parse_args()
    {"download": lambda a: download(), "train": train, "publish": publish}[args.command](args)


if __name__ == "__main__":
    main()
