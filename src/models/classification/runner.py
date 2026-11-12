"""One classifier experiment: Optuna search, retraining of the best trial, test, ONNX export.

    python -m src.models.classification.runner --config config/hpo/classifier_resnet50.yaml \
        --data data/processed/c3pi/datasets_v2/classify --out runs/classifier-resnet50
    python -m src.models.classification.runner --config ... --smoke    # local check, tiny subset

This is what each Kaggle notebook (notebooks/kaggle/classifier_<arch>.ipynb) runs on the GPU,
the same steps as the detectors (src/models/detection/runner.py):

1. Optuna study (seeded TPE sampler, config/hpo/classifier_<arch>.yaml): each trial trains the
   model for `search.epochs` epochs on `search.fraction` of the train crops of every class and
   is scored on the validation split (`val_top1`);
2. the best trial's hyperparameters are retrained on the whole train split (`final.epochs`,
   early stopping on validation, time limit `final.max_hours`);
3. the retrained model is scored on the comparison split `test_holdout.txt` (`test_v2_*`,
   promotion metric `test_v2_top1`), on the whole v2 test split (`test_v2_all_*`) and on the
   comparison crops of the 150 classes of v1 with the prediction restricted to them
   (`test_v2_v1c_*`, the like-for-like comparison with the v1 champion, D100); it is exported
   to ONNX (parity with PyTorch checked, CPU latency), with its confusion matrix, per-class
   scores and Grad-CAM pictures.

Nothing is logged to a server from Kaggle: everything goes into the export folder (`run.json`,
weights, ONNX model, class list, trials, figures), which
src/models/classification/import_kaggle.py turns into MLflow runs locally.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import platform
import shutil
import sys
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import yaml

from src.models.classification import data as D
from src.models.classification import hpo
from src.models.classification.metrics import (
    confusion_matrix,
    evaluate,
    per_class,
    ranking,
    top_confusions,
)
from src.models.detection.latency import cpu_name, isolated_latency
from src.models.detection.metrics import prefixed
from src.models.detection.runner import code_info, frozen_requirements

SCHEMA = 1
ROOT = D.ROOT
BACKENDS = {"torch": "src.models.classification.torch_backend"}
LATENCY_IMAGES = 20
PARITY_IMAGES = 100  # test crops scored with both PyTorch and ONNX Runtime
GRADCAM_IMAGES = 8  # Grad-CAM pictures in the export (half right, half wrong when possible)
SMOKE_IMAGES = {"train": 96, "val": 32, "test": 32}


def backend(family: str) -> Any:
    """Module with train / predict / export_onnx / parameter_count / gradcam_pictures."""
    return importlib.import_module(BACKENDS[family])


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # the fake backend of the tests does not need torch
        pass


def free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def environment(family: str) -> dict[str, str]:
    info = {
        "platform": "kaggle" if Path("/kaggle").is_dir() else "local",
        "python": platform.python_version(),
        "cpu": cpu_name(),
        "family": family,
    }
    for package in ("torch", "torchvision", "ultralytics", "optuna", "onnxruntime"):
        try:
            info[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue
    try:
        import torch

        info["cuda"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


# --- evaluation ---------------------------------------------------------------------------


def split_scores(probs: np.ndarray, items: list[D.Crop], v1_ids: list[int]) -> dict[str, float]:
    """`test_v2_*` (comparison split), `test_v2_all_*` (whole test), `test_v2_v1c_*`
    (comparison crops of the v1 classes, prediction restricted to the v1 classes)."""
    truth = np.asarray([c.label for c in items])
    holdout = np.asarray([c.holdout for c in items], dtype=bool)
    if not holdout.any():  # a dataset without holdout list: the whole test split
        holdout = np.ones(len(items), dtype=bool)
    v1 = np.isin(truth, v1_ids) & holdout
    out = {
        **prefixed(evaluate(probs[holdout], truth[holdout]), "test_v2"),
        **prefixed(evaluate(probs, truth), "test_v2_all"),
    }
    if v1.any():
        out.update(prefixed(evaluate(probs[v1], truth[v1], allowed=v1_ids), "test_v2_v1c"))
    return out


def write_confusion(
    probs: np.ndarray, items: list[D.Crop], classes: list[dict[str, Any]], folder: Path
) -> dict[str, str]:
    """Confusion matrix (CSV and PNG), per-class scores and the most frequent errors of the
    comparison split."""
    chosen = [i for i, c in enumerate(items) if c.holdout] or list(range(len(items)))
    truth = np.asarray([items[i].label for i in chosen])
    predicted = ranking(probs[chosen])[:, 0]
    matrix = confusion_matrix(truth, predicted, len(classes))
    folder.mkdir(parents=True, exist_ok=True)
    np.savetxt(folder / "confusion_test_v2.csv", matrix, fmt="%d", delimiter=",")
    scores = per_class(matrix)
    with (folder / "per_class_test_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_key", "name", "support", "precision", "recall", "f1"])
        for row in classes:
            k = row["class_id"]
            writer.writerow(
                [
                    k,
                    row.get("class_key", ""),
                    row.get("name", ""),
                    int(scores["support"][k]),
                    round(float(scores["precision"][k]), 4),
                    round(float(scores["recall"][k]), 4),
                    round(float(scores["f1"][k]), 4),
                ]
            )
    with (folder / "top_confusions_test_v2.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_class", "true_name", "predicted_class", "predicted_name", "count"])
        for t, p, n in top_confusions(matrix, 30):
            writer.writerow([t, classes[t].get("name", ""), p, classes[p].get("name", ""), n])
    files = {
        "confusion_csv": "confusion/confusion_test_v2.csv",
        "per_class": "confusion/per_class_test_v2.csv",
        "top_confusions": "confusion/top_confusions_test_v2.csv",
    }
    if plot_confusion(matrix, folder / "confusion_test_v2.png"):
        files["confusion_png"] = "confusion/confusion_test_v2.png"
    return files


def plot_confusion(matrix: np.ndarray, path: Path) -> bool:
    """Row-normalised confusion matrix as an image (False without matplotlib)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    rows = matrix.sum(axis=1, keepdims=True)
    share = np.divide(matrix, rows, out=np.zeros(matrix.shape, float), where=rows > 0)
    fig, ax = plt.subplots(figsize=(7, 6.4), dpi=150)
    image = ax.imshow(share, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("classe prédite (id v2)")
    ax.set_ylabel("vraie classe (id v2)")
    ax.set_title("Matrice de confusion, split de comparaison (part de chaque vraie classe)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def onnx_scores(onnx_path: Path, files: list[Path]) -> np.ndarray:
    """Probabilities of the exported model, fed exactly as the API feeds it."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    size = int(session.get_inputs()[0].shape[-1])
    return np.concatenate([session.run(None, {name: D.load_tensor(f, size)})[0] for f in files])


def onnx_parity(torch_probs: np.ndarray, onnx_path: Path, items: list[D.Crop]) -> dict[str, float]:
    """Agreement of PyTorch and ONNX Runtime on the first test crops."""
    chosen = items[:PARITY_IMAGES]
    if not chosen:
        return {}
    exported = onnx_scores(onnx_path, [c.path for c in chosen])
    reference = torch_probs[: len(chosen)]
    truth = np.asarray([c.label for c in chosen])
    return {
        "parity_images": float(len(chosen)),
        "parity_top1_agreement": float(np.mean(exported.argmax(1) == reference.argmax(1))),
        "parity_max_abs_diff": float(np.abs(exported - reference).max()),
        "parity_torch_top1": float(np.mean(reference.argmax(1) == truth)),
        "parity_onnx_top1": float(np.mean(exported.argmax(1) == truth)),
    }


def gradcam_items(probs: np.ndarray, items: list[D.Crop], n: int = GRADCAM_IMAGES) -> list[D.Crop]:
    """Comparison crops for the Grad-CAM pictures: the first n/2 right and n/2 wrong ones."""
    pool = [(i, c) for i, c in enumerate(items) if c.holdout] or list(enumerate(items))
    right = [c for i, c in pool if probs[i].argmax() == c.label]
    wrong = [c for i, c in pool if probs[i].argmax() != c.label]
    half = n // 2
    chosen = right[:half] + wrong[:half]
    extra = (right[half:] + wrong[half:])[: n - len(chosen)]
    return chosen + extra


# --- experiment ---------------------------------------------------------------------------


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", "utf-8")


def train_args(config: dict[str, Any]) -> dict[str, Any]:
    return {"model": config["model"], "pretrained": bool(config.get("pretrained", True))}


def make_objective(
    config: dict[str, Any], module: Any, data_dir: Path, out: Path, log: Callable[[str], None]
) -> Callable[[optuna.Trial], float]:
    search = config["search"]

    def objective(trial: optuna.Trial) -> float:
        suggested = hpo.suggest(trial, search["space"])
        hp = hpo.trial_hyperparameters(config, suggested)
        record: dict[str, Any] = {
            "number": trial.number,
            "params": hp,
            "optuna_params": dict(trial.params),
            "state": "RUNNING",
        }
        log(f"trial {trial.number}: {suggested}")
        started = time.monotonic()
        try:
            seed_everything(int(config["seed"]))
            result = module.train(
                hp,
                data_dir,
                out / "work",
                **train_args(config),
                epochs=int(search["epochs"]),
                name=f"trial-{trial.number:03d}",
                seed=int(config["seed"]),
                fraction=float(search.get("fraction", 1.0)),
                patience=int(search.get("patience", search["epochs"])),
                log=log,
            )
            val = D.crops(data_dir, "val")
            probs = module.predict(result.weights, [c.path for c in val], int(hp["imgsz"]))
            metrics = prefixed(evaluate(probs, [c.label for c in val]), "val")
            record.update(state="COMPLETE", metrics=metrics, history=result.history)
            value = float(metrics[search["metric"]])
            record["value"] = value
            log(f"trial {trial.number}: {search['metric']}={value:.4f}")
            return value
        except Exception as error:  # a failed trial (e.g. out of memory) is recorded, not fatal
            record.update(state="FAIL", error=f"{type(error).__name__}: {error}")
            log(f"trial {trial.number} failed: {record['error']}")
            raise
        finally:
            record["minutes"] = (time.monotonic() - started) / 60
            _write_json(out / "trials" / f"trial-{trial.number:03d}.json", record)
            shutil.rmtree(out / "work" / f"trial-{trial.number:03d}", ignore_errors=True)
            free_memory()

    return objective


def write_trials_csv(trials: list[dict[str, Any]], path: Path) -> None:
    keys = sorted({k for t in trials for k in t["params"]})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["number", "state", "value", "minutes", *keys])
        for t in trials:
            writer.writerow(
                [t["number"], t["state"], t.get("value", ""), round(t.get("minutes", 0), 2)]
                + [t["params"].get(k, "") for k in keys]
            )


def run_experiment(
    config_path: str | Path,
    data_dir: str | Path = D.DEFAULT_DATA,
    out_dir: str | Path | None = None,
    *,
    smoke: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Search, retrain, test and export one architecture; returns the content of run.json."""
    config = hpo.load_config(config_path)
    if smoke:
        config = hpo.apply_smoke(config)
    data_dir = Path(data_dir)
    out = Path(out_dir or ROOT / "runs" / f"classifier-{config['arch']}")
    out.mkdir(parents=True, exist_ok=True)
    dataset = D.dataset_info(data_dir)
    if smoke:
        sizes = {**SMOKE_IMAGES, **(config["smoke"].get("images") or {})}
        data_dir = D.make_subset(data_dir, out / "subset", sizes, seed=int(config["seed"]))
        dataset = {**dataset, "subset": sizes}
    classes = D.read_classes(data_dir)
    v1_ids = D.v1_classes(classes)
    module = backend(config["family"])
    seed_everything(int(config["seed"]))
    started = time.monotonic()

    # 1. Optuna search on the validation split.
    storage = f"sqlite:///{(out / 'optuna.db').as_posix()}"
    study = hpo.create_study(config, storage)
    remaining = int(config["search"]["n_trials"]) - len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    if remaining > 0:
        timeout = config["search"].get("timeout_hours")
        study.optimize(
            make_objective(config, module, data_dir, out, log),
            n_trials=remaining,
            timeout=float(timeout) * 3600 if timeout else None,
            catch=(Exception,),
            gc_after_trial=True,
        )
    trials = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "trials").glob("*.json"))
    ]
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        errors = sorted({t.get("error", "") for t in trials if t.get("state") == "FAIL"})
        raise RuntimeError(f"no Optuna trial completed; errors: {errors}")
    best = study.best_trial
    best_hp = hpo.final_hyperparameters(
        config, next(t["params"] for t in trials if t["number"] == best.number)
    )
    search_minutes = (time.monotonic() - started) / 60
    log(f"best trial {best.number}: {config['search']['metric']}={best.value:.4f} {best_hp}")

    # 2. Retraining of the best hyperparameters on the whole train set.
    final_cfg = config["final"]
    seed_everything(int(config["seed"]))
    retrain_started = time.monotonic()
    max_hours = final_cfg.get("max_hours")
    result = module.train(
        best_hp,
        data_dir,
        out / "work",
        **train_args(config),
        epochs=int(final_cfg["epochs"]),
        name="final",
        seed=int(config["seed"]),
        patience=int(final_cfg.get("patience", final_cfg["epochs"])),
        max_minutes=float(max_hours) * 60 if max_hours else None,
        log=log,
    )
    retrain_minutes = (time.monotonic() - retrain_started) / 60

    # 3. Validation and test scores, ONNX export, parity, latency, figures.
    imgsz = int(best_hp["imgsz"])
    val, test = D.crops(data_dir, "val"), D.crops(data_dir, "test")
    val_probs = module.predict(result.weights, [c.path for c in val], imgsz)
    test_probs = module.predict(result.weights, [c.path for c in test], imgsz)
    metrics = {
        **prefixed(evaluate(val_probs, [c.label for c in val]), "val"),
        **split_scores(test_probs, test, v1_ids),
    }
    (out / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.weights, out / "weights" / "best.pt")
    shutil.copy2(data_dir / "classes.csv", out / "classes.csv")
    onnx_path = module.export_onnx(
        out / "weights" / "best.pt",
        imgsz,
        out / "model" / "classifier.onnx",
        onnx_metadata(config, classes, imgsz),
    )
    metrics.update(onnx_parity(test_probs, onnx_path, test))
    files = [c.path for c in test[:LATENCY_IMAGES]]
    latency = {f"kaggle_{k}": v for k, v in isolated_latency(onnx_path, files).items()}
    figures = write_confusion(test_probs, test, classes, out / "confusion")
    names = {r["class_id"]: f"{r['class_id']:03d} {str(r.get('name', ''))[:22]}" for r in classes}
    pictures = module.gradcam_pictures(
        out / "weights" / "best.pt",
        [(c.path, c.label) for c in gradcam_items(test_probs, test)],
        imgsz,
        out / "gradcam",
        names,
    )
    training = out / "training"
    training.mkdir(exist_ok=True)
    for extra in [*result.run_dir.glob("*.png"), *result.run_dir.glob("*.csv")]:
        shutil.copy2(extra, training / extra.name)
    write_trials_csv(trials, out / "trials.csv")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (out / "requirements.txt").write_text(frozen_requirements(), encoding="utf-8")

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "task": "classify",
        "arch": config["arch"],
        "model": config["model"],
        "family": config["family"],
        "pretrained": bool(config.get("pretrained", True)),
        "branch": config.get("branch", ""),
        "seed": int(config["seed"]),
        "smoke": smoke,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "code": code_info(),
        "dataset": {
            **dataset,
            "images": {s: len(D.crops(data_dir, s)) for s in D.SPLITS},
            "holdout": sum(c.holdout for c in test),
            "classes": len(classes),
            "label_space": D.LABEL_SPACE,
        },
        "environment": environment(config["family"]),
        "search": {
            **{k: v for k, v in config["search"].items() if k != "space"},
            "space": config["search"]["space"],
            "best_trial": best.number,
            "best_value": float(best.value),
            "minutes": search_minutes,
        },
        "trials": trials,
        "final": {
            "params": best_hp,
            "epochs": int(final_cfg["epochs"]),
            "patience": int(final_cfg.get("patience", final_cfg["epochs"])),
            "history": result.history,
            "minutes": retrain_minutes,
            "parameters": module.parameter_count(out / "weights" / "best.pt"),
            "metrics": metrics,
            "latency": {**latency, "kaggle_cpu": cpu_name()},
        },
        "artifacts": {
            "weights": "weights/best.pt",
            "onnx": "model/classifier.onnx",
            "classes": "classes.csv",
            "training": "training",
            "trials_csv": "trials.csv",
            "config": "config.yaml",
            "requirements": "requirements.txt",
            "confusion": "confusion",
            "gradcam": "gradcam",
            **figures,
            "gradcam_pictures": [p.relative_to(out).as_posix() for p in pictures],
        },
        "minutes": (time.monotonic() - started) / 60,
    }
    _write_json(out / "run.json", summary)
    shutil.rmtree(out / "work", ignore_errors=True)
    if smoke:
        shutil.rmtree(out / "subset", ignore_errors=True)  # a copy of dataset images
    log(f"test_v2_top1={metrics.get('test_v2_top1', float('nan')):.4f}; export: {out}")
    return summary


def onnx_metadata(
    config: dict[str, Any], classes: list[dict[str, Any]], imgsz: int
) -> dict[str, str]:
    """The class list travels with the model (D101): `names` as Ultralytics writes it (what
    the API reads: output index -> class folder), the class keys of C3PI (NDC9 + part, stable
    across dataset versions) and the label space."""
    names = D.class_names(classes)
    keys = {r["class_id"]: r.get("class_key", "") for r in classes}
    return {
        "task": "classify",
        "imgsz": f"[{imgsz}, {imgsz}]",
        "batch": "1",
        "names": repr(names),
        "class_keys": json.dumps({str(k): v for k, v in keys.items()}),
        "label_space": D.LABEL_SPACE,
        "num_classes": str(len(classes)),
        "arch": str(config["arch"]),
        "model": str(config["model"]),
        "description": f"Tujya pill classifier {config['arch']} (C3PI v2, {len(classes)} classes)",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.classification.runner")
    parser.add_argument("--config", required=True, help="config/hpo/classifier_<arch>.yaml")
    parser.add_argument("--data", default=str(D.DEFAULT_DATA), help="crop dataset folder")
    parser.add_argument("--out", help="export folder, default runs/classifier-<arch>")
    parser.add_argument("--smoke", action="store_true", help="tiny subset, 2 trials, 1 epoch")
    args = parser.parse_args(argv)
    run_experiment(args.config, args.data, args.out, smoke=args.smoke)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
