"""One detector experiment: Optuna search, retraining of the best trial, test, ONNX export.

    python -m src.models.detection.runner --config config/hpo/detector_yolov8n.yaml \
        --data data/processed/c3pi/datasets_v2/detect --out runs/detector-yolov8n
    python -m src.models.detection.runner --config ... --smoke     # local check, tiny subset

This is what each Kaggle notebook (notebooks/kaggle/detector_<arch>.ipynb) runs on the GPU:

1. Optuna study (seeded TPE sampler, config/hpo/detector_<arch>.yaml): each trial trains the
   pretrained model for `search.epochs` epochs on `search.fraction` of the train images and is
   scored on the validation split (`val_mAP50_95`, src/models/detection/metrics.py);
2. the best trial's hyperparameters are retrained on the whole train set (`final.epochs`,
   early stopping on validation);
3. the retrained model is scored on the comparison split `test_holdout.txt` (metrics
   `test_v2_*`, the promotion metric is `test_v2_mAP50_95`) and on the whole v2 test split
   (`test_v2_all_*`), exported to ONNX and timed on the CPU.

Nothing is logged to a server from Kaggle: everything goes into the export folder (`run.json`,
weights, ONNX model, per-epoch histories, trials), which src/models/detection/import_kaggle.py
turns into MLflow runs locally (one parent run, one child run per trial).
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import yaml

from src.models.detection import hpo
from src.models.detection.labels import holdout_names, make_subset, read_bgr, samples
from src.models.detection.latency import cpu_name, onnx_cpu_latency, onnx_predict
from src.models.detection.metrics import evaluate, prefixed

SCHEMA = 1
ROOT = Path(__file__).resolve().parents[3]  # repository root
DEFAULT_DATA = ROOT / "data" / "processed" / "c3pi" / "datasets_v2" / "detect"
DATASET_PATH = "data/processed/c3pi/datasets_v2/detect"  # DVC output (dvc.lock)
MANIFEST = "MANIFEST.json"  # written by package_kaggle next to the data
BACKENDS = {
    "yolo": "src.models.detection.yolo_backend",
    "fasterrcnn": "src.models.detection.fasterrcnn",
}
LATENCY_IMAGES = 20
PARITY_IMAGES = 100  # test images scored with both PyTorch and ONNX Runtime
SMOKE_IMAGES = {"train": 64, "val": 16, "test": 16}


def backend(family: str) -> Any:
    """Module with train / predict / export_onnx / parameter_count for a model family."""
    return importlib.import_module(BACKENDS[family])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # the metrics and the export folder do not need torch
        pass


def free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# --- provenance ---------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str | None:
    try:
        done = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def code_info(root: Path = ROOT) -> dict[str, Any]:
    """Commit and branch of the code that runs (the clone of the experiment branch on Kaggle)."""
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "git_commit": _git(root, "rev-parse", "HEAD") or "unknown",
        "git_branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown",
        "git_dirty": bool(status),
    }


def dataset_info(data_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    """Name, DVC hash and dvc.lock id of the data: MANIFEST.json of the Kaggle dataset, else
    the repository's dvc.lock (local runs)."""
    manifest = data_dir / MANIFEST
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8"))
    info: dict[str, Any] = {"name": "c3pi-v2-detect", "path": DATASET_PATH}
    lock = root / "dvc.lock"
    if lock.is_file():
        data = yaml.safe_load(lock.read_text(encoding="utf-8")) or {}
        for stage in (data.get("stages") or {}).values():
            for out in stage.get("outs") or []:
                if out.get("path") == DATASET_PATH:
                    info["md5"] = out.get("md5")
        info["dvc_rev"] = _git(root, "hash-object", "dvc.lock") or "unknown"
    return info


def environment(family: str) -> dict[str, str]:
    info = {
        "platform": "kaggle" if Path("/kaggle").is_dir() else "local",
        "python": platform.python_version(),
        "cpu": cpu_name(),
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
    info["family"] = family
    return info


def frozen_requirements() -> str:
    seen: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            seen.setdefault(name.lower().replace("_", "-"), f"{name}=={dist.version}")
    return "\n".join(seen[key] for key in sorted(seen)) + "\n"


# --- evaluation ---------------------------------------------------------------------------


def score(
    module: Any, weights: Path, data_dir: Path, split: str, imgsz: int, names: set[str] | None
) -> dict[str, float]:
    """metrics.evaluate of a model's raw predictions on a split (or the named images)."""
    chosen = samples(data_dir, split, names)
    if not chosen:
        return {}
    predictions = module.predict(weights, [s.image for s in chosen], imgsz)
    return evaluate(predictions, [s.truth() for s in chosen])


def test_metrics(module: Any, weights: Path, data_dir: Path, imgsz: int) -> dict[str, float]:
    """`test_v2_*` on the comparison split (holdout) and `test_v2_all_*` on the whole test."""
    holdout = holdout_names(data_dir)
    everything = score(module, weights, data_dir, "test", imgsz, None)
    compared = (
        everything if holdout is None else score(module, weights, data_dir, "test", imgsz, holdout)
    )
    return {**prefixed(compared, "test_v2"), **prefixed(everything, "test_v2_all")}


def onnx_parity(
    module: Any, weights: Path, onnx_path: Path, data_dir: Path, imgsz: int
) -> dict[str, float]:
    """mAP of the PyTorch model and of its ONNX export (decoded like the API) on the same
    first test images: `parity_torch_mAP50_95` and `parity_onnx_mAP50_95` should agree."""
    chosen = samples(data_dir, "test")[:PARITY_IMAGES]
    if not chosen:
        return {}
    truths = [s.truth() for s in chosen]
    torch_scores = evaluate(module.predict(weights, [s.image for s in chosen], imgsz), truths)
    onnx_scores = evaluate(onnx_predict(onnx_path, [read_bgr(s.image) for s in chosen]), truths)
    return {
        "parity_images": float(len(chosen)),
        "parity_torch_mAP50_95": torch_scores["mAP50_95"],
        "parity_onnx_mAP50_95": onnx_scores["mAP50_95"],
        "parity_torch_mAP50": torch_scores["mAP50"],
        "parity_onnx_mAP50": onnx_scores["mAP50"],
    }


# --- experiment ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", "utf-8")


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
                weights=config["weights"],
                epochs=int(search["epochs"]),
                name=f"trial-{trial.number:03d}",
                seed=int(config["seed"]),
                fraction=float(search.get("fraction", 1.0)),
                patience=int(search.get("patience", search["epochs"])),
            )
            metrics = prefixed(
                score(module, result.weights, data_dir, "val", int(hp["imgsz"]), None), "val"
            )
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
    data_dir: str | Path = DEFAULT_DATA,
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
    out = Path(out_dir or ROOT / "runs" / f"detector-{config['arch']}")
    out.mkdir(parents=True, exist_ok=True)
    dataset = dataset_info(data_dir)
    if smoke:
        counts = {**SMOKE_IMAGES, **(config["smoke"].get("images") or {})}
        data_dir = make_subset(data_dir, out / "subset", counts, seed=int(config["seed"]))
        dataset = {**dataset, "subset": counts}
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
    best = study.best_trial
    best_hp = next(t["params"] for t in trials if t["number"] == best.number)
    search_minutes = (time.monotonic() - started) / 60
    log(f"best trial {best.number}: {config['search']['metric']}={best.value:.4f} {best_hp}")

    # 2. Retraining of the best hyperparameters on the whole train set.
    final_cfg = config["final"]
    seed_everything(int(config["seed"]))
    retrain_started = time.monotonic()
    result = module.train(
        best_hp,
        data_dir,
        out / "work",
        weights=config["weights"],
        epochs=int(final_cfg["epochs"]),
        name="final",
        seed=int(config["seed"]),
        patience=int(final_cfg.get("patience", final_cfg["epochs"])),
        plots=True,
    )
    retrain_minutes = (time.monotonic() - retrain_started) / 60

    # 3. Validation and test scores, ONNX export, CPU latency.
    imgsz = int(best_hp["imgsz"])
    metrics = {
        **prefixed(score(module, result.weights, data_dir, "val", imgsz, None), "val"),
        **test_metrics(module, result.weights, data_dir, imgsz),
    }
    (out / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(result.weights, out / "weights" / "best.pt")
    onnx_path = module.export_onnx(
        out / "weights" / "best.pt", imgsz, out / "model" / "detector.onnx"
    )
    images = [read_bgr(s.image) for s in samples(data_dir, "test")[:LATENCY_IMAGES]]
    latency = {f"kaggle_{k}": v for k, v in onnx_cpu_latency(onnx_path, images=images).items()}
    metrics.update(onnx_parity(module, result.weights, onnx_path, data_dir, imgsz))
    training = out / "training"
    training.mkdir(exist_ok=True)
    for extra in [*result.run_dir.glob("*.png"), *result.run_dir.glob("*.csv")]:
        shutil.copy2(extra, training / extra.name)
    write_trials_csv(trials, out / "trials.csv")
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (out / "requirements.txt").write_text(frozen_requirements(), encoding="utf-8")

    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "arch": config["arch"],
        "family": config["family"],
        "weights": config["weights"],
        "branch": config.get("branch", ""),
        "seed": int(config["seed"]),
        "smoke": smoke,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "code": code_info(),
        "dataset": {
            **dataset,
            "images": {s: len(samples(data_dir, s)) for s in ("train", "val", "test")},
            "holdout": len(holdout_names(data_dir) or ()),
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
            "onnx": "model/detector.onnx",
            "training": "training",
            "trials_csv": "trials.csv",
            "config": "config.yaml",
            "requirements": "requirements.txt",
        },
        "minutes": (time.monotonic() - started) / 60,
    }
    _write_json(out / "run.json", summary)
    shutil.rmtree(out / "work", ignore_errors=True)
    if smoke:
        shutil.rmtree(out / "subset", ignore_errors=True)  # a copy of dataset images
    log(f"test_v2_mAP50_95={metrics.get('test_v2_mAP50_95', float('nan')):.4f}; export: {out}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.detection.runner")
    parser.add_argument("--config", required=True, help="config/hpo/detector_<arch>.yaml")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="YOLO dataset folder")
    parser.add_argument("--out", help="export folder, default runs/detector-<arch>")
    parser.add_argument("--smoke", action="store_true", help="tiny subset, 2 trials, 1 epoch")
    args = parser.parse_args(argv)
    run_experiment(args.config, args.data, args.out, smoke=args.smoke)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
