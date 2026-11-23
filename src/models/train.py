"""Train and test the two Tujya vision models (DVC stages train_detector, train_classifier).

    python -m src.models.train detect                  # hyperparameters: params.yaml train.detector
    python -m src.models.train classify --epochs 5     # a flag overrides params.yaml (experiments)
    python -m src.models.train baseline                # record the runs of 2026-10-28 (D91)

Each run:
1. trains an Ultralytics YOLO model from pretrained weights (transfer learning) with the
   hyperparameters of params.yaml (`train.detector` / `train.classifier`, seed `train.seed`);
2. tests the best weights on the test split (mAP for detection, top-1 / top-5 for
   identification);
3. logs it to MLflow (experiment tujya-vision) through src/utils/mlflow_utils.py: tags
   git_commit, dvc_rev, dataset, seed; requirements.txt, dvc.lock and params.yaml; per-epoch
   metrics, test metrics, weights and plots;
4. writes the stage outputs: models/train/<model>/ (weights/best.pt, results.csv, args.yaml)
   and models/train/<model>.json (test metrics and MLflow run_id, read by `dvc metrics show`
   and by the next stage).

The next stage, src/models/export_onnx.py, exports the weights to ONNX, checks ONNX Runtime
against PyTorch, registers the ONNX model and promotes it if it beats the champion.

MLflow server: MLFLOW_TRACKING_URI, default http://127.0.0.1:5000 (docker compose up -d mlflow).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from ultralytics import YOLO, settings

from src.data.prepare.v1 import CLASSIFY, DETECT
from src.utils import mlflow_utils as mu

ROOT = Path(__file__).resolve().parents[2]  # repository root
# Ultralytics training runs (git-ignored; the results that matter are in MLflow and DVC).
RUNS = ROOT / "runs"
CLASSES = ROOT / "config" / "pill_classes_v1.csv"
PARAMS = ROOT / "params.yaml"
# Outputs of the DVC stages train_detector / train_classifier (git-ignored, cached by DVC).
OUT = ROOT / "models" / "train"
# The DVC output the training images come from (its hash becomes the run tag dataset_md5).
DATASET_PATH = "data/processed/c3pi/datasets"
MODELS = {"detect": "detector", "classify": "classifier"}  # keys of params.yaml, file names
EXPERIMENT = "tujya-vision"
# Hyperparameters read from params.yaml train.<model> (a command-line flag overrides them).
HYPERPARAMETERS = ("weights", "epochs", "imgsz", "batch", "patience", "fraction", "workers")


def device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def hardware() -> dict[str, str]:
    info = {
        "torch": torch.__version__,
        "cuda": str(torch.cuda.is_available()),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": str(os.cpu_count()),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def load_params(path: Path = PARAMS) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def hyperparameters(
    task: str, params: dict[str, Any], overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """params.yaml train.<model> plus the seed, with the non-empty overrides applied."""
    section = params["train"]
    hp: dict[str, Any] = {key: section[MODELS[task]][key] for key in HYPERPARAMETERS}
    hp["seed"] = section["seed"]
    for key, value in (overrides or {}).items():
        if key in hp and value not in (None, ""):
            hp[key] = value
    return hp


def data_source(task: str) -> str:
    return str(CLASSIFY) if task == "classify" else str(DETECT / "data.yaml")


def split_sizes(task: str) -> dict[str, int]:
    sizes = {}
    for split in ("train", "val", "test"):
        folder = CLASSIFY / split if task == "classify" else DETECT / "images" / split
        pattern = "*/*.jpg" if task == "classify" else "*.jpg"
        sizes[f"n_{split}_images"] = len(list(folder.glob(pattern)))
    return sizes


def _log_epochs(results_csv: Path) -> None:
    """Per-epoch metrics from Ultralytics' results.csv, as MLflow metric histories."""
    with results_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row = {k.strip(): v for k, v in row.items()}
            step = int(float(row.pop("epoch")))
            metrics = {}
            for key, value in row.items():
                try:
                    metrics[key.replace("(B)", "").replace(":", "_")] = float(value)
                except (TypeError, ValueError):
                    continue
            mlflow.log_metrics(metrics, step=step)


def classify_test(model: YOLO, folder: Path, imgsz: int, batch: int) -> dict[str, float]:
    """Top-1 / top-5 on the test crops, matching folder names to the model's class names.

    Done here rather than with `model.val(split="test")` because a class can have no test
    crop (its test photos were all doubtful), and a per-folder dataset would then number
    the classes differently from training.
    """
    index = {name: i for i, name in model.names.items()}
    files = sorted(folder.glob("*/*.jpg"))
    top1 = top5 = 0
    for start in range(0, len(files), batch):
        chunk = files[start : start + batch]
        results = model.predict(
            [str(f) for f in chunk], imgsz=imgsz, verbose=False, device=device()
        )
        for path, result in zip(chunk, results, strict=True):
            truth = index[path.parent.name]
            top1 += int(result.probs.top1 == truth)
            top5 += int(truth in result.probs.top5)
    n = max(1, len(files))
    classes = len({f.parent.name for f in files})
    return {
        "test_top1": top1 / n,
        "test_top5": top5 / n,
        "test_images": float(len(files)),
        "test_classes": float(classes),
    }


def test_metrics(task: str, model: YOLO, imgsz: int, batch: int, name: str) -> dict[str, float]:
    """Scores of `model` on the test split: mAP (detection) or top-1 / top-5 (identification)."""
    if task == "classify":
        return classify_test(model, CLASSIFY / "test", imgsz, batch)
    test = model.val(
        data=data_source(task),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device(),
        project=str(RUNS),
        name=f"{name}-test",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    return {
        "test_mAP50": float(test.box.map50),
        "test_mAP50_95": float(test.box.map),
        "test_precision": float(test.box.mp),
        "test_recall": float(test.box.mr),
    }


def log_training_artifacts(run_dir: Path) -> None:
    """Per-epoch history, best weights, plots and Ultralytics arguments of a training run."""
    _log_epochs(run_dir / "results.csv")
    mlflow.log_artifact(str(run_dir / "weights" / "best.pt"), "weights")
    for extra in sorted(run_dir.glob("*.png")):
        mlflow.log_artifact(str(extra), "plots")
    mlflow.log_artifact(str(run_dir / "results.csv"), "training")
    mlflow.log_artifact(str(run_dir / "args.yaml"), "training")
    if CLASSES.exists():
        mlflow.log_artifact(str(CLASSES), "config")


def write_outputs(out: Path, run_dir: Path, summary: dict[str, Any]) -> Path:
    """Stage outputs: out/ (weights/best.pt, results.csv, args.yaml) and out.json (summary)."""
    if out.exists():
        shutil.rmtree(out)
    (out / "weights").mkdir(parents=True)
    shutil.copy2(run_dir / "weights" / "best.pt", out / "weights" / "best.pt")
    for name in ("results.csv", "args.yaml"):
        shutil.copy2(run_dir / name, out / name)
    metrics = out.with_suffix(".json")
    metrics.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metrics


def run(args: argparse.Namespace) -> dict[str, Any]:
    task = args.task
    settings.update({"mlflow": False})  # MLflow logging is done here, explicitly
    params = load_params()
    hp = hyperparameters(task, params, vars(args))
    dataset = params["train"]["dataset"]
    data = data_source(task)
    name = f"{task}-{time.strftime('%Y%m%d-%H%M%S')}"
    logged = {
        **hp,
        "task": task,
        "device": device(),
        "data": Path(data).relative_to(ROOT).as_posix(),
        **split_sizes(task),
    }
    mu.configure()
    with mu.start_tracked_run(
        EXPERIMENT,
        name,
        seed=hp["seed"],
        dataset=dataset,
        dataset_path=DATASET_PATH,
        params={k: v for k, v in logged.items() if k != "seed"},
        tags={**hardware(), "task": task, "model": MODELS[task], "stage": "train"},
    ) as active:
        model = YOLO(hp["weights"])
        started = time.monotonic()
        model.train(
            data=data,
            epochs=hp["epochs"],
            imgsz=hp["imgsz"],
            batch=hp["batch"],
            device=device(),
            workers=hp["workers"],
            project=str(RUNS),
            name=name,
            seed=hp["seed"],
            deterministic=True,
            fraction=hp["fraction"],
            patience=hp["patience"],
            plots=True,
            verbose=False,
        )
        mlflow.log_metric("train_minutes", (time.monotonic() - started) / 60)
        run_dir = RUNS / name
        model = YOLO(str(run_dir / "weights" / "best.pt"))
        metrics = test_metrics(task, model, hp["imgsz"], hp["batch"], name)
        mlflow.log_metrics(metrics)
        log_training_artifacts(run_dir)
        summary = {
            "run_id": active.info.run_id,
            "task": task,
            "model": MODELS[task],
            "weights": hp["weights"],
            "imgsz": hp["imgsz"],
            "dataset": dataset,
            **metrics,
        }
        write_outputs(Path(args.out) if args.out else OUT / MODELS[task], run_dir, summary)
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if args and args[0] == "baseline":  # python -m src.models.train baseline [--task ...]
        baseline_main(args[1:])
        return
    parser = argparse.ArgumentParser(prog="python -m src.models.train")
    parser.add_argument("task", choices=["classify", "detect"])
    parser.add_argument("--epochs", type=int, help="default: params.yaml")
    parser.add_argument("--imgsz", type=int, help="default: params.yaml")
    parser.add_argument("--batch", type=int, help="default: params.yaml")
    parser.add_argument("--workers", type=int, help="default: params.yaml")
    parser.add_argument("--weights", help="pretrained weights, default: params.yaml")
    parser.add_argument("--fraction", type=float, help="share of the train set used")
    parser.add_argument("--patience", type=int, help="default: params.yaml")
    parser.add_argument("--seed", type=int, help="default: params.yaml train.seed")
    parser.add_argument("--out", help="output folder, default models/train/<model>")
    run(parser.parse_args(args))


# --- Baseline: the GPU runs of 2026-10-28 recorded as the stages' output -----------------------
# Record the models trained on 2026-10-28 as the BASELINE of the pipeline and of MLflow.
#
#     python -m src.models.train baseline        # training venv, MLflow server running
#     dvc commit -f train_detector train_classifier # the baseline becomes the stages' output
#     dvc repro export_onnx evaluate                # ONNX export, Registry, alias champion
#
# The detector (YOLO11n) and the classifier (YOLO11s-cls) served since 2026-10-28 were trained
# by the GPU runs kept on disk under runs/ (Ultralytics output: weights, results.csv,
# args.yaml), but their MLflow runs are on the server of the first copy of the project, not on
# this one (docs/decisions.md D83). Rather than retrain them, this script, for each model:
#
# 1. checks that the run's hyperparameters (args.yaml) are those of params.yaml `train.*`, so
#    the DVC stage train_<model> describes exactly how the weights were made;
# 2. starts a tracked run (src/utils/mlflow_utils.py: tags git_commit, dvc_rev, dataset, seed;
#    requirements.txt, dvc.lock, params.yaml) in the experiment tujya-vision, with the
#    training parameters, the per-epoch history, the weights and plots, and tags pointing to the
#    original run (`origin_*`, `baseline=true`);
# 3. evaluates the weights again on the test split, now (GPU if available, tag `eval_device`),
#    and logs these test metrics (the original ones are kept as `origin_test_*` tags);
# 4. writes the outputs of the stage train_<model> (models/train/<model>/ and
#    models/train/<model>.json), exactly as src/models/train.py does.

# Ultralytics runs of the models served since 2026-10-28 (models/models.json).
BASELINE_RUNS = {
    "detect": RUNS / "detect-20261028-023214",
    "classify": RUNS / "classify-20261028-021601",
}
# args.yaml key -> params.yaml key, checked to be equal.
ARGS_TO_PARAMS = {
    "model": "weights",
    "epochs": "epochs",
    "imgsz": "imgsz",
    "batch": "batch",
    "patience": "patience",
    "fraction": "fraction",
    "workers": "workers",
    "seed": "seed",
}


def check_hyperparameters(task: str, args: dict[str, Any], hp: dict[str, Any]) -> None:
    """Refuse to record a run whose hyperparameters differ from params.yaml."""
    diff = {
        ours: (args.get(theirs), hp[ours])
        for theirs, ours in ARGS_TO_PARAMS.items()
        if args.get(theirs) != hp[ours]
    }
    if diff:
        raise SystemExit(f"{task}: args.yaml differs from params.yaml (run, params): {diff}")


def register_baseline(task: str, run_dir: Path, params: dict[str, Any]) -> dict[str, Any]:
    args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))
    origin = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    hp = hyperparameters(task, params)
    check_hyperparameters(task, args, hp)
    dataset = params["train"]["dataset"]
    model_name = MODELS[task]
    arch = Path(hp["weights"]).stem
    logged = {
        **{k: v for k, v in hp.items() if k != "seed"},
        "task": task,
        "device": str(args["device"]),
        "data": Path(data_source(task)).relative_to(mu.ROOT).as_posix(),
        "optimizer": args["optimizer"],
        "lr0": args["lr0"],
        "deterministic": args["deterministic"],
        **split_sizes(task),
    }
    tags = {
        **hardware(),
        "task": task,
        "model": model_name,
        "stage": "train",
        "baseline": "true",
        "origin": "GPU training of 2026-10-28, re-registered without retraining",
        "origin_run_dir": f"runs/{run_dir.name}",
        "origin_mlflow_run_id": origin["run_id"],
        "eval_device": device(),
    }
    tags.update({f"origin_{k}": f"{v:.6f}" for k, v in origin.items() if k.startswith("test_")})

    with mu.start_tracked_run(
        EXPERIMENT,
        f"baseline-{model_name}-{arch}",
        seed=hp["seed"],
        dataset=dataset,
        dataset_path=DATASET_PATH,
        params=logged,
        tags=tags,
    ) as run:
        model = YOLO(str(run_dir / "weights" / "best.pt"))
        metrics = test_metrics(task, model, hp["imgsz"], hp["batch"], f"baseline-{task}")
        mlflow.log_metrics(metrics)
        log_training_artifacts(run_dir)
        summary = {
            "run_id": run.info.run_id,
            "task": task,
            "model": model_name,
            "weights": hp["weights"],
            "imgsz": hp["imgsz"],
            "dataset": dataset,
            "baseline": True,
            **metrics,
        }
        write_outputs(OUT / model_name, run_dir, summary)
    return summary


def baseline_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.train baseline")
    parser.add_argument("--task", choices=["detect", "classify"], action="append")
    args = parser.parse_args(argv)
    settings.update({"mlflow": False})
    params = load_params()
    mu.configure()
    for task in args.task or BASELINE_RUNS:
        summary = register_baseline(task, BASELINE_RUNS[task], params)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
