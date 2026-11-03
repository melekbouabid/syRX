"""Record the models trained on 2026-10-28 as the BASELINE of the pipeline and of MLflow.

    python -m src.models.register_baseline        # training venv, MLflow server running
    dvc commit -f train_detector train_classifier # the baseline becomes the stages' output
    dvc repro export_onnx evaluate                # ONNX export, Registry, alias champion

The detector (YOLO11n) and the classifier (YOLO11s-cls) served since 2026-10-28 were trained
by the GPU runs kept on disk under runs/ (Ultralytics output: weights, results.csv,
args.yaml), but their MLflow runs are on the server of the first copy of the project, not on
this one (docs/decisions.md D83). Rather than retrain them, this script, for each model:

1. checks that the run's hyperparameters (args.yaml) are those of params.yaml `train.*`, so
   the DVC stage train_<model> describes exactly how the weights were made;
2. starts a tracked run (src/utils/mlflow_utils.py: tags git_commit, dvc_rev, dataset, seed;
   requirements.txt, dvc.lock, params.yaml) in the experiment tujya-vision, with the
   training parameters, the per-epoch history, the weights and plots, and tags pointing to the
   original run (`origin_*`, `baseline=true`);
3. evaluates the weights again on the test split, now, and logs these test metrics (the
   original ones are kept as `origin_test_*` tags for comparison);
4. writes the outputs of the stage train_<model> (models/train/<model>/ and
   models/train/<model>.json), exactly as src/models/train.py does.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import yaml
from ultralytics import YOLO, settings

from src.models.train import (
    DATASET_PATH,
    EXPERIMENT,
    MODELS,
    OUT,
    RUNS,
    data_source,
    device,
    hardware,
    hyperparameters,
    load_params,
    log_training_artifacts,
    split_sizes,
    test_metrics,
    write_outputs,
)
from src.utils import mlflow_utils as mu

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.register_baseline")
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
