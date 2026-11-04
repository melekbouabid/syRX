"""Import a detector experiment run on Kaggle (export folder of runner.py) into MLflow.

    # training venv, MLflow server running (docker compose up -d mlflow)
    python -m src.models.detection.import_kaggle runs/kaggle/out/detector-yolov8n
    python -m src.models.detection.import_kaggle <folder or .zip> --smoke   # never registered

What it creates in the experiment `tujya-vision`:

- one parent run `detector-<arch>` (tag `stage=compare`): the provenance of the Kaggle run
  (`git_commit` and `git_branch` of the code that ran, `dvc_rev` and `dataset_md5` of the
  data, `seed`, GPU), the retained hyperparameters, the per-epoch history of the final
  training, the validation and test metrics (`test_v2_*` on the comparison split,
  `test_v2_all_*` on the whole v2 test), the Kaggle CPU latency and the latency measured
  again here (`onnx_cpu_latency_ms`, the same PC for every model), and the artefacts
  (weights, ONNX model, trials, Kaggle requirements, dvc.lock, params.yaml);
- one child run per Optuna trial (`trial-NNN`, tag `stage=hpo`, MLflow parent link): its
  hyperparameters, its validation metrics and its per-epoch history, so the UI's Compare and
  Parallel Coordinates views show the search;
- unless `--smoke` (or a smoke export): the ONNX model as an MLflow 3 LoggedModel, a new
  version of `tujya-pill-detector` and `promote_if_better` on `test_v2_mAP50_95`
  (params.yaml `compare.detector`), against a champion re-evaluated on the same split by
  src/models/detection/evaluate_champion.py.

A smoke export (local check of the code) gets the tag `smoke=true`, a `smoke-` run name, and
never reaches the Model Registry.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow import MlflowClient

from src.models.detection.labels import read_bgr, samples
from src.models.detection.latency import cpu_name, onnx_cpu_latency
from src.utils import mlflow_utils as mu

ROOT = mu.ROOT
PARAMS = ROOT / "params.yaml"
EXPERIMENT = "tujya-vision"
LOCAL_DATA = ROOT / "data" / "processed" / "c3pi" / "datasets_v2" / "detect"
SUMMARY_DIR = ROOT / "models" / "compare"  # git-ignored
SUPPORTED_SCHEMA = 1
LATENCY_IMAGES = 20


class ImportErrorKaggle(RuntimeError):
    """The export folder cannot be imported (schema, files, champion not re-evaluated)."""


def load_export(path: Path, work: Path) -> Path:
    """The export folder: `path` itself, or the folder holding run.json inside a .zip."""
    if path.is_file() and path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                target = (work / member).resolve()
                if not target.is_relative_to(work.resolve()):
                    raise ImportErrorKaggle(f"unsafe path in the archive: {member}")
            archive.extractall(work)
        found = sorted(work.rglob("run.json"))
        if not found:
            raise ImportErrorKaggle(f"no run.json in {path}")
        return found[0].parent
    if not (path / "run.json").is_file():
        raise ImportErrorKaggle(f"no run.json in {path}")
    return path


def read_run(folder: Path) -> dict[str, Any]:
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if run.get("schema") != SUPPORTED_SCHEMA:
        raise ImportErrorKaggle(f"run.json schema {run.get('schema')} (expected 1)")
    for key in ("arch", "family", "trials", "final", "code", "dataset"):
        if key not in run:
            raise ImportErrorKaggle(f"run.json has no {key!r}")
    return run


def _flat(values: Mapping[str, Any]) -> dict[str, float]:
    return {
        k: float(v)
        for k, v in values.items()
        if isinstance(v, int | float) and not isinstance(v, bool) and v == v  # drop NaN
    }


def _log_history(history: list[dict[str, float]], prefix: str = "") -> None:
    for step, row in enumerate(history):
        metrics = _flat({f"{prefix}{k}": v for k, v in row.items() if k != "epoch"})
        if metrics:
            mlflow.log_metrics(metrics, step=int(row.get("epoch", step)))


def _params(values: Mapping[str, Any]) -> dict[str, str]:
    return {k: str(v) for k, v in values.items()}


def parent_tags(run: dict[str, Any], smoke: bool) -> dict[str, str]:
    env, code, data = run.get("environment", {}), run["code"], run["dataset"]
    tags = {
        "git_commit": str(code.get("git_commit", "unknown")),
        "git_branch": str(code.get("git_branch", run.get("branch", ""))),
        "dvc_rev": str(data.get("dvc_rev", "unknown")),
        "dataset": str(data.get("name", "c3pi-v2-detect")),
        "seed": str(run.get("seed", 0)),
        "task": "detect",
        "model": "detector",
        "arch": run["arch"],
        "family": run["family"],
        "stage": "compare",
        "plan_step": "2.2",
        "source": str(env.get("platform", "unknown")),
        "smoke": str(smoke).lower(),
        "import_git_commit": mu.git_commit(),
        "experiment_branch": str(run.get("branch", "")),
    }
    if data.get("md5"):
        tags["dataset_md5"] = str(data["md5"])
    if code.get("git_dirty"):
        tags["git_dirty"] = "true"
    for key in ("gpu", "torch", "ultralytics", "torchvision", "python", "cpu"):
        if env.get(key):
            tags[f"env_{key}"] = str(env[key])
    return tags


def local_latency(onnx_path: Path, data_dir: Path = LOCAL_DATA) -> dict[str, float]:
    """CPU latency of the ONNX model on this PC (real test images when they are here)."""
    images = []
    if (data_dir / "images" / "test").is_dir():
        images = [read_bgr(s.image) for s in samples(data_dir, "test")[:LATENCY_IMAGES]]
    return onnx_cpu_latency(onnx_path, images=images)


def log_trials(run: dict[str, Any], smoke: bool) -> list[str]:
    """One nested run per Optuna trial; returns their run ids."""
    ids = []
    for trial in run["trials"]:
        tags = {
            "stage": "hpo",
            "arch": run["arch"],
            "trial_number": str(trial["number"]),
            "trial_state": trial.get("state", "UNKNOWN"),
            "smoke": str(smoke).lower(),
            "git_commit": str(run["code"].get("git_commit", "unknown")),
            "dataset": str(run["dataset"].get("name", "")),
        }
        if trial.get("error"):
            tags["error"] = str(trial["error"])[:5000]
        name = f"{'smoke-' if smoke else ''}trial-{int(trial['number']):03d}"
        with mlflow.start_run(run_name=name, nested=True, tags=tags) as child:
            mlflow.log_params(_params(trial.get("params", {})))
            metrics = _flat(trial.get("metrics", {}))
            if "minutes" in trial:
                metrics["train_minutes"] = float(trial["minutes"])
            if metrics:
                mlflow.log_metrics(metrics)
            _log_history(trial.get("history", []))
            ids.append(child.info.run_id)
            if trial.get("state") == "FAIL":
                mlflow.end_run(status="FAILED")
    return ids


def compare_rules(params_path: Path = PARAMS) -> dict[str, Any]:
    params = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
    registry = params["registry"]["detector"]
    compare = (params.get("compare") or {}).get("detector") or {}
    return {
        "name": registry["name"],
        "metric": compare.get("metric", "test_v2_mAP50_95"),
        "min_gain": float(registry.get("min_gain", 0.0)),
    }


def import_export(
    folder: Path,
    *,
    experiment: str = EXPERIMENT,
    smoke: bool = False,
    register: bool = True,
    measure_latency: bool = True,
    data_dir: Path = LOCAL_DATA,
    params_path: Path = PARAMS,
    summary_dir: Path | None = SUMMARY_DIR,
    client: MlflowClient | None = None,
) -> dict[str, Any]:
    """Create the MLflow runs of one export folder; register and promote unless smoke."""
    run = read_run(folder)
    smoke = smoke or bool(run.get("smoke"))
    client = client or MlflowClient()
    final = run["final"]
    onnx_path = folder / run.get("artifacts", {}).get("onnx", "model/detector.onnx")
    if not onnx_path.is_file():
        raise ImportErrorKaggle(f"missing ONNX model {onnx_path}")
    rules = compare_rules(params_path)
    if register and not smoke:
        champion = mu.alias_version(rules["name"], mu.CHAMPION, client)
        if champion is not None and (
            mu.version_metric(rules["name"], champion, rules["metric"], client) is None
        ):
            raise ImportErrorKaggle(
                f"the champion (version {champion}) has no {rules['metric']}: re-evaluate it "
                "on the same split first: python -m src.models.detection.evaluate_champion"
            )

    mlflow.set_experiment(experiment)
    name = f"{'smoke-' if smoke else ''}detector-{run['arch']}"
    with mlflow.start_run(run_name=name, tags=parent_tags(run, smoke)) as parent:
        search = run.get("search", {})
        mlflow.log_params(
            {
                **_params(final.get("params", {})),
                "arch": run["arch"],
                "family": run["family"],
                "weights": run.get("weights", ""),
                "seed": run.get("seed", 0),
                "search_n_trials": search.get("n_trials", len(run["trials"])),
                "search_epochs": search.get("epochs", ""),
                "search_fraction": search.get("fraction", ""),
                "search_sampler": search.get("sampler", "tpe"),
                "search_metric": search.get("metric", "val_mAP50_95"),
                "search_best_trial": search.get("best_trial", ""),
                "final_epochs": final.get("epochs", ""),
                "final_patience": final.get("patience", ""),
                **{f"n_{k}_images": v for k, v in run["dataset"].get("images", {}).items()},
                "n_test_holdout_images": run["dataset"].get("holdout", ""),
            }
        )
        metrics = _flat(final.get("metrics", {}))
        metrics.update(_flat(final.get("latency", {})))
        if "best_value" in search:
            metrics["search_best_value"] = float(search["best_value"])
        for key, value in (
            ("search_minutes", search.get("minutes")),
            ("train_minutes", final.get("minutes")),
            ("total_minutes", run.get("minutes")),
            ("parameters", final.get("parameters")),
        ):
            if value is not None:
                metrics[key] = float(value)
        if measure_latency:
            metrics.update(local_latency(onnx_path, data_dir))
            mlflow.set_tag("latency_cpu", cpu_name())
        mlflow.log_metrics(metrics)
        _log_history(final.get("history", []))
        _log_artifacts(folder, run)
        child_ids = log_trials(run, smoke)
        parent_id = parent.info.run_id

        result: dict[str, Any] = {
            "arch": run["arch"],
            "run_id": parent_id,
            "trial_run_ids": child_ids,
            "smoke": smoke,
            "metrics": metrics,
        }
        if register and not smoke:
            import onnx

            info = mu.log_model(
                onnx.load(str(onnx_path)),
                name="detector",
                metrics={k: v for k, v in metrics.items() if k.startswith(("test_", "val_"))},
                params={
                    "arch": run["arch"],
                    "imgsz": final.get("params", {}).get("imgsz", ""),
                    "task": "detect",
                },
            )
            result["model_id"] = info.model_id

    if register and not smoke:
        imgsz = final.get("params", {}).get("imgsz", "")
        version = mu.register(
            info.model_uri,
            rules["name"],
            tags={
                "task": "detect",
                "arch": run["arch"],
                "family": run["family"],
                "dataset": str(run["dataset"].get("name", "")),
                "git_commit": str(run["code"].get("git_commit", "unknown")),
                "experiment_branch": str(run.get("branch", "")),
                "baseline": "false",
            },
            description=(
                f"{run['arch']} (detect, {run['family']}), ONNX {imgsz}px, dataset "
                f"{run['dataset'].get('name', '')}, {rules['metric']}="
                f"{metrics.get(rules['metric'], float('nan')):.4f}, run {parent_id}, "
                f"branch {run.get('branch', '')}"
            ),
            client=client,
        )
        promotion = mu.promote_if_better(
            rules["name"], version.version, rules["metric"], rules["min_gain"], client=client
        )
        result.update(
            registered_model=rules["name"],
            version=int(version.version),
            alias=promotion.status,
            promoted=promotion.promoted,
            promotion=promotion.reason,
        )
    if summary_dir is not None:
        summary_dir.mkdir(parents=True, exist_ok=True)
        prefix = "smoke-" if smoke else ""
        (summary_dir / f"{prefix}detector_{run['arch']}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def _log_artifacts(folder: Path, run: dict[str, Any]) -> None:
    artifacts = run.get("artifacts", {})
    mlflow.log_artifact(str(folder / "run.json"), "export")
    for key, target in (("config", "export"), ("trials_csv", "export")):
        path = folder / artifacts.get(key, "")
        if artifacts.get(key) and path.is_file():
            mlflow.log_artifact(str(path), target)
    requirements = folder / artifacts.get("requirements", "requirements.txt")
    if requirements.is_file():
        mlflow.log_artifact(str(requirements), "reproducibility")  # the Kaggle environment
    for name in mu.REPRODUCIBILITY_FILES:
        if (ROOT / name).is_file():
            mlflow.log_artifact(str(ROOT / name), "reproducibility")
    weights = folder / artifacts.get("weights", "weights/best.pt")
    if weights.is_file():
        mlflow.log_artifact(str(weights), "weights")
    onnx_file = folder / artifacts.get("onnx", "model/detector.onnx")
    mlflow.log_artifact(str(onnx_file), "onnx")
    training = folder / artifacts.get("training", "training")
    if training.is_dir():
        mlflow.log_artifacts(str(training), "training")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.detection.import_kaggle")
    parser.add_argument("exports", nargs="+", type=Path, help="export folders or .zip files")
    parser.add_argument("--smoke", action="store_true", help="tag smoke=true, no Registry")
    parser.add_argument("--no-register", action="store_true", help="runs only, no Registry")
    parser.add_argument("--no-latency", action="store_true", help="skip the local CPU timing")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--data", type=Path, default=LOCAL_DATA, help="test images for timing")
    args = parser.parse_args(argv)
    mu.configure()
    for export in args.exports:
        with tempfile.TemporaryDirectory() as tmp:
            folder = load_export(export, Path(tmp))
            result = import_export(
                folder,
                experiment=args.experiment,
                smoke=args.smoke,
                register=not args.no_register,
                measure_latency=not args.no_latency,
                data_dir=args.data,
            )
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
