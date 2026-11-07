"""Import a classifier experiment run on Kaggle (export folder of runner.py) into MLflow.

    # training venv, MLflow server running (docker compose up -d mlflow)
    python -m src.models.classification.import_kaggle --no-latency classifier_*_export.zip
    python -m src.models.classification.import_kaggle <folder or .zip> --smoke   # never registered

Same steps as the detectors (src/models/detection/import_kaggle.py, whose export reading,
ordering and trial logging are reused): several exports are imported best first (promotion
metric of their run.json, D98), so the result does not depend on the order of the arguments.

What it creates in the experiment `tujya-vision`:

- one parent run `classifier-<arch>` (tags `stage=compare`, `task=classify`): provenance of
  the Kaggle run (`git_commit`, `git_branch`, `dvc_rev`, `dataset_md5`, `seed`, GPU), the
  retained hyperparameters, the per-epoch history of the final training, the validation and
  test metrics (`test_v2_*` comparison split, `test_v2_all_*` whole v2 test, `test_v2_v1c_*`
  like-for-like with the v1 champion), latency, and the artefacts (weights, ONNX model, class
  list, confusion matrix, per-class scores, Grad-CAM pictures, trials, requirements);
- one child run per Optuna trial (`trial-NNN`, tag `stage=hpo`);
- unless `--smoke` (or a smoke export): the ONNX model as a LoggedModel, a new version of
  `tujya-pill-classifier` (tags `label_space`, `num_classes`) and `promote_if_better` on
  `test_v2_top1` (params.yaml `compare.classifier`) with the guard of D100: a model trained on
  another class list than the reference (the v1 champion) must not be worse than it on the
  like-for-like metric `test_v2_v1c_top1`, otherwise it is archived.

A smoke export gets the tag `smoke=true`, a `smoke-` run name and never reaches the Registry.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow import MlflowClient

from src.models.classification import data as D
from src.models.detection.import_kaggle import (
    ImportErrorKaggle,
    _flat,
    _log_history,
    _params,
    load_export,
    log_trials,
    peek_run,
    promotion_order,
    read_run,
)
from src.models.detection.latency import cpu_name, isolated_latency
from src.utils import mlflow_utils as mu
from src.utils.console import console_utf8

__all__ = ["ImportErrorKaggle", "import_export", "load_export", "peek_run", "promotion_order"]

ROOT = mu.ROOT
PARAMS = ROOT / "params.yaml"
EXPERIMENT = "tujya-vision"
LOCAL_DATA = D.DEFAULT_DATA
SUMMARY_DIR = ROOT / "models" / "compare"  # git-ignored
LATENCY_IMAGES = 20


def compare_rules(params_path: Path = PARAMS) -> dict[str, Any]:
    params = yaml.safe_load(params_path.read_text(encoding="utf-8")) or {}
    registry = params["registry"]["classifier"]
    compare = (params.get("compare") or {}).get("classifier") or {}
    return {
        "name": registry["name"],
        "metric": compare.get("metric", "test_v2_top1"),
        "min_gain": float(registry.get("min_gain", 0.0)),
        "dataset": compare.get("dataset", D.DATASET_NAME),
        "guard_metric": compare.get("guard_metric", "test_v2_v1c_top1"),
        "guard_max_drop": float(compare.get("guard_max_drop", 0.0)),
    }


def guard_reference(name: str, client: MlflowClient) -> str | None:
    """The version the guard compares with: the baseline (the v1 champion, step 2.1)."""
    for mv in client.search_model_versions(f"name='{name}'"):
        tags = mv.tags or {}
        if tags.get("baseline") == "true" and "precision" not in tags:
            return str(mv.version)
    return None


def check_guard(
    metrics: dict[str, float], rules: dict[str, Any], client: MlflowClient
) -> tuple[bool, str, str | None]:
    """(eligible, reason, reference version) of a candidate (D100).

    The reference serves another class list (150 classes of v1): the candidate must not be
    worse than it on the like-for-like metric (same comparison crops of the v1 classes, the
    prediction restricted to them), by more than `guard_max_drop`.
    """
    reference = guard_reference(rules["name"], client)
    if reference is None:
        return True, "no reference version", None
    reference_value = mu.version_metric(rules["name"], reference, rules["guard_metric"], client)
    if reference_value is None:
        raise ImportErrorKaggle(
            f"the reference (version {reference}) has no {rules['guard_metric']}: score it on "
            "the same split first: python -m src.models.classification.evaluate_champion"
        )
    value = metrics.get(rules["guard_metric"])
    if value is None:
        return False, f"no {rules['guard_metric']} in the export", reference
    limit = reference_value - rules["guard_max_drop"]
    if value >= limit:
        return True, f"{rules['guard_metric']}={value:.4f} >= {limit:.4f} (v{reference})", reference
    return (
        False,
        f"{rules['guard_metric']}={value:.4f} < {limit:.4f} of the reference v{reference}",
        reference,
    )


def parent_tags(run: dict[str, Any], smoke: bool) -> dict[str, str]:
    env, code, data = run.get("environment", {}), run["code"], run["dataset"]
    tags = {
        "git_commit": str(code.get("git_commit", "unknown")),
        "git_branch": str(code.get("git_branch", run.get("branch", ""))),
        "dvc_rev": str(data.get("dvc_rev", "unknown")),
        "dataset": str(data.get("name", D.DATASET_NAME)),
        "label_space": str(data.get("label_space", D.LABEL_SPACE)),
        "seed": str(run.get("seed", 0)),
        "task": "classify",
        "model": "classifier",
        "arch": run["arch"],
        "family": run["family"],
        "pretrained": str(run.get("pretrained", True)).lower(),
        "stage": "compare",
        "plan_step": "2.3",
        "source": str(env.get("platform", "unknown")),
        "smoke": str(smoke).lower(),
        "import_git_commit": mu.git_commit(),
        "experiment_branch": str(run.get("branch", "")),
    }
    if data.get("md5"):
        tags["dataset_md5"] = str(data["md5"])
    if code.get("git_dirty"):
        tags["git_dirty"] = "true"
    for key in ("gpu", "torch", "torchvision", "ultralytics", "python", "cpu"):
        if env.get(key):
            tags[f"env_{key}"] = str(env[key])
    return tags


def local_latency(onnx_path: Path, data_dir: Path = LOCAL_DATA) -> dict[str, float]:
    """CPU latency of the ONNX model on this PC (real test crops when they are here)."""
    files: list[Path] = []
    if (data_dir / "index.csv").is_file():
        files = [c.path for c in D.crops(data_dir, "test")[:LATENCY_IMAGES]]
    return isolated_latency(onnx_path, files)


def _log_artifacts(folder: Path, run: dict[str, Any]) -> None:
    artifacts = run.get("artifacts", {})
    mlflow.log_artifact(str(folder / "run.json"), "export")
    for key in ("config", "trials_csv", "classes"):
        path = folder / str(artifacts.get(key, ""))
        if artifacts.get(key) and path.is_file():
            mlflow.log_artifact(str(path), "export")
    requirements = folder / artifacts.get("requirements", "requirements.txt")
    if requirements.is_file():
        mlflow.log_artifact(str(requirements), "reproducibility")  # the Kaggle environment
    for name in mu.REPRODUCIBILITY_FILES:
        if (ROOT / name).is_file():
            mlflow.log_artifact(str(ROOT / name), "reproducibility")
    weights = folder / artifacts.get("weights", "weights/best.pt")
    if weights.is_file():
        mlflow.log_artifact(str(weights), "weights")
    mlflow.log_artifact(str(folder / artifacts.get("onnx", "model/classifier.onnx")), "onnx")
    for key in ("training", "confusion", "gradcam"):
        path = folder / str(artifacts.get(key, key))
        if path.is_dir():
            mlflow.log_artifacts(str(path), key)


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
    if run.get("task", "classify") != "classify":
        raise ImportErrorKaggle(f"not a classifier export (task {run.get('task')!r})")
    smoke = smoke or bool(run.get("smoke"))
    client = client or MlflowClient()
    final = run["final"]
    onnx_path = folder / run.get("artifacts", {}).get("onnx", "model/classifier.onnx")
    if not onnx_path.is_file():
        raise ImportErrorKaggle(f"missing ONNX model {onnx_path}")
    rules = compare_rules(params_path)
    eligible, guard_reason, reference = True, "", None
    if register and not smoke:
        champion = mu.alias_version(rules["name"], mu.CHAMPION, client)
        if champion is not None and (
            mu.version_metric(rules["name"], champion, rules["metric"], client) is None
        ):
            raise ImportErrorKaggle(
                f"the champion (version {champion}) has no {rules['metric']}: re-evaluate it "
                "on the same split first: python -m src.models.classification.evaluate_champion"
            )
        eligible, guard_reason, reference = check_guard(
            _flat(final.get("metrics", {})), rules, client
        )

    mlflow.set_experiment(experiment)
    name = f"{'smoke-' if smoke else ''}classifier-{run['arch']}"
    with mlflow.start_run(run_name=name, tags=parent_tags(run, smoke)) as parent:
        search = run.get("search", {})
        mlflow.log_params(
            {
                **_params(final.get("params", {})),
                "arch": run["arch"],
                "model": run.get("model", run["arch"]),
                "family": run["family"],
                "pretrained": run.get("pretrained", True),
                "seed": run.get("seed", 0),
                "search_n_trials": search.get("n_trials", len(run["trials"])),
                "search_epochs": search.get("epochs", ""),
                "search_fraction": search.get("fraction", ""),
                "search_sampler": search.get("sampler", "tpe"),
                "search_metric": search.get("metric", "val_top1"),
                "search_best_trial": search.get("best_trial", ""),
                "final_epochs": final.get("epochs", ""),
                "final_patience": final.get("patience", ""),
                "num_classes": run["dataset"].get("classes", ""),
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
        if guard_reason:
            mlflow.set_tag("guard", guard_reason[:5000])

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
                name="classifier",
                metrics={k: v for k, v in metrics.items() if k.startswith(("test_", "val_"))},
                params={
                    "arch": run["arch"],
                    "imgsz": final.get("params", {}).get("imgsz", ""),
                    "task": "classify",
                },
            )
            result["model_id"] = info.model_id

    if register and not smoke:
        imgsz = final.get("params", {}).get("imgsz", "")
        version = mu.register(
            info.model_uri,
            rules["name"],
            tags={
                "task": "classify",
                "arch": run["arch"],
                "family": run["family"],
                "dataset": str(run["dataset"].get("name", "")),
                "label_space": str(run["dataset"].get("label_space", D.LABEL_SPACE)),
                "num_classes": str(run["dataset"].get("classes", "")),
                "git_commit": str(run["code"].get("git_commit", "unknown")),
                "experiment_branch": str(run.get("branch", "")),
                "baseline": "false",
                **({"guard": guard_reason[:5000]} if guard_reason else {}),
            },
            description=(
                f"{run['arch']} (classify, {run['dataset'].get('classes', '')} classes of "
                f"{run['dataset'].get('label_space', D.LABEL_SPACE)}), ONNX {imgsz}px, "
                f"{rules['metric']}={metrics.get(rules['metric'], float('nan')):.4f}, run "
                f"{parent_id}, branch {run.get('branch', '')}"
            ),
            client=client,
        )
        promotion = mu.promote_if_better(
            rules["name"],
            version.version,
            rules["metric"],
            rules["min_gain"],
            eligible=eligible,
            ineligible_reason=guard_reason,
            client=client,
        )
        result.update(
            registered_model=rules["name"],
            version=int(version.version),
            alias=promotion.status,
            promoted=promotion.promoted,
            promotion=promotion.reason,
            eligible=eligible,
            guard=guard_reason,
            guard_reference=reference,
        )
    if summary_dir is not None:
        summary_dir.mkdir(parents=True, exist_ok=True)
        prefix = "smoke-" if smoke else ""
        (summary_dir / f"{prefix}classifier_{run['arch']}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.classification.import_kaggle")
    parser.add_argument("exports", nargs="+", type=Path, help="export folders or .zip files")
    parser.add_argument("--smoke", action="store_true", help="tag smoke=true, no Registry")
    parser.add_argument("--no-register", action="store_true", help="runs only, no Registry")
    parser.add_argument("--no-latency", action="store_true", help="skip the local CPU timing")
    parser.add_argument("--experiment", default=EXPERIMENT)
    parser.add_argument("--data", type=Path, default=LOCAL_DATA, help="test crops for timing")
    parser.add_argument(
        "--keep-order",
        action="store_true",
        help="import in the order given (default: best promotion metric first, D98)",
    )
    args = parser.parse_args(argv)
    console_utf8()
    mu.configure()
    exports = list(args.exports)
    if not args.keep_order:
        exports = promotion_order(exports, compare_rules()["metric"])
        print("import order:", ", ".join(str(e) for e in exports))
    for export in exports:
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
