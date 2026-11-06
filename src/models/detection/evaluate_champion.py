"""Re-evaluate the current champion detector on the comparison split of step 2.2.

    # training venv, MLflow server running, dataset of stage prepare_detect_v2 on disk
    python -m src.models.detection.evaluate_champion            # alias champion
    python -m src.models.detection.evaluate_champion --version 1

The baseline (YOLO11n, trained on C3PI v1) was scored on the v1 test split; the candidates of
step 2.2 are trained on C3PI v2 and scored on the v2 comparison split (`test_holdout.txt`:
v2 test sessions the baseline never saw, D93). To compare like with like, this script scores
the registered version again, with the same code as the candidates
(src/models/detection/runner.py `test_metrics`, src/models/detection/metrics.py):

1. downloads the weights logged in the version's run (`weights/best.pt`) and its ONNX model;
2. computes `test_v2_*` (comparison split) and `test_v2_all_*` (whole v2 test) and the CPU
   latency of the ONNX model on this PC (`onnx_cpu_latency_ms`);
3. logs them in a run `evaluate-detector-v<version>-c3pi-v2` (tag `stage=compare-eval`) and
   adds them to the version's own run, where promote_if_better reads the metric.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from src.models.detection.import_kaggle import EXPERIMENT, LOCAL_DATA, compare_rules, local_latency
from src.models.detection.latency import cpu_name
from src.models.detection.runner import backend, dataset_info, test_metrics
from src.utils import mlflow_utils as mu
from src.utils.console import console_utf8


def family_of(arch: str, tags: dict[str, str]) -> str:
    if tags.get("family"):
        return tags["family"]
    return "fasterrcnn" if "rcnn" in arch.lower() else "yolo"


def evaluate_version(
    version: str | None = None,
    *,
    alias: str = mu.CHAMPION,
    data_dir: Path = LOCAL_DATA,
    weights: Path | None = None,
    client: MlflowClient | None = None,
) -> dict[str, Any]:
    client = client or MlflowClient()
    rules = compare_rules()
    name = rules["name"]
    version = version or mu.alias_version(name, alias, client)
    if version is None:
        raise SystemExit(f"no version of {name} holds the alias {alias!r}")
    mv = client.get_model_version(name, str(version))
    run = client.get_run(mv.run_id)
    arch = mv.tags.get("arch") or run.data.params.get("arch") or "unknown"
    imgsz = int(run.data.params.get("imgsz", 640))
    module = backend(family_of(arch, mv.tags))

    with tempfile.TemporaryDirectory() as tmp:
        if weights is None:
            weights = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=mv.run_id, artifact_path="weights/best.pt", dst_path=tmp
                )
            )
        metrics = test_metrics(module, weights, data_dir, imgsz)
        model_dir = Path(
            mlflow.artifacts.download_artifacts(f"models:/{name}/{version}", dst_path=tmp)
        )
        onnx_files = sorted(model_dir.rglob("*.onnx"))
        if onnx_files:
            metrics.update(local_latency(onnx_files[0], data_dir))

    dataset = dataset_info(data_dir)
    tags = {
        "stage": "compare-eval",
        "task": "detect",
        "arch": arch,
        "evaluated_model": name,
        "evaluated_version": str(version),
        "evaluated_run_id": mv.run_id,
        "dataset": "c3pi-v2-detect",
        "plan_step": "2.2",
        "smoke": "false",
    }
    if dataset.get("md5"):
        tags["dataset_md5"] = str(dataset["md5"])
    with mu.start_tracked_run(
        EXPERIMENT,
        f"evaluate-detector-v{version}-c3pi-v2",
        seed=int(run.data.params.get("seed", run.data.tags.get("seed", 0))),
        dataset="c3pi-v2-detect",
        dataset_path="data/processed/c3pi/datasets_v2/detect",
        params={"imgsz": imgsz, "arch": arch, "evaluated_version": version},
        tags=tags,
    ) as eval_run:
        mlflow.log_metrics(metrics)
        mlflow.set_tag("latency_cpu", cpu_name())
    # The version's own run receives the metrics too: promote_if_better reads them there.
    with mlflow.start_run(run_id=mv.run_id):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("test_v2_eval_run_id", eval_run.info.run_id)
    client.set_model_version_tag(
        name, str(version), rules["metric"], f"{metrics[rules['metric']]:.4f}"
    )
    return {
        "model": name,
        "version": str(version),
        "arch": arch,
        "run_id": mv.run_id,
        "eval_run_id": eval_run.info.run_id,
        **metrics,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.detection.evaluate_champion")
    parser.add_argument("--version", help="registered version (default: the alias)")
    parser.add_argument("--alias", default=mu.CHAMPION)
    parser.add_argument("--data", type=Path, default=LOCAL_DATA)
    parser.add_argument("--weights", type=Path, help="local weights instead of the run's")
    args = parser.parse_args(argv)
    console_utf8()
    mu.configure()
    result = evaluate_version(
        args.version, alias=args.alias, data_dir=args.data, weights=args.weights
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
