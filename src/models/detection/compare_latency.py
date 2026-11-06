"""CPU latency, size and parameters of every compared detector, measured together (step 2.2).

    # training venv, MLflow server running, dataset of stage prepare_detect_v2 on disk
    python -m src.models.detection.compare_latency             # measure and log
    python -m src.models.detection.compare_latency --dry-run   # measure and print only

The development PC is a laptop whose CPU slows down under a sustained load (D96): a model
timed alone, right after another one, is measured on a hotter CPU. So the FP32 detectors of the
comparison (every version of `tujya-pill-detector` without a `precision` tag, baseline
included, superseded versions excluded) are timed together with the protocol of step 2.4
(`src.models.optimize.measure_latency`, params.yaml `optimize`): interleaved rounds, each model
once per round in a fresh process after a rest, medians over the rounds, on the same test
images of the comparison split. The models are timed from the smallest file to the largest,
so the slow two-stage detector never runs just before a small one within a round.

For each model the figures are logged to the run of its Registry version and to the
re-evaluation run of the baseline (`stage=compare-eval`, read by compare.py):
`onnx_cpu_latency_ms` (mean, median of the rounds), `onnx_cpu_latency_median_ms`,
`onnx_cpu_latency_p95_ms`, the mean of every round (`onnx_cpu_latency_round_ms`, one step per
round, to check that no round was throttled), `model_size_mb` (self-contained FP32 ONNX file)
and `onnx_parameters` (floating-point weights of the ONNX graph), with the tags
`latency_protocol`, `latency_cpu` and `latency_git_commit`. A summary is written to
models/compare/latency.json (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from mlflow import MlflowClient

from src.models import optimize
from src.models.detection.latency import cpu_name
from src.models.quantize import size_mb
from src.models.serving import PRECISION_TAG, REGISTERED
from src.utils import mlflow_utils as mu
from src.utils.console import console_utf8

ROOT = mu.ROOT
SUMMARY = ROOT / "models" / "compare" / "latency.json"
FLOAT_TYPES = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16, onnx.TensorProto.DOUBLE}


def onnx_parameters(path: Path) -> int:
    """Number of floating-point weights stored in the ONNX graph (its initializers)."""
    model = onnx.load(str(path))
    return int(
        sum(
            int(np.prod(tensor.dims)) if tensor.dims else 1
            for tensor in model.graph.initializer
            if tensor.data_type in FLOAT_TYPES
        )
    )


def compared_versions(name: str, client: MlflowClient) -> list[Any]:
    """FP32 versions of the comparison: no `precision` tag (variants), not superseded."""
    versions = [
        v
        for v in client.search_model_versions(f"name='{name}'")
        if PRECISION_TAG not in (v.tags or {}) and (v.tags or {}).get("status") != "superseded"
    ]
    return sorted(versions, key=lambda v: int(v.version))


def target_runs(run_id: str, experiment_ids: list[str], client: MlflowClient) -> list[str]:
    """The version's run and the re-evaluation runs of it (`evaluated_run_id`)."""
    found = client.search_runs(experiment_ids, f"tags.evaluated_run_id = '{run_id}'")
    return [run_id, *(run.info.run_id for run in found)]


def protocol(settings: dict[str, Any]) -> str:
    return (
        f"interleaved rounds (D96): {int(settings.get('latency_rounds', 1))} rounds x "
        f"{int(settings.get('latency_runs', 30))} passes after "
        f"{int(settings.get('latency_warmup', 5))} warm-up, "
        f"{float(settings.get('latency_cooldown_s', 0)):g} s rest before each, at most "
        f"{float(settings.get('latency_budget_s', 0)):g} s of load each (0 = no limit), "
        f"{int(settings.get('latency_images', 20))} test images, medians of the rounds"
    )


def measure_all(
    *,
    name: str = REGISTERED["detect"],
    experiment: str = optimize.EXPERIMENT,
    data: Path = optimize.DETECT_DATA,
    settings: dict[str, Any] | None = None,
    log: bool = True,
    client: MlflowClient | None = None,
) -> dict[str, Any]:
    """Time every compared detector together; log the figures unless `log` is False."""
    import mlflow

    client = client or MlflowClient()
    settings = settings or optimize.load_settings()
    versions = compared_versions(name, client)
    if not versions:
        raise SystemExit(f"no FP32 version of {name} to measure")
    exp = client.get_experiment_by_name(experiment)
    experiment_ids = [exp.experiment_id] if exp is not None else []
    rows: dict[str, dict[str, Any]] = {}
    with tempfile.TemporaryDirectory() as tmp:
        paths: dict[str, Path] = {}
        for v in versions:
            key = f"v{v.version}"
            downloaded = Path(
                mlflow.artifacts.download_artifacts(
                    f"models:/{name}/{v.version}", dst_path=str(Path(tmp) / key)
                )
            )
            path = Path(tmp) / f"{key}.onnx"
            # One self-contained file (MLflow stores the weights as external data).
            onnx.save_model(onnx.load(str(sorted(downloaded.rglob("*.onnx"))[0])), str(path))
            paths[key] = path
            rows[key] = {
                "version": int(v.version),
                "arch": (v.tags or {}).get("arch", "?"),
                "run_id": v.run_id,
                "model_size_mb": size_mb(path),
                "onnx_parameters": onnx_parameters(path),
            }
        order = dict(sorted(paths.items(), key=lambda item: rows[item[0]]["model_size_mb"]))
        rounds: dict[str, list[dict[str, float]]] = {}
        start = time.time()
        medians = optimize.measure_latency("detect", order, data, settings, rounds_out=rounds)
    for key, row in rows.items():
        row["latency"] = medians[key]
        row["round_means_ms"] = [r["onnx_cpu_latency_ms"] for r in rounds[key]]
        row["targets"] = target_runs(row["run_id"], experiment_ids, client) if log else []
    summary = {
        "registered_model": name,
        "protocol": protocol(settings),
        "cpu": cpu_name(),
        "git_commit": mu.git_commit(),
        "order": list(order),
        "minutes": (time.time() - start) / 60,
        "models": rows,
    }
    if log:
        for row in rows.values():
            _log(row, summary, client)
    return summary


def _log(row: dict[str, Any], summary: dict[str, Any], client: MlflowClient) -> None:
    from mlflow.entities import Metric

    now = int(time.time() * 1000)
    values = {
        **row["latency"],
        "model_size_mb": row["model_size_mb"],
        "onnx_parameters": float(row["onnx_parameters"]),
    }
    metrics = [Metric(key, float(value), now, 0) for key, value in values.items()]
    metrics += [
        Metric("onnx_cpu_latency_round_ms", float(value), now, step)
        for step, value in enumerate(row["round_means_ms"])
    ]
    for run_id in row["targets"]:
        client.log_batch(run_id, metrics=metrics)
        client.set_tag(run_id, "latency_protocol", summary["protocol"])
        client.set_tag(run_id, "latency_cpu", summary["cpu"])
        client.set_tag(run_id, "latency_git_commit", summary["git_commit"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.detection.compare_latency")
    parser.add_argument("--dry-run", action="store_true", help="measure and print, log nothing")
    parser.add_argument("--data", type=Path, default=optimize.DETECT_DATA)
    args = parser.parse_args(argv)
    console_utf8()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mu.configure()
    summary = measure_all(data=args.data, log=not args.dry_run)
    for key, row in summary["models"].items():
        lat = row["latency"]
        print(
            f"{key} {row['arch']}: {lat['onnx_cpu_latency_ms']:.1f} ms "
            f"(p95 {lat['onnx_cpu_latency_p95_ms']:.1f}; rounds "
            f"{', '.join(f'{m:.1f}' for m in row['round_means_ms'])}), "
            f"{row['model_size_mb']:.1f} MB, {row['onnx_parameters'] / 1e6:.2f} M parameters"
        )
    if not args.dry_run:
        SUMMARY.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
