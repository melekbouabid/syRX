"""Optimise the two champions for CPU serving: FP32 / FP16 / INT8 ONNX variants (step 2.4).

    # training venv (.venv-train), MLflow server running, datasets of the DVC stages on disk
    python -m src.models.optimize                     # both champions, register, alias served
    python -m src.models.optimize --task classify     # one of them
    python -m src.models.optimize --no-register       # measure only (runs, no Registry change)
    python -m src.models.optimize --publish           # then copy the served variants to models/

For the version holding the alias `champion` of each registered model:

1. downloads its ONNX model (FP32, the export of the Registry) and builds the variants of
   src/models/quantize.py: FP16, INT8 static (QDQ, calibrated on images of the validation
   split, never the test split) and INT8 dynamic;
2. scores every variant with the same code: the detector with src/models/detection/metrics.py
   on the comparison split of step 2.2 (`test_holdout.txt` of C3PI v2, decoded like the API by
   src/models/detection/latency.py `onnx_predict`), the classifier on the test crops of C3PI v1
   (top-1 / top-5, preprocessed like the API); CPU latency (mean, p95) in a fresh process
   (`isolated_latency`) on the same test images; file size;
3. logs one MLflow run per variant (experiment tujya-vision, tag stage=optimize, provenance tags
   of src/utils/mlflow_utils.py) and registers each FP16 / INT8 variant as a new version of
   the same registered model, tagged `precision` and `variant_of`;
4. picks the served variant (`choose_variant`, params.yaml `optimize`) and gives it the alias
   `served` (src/models/serving.py): the alias `champion` stays on the FP32 model, so
   promote_if_better keeps comparing FP32 candidates with an FP32 champion.

Writes models/optimize.json (git-ignored) and prints the comparison table of
docs/module1/optimisation_onnx.md. A new champion (step 2.2 or 2.3) only needs this command
again, then `python -m src.models.publish`.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import yaml
from mlflow import MlflowClient

from src.models import quantize as q
from src.models import serving
from src.models.detection.labels import holdout_names, read_bgr, samples
from src.models.detection.latency import (
    cpu_name,
    input_size,
    isolated_latency,
    letterbox,
    onnx_predict,
    to_tensor,
)
from src.models.detection.metrics import evaluate, prefixed
from src.utils import mlflow_utils as mu
from src.utils.console import console_utf8

ROOT = mu.ROOT
PARAMS = ROOT / "params.yaml"
REPORT = ROOT / "models" / "optimize.json"
EXPERIMENT = "tujya-vision"
DETECT_DATA = ROOT / "data" / "processed" / "c3pi" / "datasets_v2" / "detect"
CLASSIFY_DATA = ROOT / "data" / "processed" / "c3pi" / "datasets" / "classify"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class TaskSpec:
    """What differs between the detector and the classifier."""

    task: str  # detect / classify
    model: str  # params.yaml key: detector / classifier
    dataset: str
    dataset_path: str  # DVC output (tag dataset_md5)
    metric: str  # accuracy figure compared between variants
    test_split: str


SPECS = {
    "detect": TaskSpec(
        "detect",
        "detector",
        "c3pi-v2-detect",
        "data/processed/c3pi/datasets_v2/detect",
        "test_v2_mAP50_95",
        "test_holdout.txt of C3PI v2 (comparison split of step 2.2)",
    ),
    "classify": TaskSpec(
        "classify",
        "classifier",
        "c3pi-v1",
        "data/processed/c3pi/datasets",
        "test_top1",
        "test crops of C3PI v1",
    ),
}


def load_settings(path: Path = PARAMS) -> dict[str, Any]:
    params = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(params.get("optimize") or {})


def spread(items: Sequence[Any], n: int) -> list[Any]:
    """`n` items evenly spread over a sorted list (all of them when n <= 0 or too few)."""
    if n <= 0 or len(items) <= n:
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


# --- data ---------------------------------------------------------------------------------


def crop_files(root: Path, split: str) -> list[Path]:
    """Classification crops of a split: <root>/<split>/<class folder>/<image>, sorted."""
    folder = root / split
    return sorted(p for p in folder.glob("*/*") if p.suffix.lower() in IMAGE_SUFFIXES)


def crop_tensor(image: np.ndarray, size: int) -> np.ndarray:
    """A square crop as the API feeds the classifier (backend/app/inference.py)."""
    if image.shape[:2] != (size, size):
        image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    return to_tensor(image)


def class_names(model: Path) -> list[str]:
    """Output index -> class folder name, from the Ultralytics metadata (`names`)."""
    import ast

    meta = {p.key: p.value for p in onnx.load(str(model), load_external_data=False).metadata_props}
    names = ast.literal_eval(meta.get("names", "{}"))
    return [str(names[i]) for i in sorted(names)]


def calibration_tensors(task: str, model: Path, n: int, data: Path) -> list[np.ndarray]:
    """Validation images preprocessed as served (never the test split)."""
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    size = input_size(session)
    if task == "detect":
        files = [s.image for s in spread(samples(data, "val"), n)]
        return [to_tensor(letterbox(read_bgr(f), size)) for f in files]
    return [crop_tensor(read_bgr(f), size) for f in spread(crop_files(data, "val"), n)]


# --- scoring ------------------------------------------------------------------------------


def score_detector(model: Path, data: Path) -> dict[str, float]:
    """test_v2_* on the comparison split, decoded like the API (conf 0.001, as step 2.2)."""
    chosen = samples(data, "test", holdout_names(data))
    predictions = onnx_predict(model, [read_bgr(s.image) for s in chosen])
    return prefixed(evaluate(predictions, [s.truth() for s in chosen]), "test_v2")


def score_classifier(model: Path, data: Path) -> dict[str, float]:
    """test_top1 / test_top5 of the ONNX model on the test crops."""
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    size, name = input_size(session), session.get_inputs()[0].name
    index = {folder: i for i, folder in enumerate(class_names(model))}
    files = crop_files(data, "test")
    top1 = top5 = 0
    for path in files:
        probs = session.run(None, {name: crop_tensor(read_bgr(path), size)})[0][0]
        order = np.argsort(-probs)
        truth = index[path.parent.name]
        top1 += int(order[0] == truth)
        top5 += int(truth in order[:5])
    n = max(1, len(files))
    return {"test_top1": top1 / n, "test_top5": top5 / n, "test_images": float(len(files))}


def latency_files(task: str, data: Path, n: int) -> list[Path]:
    if task == "detect":
        return [s.image for s in samples(data, "test", holdout_names(data))[:n]]
    return spread(crop_files(data, "test"), n)


# --- variants -----------------------------------------------------------------------------


def build_variants(
    task: str, fp32: Path, work: Path, settings: dict[str, Any], data: Path
) -> dict[str, tuple[Path, dict[str, Any]]]:
    """The four precisions of one model and the parameters that describe how each was made."""
    model_settings = settings[SPECS[task].model]
    n_cal = int(model_settings["calibration_images"])
    method = str(model_settings.get("calibration_method", "minmax"))
    exclude: list[str] = []
    if model_settings.get("exclude_head"):
        exclude = q.head_nodes(fp32, q.output_module_prefix(fp32))
    tensors = calibration_tensors(task, fp32, n_cal, data)
    out: dict[str, tuple[Path, dict[str, Any]]] = {
        q.FP32: (fp32, {"method": "onnx-export", "calibration_images": 0}),
        q.FP16: (
            q.to_fp16(fp32, work / f"{task}-fp16.onnx"),
            {"method": "float16 (keep_io_types)", "calibration_images": 0},
        ),
        q.INT8_STATIC: (
            q.to_int8_static(
                fp32,
                work / f"{task}-int8-static.onnx",
                tensors,
                method=method,
                exclude_nodes=exclude,
            ),
            {
                "method": "static QDQ, int8 per-channel weights, int8 activations",
                "calibration_method": method,
                "calibration_images": len(tensors),
                "calibration_split": "val",
                "excluded_nodes": len(exclude),
            },
        ),
        q.INT8_DYNAMIC: (
            q.to_int8_dynamic(fp32, work / f"{task}-int8-dynamic.onnx"),
            {"method": "dynamic, uint8 weights", "calibration_images": 0},
        ),
    }
    return out


def measure(task: str, path: Path, data: Path, settings: dict[str, Any]) -> dict[str, float]:
    scorer: Callable[[Path, Path], dict[str, float]] = (
        score_detector if task == "detect" else score_classifier
    )
    metrics = scorer(path, data)
    files = latency_files(task, data, int(settings.get("latency_images", 20)))
    latency = isolated_latency(
        path,
        files,
        runs=int(settings.get("latency_runs", 200)),
        warmup=int(settings.get("latency_warmup", 20)),
    )
    return {**metrics, **latency, "model_size_mb": q.size_mb(path)}


# --- MLflow -------------------------------------------------------------------------------


def log_variant(
    spec: TaskSpec,
    precision: str,
    path: Path,
    how: dict[str, Any],
    metrics: dict[str, float],
    champion: Any,
    *,
    register: bool,
    seed: int,
    client: MlflowClient,
) -> dict[str, Any]:
    """One run per variant; FP16 / INT8 variants become versions of the registered model."""
    name = champion.name
    run_name = f"optimize-{spec.model}-v{champion.version}-{precision}"
    tags = {
        "stage": "optimize",
        "plan_step": "2.4",
        "task": spec.task,
        serving.PRECISION_TAG: precision,
        serving.VARIANT_OF_TAG: str(champion.version),
        "source_run_id": champion.run_id,
        "latency_cpu": cpu_name(),
    }
    params = {
        "precision": precision,
        "source_model": f"{name}/{champion.version}",
        "test_split": spec.test_split,
        **{k: str(v) for k, v in how.items()},
    }
    with mu.start_tracked_run(
        EXPERIMENT,
        run_name,
        seed=seed,
        dataset=spec.dataset,
        dataset_path=spec.dataset_path,
        params=params,
        tags=tags,
    ) as run:
        mlflow.log_metrics(metrics)
        result: dict[str, Any] = {"run_id": run.info.run_id, "version": None}
        if register and precision != q.FP32:
            info = mu.log_model(
                onnx.load(str(path)),
                name=f"{spec.model}-{precision}",
                metrics=metrics,
                params={"precision": precision, "variant_of": str(champion.version)},
            )
            arch = (champion.tags or {}).get("arch", "unknown")
            version = mu.register(
                info.model_uri,
                name,
                tags={
                    "task": spec.task,
                    "arch": arch,
                    "dataset": (champion.tags or {}).get("dataset", spec.dataset),
                    serving.PRECISION_TAG: precision,
                    serving.VARIANT_OF_TAG: str(champion.version),
                    "status": "variant",
                    "git_commit": mu.git_commit(),
                },
                description=(
                    f"{precision} variant of {name} v{champion.version} ({arch}), "
                    f"{spec.metric}={metrics.get(spec.metric, float('nan')):.4f} on "
                    f"{spec.test_split}, {metrics['onnx_cpu_latency_ms']:.1f} ms CPU, "
                    f"run {run.info.run_id}"
                ),
                client=client,
            )
            result["version"] = str(version.version)
    return result


def optimise(
    task: str,
    *,
    register: bool = True,
    settings: dict[str, Any] | None = None,
    data: Path | None = None,
    client: MlflowClient | None = None,
) -> dict[str, Any]:
    spec = SPECS[task]
    settings = settings or load_settings()
    data = data or (DETECT_DATA if task == "detect" else CLASSIFY_DATA)
    client = client or MlflowClient()
    name = serving.REGISTERED[task]
    champion = client.get_model_version_by_alias(name, serving.CHAMPION)
    seed = int(settings.get("seed", 0))
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                f"models:/{name}/{champion.version}", dst_path=str(work / "download")
            )
        )
        fp32 = work / f"{task}-fp32.onnx"
        # One self-contained file (MLflow stores the weights as external data).
        onnx.save_model(onnx.load(str(sorted(downloaded.rglob("*.onnx"))[0])), str(fp32))
        variants = build_variants(task, fp32, work, settings, data)
        rows: dict[str, dict[str, Any]] = {}
        for precision, (path, how) in variants.items():
            metrics = measure(task, path, data, settings)
            print(
                f"{task} {precision}: {spec.metric}={metrics[spec.metric]:.4f} "
                f"{metrics['onnx_cpu_latency_ms']:.1f} ms",
                flush=True,
            )
            rows[precision] = {"metrics": metrics, "how": how, "path": path}
        reference = rows[q.FP32]["metrics"][spec.metric]
        for row in rows.values():
            row["metrics"]["accuracy_drop"] = reference - row["metrics"][spec.metric]
        results = [
            q.VariantResult(
                precision,
                row["metrics"][spec.metric],
                row["metrics"]["onnx_cpu_latency_ms"],
                row["metrics"]["onnx_cpu_latency_p95_ms"],
                row["metrics"]["model_size_mb"],
            )
            for precision, row in rows.items()
        ]
        choice = q.choose_variant(
            results,
            max_drop=float(settings["max_drop"]),
            min_speedup=float(settings["min_speedup"]),
        )
        logged = {
            precision: log_variant(
                spec,
                precision,
                row["path"],
                row["how"],
                row["metrics"],
                champion,
                register=register,
                seed=seed,
                client=client,
            )
            for precision, row in rows.items()
        }
    served = None
    if register:
        chosen = logged[choice.precision]["version"] or str(champion.version)
        serving.set_served(name, chosen, str(champion.version), choice.precision, client)
        client.set_model_version_tag(name, chosen, "served_reason", choice.reason[:5000])
        served = chosen
    return {
        "registered_model": name,
        "champion_version": str(champion.version),
        "champion_run_id": champion.run_id,
        "metric": spec.metric,
        "chosen": choice.precision,
        "reason": choice.reason,
        "served_version": served,
        "variants": {
            precision: {
                **logged[precision],
                **row["how"],
                "metrics": row["metrics"],
            }
            for precision, row in rows.items()
        },
        "results": [asdict(r) for r in results],
    }


def markdown_table(report: dict[str, Any]) -> str:
    lines = [
        "| Modèle | Variante | Précision (métrique) | Écart vs FP32 | Latence moyenne (ms) "
        "| Latence p95 (ms) | Taille (Mo) | Version Registry | Run MLflow |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for entry in report.values():
        for precision, variant in entry["variants"].items():
            m = variant["metrics"]
            version = variant["version"] or entry["champion_version"]
            served = " (servie)" if precision == entry["chosen"] else ""
            lines.append(
                f"| {entry['registered_model']} | {precision}{served} "
                f"| {m[entry['metric']]:.4f} ({entry['metric']}) | {-m['accuracy_drop']:+.4f} "
                f"| {m['onnx_cpu_latency_ms']:.1f} | {m['onnx_cpu_latency_p95_ms']:.1f} "
                f"| {m['model_size_mb']:.1f} | v{version} | `{variant['run_id'][:8]}` |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.optimize")
    parser.add_argument("--task", choices=sorted(SPECS), action="append")
    parser.add_argument("--no-register", action="store_true", help="runs only, no Registry")
    parser.add_argument("--publish", action="store_true", help="then run src.models.publish")
    args = parser.parse_args(argv)
    console_utf8()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mu.configure()
    settings = load_settings()
    report = {
        task: optimise(task, register=not args.no_register, settings=settings)
        for task in args.task or ["detect", "classify"]
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(markdown_table(report))
    if args.publish:
        from src.models import publish

        publish.main([])


if __name__ == "__main__":
    main()
