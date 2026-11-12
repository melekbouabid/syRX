"""Score the current champion classifier (150 classes of C3PI v1) on the split of step 2.3.

    # training venv, MLflow server running, dataset of stage prepare_classify_v2 on disk
    python -m src.models.classification.evaluate_champion            # alias champion
    python -m src.models.classification.evaluate_champion --version 1

The champion (YOLO11s-cls, step 2.1, trained on the 150 classes of v1) was scored on the v1
test crops; the candidates of step 2.3 are trained on the 380 classes of v2 and scored on the
v2 comparison split (`test_holdout.txt`: test sessions the champion never saw, D99). To
compare like with like (D100), this script scores the registered ONNX model, the file the API
serves, with the preprocessing of the API and the metrics of the candidates
(src/models/classification/metrics.py):

1. its outputs (v1 class folders, ONNX metadata `names`) are mapped to v2 class ids through
   the C3PI class key (classes.csv `v1_class_id`); the 230 classes it does not know get no
   probability, so their crops count as errors in `test_v2_*` and `test_v2_all_*`;
2. `test_v2_v1c_*`: the comparison crops of the 150 v1 classes, the prediction restricted to
   these classes: the same label space as the champion (the guard of the promotion);
3. the metrics and the CPU latency are logged in a run `evaluate-classifier-v<n>-c3pi-v2`
   (tag `stage=compare-eval`); the scores are also added to the version's own run, where
   promote_if_better reads them.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import onnx
import onnxruntime as ort
from mlflow import MlflowClient

from src.models.classification import data as D
from src.models.classification.import_kaggle import (
    EXPERIMENT,
    LOCAL_DATA,
    compare_rules,
    local_latency,
)
from src.models.classification.runner import split_scores
from src.models.detection.latency import cpu_name
from src.utils import mlflow_utils as mu
from src.utils.console import console_utf8

UNKNOWN = -1.0  # score of the classes the model does not know (never ranked before a known one)


def output_classes(model: Path) -> list[int]:
    """Class folder (v1 class id) of every output, from the ONNX metadata `names`."""
    meta = {p.key: p.value for p in onnx.load(str(model), load_external_data=False).metadata_props}
    names = ast.literal_eval(meta.get("names", "{}"))
    return [int(names[i]) for i in sorted(names)]


def to_v2(probs: np.ndarray, outputs: list[int], classes: list[dict[str, Any]]) -> np.ndarray:
    """Probabilities over the v1 outputs -> scores over the v2 classes (UNKNOWN elsewhere)."""
    by_v1 = {r["v1_class_id"]: r["class_id"] for r in classes if r["v1_class_id"] is not None}
    missing = [c for c in outputs if c not in by_v1]
    if missing:
        raise ValueError(f"v1 classes without a v2 class: {missing[:5]}")
    out = np.full((probs.shape[0], len(classes)), UNKNOWN, dtype=np.float32)
    out[:, [by_v1[c] for c in outputs]] = probs
    return out


def score_onnx(model: Path, data_dir: Path) -> dict[str, float]:
    """test_v2_*, test_v2_all_* and test_v2_v1c_* of an ONNX classifier of another class list."""
    classes = D.read_classes(data_dir)
    test = D.crops(data_dir, "test")
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    name, size = session.get_inputs()[0].name, int(session.get_inputs()[0].shape[-1])
    probs = np.concatenate(
        [session.run(None, {name: D.load_tensor(c.path, size)})[0] for c in test]
    )
    scores = to_v2(probs, output_classes(model), classes)
    return split_scores(scores, test, D.v1_classes(classes))


def evaluate_version(
    version: str | None = None,
    *,
    alias: str = mu.CHAMPION,
    data_dir: Path = LOCAL_DATA,
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
    arch = (mv.tags or {}).get("arch") or run.data.params.get("arch") or "unknown"
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(
            mlflow.artifacts.download_artifacts(f"models:/{name}/{version}", dst_path=tmp)
        )
        model = Path(tmp) / "classifier.onnx"
        onnx.save_model(onnx.load(str(sorted(folder.rglob("*.onnx"))[0])), str(model))
        metrics = score_onnx(model, data_dir)
        metrics.update(local_latency(model, data_dir))
        imgsz = int(ort.InferenceSession(str(model)).get_inputs()[0].shape[-1])

    dataset = D.dataset_info(data_dir)
    tags = {
        "stage": "compare-eval",
        "task": "classify",
        "arch": arch,
        "evaluated_model": name,
        "evaluated_version": str(version),
        "evaluated_run_id": mv.run_id,
        "dataset": D.DATASET_NAME,
        "label_space": (mv.tags or {}).get("dataset", "c3pi-v1"),
        "plan_step": "2.3",
        "smoke": "false",
        "scored_as": "onnx (registered file, API preprocessing)",
    }
    if dataset.get("md5"):
        tags["dataset_md5"] = str(dataset["md5"])
    with mu.start_tracked_run(
        EXPERIMENT,
        f"evaluate-classifier-v{version}-c3pi-v2",
        seed=int(run.data.params.get("seed", run.data.tags.get("seed", 0))),
        dataset=D.DATASET_NAME,
        dataset_path=D.DATASET_PATH,
        params={"imgsz": imgsz, "arch": arch, "evaluated_version": version},
        tags=tags,
    ) as eval_run:
        mlflow.log_metrics(metrics)
        mlflow.set_tag("latency_cpu", cpu_name())
    # The version's own run receives the scores too: promote_if_better reads them there. Its
    # latency stays the one of the interleaved protocol (step 2.4, then compare_latency
    # --task classify, D96): the isolated timing above goes to the evaluation run only.
    with mlflow.start_run(run_id=mv.run_id):
        mlflow.log_metrics({k: v for k, v in metrics.items() if k.startswith("test_v2")})
        mlflow.set_tag("test_v2_eval_run_id", eval_run.info.run_id)
    for key in (rules["metric"], rules["guard_metric"]):
        if key in metrics:
            client.set_model_version_tag(name, str(version), key, f"{metrics[key]:.4f}")
    return {
        "model": name,
        "version": str(version),
        "arch": arch,
        "run_id": mv.run_id,
        "eval_run_id": eval_run.info.run_id,
        **metrics,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.classification.evaluate_champion")
    parser.add_argument("--version", help="registered version (default: the alias)")
    parser.add_argument("--alias", default=mu.CHAMPION)
    parser.add_argument("--data", type=Path, default=LOCAL_DATA)
    parser.add_argument("--dry-run", action="store_true", help="score only, log nothing")
    args = parser.parse_args(argv)
    console_utf8()
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mu.configure()
    if args.dry_run:
        client = MlflowClient()
        name = compare_rules()["name"]
        version = args.version or mu.alias_version(name, args.alias, client)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(
                mlflow.artifacts.download_artifacts(f"models:/{name}/{version}", dst_path=tmp)
            )
            print(json.dumps(score_onnx(sorted(folder.rglob("*.onnx"))[0], args.data), indent=2))
        return
    print(
        json.dumps(evaluate_version(args.version, alias=args.alias, data_dir=args.data), indent=2)
    )


if __name__ == "__main__":
    main()
