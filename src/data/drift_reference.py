"""Reference statistics of the data drift monitor (step 2.8, docs/module1/derive.md).

    # training venv (.venv-train), served models in models/, MLflow server running
    PYTHONPATH="backend;." python -m src.data.drift_reference
    # or through DVC: dvc repro drift_reference

Takes `drift.reference_images` photos spread over the `drift.reference_split` split of the
detection dataset of C3PI v2 (consumer photos, the kind of photo a user takes), runs the served
scan on each (app/inference.py: the models the API serves), and computes the statistics of
app/drift.py `scan_stats`. The split is `val`, not `train` (D122): on the training photos the
models are more confident than on any new photo (classifier 0.93 against 0.81 on the test
split), and the split job leaves the largest shooting sessions (12-megapixel cameras) in train,
so the test photos themselves drifted from a train reference; val holds photos never trained
on, chosen like the test ones. Writes data/processed/c3pi/drift/reference.json
(per-statistic sample and the thresholds of params.yaml `drift`, read by the API) and
summary.json (the DVC metrics), and logs one MLflow run in the experiment `tujya-monitoring`
(thresholds as params, mean / quantiles as metrics, the files and a histogram per statistic
as artifacts, provenance tags of src/utils/mlflow_utils.py).

The reference depends on the served models (the two confidences): publishing a new model
makes `dvc status` show this stage, and it is run again.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:  # the training venv has no editable root package
    sys.path.insert(0, str(ROOT / "backend"))

from app.drift import FEATURES, Reference, scan_stats  # noqa: E402
from app.inference import PillScanner  # noqa: E402

PARAMS = ROOT / "params.yaml"
INDEX = ROOT / "data" / "processed" / "c3pi" / "datasets_v2" / "detect" / "index.csv"
IMAGES = ROOT / "data" / "raw" / "c3pi" / "images"
MODELS = ROOT / "models"
OUT = ROOT / "data" / "processed" / "c3pi" / "drift"
EXPERIMENT = "tujya-monitoring"
THRESHOLD_KEYS = ("window", "min_samples", "ks_alpha", "psi_warning", "psi_drift", "bins")


def load_params(path: Path = PARAMS) -> dict[str, Any]:
    params = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return dict(params.get("drift") or {})


def raw_photo(index_path: str, images: Path = IMAGES) -> Path:
    """`PillProjectDisc1/images/X.JPG` (index of the dataset) -> images/PillProjectDisc1/X.JPG."""
    parts = index_path.split("/")
    return images / parts[0] / parts[-1]


def photos(split: str, n: int, *, index: Path = INDEX, images: Path = IMAGES) -> list[Path]:
    """`n` photos of a split of the detection dataset v2, evenly spread over the sorted list."""
    with index.open(newline="", encoding="utf-8") as handle:
        chosen = sorted(r["path"] for r in csv.DictReader(handle) if r["split"] == split)
    if 0 < n < len(chosen):
        step = len(chosen) / n
        chosen = [chosen[int(i * step)] for i in range(n)]
    return [raw_photo(p, images) for p in chosen]


def read_bgr(path: Path) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode {path}")
    return image


def photo_stats(scanner: PillScanner, image: np.ndarray) -> dict[str, float]:
    """The statistics of a photo with the confidences of the served scan (as the API does)."""
    return scan_stats(image, scanner.scan(image))


def build(files: list[Path], scanner: PillScanner, params: dict[str, Any]) -> Reference:
    rows = [photo_stats(scanner, read_bgr(f)) for f in files]
    values = {name: [r[name] for r in rows if name in r] for name in FEATURES}
    manifest = json.loads((MODELS / "models.json").read_text(encoding="utf-8"))
    split = str(params.get("reference_split", "val"))
    meta = {
        "dataset": f"C3PI v2, {split} split of the detection dataset (consumer photos)",
        "split": split,
        "photos": len(rows),
        "models": {
            task: {k: (manifest.get(task) or {}).get(k) for k in ("arch", "version", "precision")}
            for task in ("detect", "classify")
        },
    }
    thresholds = {k: float(params[k]) for k in THRESHOLD_KEYS if k in params}
    return Reference(values, thresholds, meta)


def summary(reference: Reference) -> dict[str, Any]:
    out: dict[str, Any] = {"photos": reference.meta.get("photos", 0)}
    for name, values in reference.values.items():
        array = np.asarray(values, float)
        if array.size:
            out[name] = {
                "n": int(array.size),
                "mean": float(array.mean()),
                "p05": float(np.quantile(array, 0.05)),
                "p50": float(np.quantile(array, 0.5)),
                "p95": float(np.quantile(array, 0.95)),
            }
    return out


def histograms(reference: Reference, path: Path) -> Path | None:
    try:
        import matplotlib
    except ImportError:  # project venv: the histograms are optional, the numbers are logged
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=(13, 5.5), dpi=120)
    for ax, name in zip(axes.flat, FEATURES, strict=True):
        ax.hist(reference.values.get(name, []), bins=30, color="#2a78d6")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    split = reference.meta.get("split", "val")
    fig.suptitle(f"Référence de la dérive : photos {split} de C3PI v2", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def log_mlflow(reference: Reference, out: Path, params: dict[str, Any]) -> str:
    import mlflow

    from src.utils import mlflow_utils as mu

    mu.configure()
    stats = summary(reference)
    with mu.start_tracked_run(
        EXPERIMENT,
        "drift-reference",
        seed=int(params.get("seed", 0)),
        dataset=f"c3pi-v2-detect-{reference.meta.get('split', 'val')}",
        dataset_path="data/processed/c3pi/datasets_v2/detect",
        params={**{f"drift_{k}": v for k, v in params.items()}},
        tags={"stage": "drift-reference", "plan_step": "2.8"},
    ) as run:
        mlflow.log_metric("reference_photos", float(stats["photos"]))
        for name in FEATURES:
            for key in ("mean", "p05", "p50", "p95"):
                if name in stats:
                    mlflow.log_metric(f"{name}_{key}", stats[name][key])
        mlflow.log_artifact(str(out / "reference.json"))
        mlflow.log_artifact(str(out / "summary.json"))
        with tempfile.TemporaryDirectory() as tmp:
            figure = histograms(reference, Path(tmp) / "reference_histograms.png")
            if figure:
                mlflow.log_artifact(str(figure))
        return str(run.info.run_id)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.data.drift_reference")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    params = load_params()
    files = photos(
        str(params.get("reference_split", "val")), int(params.get("reference_images", 600))
    )
    scanner = PillScanner.from_dir(MODELS)
    reference = build(files, scanner, params)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "reference.json").write_text(
        json.dumps(reference.to_json(), indent=1) + "\n", encoding="utf-8"
    )
    (args.out / "summary.json").write_text(
        json.dumps(summary(reference), indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary(reference), indent=2))
    if not args.no_mlflow:
        print("MLflow run", log_mlflow(reference, args.out, params))


if __name__ == "__main__":
    main()
