"""Ultralytics detectors (YOLOv8, YOLO11, YOLOv10): training, prediction and ONNX export.

The same three functions exist for every family (`fasterrcnn` on its experiment branch), so
src/models/detection/runner.py drives all of them the same way:

- `train(hp, data_root, work, ...)` -> `TrainResult` (best weights, per-epoch history);
- `predict(weights, images, imgsz)` -> one `ImagePrediction` per image (raw boxes and scores
  at confidence 0.001, scored by src/models/detection/metrics.py);
- `export_onnx(weights, imgsz, target)` -> ONNX file with a static 1x3xSxS input, the input
  the API feeds (backend/app/inference.py).

Ultralytics' own logger integrations (MLflow, W&B, ...) are switched off: the runner writes
its export folder, imported into MLflow afterwards (src/models/detection/import_kaggle.py).
"""

from __future__ import annotations

import csv
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.models.detection.labels import write_data_yaml
from src.models.detection.metrics import ImagePrediction

# Ultralytics integrations that would log elsewhere or ask for a login on Kaggle.
INTEGRATIONS = ("mlflow", "wandb", "comet", "clearml", "neptune", "dvc", "tensorboard", "raytune")
# Hyperparameters that are not Ultralytics training arguments.
NOT_TRAIN_ARGS = {"device"}
PREDICT_CONF = 0.001  # as validation: the whole precision / recall curve is needed
PREDICT_IOU = 0.7
MAX_DET = 300


@dataclass
class TrainResult:
    weights: Path  # best weights (validation fitness)
    run_dir: Path
    history: list[dict[str, float]] = field(default_factory=list)


def device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def quiet_integrations() -> None:
    from ultralytics import settings

    settings.update(dict.fromkeys(INTEGRATIONS, False))


def read_history(results_csv: Path) -> list[dict[str, float]]:
    """Per-epoch metrics of Ultralytics' results.csv (`metrics/mAP50-95(B)` -> `mAP50-95`)."""
    if not results_csv.is_file():
        return []
    rows = []
    with results_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            clean: dict[str, float] = {}
            for key, value in row.items():
                name = key.strip().replace("(B)", "").replace("metrics/", "").replace(":", "_")
                try:
                    clean[name.replace("/", "_")] = float(value)
                except (TypeError, ValueError):
                    continue
            rows.append(clean)
    return rows


def train(
    hp: dict[str, Any],
    data_root: Path,
    work: Path,
    *,
    weights: str,
    epochs: int,
    name: str,
    seed: int,
    fraction: float = 1.0,
    patience: int = 100,
    plots: bool = False,
) -> TrainResult:
    """Fine-tune pretrained `weights` with the hyperparameters `hp` (Ultralytics arguments).

    The dataset folder may be read-only (Kaggle input): data.yaml is written in `work`.
    """
    from ultralytics import YOLO

    quiet_integrations()
    data_yaml = write_data_yaml(data_root, work / "data.yaml")
    args = {k: v for k, v in hp.items() if k not in NOT_TRAIN_ARGS}
    model = YOLO(weights)
    model.train(
        data=str(data_yaml),
        epochs=epochs,
        fraction=fraction,
        patience=patience,
        seed=seed,
        deterministic=True,
        device=hp.get("device", device()),
        project=str(work),
        name=name,
        exist_ok=True,
        plots=plots,
        verbose=False,
        **args,
    )
    run_dir = work / name
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        best = run_dir / "weights" / "last.pt"
    return TrainResult(best, run_dir, read_history(run_dir / "results.csv"))


def _chunks(items: Sequence[Path], size: int) -> Iterator[Sequence[Path]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def predict(
    weights: Path | str, images: Sequence[Path], imgsz: int, *, batch: int = 16
) -> list[ImagePrediction]:
    """Raw predictions (confidence >= 0.001, NMS IoU 0.7) in pixels of each image file."""
    from ultralytics import YOLO

    model = YOLO(str(weights), task="detect")
    out: list[ImagePrediction] = []
    for chunk in _chunks(list(images), batch):
        results = model.predict(
            [str(p) for p in chunk],
            imgsz=imgsz,
            conf=PREDICT_CONF,
            iou=PREDICT_IOU,
            max_det=MAX_DET,
            device=device(),
            verbose=False,
        )
        for result in results:
            boxes = result.boxes
            out.append(
                ImagePrediction(
                    boxes.xyxy.cpu().numpy().astype(np.float32),
                    boxes.conf.cpu().numpy().astype(np.float32),
                )
            )
    return out


def export_onnx(weights: Path, imgsz: int, target: Path, *, simplify: bool = True) -> Path:
    """Static-shape ONNX export (the export runs on a copy: the weights stay untouched)."""
    from ultralytics import YOLO

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "best.pt"
        shutil.copy2(weights, copy)
        produced = YOLO(str(copy)).export(
            format="onnx", imgsz=imgsz, simplify=simplify, dynamic=False, device="cpu"
        )
        shutil.move(str(produced), target)
    return target


def parameter_count(weights: Path) -> int:
    from ultralytics import YOLO

    return int(sum(p.numel() for p in YOLO(str(weights)).model.parameters()))
