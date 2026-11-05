"""Synthetic detection datasets, tiny ONNX detectors and a fake model family for the tests.

Nothing here needs PyTorch or Ultralytics: the fake family "trains" instantly and predicts
the ground-truth boxes shifted by an amount that depends on the learning rate, so an Optuna
search has something to optimise and the whole runner -> export -> MLflow import chain runs
in a few seconds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.models.detection.labels import SPLITS, parse_yolo, read_bgr, yolo_to_xyxy
from src.models.detection.metrics import ImagePrediction
from src.models.detection.yolo_backend import TrainResult

WIDTH, HEIGHT = 96, 64


def make_dataset(root: Path, counts: dict[str, int] | None = None, holdout: int = 2) -> Path:
    """YOLO dataset of grey images, each with one or two bright rectangles (the 'pills')."""
    counts = counts or {"train": 6, "val": 3, "test": 4}
    rng = np.random.default_rng(0)
    test_names = []
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i in range(counts.get(split, 0)):
            image = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
            lines = []
            for _ in range(1 + i % 2):
                x1, y1 = int(rng.integers(0, WIDTH - 30)), int(rng.integers(0, HEIGHT - 20))
                x2, y2 = x1 + int(rng.integers(10, 28)), y1 + int(rng.integers(8, 18))
                cv2.rectangle(image, (x1, y1), (x2, y2), (240, 240, 240), -1)
                cx, cy = (x1 + x2) / 2 / WIDTH, (y1 + y2) / 2 / HEIGHT
                lines.append(
                    f"0 {cx:.6f} {cy:.6f} {(x2 - x1) / WIDTH:.6f} {(y2 - y1) / HEIGHT:.6f}"
                )
            name = f"{split}_{i:03d}"
            cv2.imwrite(str(root / "images" / split / f"{name}.jpg"), image)
            (root / "labels" / split / f"{name}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            if split == "test":
                test_names.append(f"{name}.jpg")
    (root / "test_holdout.txt").write_text("\n".join(test_names[:holdout]) + "\n", "utf-8")
    return root


def tiny_detector(size: int = 32, boxes: int = 4) -> onnx.ModelProto:
    """ONNX graph with the API's detector layout: input 1x3xSxS, output 1x5xN (cx, cy, w,
    h, score), here a constant of empty-score boxes (0 * mean(images) keeps it connected)."""
    raw = np.zeros((1, 5, boxes), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
            helper.make_node("Mul", ["mean", "zero"], ["nothing"]),
            helper.make_node("Add", ["raw", "nothing"], ["output0"]),
        ],
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(raw.shape))],
        [
            numpy_helper.from_array(raw, "raw"),
            numpy_helper.from_array(np.array(0, dtype=np.float32), "zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


def _truth_of(image: Path) -> np.ndarray:
    label = image.parent.parent.parent / "labels" / image.parent.name / f"{image.stem}.txt"
    height, width = read_bgr(image).shape[:2]
    return yolo_to_xyxy(parse_yolo(label.read_text(encoding="utf-8"))[1], width, height)


# --- fake model family (same functions as src/models/detection/yolo_backend.py) -----------


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
    run_dir = work / name
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    best = run_dir / "weights" / "best.pt"
    # The "model" is the shift applied to the true boxes: the closer lr0 is to 0.01, the better.
    shift = abs(np.log10(float(hp["lr0"])) + 2) * 10
    best.write_text(json.dumps({"shift": shift, "weights": weights}), encoding="utf-8")
    (run_dir / "results.csv").write_text("epoch,loss\n1,0.5\n", encoding="utf-8")
    history = [{"epoch": float(e + 1), "loss": 1.0 / (e + 1)} for e in range(epochs)]
    return TrainResult(best, run_dir, history)


def predict(weights: Path, images: list[Path], imgsz: int) -> list[ImagePrediction]:
    shift = json.loads(Path(weights).read_text(encoding="utf-8"))["shift"]
    out = []
    for image in images:
        truth = _truth_of(image)
        out.append(ImagePrediction(truth + shift, np.linspace(0.9, 0.5, len(truth))))
    return out


def export_onnx(weights: Path, imgsz: int, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(tiny_detector(imgsz), str(target))
    return target


def parameter_count(weights: Path) -> int:
    return 1234
