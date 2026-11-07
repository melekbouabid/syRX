"""Synthetic crop datasets, tiny ONNX classifiers and a fake model family for the tests.

Nothing here needs PyTorch: the fake family "trains" instantly and predicts the true class
of a crop (its folder) except for a share of errors that depends on the learning rate, so an
Optuna search has something to optimise and the whole runner -> export -> MLflow import chain
runs in a few seconds.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from src.models.classification.backend import TrainResult

# Six classes; classes 0..2 existed in v1 under other ids (10, 11, 12).
CLASSES = 6
V1_IDS = {0: 10, 1: 11, 2: 12}
PER_SPLIT = {"train": 4, "val": 2, "test": 2}
SIZE = 32


def make_dataset(root: Path, per_split: dict[str, int] | None = None) -> Path:
    """Crop dataset of plain-coloured squares (one colour per class), stage layout."""
    per_split = per_split or PER_SPLIT
    rng = np.random.default_rng(0)
    rows = []
    for split, n in per_split.items():
        for label in range(CLASSES):
            colour = np.array([40 * label, 255 - 40 * label, 120], np.float32)
            for i in range(n):
                image = np.clip(colour + rng.normal(0, 8, (SIZE, SIZE, 3)), 0, 255)
                name = f"{split}_{label}_{i}.jpg"
                folder = root / split / f"{label:03d}"
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / name), image.astype(np.uint8))
                holdout = split == "test" and i == 0
                rows.append(
                    {
                        "file": f"{split}/{label:03d}/{name}",
                        "split": split,
                        "class_id": label,
                        "group_id": f"g{label}{i}",
                        "role": "consumer",
                        "source": f"Disc/images/{name}",
                        "box": "0 0 10 10",
                        "holdout": int(holdout),
                    }
                )
    with (root / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (root / "classes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_key", "name", "generic_name", "in_v1", "v1_class_id"])
        for label in range(CLASSES):
            v1 = V1_IDS.get(label)
            writer.writerow(
                [label, f"00000000{label}_1", f"PILL {label}", "X", v1 is not None, v1 or ""]
            )
    holdout = sorted(r["file"] for r in rows if r["holdout"])
    (root / "test_holdout.txt").write_text("\n".join(holdout) + "\n", encoding="utf-8")
    return root


def tiny_classifier(
    size: int = 32, classes: int = CLASSES, metadata: dict[str, str] | None = None
) -> onnx.ModelProto:
    """ONNX graph with the API's classifier layout: input 1x3xSxS, output 1xN probabilities
    (softmax of a linear map of the mean colour)."""
    weights = np.random.default_rng(0).normal(size=(3, classes)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["images"], ["mean"], axes=[2, 3], keepdims=0),
            helper.make_node("MatMul", ["mean", "w"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["output0"], axis=1),
        ],
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, classes])],
        [numpy_helper.from_array(weights, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    if metadata:
        helper.set_model_props(model, metadata)
    return model


# --- fake model family (same functions as src/models/classification/torch_backend.py) -----


def _error_rate(lr: float) -> float:
    """The closer lr is to 1e-3, the fewer errors."""
    return min(0.9, abs(np.log10(lr) + 3) / 4)


def _wrong(path: Path, rate: float) -> bool:
    digest = int(hashlib.sha1(path.name.encode()).hexdigest()[:8], 16)
    return (digest % 1000) / 1000 < rate


def train(
    hp: dict[str, Any],
    data_root: Path,
    work: Path,
    *,
    model: str,
    pretrained: bool,
    epochs: int,
    name: str,
    seed: int,
    fraction: float = 1.0,
    patience: int = 100,
    max_minutes: float | None = None,
    log: Any = print,
) -> TrainResult:
    run_dir = work / name
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    best = run_dir / "weights" / "best.pt"
    best.write_text(
        json.dumps({"rate": _error_rate(float(hp["lr"])), "model": model}), encoding="utf-8"
    )
    (run_dir / "results.csv").write_text("epoch,val_top1\n1,0.5\n", encoding="utf-8")
    history = [{"epoch": float(e + 1), "train_loss": 1.0 / (e + 1)} for e in range(epochs)]
    return TrainResult(best, run_dir, history)


def predict(weights: Path, files: list[Path], imgsz: int) -> np.ndarray:
    rate = json.loads(Path(weights).read_text(encoding="utf-8"))["rate"]
    probs = np.full((len(files), CLASSES), 0.01, np.float32)
    for row, path in enumerate(files):
        label = int(path.parent.name)
        chosen = (label + 1) % CLASSES if _wrong(path, rate) else label
        probs[row, chosen] = 0.9
    return probs / probs.sum(axis=1, keepdims=True)


def export_onnx(
    weights: Path, imgsz: int, target: Path, metadata: dict[str, str] | None = None
) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(tiny_classifier(imgsz, metadata=metadata), str(target))
    return target


def parameter_count(weights: Path) -> int:
    return 4321


def gradcam_pictures(
    weights: Path,
    items: list[tuple[Path, int]],
    imgsz: int,
    out_dir: Path,
    names: dict[int, str] | None = None,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index, (path, _) in enumerate(items):
        target = out_dir / f"gradcam_{index:02d}.png"
        cv2.imwrite(str(target), cv2.imread(str(path)))
        written.append(target)
    return written
