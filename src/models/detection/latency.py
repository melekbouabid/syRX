"""ONNX Runtime CPU latency of an exported detector (the metric `onnx_cpu_latency_ms`).

The API serves the models with ONNX Runtime on the CPU (backend/app/inference.py), so the
comparison measures exactly that: one image per call, `CPUExecutionProvider`, after a few
warm-up calls. The input is the real preprocessing of the API when images are given (letterbox
to the model's square input, RGB, [0, 1]), otherwise random noise of the right shape.

Latency depends on the machine: the notebooks log the Kaggle CPU figure
(`kaggle_onnx_cpu_latency_ms`), and the import script measures every model again on the
same local PC (`onnx_cpu_latency_ms`), which is the figure compared. Both are measured in a
fresh Python process (`isolated_latency`): in the process that just ran PyTorch, its idle
worker threads slow ONNX Runtime down by an order of magnitude (552 ms instead of 26 ms for
YOLO11n on the development PC).

`onnx_predict` decodes the exported model exactly as the API does (letterbox, output
1 x (4 + classes) x N in cx cy w h, non-maximum suppression): the runner scores it on a few
test images next to the PyTorch predictions (`parity_*` metrics), so a gap between the
evaluated model and the served one shows up.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

LETTERBOX_FILL = 114  # grey padding, as Ultralytics and backend/app/inference.py


def letterbox(image: np.ndarray, size: int) -> np.ndarray:
    """Resize keeping the aspect ratio, pad to size x size (as backend/app/inference.py)."""
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    if (new_w, new_h) != (width, height):
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    return cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(LETTERBOX_FILL,) * 3
    )


def to_tensor(image: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 -> RGB float32 1x3xHxW in [0, 1]."""
    rgb = image[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(rgb, dtype=np.float32)[None] / 255.0


def input_size(session: ort.InferenceSession, default: int = 640) -> int:
    """Square input side of the model (its static shape, else `default`)."""
    shape = session.get_inputs()[0].shape
    return int(shape[-1]) if isinstance(shape[-1], int) else default


def onnx_cpu_latency(
    model: str | Path,
    *,
    images: Sequence[np.ndarray] = (),
    imgsz: int | None = None,
    warmup: int = 5,
    runs: int = 30,
    threads: int = 0,
    seed: int = 0,
) -> dict[str, float]:
    """Mean, median and 95th percentile latency (ms) of one forward pass on the CPU.

    `images` (BGR) are cycled through; without them the input is random noise. `threads` = 0
    lets ONNX Runtime choose (as the API does by default).
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
    size = imgsz or input_size(session)
    name = session.get_inputs()[0].name
    if images:
        tensors = [to_tensor(letterbox(image, size)) for image in images]
    else:
        rng = np.random.default_rng(seed)
        tensors = [rng.random((1, 3, size, size), dtype=np.float32)]
    for i in range(warmup):
        session.run(None, {name: tensors[i % len(tensors)]})
    timings = []
    for i in range(runs):
        start = time.perf_counter()
        session.run(None, {name: tensors[i % len(tensors)]})
        timings.append((time.perf_counter() - start) * 1000)
    values = np.asarray(timings)
    return {
        "onnx_cpu_latency_ms": float(values.mean()),
        "onnx_cpu_latency_median_ms": float(np.median(values)),
        "onnx_cpu_latency_p95_ms": float(np.percentile(values, 95)),
        "onnx_cpu_runs": float(runs),
        "onnx_input_size": float(size),
    }


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy non-maximum suppression of x1 y1 x2 y2 boxes (as backend/app/inference.py)."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        best, rest = order[0], order[1:]
        keep.append(int(best))
        xx1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[best, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[best, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        overlap = inter / np.maximum(areas[best] + areas[rest] - inter, 1e-9)
        order = rest[overlap <= iou]
    return keep


def onnx_predict(
    model: str | Path,
    images: Sequence[np.ndarray],
    *,
    conf: float = 0.001,
    iou: float = 0.7,
    max_det: int = 300,
) -> list[Any]:
    """Boxes and scores of an exported detector, decoded like the API, in image pixels."""
    from src.models.detection.metrics import ImagePrediction

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    size, name = input_size(session), session.get_inputs()[0].name
    out = []
    for image in images:
        height, width = image.shape[:2]
        ratio = min(size / height, size / width)
        left = round((size - round(width * ratio)) / 2 - 0.1)
        top = round((size - round(height * ratio)) / 2 - 0.1)
        raw = session.run(None, {name: to_tensor(letterbox(image, size))})[0][0].T
        scores = raw[:, 4:].max(axis=1) if raw.shape[0] else np.zeros(0, np.float32)
        mask = scores >= conf
        cx, cy, w, h = raw[mask, :4].T
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        kept = nms(boxes, scores[mask], iou)[:max_det] if mask.any() else []
        boxes = (boxes[kept] - [left, top, left, top]) / ratio
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
        out.append(ImagePrediction(boxes.astype(np.float32), scores[mask][kept].astype(np.float32)))
    return out


def cpu_name() -> str:
    """A short description of the CPU the latency was measured on (MLflow tag)."""
    return platform.processor() or platform.machine() or "unknown"


def isolated_latency(
    model: str | Path, image_files: Sequence[Path] = (), *, runs: int = 30, warmup: int = 5
) -> dict[str, float]:
    """onnx_cpu_latency in a new Python process (no PyTorch threads competing for the CPU)."""
    command = [
        sys.executable,
        "-m",
        "src.models.detection.latency",
        str(model),
        "--runs",
        str(runs),
        "--warmup",
        str(warmup),
        *[str(f) for f in image_files],
    ]
    root = Path(__file__).resolve().parents[3]
    # The current interpreter on this repository's own module, fixed arguments.
    done = subprocess.run(  # noqa: S603
        command, cwd=root, capture_output=True, text=True, check=True, timeout=3600
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.detection.latency")
    parser.add_argument("model", type=Path)
    parser.add_argument("images", nargs="*", type=Path, help="images (default: random noise)")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args(argv)
    images = [cv2.imdecode(np.fromfile(str(f), np.uint8), cv2.IMREAD_COLOR) for f in args.images]
    result = onnx_cpu_latency(args.model, images=images, runs=args.runs, warmup=args.warmup)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
