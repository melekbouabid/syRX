"""Tiny ONNX models with the same inputs, outputs and metadata as the Ultralytics exports.

- Detector: input 1x3xSxS, output 1x5xN, always the boxes given (in letterbox pixels).
- Classifier: input 1x3xCxC, output 1xK probabilities. The class is chosen by the dominant
  colour of the crop: red -> index 0, green -> index 1, blue -> index 2 (the model sees RGB),
  so a test can check that each box was cropped at the right place.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = [helper.make_opsetid("", 17)]


def _save(graph: onnx.GraphProto, path: Path, metadata: dict[str, str]) -> None:
    model = helper.make_model(graph, opset_imports=OPSET)
    model.ir_version = 8
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.checker.check_model(model)
    onnx.save_model(model, str(path))


def write_detector(
    path: Path, boxes: list[tuple[float, float, float, float, float]], size: int = 64
) -> None:
    """`boxes`: (cx, cy, w, h, score) in the size x size letterboxed image."""
    raw = np.array(boxes, dtype=np.float32).T[None]  # 1 x 5 x N
    graph = helper.make_graph(
        [
            # 0 * mean(images) keeps the input connected; the output is the constant.
            helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
            helper.make_node("Mul", ["mean", "zero"], ["nothing"]),
            helper.make_node("Add", ["raw", "nothing"], ["output0"]),
        ],
        "fake-detector",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, list(raw.shape))],
        [
            numpy_helper.from_array(raw, "raw"),
            numpy_helper.from_array(np.array(0, dtype=np.float32), "zero"),
        ],
    )
    meta = {"task": "detect", "names": "{0: 'pill'}", "imgsz": f"[{size}, {size}]"}
    _save(graph, path, meta)


def write_classifier(path: Path, names: list[str], size: int = 16) -> None:
    weights = np.zeros((3, len(names)), dtype=np.float32)
    weights[0, 0] = weights[1, 1] = weights[2, 2] = 20.0
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["images"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["colour"]),
            # Subtracting the grey level makes the dominant channel stand out.
            helper.make_node("ReduceMean", ["colour"], ["grey"], axes=[1], keepdims=1),
            helper.make_node("Sub", ["colour", "grey"], ["tint"]),
            helper.make_node("MatMul", ["tint", "weights"], ["logits"]),
            helper.make_node("Softmax", ["logits"], ["output0"], axis=1),
        ],
        "fake-classifier",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, len(names)])],
        [numpy_helper.from_array(weights, "weights")],
    )
    meta = {
        "task": "classify",
        "names": repr(dict(enumerate(names))),
        "imgsz": f"[{size}, {size}]",
    }
    _save(graph, path, meta)


def write_models(
    directory: Path,
    boxes: list[tuple[float, float, float, float, float]],
    names: list[str],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_detector(directory / "detector.onnx", boxes)
    write_classifier(directory / "classifier.onnx", names)
    manifest = {"detect": {"version": 0, "source": "test"}, "classify": {"version": 0}}
    (directory / "models.json").write_text(json.dumps(manifest), encoding="utf-8")
