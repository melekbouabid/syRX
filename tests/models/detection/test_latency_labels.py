"""ONNX CPU latency and YOLO label conversions (src/models/detection latency.py, labels.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

from src.models.detection.labels import (
    holdout_names,
    make_subset,
    parse_yolo,
    samples,
    to_torchvision,
    write_data_yaml,
    xyxy_to_yolo,
    yolo_to_xyxy,
)
from src.models.detection.latency import letterbox, onnx_cpu_latency, onnx_predict, to_tensor
from tests.models.detection.helpers import make_dataset, tiny_detector


def test_latency_of_a_tiny_model(tmp_path: Path) -> None:
    path = tmp_path / "tiny.onnx"
    onnx.save(tiny_detector(32), str(path))
    noise = onnx_cpu_latency(path, warmup=1, runs=5)
    assert noise["onnx_input_size"] == 32
    assert noise["onnx_cpu_runs"] == 5
    assert 0 < noise["onnx_cpu_latency_median_ms"] <= noise["onnx_cpu_latency_p95_ms"]
    # Real images go through the API's preprocessing (letterbox to the model input).
    image = np.zeros((48, 80, 3), np.uint8)
    timed = onnx_cpu_latency(path, images=[image, image], warmup=1, runs=3)
    assert timed["onnx_cpu_latency_ms"] > 0


def test_letterbox_and_tensor_shape() -> None:
    image = np.full((48, 80, 3), 255, np.uint8)
    padded = letterbox(image, 64)
    assert padded.shape == (64, 64, 3)
    assert padded[0, 0, 0] == 114  # grey padding above the resized image
    tensor = to_tensor(padded)
    assert tensor.shape == (1, 3, 64, 64)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0


def test_yolo_boxes_round_trip() -> None:
    classes, boxes = parse_yolo("0 0.5 0.5 0.2 0.4\n\n0 0.25 0.75 0.1 0.1\n")
    assert classes.tolist() == [0, 0]
    xyxy = yolo_to_xyxy(boxes, 200, 100)
    np.testing.assert_allclose(xyxy[0], [80, 30, 120, 70])
    np.testing.assert_allclose(xyxy_to_yolo(xyxy, 200, 100), boxes, atol=1e-6)


def test_parse_rejects_malformed_lines() -> None:
    with pytest.raises(ValueError, match="5 values"):
        parse_yolo("0 0.5 0.5 0.2\n")


def test_torchvision_targets_shift_labels_and_drop_degenerate_boxes() -> None:
    text = "0 0.5 0.5 0.5 0.5\n2 0.1 0.1 0.2 0.2\n0 1.2 0.5 0.1 0.1\n"
    boxes, labels = to_torchvision(text, 100, 80)
    # Third box lies outside the image: zero width after clipping, dropped.
    assert boxes.shape == (2, 4)
    assert boxes.dtype == np.float32
    np.testing.assert_allclose(boxes[0], [25, 20, 75, 60])
    np.testing.assert_allclose(boxes[1], [0, 0, 20, 16])
    assert labels.tolist() == [1, 3]  # class + 1: 0 is the background
    assert labels.dtype == np.int64
    empty_boxes, empty_labels = to_torchvision("", 10, 10)
    assert empty_boxes.shape == (0, 4)
    assert empty_labels.shape == (0,)


def test_samples_truth_and_holdout(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    test = samples(root, "test")
    assert len(test) == 4
    assert test[0].truth().shape[1] == 4
    assert holdout_names(root) == {"test_000.jpg", "test_001.jpg"}
    assert len(samples(root, "test", holdout_names(root))) == 2
    assert holdout_names(tmp_path) is None


def test_make_subset_copies_and_keeps_the_holdout(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    subset = make_subset(root, tmp_path / "subset", {"train": 2, "val": 1, "test": 4})
    assert len(samples(subset, "train")) == 2
    assert len(samples(subset, "val")) == 1
    assert holdout_names(subset) == holdout_names(root)
    assert "path: " + subset.resolve().as_posix() in (subset / "data.yaml").read_text("utf-8")
    yaml_file = write_data_yaml(root, tmp_path / "work" / "data.yaml")
    assert yaml_file.parent.name == "work"


def test_onnx_predict_decodes_like_the_api(tmp_path: Path) -> None:
    """Boxes given in the letterboxed input come back in image pixels, after NMS."""
    from onnx import TensorProto, helper, numpy_helper

    # Two overlapping boxes and a separate one in a 64 x 64 input (cx, cy, w, h, score).
    raw = np.array(
        [[[20, 20.5, 50], [20, 20.5, 40], [10, 10, 8], [10, 10, 8], [0.9, 0.8, 0.7]]],
        dtype=np.float32,
    )
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
            helper.make_node("Mul", ["mean", "zero"], ["nothing"]),
            helper.make_node("Add", ["raw", "nothing"], ["output0"]),
        ],
        "boxes",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 5, 3])],
        [
            numpy_helper.from_array(raw, "raw"),
            numpy_helper.from_array(np.array(0, dtype=np.float32), "zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path = tmp_path / "boxes.onnx"
    onnx.save(model, str(path))
    # A 128 x 64 image: scale 0.5, 16 px of padding above.
    (pred,) = onnx_predict(path, [np.zeros((64, 128, 3), np.uint8)])
    assert pred.scores.tolist() == pytest.approx([0.9, 0.7])  # the 0.8 overlap is suppressed
    np.testing.assert_allclose(pred.boxes[0], [30, 0, 50, 18])  # y clipped at the border
    np.testing.assert_allclose(pred.boxes[1], [92, 40, 108, 56])
