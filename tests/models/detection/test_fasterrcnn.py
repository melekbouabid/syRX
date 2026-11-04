"""Faster R-CNN (src/models/detection/fasterrcnn.py): model, YOLO labels -> torchvision
targets, augmentation, training step, ONNX export with the API's layout.

Needs PyTorch and torchvision (CPU is enough): the CI job `vision` installs them; the job
`src` has no PyTorch and skips this file. No pretrained weights are downloaded.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.models.detection import fasterrcnn as frcnn  # noqa: E402
from src.models.detection.labels import samples  # noqa: E402
from tests.models.detection.helpers import HEIGHT, WIDTH, make_dataset  # noqa: E402

SIZE = 64


@pytest.fixture(scope="module")
def model() -> torch.nn.Module:
    torch.manual_seed(0)
    return frcnn.build_model(SIZE, pretrained=False)


def test_model_has_a_two_class_head(model: torch.nn.Module) -> None:
    predictor = model.roi_heads.box_predictor
    assert predictor.cls_score.out_features == frcnn.NUM_CLASSES
    assert predictor.bbox_pred.out_features == 4 * frcnn.NUM_CLASSES
    assert model.transform.min_size == (SIZE,)
    assert model.transform.max_size == SIZE
    assert model.roi_heads.score_thresh == frcnn.SCORE_THRESH


def test_dataset_converts_yolo_labels_to_letterboxed_targets(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    items = samples(root, "train")
    image, target = frcnn.YoloBoxDataset(items, SIZE)[1]
    assert image.shape == (3, SIZE, SIZE)
    assert image.dtype == torch.float32
    assert float(image.min()) >= 0.0 and float(image.max()) <= 1.0
    assert target["labels"].tolist() == [1] * len(target["labels"])  # pill = 1
    assert len(target["labels"]) == 2  # two boxes in odd images
    # Letterbox of a 96 x 64 image to 64: scale 2/3, 64 x 43, 10 px of padding above.
    ratio, left, top = frcnn.letterbox_geometry(HEIGHT, WIDTH, SIZE)
    assert (round(ratio, 4), left, top) == (0.6667, 0, 10)
    boxes = target["boxes"].numpy()
    assert (boxes[:, [1, 3]] >= top - 1).all() and (boxes[:, [1, 3]] <= SIZE - top + 1).all()
    back = frcnn.from_letterbox(boxes, ratio, left, top, HEIGHT, WIDTH)
    np.testing.assert_allclose(back, items[1].truth(), atol=1.0)


def test_flips_move_the_boxes() -> None:
    image = np.zeros((10, 20, 3), np.uint8)
    boxes = np.array([[2, 1, 6, 4]], dtype=np.float32)
    hp = {"fliplr": 1.0, "flipud": 1.0}
    flipped, moved = frcnn.augment(image, boxes, hp, random.Random(0))
    assert flipped.shape == image.shape
    np.testing.assert_allclose(moved, [[14, 6, 18, 9]])
    _, kept = frcnn.augment(image, boxes, {}, random.Random(0))
    np.testing.assert_allclose(kept, boxes)


def test_hsv_jitter_keeps_the_shape() -> None:
    image = np.full((8, 8, 3), 120, np.uint8)
    out = frcnn.hsv_jitter(image, 0.015, 0.7, 0.4, random.Random(1))
    assert out.shape == image.shape and out.dtype == np.uint8


def test_training_step_returns_the_four_losses(model: torch.nn.Module, tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    images, targets = frcnn.collate(
        [frcnn.YoloBoxDataset(samples(root, "train"), SIZE)[i] for i in range(2)]
    )
    model.train()
    losses = model(images, targets)
    assert set(losses) == {"loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"}
    assert all(torch.isfinite(v) for v in losses.values())


def test_optimizer_and_schedule() -> None:
    net = torch.nn.Linear(2, 2)
    assert isinstance(
        frcnn.make_optimizer(net, {"optimizer": "AdamW", "lr0": 1e-4}), torch.optim.AdamW
    )
    sgd = frcnn.make_optimizer(net, {"optimizer": "SGD", "lr0": 0.02})
    assert isinstance(sgd, torch.optim.SGD)
    assert sgd.param_groups[0]["lr"] == 0.02
    assert frcnn.learning_rate_factor(0, 100, 10) == pytest.approx(0.001)
    assert frcnn.learning_rate_factor(10, 100, 10) == pytest.approx(1.0)
    assert frcnn.learning_rate_factor(100, 100, 10) == pytest.approx(0.01)


def test_train_predict_and_export(tmp_path: Path) -> None:
    """One tiny epoch on CPU, then predictions and the ONNX export the API can read."""
    ort = pytest.importorskip("onnxruntime")
    root = make_dataset(tmp_path / "data", {"train": 2, "val": 1, "test": 1})
    hp = {"imgsz": SIZE, "batch": 2, "lr0": 0.001, "optimizer": "SGD", "workers": 0, "fliplr": 0.5}
    result = frcnn.train(hp, root, tmp_path / "work", weights="none", epochs=1, name="t", seed=0)
    assert result.weights.is_file()
    assert result.history[0]["epoch"] == 1.0
    assert (result.run_dir / "results.csv").is_file()
    predictions = frcnn.predict(result.weights, [samples(root, "test")[0].image], SIZE)
    assert len(predictions) == 1
    assert predictions[0].boxes.shape[1] == 4
    if len(predictions[0].boxes):
        assert predictions[0].boxes[:, [0, 2]].max() <= WIDTH
    onnx_path = frcnn.export_onnx(result.weights, SIZE, tmp_path / "detector.onnx")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 3, SIZE, SIZE]
    output = session.run(None, {"images": np.random.rand(1, 3, SIZE, SIZE).astype("float32")})
    assert output[0].shape[:2] == (1, 5)  # cx, cy, w, h, score: the API's layout
    meta = session.get_modelmeta().custom_metadata_map
    assert meta["task"] == "detect" and meta["imgsz"] == f"[{SIZE}, {SIZE}]"
    assert frcnn.parameter_count(result.weights) > 1_000_000
