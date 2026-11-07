"""The PyTorch backend of the classifiers (src/models/classification/torch_backend.py):
losses, augmentation of the train split only, a real training of the baseline, ONNX export
and Grad-CAM pictures.

Needs PyTorch and torchvision (CI job `vision`, CPU wheels; the job `src` skips this file).
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
onnxruntime = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from src.models.classification import data as D  # noqa: E402
from src.models.classification import torch_backend as T  # noqa: E402
from tests.models.classification.helpers import make_dataset  # noqa: E402


def test_loss_is_the_cross_entropy_without_options() -> None:
    logits = torch.randn(5, 4)
    labels = torch.tensor([0, 1, 2, 3, 1])
    targets = torch.nn.functional.one_hot(labels, 4).float()
    expected = torch.nn.functional.cross_entropy(logits, labels)
    assert T.soft_target_loss(logits, targets) == pytest.approx(float(expected), rel=1e-5)
    # Focal loss: an easy example (high p) weighs less than in the cross-entropy.
    easy = torch.tensor([[6.0, 0.0, 0.0, 0.0]])
    one = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert T.soft_target_loss(easy, one, gamma=2.0) < 0.01 * T.soft_target_loss(easy, one)
    # Class weights multiply the term of the true class; label smoothing spreads the target.
    weights = torch.tensor([2.0, 1.0, 1.0, 1.0])
    assert T.soft_target_loss(easy, one, weights) == pytest.approx(
        2 * float(T.soft_target_loss(easy, one))
    )
    smoothed = T.smooth(one, 0.2)
    assert smoothed.shape == one.shape
    assert smoothed[0].tolist() == pytest.approx([0.85, 0.05, 0.05, 0.05])


def test_class_weights_favour_rare_classes() -> None:
    weights = T.class_weights([0, 0, 0, 0, 1], 3, power=0.5)
    assert weights[1] > weights[0]
    assert weights[2] == 1.0  # absent from train
    assert weights[:2].mean() == pytest.approx(1.0)


def test_random_view_rotates_and_crops_in_one_warp() -> None:
    image = np.full((256, 256, 3), 200, np.uint8)
    image[:, :128] = 30  # left half dark: a rotation moves it
    views = [T.random_view(image, 64, np.random.default_rng(seed)) for seed in range(12)]
    assert all(v.shape == (64, 64, 3) and v.dtype == np.uint8 for v in views)
    # Same generator, same view; different generators, different views.
    assert np.array_equal(views[0], T.random_view(image, 64, np.random.default_rng(0)))
    assert len({v.tobytes() for v in views}) == len(views)
    # Rotations by large angles leave grey corners (the fill of RandomRotation).
    assert any((v[0, 0] == T.ROTATION_FILL).all() for v in views)
    # A large downscale (256 -> 64 px) keeps the mean colour (INTER_AREA, no aliasing).
    flat = np.full((256, 256, 3), 77, np.uint8)
    view = T.random_view(flat, 64, np.random.default_rng(1))
    assert (np.abs(view[28:36, 28:36].astype(int) - 77) <= 1).all()


def test_augmentation_only_touches_the_train_split(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    train = T.TrainCrops(D.crops(root, "train")[:1], T.train_transform(24, randaugment=9))
    torch.manual_seed(0)
    first = train[0][0]
    second = train[0][0]
    assert first.shape == (3, 24, 24) and first.dtype == torch.float32
    assert not torch.equal(first, second)  # random rotation, crop, RandAugment
    files = [c.path for c in D.crops(root, "val")]
    evaluation = T.EvalCrops(files, 24)
    # Validation and test: exactly the API's preprocessing, the same every time.
    assert torch.equal(evaluation[0], evaluation[0])
    assert np.allclose(evaluation[0].numpy(), D.load_tensor(files[0], 24)[0])


def test_mixup_and_cutmix_give_soft_targets() -> None:
    mix = T.mixer(4, 0.2, 1.0)
    images, targets = mix(torch.rand(6, 3, 16, 16), torch.tensor([0, 1, 2, 3, 0, 1]))
    assert images.shape == (6, 3, 16, 16) and targets.shape == (6, 4)
    assert torch.allclose(targets.sum(dim=1), torch.ones(6))


def test_learning_rate_schedule() -> None:
    constant = T.lr_factor(0)
    assert constant(0) == constant(100) == 1.0
    cosine = T.lr_factor(10)
    assert cosine(0) == pytest.approx(1.0)
    assert cosine(10) == pytest.approx(0.01) and cosine(20) == pytest.approx(0.01)


def test_baseline_trains_predicts_and_exports_like_the_api(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    hp = {
        "lr": 3e-3,
        "imgsz": 32,
        "batch": 8,
        "workers": 0,
        "width": 8,
        "randaugment": 3,
        "mix_prob": 0.5,
        "imbalance": "focal",
        "label_smoothing": 0.1,
    }
    result = T.train(
        hp,
        root,
        tmp_path / "work",
        model="baseline-cnn",
        pretrained=False,
        epochs=2,
        name="final",
        seed=0,
        log=lambda _: None,
    )
    assert result.weights.is_file() and len(result.history) == 2
    assert {"train_loss", "val_top1", "val_top5", "lr"} <= set(result.history[0])
    assert (result.run_dir / "results.csv").is_file()
    info = T.checkpoint_info(result.weights)
    assert info["arch"] == "baseline-cnn" and info["options"] == {"width": 8}

    test = D.crops(root, "test")
    probs = T.predict(result.weights, [c.path for c in test], 32)
    assert probs.shape == (12, 6) and np.allclose(probs.sum(axis=1), 1, atol=1e-5)

    onnx_path = T.export_onnx(
        result.weights, 32, tmp_path / "classifier.onnx", {"names": "{0: '000'}"}
    )
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].shape == [1, 3, 32, 32]
    assert session.get_outputs()[0].shape == [1, 6]
    exported = np.concatenate(
        [session.run(None, {"images": D.load_tensor(c.path, 32)})[0] for c in test]
    )
    assert np.abs(exported - probs).max() < 1e-4  # parity PyTorch / ONNX Runtime
    meta = session.get_modelmeta().custom_metadata_map
    assert ast.literal_eval(meta["imgsz"]) == [32, 32] and meta["task"] == "classify"

    pictures = T.gradcam_pictures(
        result.weights, [(c.path, c.label) for c in test[:2]], 32, tmp_path / "cam"
    )
    assert len(pictures) == 2 and all(p.stat().st_size > 0 for p in pictures)
    assert T.parameter_count(result.weights) > 0
