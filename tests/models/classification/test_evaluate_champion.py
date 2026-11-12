"""The v1 champion (150 classes) scored on the v2 split
(src/models/classification/evaluate_champion.py): mapping of its outputs to the v2 classes,
both label spaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")

from src.models.classification import evaluate_champion  # noqa: E402
from tests.models.classification.helpers import make_dataset, tiny_classifier  # noqa: E402


def test_v1_outputs_are_mapped_to_v2_classes(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    classes = evaluate_champion.D.read_classes(root)
    probs = np.array([[0.2, 0.7, 0.1]])  # outputs: v1 classes 10, 11, 12
    scores = evaluate_champion.to_v2(probs, [10, 11, 12], classes)
    assert scores.shape == (1, 6)
    assert scores[0, :3].tolist() == pytest.approx([0.2, 0.7, 0.1])
    assert scores[0, 3:].tolist() == [evaluate_champion.UNKNOWN] * 3
    with pytest.raises(ValueError, match="without a v2 class"):
        evaluate_champion.to_v2(probs, [10, 11, 99], classes)


def test_a_150_class_model_is_scored_on_both_label_spaces(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    path = tmp_path / "v1.onnx"
    onnx.save(
        tiny_classifier(classes=3, metadata={"names": "{0: '010', 1: '011', 2: '012'}"}), str(path)
    )
    assert evaluate_champion.output_classes(path) == [10, 11, 12]
    scores = evaluate_champion.score_onnx(path, root)
    # Classes 3..5 are unknown to the model: at most half of the comparison crops are right.
    assert scores["test_v2_images"] == 6 and scores["test_v2_top1"] <= 0.5
    assert scores["test_v2_v1c_images"] == 3
    # On its own label space the same predictions count twice as much.
    assert scores["test_v2_v1c_top1"] == pytest.approx(2 * scores["test_v2_top1"])
