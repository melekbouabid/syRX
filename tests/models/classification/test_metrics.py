"""Top-1, top-5, macro F1 and confusion matrix (src/models/classification/metrics.py)."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.classification.metrics import (
    confusion_matrix,
    evaluate,
    per_class,
    ranking,
    top_confusions,
)


def test_top1_top5_and_macro_f1_by_hand() -> None:
    probs = np.array(
        [
            [0.7, 0.1, 0.1, 0.05, 0.05, 0.0],  # truth 0: right
            [0.1, 0.6, 0.1, 0.1, 0.05, 0.05],  # truth 1: right
            [0.5, 0.3, 0.1, 0.05, 0.03, 0.02],  # truth 2: predicted 0, true class 3rd
            [0.4, 0.2, 0.15, 0.1, 0.09, 0.06],  # truth 5: last, out of the top 5
        ]
    )
    scores = evaluate(probs, [0, 1, 2, 5])
    assert scores["top1"] == pytest.approx(0.5)
    assert scores["top5"] == pytest.approx(0.75)
    # Per class (present in truth or prediction): 0 -> P 1/3 R 1 F1 0.5; 1 -> 1; 2 -> 0; 5 -> 0.
    assert scores["f1_macro"] == pytest.approx((0.5 + 1 + 0 + 0) / 4)
    assert scores["images"] == 4 and scores["classes"] == 4


def test_restricted_label_space() -> None:
    probs = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
    # Class 0 cannot be predicted: the first crop's prediction becomes class 1.
    assert ranking(probs, allowed=[1, 2])[:, 0].tolist() == [1, 1]
    scores = evaluate(probs, [1, 2], allowed=[1, 2])
    assert scores["top1"] == pytest.approx(0.5)
    assert evaluate(np.zeros((0, 3)), []) == {}
    with pytest.raises(ValueError, match="predictions"):
        evaluate(probs, [0])


def test_confusion_matrix_and_most_frequent_errors() -> None:
    matrix = confusion_matrix(np.array([0, 0, 1, 1, 1, 2]), np.array([0, 1, 2, 2, 1, 2]), 3)
    assert matrix.tolist() == [[1, 1, 0], [0, 1, 2], [0, 0, 1]]
    scores = per_class(matrix)
    assert scores["recall"].tolist() == pytest.approx([0.5, 1 / 3, 1.0])
    assert top_confusions(matrix, 5) == [(1, 2, 2), (0, 1, 1)]
