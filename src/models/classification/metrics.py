"""Classification metrics computed the same way for every model (and for the v1 champion).

Every classifier of step 2.3 is scored by `evaluate` from its class probabilities (one row
per crop, one column per v2 class), so no library-specific shortcut can favour one model:

- `top1`, `top5`: share of crops whose true class is the first / among the first five;
- `f1_macro`: F1 of every class, averaged with the same weight per class (the classes that
  appear in the truth or in the predictions, as scikit-learn), the figure that is not
  dominated by the classes with many crops;
- `images`, `classes`: number of crops and of distinct true classes scored.

`allowed` restricts the prediction to a subset of classes (the others can never be chosen):
this is how a 380-class model and the 150-class v1 champion are compared on the same label
space (`test_v2_v1c_*`, D100). The confusion matrix and the per-class table are written to the
export folder (`confusion/`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

TOP_K = 5


def ranking(probs: np.ndarray, allowed: Sequence[int] | None = None) -> np.ndarray:
    """Class indices of every row by decreasing probability (only `allowed` when given)."""
    probs = np.asarray(probs, dtype=np.float64)
    if allowed is not None:
        masked = np.full_like(probs, -np.inf)
        index = np.asarray(list(allowed), dtype=np.int64)
        masked[:, index] = probs[:, index]
        probs = masked
    # Stable sort on the negated scores: ties keep the lower class index first.
    return np.argsort(-probs, axis=1, kind="stable")


def confusion_matrix(truth: np.ndarray, predicted: np.ndarray, num_classes: int) -> np.ndarray:
    """counts[t, p] = crops of true class t predicted as p."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (np.asarray(truth, np.int64), np.asarray(predicted, np.int64)), 1)
    return matrix


def per_class(matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Support, precision, recall and F1 of every class (0 where undefined)."""
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {
        "support": support,
        "predicted": predicted,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(
    probs: np.ndarray,
    truth: Sequence[int] | np.ndarray,
    *,
    allowed: Sequence[int] | None = None,
) -> dict[str, float]:
    """top1, top5, f1_macro, images and classes of class probabilities against the truth."""
    truth = np.asarray(truth, dtype=np.int64)
    if len(truth) == 0:
        return {}
    probs = np.asarray(probs)
    if probs.shape[0] != len(truth):
        raise ValueError(f"{probs.shape[0]} predictions for {len(truth)} crops")
    order = ranking(probs, allowed)
    top = order[:, :TOP_K]
    predicted = order[:, 0]
    matrix = confusion_matrix(truth, predicted, probs.shape[1])
    scores = per_class(matrix)
    present = (scores["support"] > 0) | (scores["predicted"] > 0)
    return {
        "top1": float(np.mean(predicted == truth)),
        "top5": float(np.mean((top == truth[:, None]).any(axis=1))),
        "f1_macro": float(scores["f1"][present].mean()) if present.any() else 0.0,
        "images": float(len(truth)),
        "classes": float(len(np.unique(truth))),
    }


def top_confusions(matrix: np.ndarray, n: int = 20) -> list[tuple[int, int, int]]:
    """The `n` most frequent errors as (true class, predicted class, count)."""
    errors = matrix.copy()
    np.fill_diagonal(errors, 0)
    flat = np.argsort(-errors, axis=None, kind="stable")[:n]
    out = []
    for index in flat:
        t, p = np.unravel_index(index, errors.shape)
        if errors[t, p] == 0:
            break
        out.append((int(t), int(p), int(errors[t, p])))
    return out
