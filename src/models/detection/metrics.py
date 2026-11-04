"""Detection metrics computed the same way for every model (COCO protocol, one class).

Every detector of step 2.2 (Ultralytics or torchvision) and the baseline are scored by
`evaluate`, from their raw predictions (boxes and scores, confidence threshold ~0), so no
library-specific shortcut can favour one model:

- AP at an IoU threshold: detections of all images sorted by score, each one matched greedily
  to the unmatched ground-truth box of highest IoU (at least the threshold), then the
  101-point interpolated area under the precision / recall curve (COCO);
- `mAP50` = AP at IoU 0.50, `mAP50_95` = mean AP over IoU 0.50, 0.55, ..., 0.95 (the metric
  of the module guide and of the Model Registry);
- `precision`, `recall`: at the confidence that maximises F1 at IoU 0.50 (as Ultralytics
  reports them), with that confidence as `best_conf`.

At most `max_det` detections per image are kept (100, the COCO default).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass(frozen=True)
class ImagePrediction:
    """Predicted boxes (N x 4, x1 y1 x2 y2 pixels) and their scores (N)."""

    boxes: np.ndarray
    scores: np.ndarray

    @classmethod
    def empty(cls) -> ImagePrediction:
        return cls(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (len(a) x len(b)) of x1 y1 x2 y2 boxes."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _match(pred: ImagePrediction, truth: np.ndarray, threshold: float) -> np.ndarray:
    """True positive flag of each prediction (in the given order) at one IoU threshold."""
    tp = np.zeros(len(pred.scores), dtype=bool)
    if not len(truth) or not len(pred.scores):
        return tp
    ious = box_iou(pred.boxes, truth)
    taken = np.zeros(len(truth), dtype=bool)
    for i in np.argsort(-pred.scores, kind="stable"):
        candidates = np.where(~taken, ious[i], -1.0)
        best = int(candidates.argmax())
        if candidates[best] >= threshold:
            taken[best] = True
            tp[i] = True
    return tp


def average_precision(tp: np.ndarray, scores: np.ndarray, n_truth: int) -> float:
    """101-point interpolated AP (COCO) from the TP flags and scores of all detections."""
    if n_truth == 0:
        return float("nan")
    if not len(scores):
        return 0.0
    order = np.argsort(-scores, kind="stable")
    hits = tp[order].astype(np.float64)
    tps, fps = np.cumsum(hits), np.cumsum(1.0 - hits)
    recall = tps / n_truth
    precision = tps / np.maximum(tps + fps, 1e-12)
    # Precision envelope: best precision at any recall at least as high.
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    index = np.searchsorted(recall, RECALL_POINTS, side="left")
    sampled = np.where(index < len(envelope), envelope[np.minimum(index, len(envelope) - 1)], 0)
    return float(sampled.mean())


def _cap(pred: ImagePrediction, max_det: int) -> ImagePrediction:
    if len(pred.scores) <= max_det:
        return pred
    keep = np.argsort(-pred.scores, kind="stable")[:max_det]
    return ImagePrediction(pred.boxes[keep], pred.scores[keep])


def evaluate(
    predictions: Sequence[ImagePrediction],
    truths: Sequence[np.ndarray],
    *,
    max_det: int = 100,
    thresholds: Sequence[float] = IOU_THRESHOLDS,
) -> dict[str, float]:
    """mAP50_95, mAP50, precision, recall (and counts) over a set of images."""
    if len(predictions) != len(truths):
        raise ValueError(f"{len(predictions)} predictions for {len(truths)} images")
    capped = [_cap(p, max_det) for p in predictions]
    truths = [np.asarray(t, dtype=np.float64).reshape(-1, 4) for t in truths]
    n_truth = sum(len(t) for t in truths)
    scores = (
        np.concatenate([p.scores for p in capped]).astype(np.float64) if capped else np.zeros(0)
    )
    aps = []
    tp50 = np.zeros(len(scores), dtype=bool)
    for threshold in thresholds:
        tp = (
            np.concatenate([_match(p, t, threshold) for p, t in zip(capped, truths, strict=True)])
            if capped
            else np.zeros(0, dtype=bool)
        )
        aps.append(average_precision(tp, scores, n_truth))
        if threshold == thresholds[0]:
            tp50 = tp
    precision, recall, best_conf = _best_f1(tp50, scores, n_truth)
    return {
        "mAP50_95": float(np.mean(aps)),
        "mAP50": float(aps[0]),
        "precision": precision,
        "recall": recall,
        "best_conf": best_conf,
        "images": float(len(truths)),
        "instances": float(n_truth),
    }


def _best_f1(tp: np.ndarray, scores: np.ndarray, n_truth: int) -> tuple[float, float, float]:
    """Precision and recall at the score threshold with the best F1 (IoU 0.50)."""
    if not len(scores) or n_truth == 0:
        return 0.0, 0.0, 0.0
    order = np.argsort(-scores, kind="stable")
    hits = tp[order].astype(np.float64)
    tps, fps = np.cumsum(hits), np.cumsum(1.0 - hits)
    precision = tps / np.maximum(tps + fps, 1e-12)
    recall = tps / n_truth
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(f1.argmax())
    return float(precision[best]), float(recall[best]), float(scores[order][best])


def prefixed(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    """`{"mAP50": x}` -> `{"<prefix>_mAP50": x}` (MLflow metric names)."""
    return {f"{prefix}_{key}": value for key, value in metrics.items()}
