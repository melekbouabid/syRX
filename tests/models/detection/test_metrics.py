"""COCO-style detection metrics (src/models/detection/metrics.py) on hand-made cases."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.detection.metrics import (
    ImagePrediction,
    average_precision,
    box_iou,
    evaluate,
    prefixed,
)

TRUTH = np.array([[10, 10, 50, 50], [60, 60, 100, 100]], dtype=np.float32)


def pred(boxes: list[list[float]], scores: list[float]) -> ImagePrediction:
    return ImagePrediction(np.asarray(boxes, np.float32).reshape(-1, 4), np.asarray(scores))


def test_box_iou_known_values() -> None:
    a = np.array([[0, 0, 10, 10]])
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]])
    np.testing.assert_allclose(box_iou(a, b), [[1.0, 50 / 150, 0.0]])


def test_perfect_predictions_score_one() -> None:
    metrics = evaluate([pred(TRUTH.tolist(), [0.9, 0.8])], [TRUTH])
    assert metrics["mAP50"] == pytest.approx(1.0)
    assert metrics["mAP50_95"] == pytest.approx(1.0)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["images"] == 1
    assert metrics["instances"] == 2


def test_no_prediction_scores_zero() -> None:
    metrics = evaluate([ImagePrediction.empty()], [TRUTH])
    assert metrics["mAP50"] == 0
    assert metrics["mAP50_95"] == 0
    assert metrics["recall"] == 0


def test_loose_box_counts_only_at_low_iou_thresholds() -> None:
    # IoU 0.64 with the truth: a hit at 0.50, 0.55 and 0.60, a miss from 0.65 on.
    truth = np.array([[0, 0, 100, 100]], dtype=np.float32)
    loose = pred([[0, 0, 100, 64]], [0.9])
    assert box_iou(loose.boxes, truth)[0, 0] == pytest.approx(0.64)
    metrics = evaluate([loose], [truth])
    assert metrics["mAP50"] == pytest.approx(1.0)
    assert metrics["mAP50_95"] == pytest.approx(0.3)


def test_false_positive_ranked_first_lowers_ap() -> None:
    truth = np.array([[0, 0, 10, 10]], dtype=np.float32)
    # A confident false positive, then the true box: precision 1/2 at full recall.
    metrics = evaluate([pred([[50, 50, 60, 60], [0, 0, 10, 10]], [0.9, 0.8])], [truth])
    assert metrics["mAP50"] == pytest.approx(0.5)
    # The best F1 is at the lowest threshold (P = 0.5, R = 1).
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["best_conf"] == pytest.approx(0.8)


def test_duplicate_detection_is_a_false_positive() -> None:
    truth = np.array([[0, 0, 10, 10]], dtype=np.float32)
    metrics = evaluate([pred([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8])], [truth])
    assert metrics["mAP50"] == pytest.approx(1.0)  # the duplicate comes after full recall
    assert metrics["precision"] == pytest.approx(1.0)  # best F1 before the duplicate


def test_detections_are_pooled_across_images() -> None:
    truths = [np.array([[0, 0, 10, 10]]), np.array([[0, 0, 10, 10]])]
    preds = [pred([[0, 0, 10, 10]], [0.9]), pred([[30, 30, 40, 40]], [0.95])]
    metrics = evaluate(preds, truths)
    # Ranked: FP (0.95), TP (0.9): recall 0.5 reached at precision 0.5.
    assert metrics["mAP50"] == pytest.approx(51 * 0.5 / 101)
    assert metrics["recall"] == pytest.approx(0.5)


def test_max_det_keeps_the_most_confident() -> None:
    truth = np.array([[0, 0, 10, 10]], dtype=np.float32)
    boxes = [[50, 50, 60, 60]] * 5 + [[0, 0, 10, 10]]
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    assert evaluate([pred(boxes, scores)], [truth], max_det=5)["mAP50"] == 0
    assert evaluate([pred(boxes, scores)], [truth], max_det=6)["mAP50"] > 0


def test_average_precision_without_truth_is_nan() -> None:
    assert np.isnan(average_precision(np.array([True]), np.array([0.5]), 0))


def test_mismatched_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="predictions"):
        evaluate([ImagePrediction.empty()], [TRUTH, TRUTH])


def test_prefixed_names() -> None:
    assert prefixed({"mAP50": 0.5}, "test_v2") == {"test_v2_mAP50": 0.5}
