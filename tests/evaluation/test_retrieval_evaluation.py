"""Retrieval evaluation helpers: means, the Recall@k curve and where it flattens."""

from __future__ import annotations

from typing import Any

import pytest

from src.evaluation import retrieval_evaluation as ev


def rows() -> list[dict[str, Any]]:
    # q1: the only relevant chunk first; q2: its two relevant chunks at ranks 2 and 3.
    return [
        {"scores": ev.scored(["a", "x", "y"], {"a"}, 3)},
        {"scores": ev.scored(["x", "b", "c"], {"b", "c"}, 3)},
    ]


def test_summary_means_every_metric() -> None:
    s = ev.summary(rows(), (1, 3))
    assert s["hit_at_1"] == 0.5
    assert s["recall_at_3"] == 1.0
    assert s["precision_at_3"] == pytest.approx(round((1 / 3 + 2 / 3) / 2, 4))
    assert s["mrr"] == pytest.approx((1.0 + 0.5) / 2)
    # AP of q2 = (1/2 + 2/3) / 2
    assert s["map"] == pytest.approx(round((1.0 + (1 / 2 + 2 / 3) / 2) / 2, 4))
    assert set(s) >= {"f1_at_1", "ndcg_at_3", "r_precision"}


def test_curve_has_one_value_per_k() -> None:
    c = ev.curve(rows(), 3)
    assert c["recall"] == [0.5, 0.75, 1.0]
    assert c["precision"] == [0.5, 0.5, 0.5]
    assert len(c["ndcg"]) == 3


def test_flattening_k_is_where_two_more_passages_add_little() -> None:
    assert ev.flattening_k([0.4, 0.6, 0.7, 0.705, 0.708, 0.709]) == 3
    assert ev.flattening_k([0.1, 0.2, 0.3, 0.4]) == 4  # never flat: the last k
    assert ev.flattening_k([0.5, 0.5, 0.5]) == 1
