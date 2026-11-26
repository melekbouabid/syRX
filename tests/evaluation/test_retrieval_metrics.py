"""Retrieval metrics on the example of the course: ranking [d3, d1, d7, d2, d9], relevant
{d1, d2, d5}, graded gains d1 = 3, d2 = 2, d5 = 1."""

from __future__ import annotations

import math

import pytest

from src.evaluation import retrieval_metrics as m

RANKED = ["d3", "d1", "d7", "d2", "d9"]
RELEVANT = {"d1", "d2", "d5"}
GAINS = {"d1": 3.0, "d2": 2.0, "d5": 1.0}


def test_set_metrics_of_the_course_example() -> None:
    assert m.hit_at_k(RANKED, RELEVANT, 5) == 1.0
    assert m.hit_at_k(RANKED, RELEVANT, 1) == 0.0  # d3 is not relevant
    assert m.precision_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.40)
    assert round(m.recall_at_k(RANKED, RELEVANT, 5), 2) == 0.67
    # F1@5 = 2 * 0.4 * (2/3) / (0.4 + 2/3) = 0.5
    assert m.f1_at_k(RANKED, RELEVANT, 5) == pytest.approx(0.5)


def test_rank_metrics_of_the_course_example() -> None:
    # R-Precision = P@|R| = P@3: only d1 in [d3, d1, d7].
    assert round(m.r_precision(RANKED, RELEVANT), 2) == 0.33
    assert m.reciprocal_rank(RANKED, RELEVANT) == pytest.approx(0.50)
    # AP: relevant at ranks 2 and 4, P@2 = 1/2, P@4 = 2/4; d5 never retrieved counts 0.
    by_hand_ap = (1 / 2 + 2 / 4) / 3
    assert m.average_precision(RANKED, RELEVANT) == pytest.approx(by_hand_ap)
    assert by_hand_ap == pytest.approx(1 / 3)


def test_graded_ndcg_of_the_course_example() -> None:
    # DCG@5: d1 (gain 3) at rank 2, d2 (gain 2) at rank 4.
    dcg = 3 / math.log2(3) + 2 / math.log2(5)
    # Ideal order d1, d2, d5 at ranks 1, 2, 3.
    idcg = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
    assert m.ndcg_at_k(RANKED, GAINS, 5) == pytest.approx(dcg / idcg)
    assert round(dcg / idcg, 4) == 0.5784


def test_binary_ndcg_gives_every_relevant_document_gain_one() -> None:
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert m.ndcg_at_k(RANKED, RELEVANT, 5) == pytest.approx(dcg / idcg)


def test_edge_cases() -> None:
    assert m.recall_at_k(RANKED, set(), 5) == 0.0
    assert m.average_precision(RANKED, set()) == 0.0
    assert m.reciprocal_rank(["x", "y"], RELEVANT) == 0.0
    assert m.ndcg_at_k([], RELEVANT, 5) == 0.0
    assert m.f1_at_k(["x"], RELEVANT, 1) == 0.0
    with pytest.raises(ValueError):
        m.precision_at_k(RANKED, RELEVANT, 0)


def test_score_and_means() -> None:
    row = m.score(RANKED, RELEVANT, ks=(1, 5), gains=GAINS)
    assert row["precision_at_5"] == pytest.approx(0.4)
    assert row["hit_at_1"] == 0.0
    assert row["rr"] == pytest.approx(0.5)
    other = m.score(["d1"], {"d1"}, ks=(1, 5))
    means = m.mean_scores([row, other])
    assert means["mrr"] == pytest.approx((0.5 + 1.0) / 2)
    assert means["map"] == pytest.approx((1 / 3 + 1.0) / 2)
    assert "rr" not in means and "ap" not in means
