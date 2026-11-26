"""Cohen's kappa of the judge against human labels."""

from __future__ import annotations

import math

import pytest

from src.evaluation.judge_agreement import cohen_kappa, confusion, interpret


def raters(both_yes: int, a_yes_b_no: int, a_no_b_yes: int, both_no: int) -> tuple[list, list]:
    a = [1] * both_yes + [1] * a_yes_b_no + [0] * a_no_b_yes + [0] * both_no
    b = [1] * both_yes + [0] * a_yes_b_no + [1] * a_no_b_yes + [0] * both_no
    return a, b


def test_textbook_example() -> None:
    # 50 items: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no. p_o = 35/50 = 0.70; A says yes
    # 25/50, B 30/50: p_e = 0.5 * 0.6 + 0.5 * 0.4 = 0.50; kappa = (0.70 - 0.50) / 0.50 = 0.40.
    a, b = raters(20, 5, 10, 15)
    result = cohen_kappa(a, b)
    assert result["confusion"] == [[20, 5], [10, 15]]
    assert result["observed_agreement"] == pytest.approx(0.70)
    assert result["expected_agreement"] == pytest.approx(0.50)
    assert result["kappa"] == pytest.approx(0.40)
    assert result["interpretation"] == "fair"


def test_high_agreement_can_be_chance_only() -> None:
    # A judge saying "faithful" 48 times out of 50 against 47 faithful human labels: 90 %
    # agreement, yet below chance.
    a, b = raters(45, 2, 3, 0)
    result = cohen_kappa(a, b)
    assert result["observed_agreement"] == pytest.approx(0.90)
    assert result["expected_agreement"] == pytest.approx(0.94 * 0.96 + 0.06 * 0.04)
    assert result["kappa"] < 0 and result["interpretation"] == "poor"


def test_perfect_and_undefined() -> None:
    assert cohen_kappa([1, 0, 1], [1, 0, 1])["kappa"] == pytest.approx(1.0)
    assert math.isnan(cohen_kappa([1, 1], [1, 1])["kappa"])  # one label only: p_e = 1


def test_landis_koch_bands() -> None:
    assert [interpret(k) for k in (-0.1, 0.1, 0.3, 0.5, 0.7, 0.9)] == [
        "poor",
        "slight",
        "fair",
        "moderate",
        "substantial",
        "almost perfect",
    ]


def test_invalid_input() -> None:
    with pytest.raises(ValueError):
        cohen_kappa([1, 0], [1])
    with pytest.raises(ValueError):
        cohen_kappa([1, 2], [1, 0])
    with pytest.raises(ValueError):
        confusion([1], [1, 0])
