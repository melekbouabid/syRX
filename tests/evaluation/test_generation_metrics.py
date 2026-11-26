"""Reference-based generation metrics, with values computed by hand."""

from __future__ import annotations

import math

import pytest

from src.evaluation import generation_metrics as g

PRED = "Warfarin raises bleeding risk."
REF = "Warfarin increases the risk of bleeding."


def test_normalisation_is_squad_style_with_accents_folded() -> None:
    assert g.normalize("The  Contre-indiquée, «sévère» !") == "contre indiquee severe"
    assert g.tokens(REF) == ["warfarin", "increases", "risk", "of", "bleeding"]


def test_exact_match_after_normalisation() -> None:
    assert g.exact_match("The cat sat on the mat.", "a cat sat on a mat") == 1.0
    assert g.exact_match("Paris is the capital", "the capital is Paris") == 0.0


def test_token_f1() -> None:
    assert g.token_f1("Paris is the capital", "the capital is Paris") == 1.0  # bag of tokens
    # common {warfarin, bleeding, risk}: P = 3/4, R = 3/5
    assert g.token_f1(PRED, REF) == pytest.approx(2 * 0.75 * 0.6 / (0.75 + 0.6))
    assert g.token_f1("", REF) == 0.0
    assert g.token_f1("zzz", REF) == 0.0


def test_rouge() -> None:
    assert g.rouge_n(PRED, REF, 1) == pytest.approx(g.token_f1(PRED, REF))
    assert g.rouge_n(PRED, REF, 2) == 0.0  # no common bigram
    # LCS = 2 (warfarin ... risk): P = 2/4, R = 2/5
    assert g.rouge_l(PRED, REF) == pytest.approx(2 * 0.5 * 0.4 / 0.9)
    assert g.lcs_length(list("abcbdab"), list("bdcaba")) == 4


def test_smoothed_bleu_by_hand() -> None:
    # 4 predicted tokens, 5 reference tokens. p1 = 3/4; no 2-, 3-, 4-gram match: the k-th
    # empty order counts 1 / (2^k * its n-grams): 1/(2*3), 1/(4*2), 1/(8*1).
    precisions = [3 / 4, 1 / 6, 1 / 8, 1 / 8]
    by_hand = math.exp(1 - 5 / 4) * math.exp(sum(math.log(p) for p in precisions) / 4)
    assert g.bleu(PRED, REF) == pytest.approx(by_hand)
    assert g.bleu(PRED, REF, smooth=False) == 0.0  # why sentence BLEU needs smoothing
    assert g.bleu(REF, REF) == pytest.approx(1.0)


def test_bleu_is_zero_without_any_common_token() -> None:
    # Smoothing only fills the higher orders: no unigram match is 0 (as sacrebleu).
    assert g.bleu("La warfarine augmente le saignement", "Bleeding increases") == 0.0


def test_bleu_effective_order_on_a_short_answer() -> None:
    # Two tokens: only 1- and 2-grams exist, both match; c = r = 2, so no brevity penalty.
    assert g.bleu("warfarin bleeding", "warfarin bleeding") == pytest.approx(1.0)


def test_scores_and_cosine() -> None:
    assert set(g.scores(PRED, REF)) == {
        "exact_match",
        "token_f1",
        "rouge_1",
        "rouge_2",
        "rouge_l",
        "bleu",
    }
    assert g.cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert g.cosine([1.0, 0.0], [0.0, 2.0]) == 0.0
