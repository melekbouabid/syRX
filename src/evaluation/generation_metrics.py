"""Reference-based generation metrics (course "Generation Evaluation"): Exact Match, token F1,
ROUGE-1 / ROUGE-2 / ROUGE-L and smoothed sentence BLEU, all on normalised tokens.

Normalisation (SQuAD style, accents folded so French tokens compare): lower case, accents
removed, punctuation removed, the articles a / an / the dropped, whitespace collapsed.

- Exact Match: 1 if the normalised strings are equal.
- Token F1: harmonic mean of the precision and recall of the common tokens (multiset).
- ROUGE-N: F1 of the clipped n-gram overlap; ROUGE-L: F1 of the longest common subsequence
  (P = LCS / |prediction|, R = LCS / |reference|).
- BLEU: geometric mean of the clipped 1- to 4-gram precisions times the brevity penalty
  BP = min(1, exp(1 - r / c)). On one sentence, an answer without any common 4-gram gets 0
  whatever its 1- and 2-gram overlap, so the score collapses to 0 for most free-form answers.
  Smoothing (Chen & Cherry 2014, the `exp` method of sacrebleu): the k-th order with no match
  counts 1 / (2^k * n-grams of that order) instead of 0, so the score keeps ranking answers.
  Orders longer than the answer are left out (effective order, as sacrebleu's sentence_bleu);
  without any common token the score is 0.
"""

from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter
from collections.abc import Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b")
# Typographic punctuation too: guillemets, curly quotes, ellipsis, bullet, en and em dashes.
TYPOGRAPHIC = "\u00ab\u00bb\u2019\u201c\u201d\u2026\u2022\u2013\u2014"
_PUNCTUATION = str.maketrans(dict.fromkeys(string.punctuation + TYPOGRAPHIC, " "))


def normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.translate(_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", folded).split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize(prediction) == normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return float(pred == ref)
    common = sum((Counter(pred) & Counter(ref)).values())
    if not common:
        return 0.0
    precision, recall = common / len(pred), common / len(ref)
    return 2 * precision * recall / (precision + recall)


def _ngrams(items: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(items[i : i + n]) for i in range(len(items) - n + 1))


def _f1(overlap: int, predicted: int, expected: int) -> float:
    if not overlap or not predicted or not expected:
        return 0.0
    precision, recall = overlap / predicted, overlap / expected
    return 2 * precision * recall / (precision + recall)


def rouge_n(prediction: str, reference: str, n: int) -> float:
    pred, ref = _ngrams(tokens(prediction), n), _ngrams(tokens(reference), n)
    return _f1(sum((pred & ref).values()), sum(pred.values()), sum(ref.values()))


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    pred, ref = tokens(prediction), tokens(reference)
    return _f1(lcs_length(pred, ref), len(pred), len(ref))


def bleu(prediction: str, reference: str, max_n: int = 4, smooth: bool = True) -> float:
    """Sentence BLEU in [0, 1] (sacrebleu reports it x 100)."""
    pred, ref = tokens(prediction), tokens(reference)
    if not pred or not ref:
        return 0.0
    logs: list[float] = []
    zero_orders = 0
    for n in range(1, max_n + 1):
        total = len(pred) - n + 1
        if total <= 0:
            break  # effective order: the answer is shorter than n
        matches = sum((_ngrams(pred, n) & _ngrams(ref, n)).values())
        if not matches and n == 1:
            return 0.0  # no common token at all: nothing to smooth (as sacrebleu)
        if matches:
            logs.append(math.log(matches / total))
        elif smooth:
            zero_orders += 1
            logs.append(math.log(1 / (2**zero_orders * total)))
        else:
            return 0.0
    brevity = 1.0 if len(pred) > len(ref) else math.exp(1 - len(ref) / len(pred))
    return brevity * math.exp(sum(logs) / len(logs))


def scores(prediction: str, reference: str) -> dict[str, float]:
    return {
        "exact_match": exact_match(prediction, reference),
        "token_f1": token_f1(prediction, reference),
        "rouge_1": rouge_n(prediction, reference, 1),
        "rouge_2": rouge_n(prediction, reference, 2),
        "rouge_l": rouge_l(prediction, reference),
        "bleu": bleu(prediction, reference),
    }


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0
