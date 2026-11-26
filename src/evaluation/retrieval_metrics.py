"""Retrieval metrics of one ranked list against its qrels, and their means over a question set.

Set metrics (order inside the top k does not matter): Hit@k, Precision@k, Recall@k, F1@k,
R-Precision. Rank metrics: reciprocal rank (RR, mean = MRR), average precision (AP, mean =
MAP), nDCG@k. Definitions, for a ranking r_1, r_2, ... and the relevant set R:

- Hit@k = 1 if at least one relevant document is in the top k, else 0.
- Precision@k = |R ∩ top k| / k; Recall@k = |R ∩ top k| / |R|; F1@k their harmonic mean.
- R-Precision = Precision@|R|.
- RR = 1 / rank of the first relevant document (0 when none is retrieved).
- AP = (1 / |R|) * sum of Precision@i over the ranks i holding a relevant document
  (relevant documents never retrieved count as 0).
- nDCG@k = DCG@k / IDCG@k with DCG@k = sum_i gain(r_i) / log2(i + 1) (linear gains,
  Järvelin & Kekäläinen 2002) and IDCG@k the DCG of the ideal order of the gains. With binary
  qrels every relevant document has gain 1.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

SET_METRICS = ("hit", "precision", "recall", "f1")


def _relevant_in_top(ranked: Sequence[str], relevant: set[str], k: int) -> int:
    return sum(1 for doc in ranked[:k] if doc in relevant)


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    return 1.0 if _relevant_in_top(ranked, set(relevant), k) else 0.0


def precision_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    return _relevant_in_top(ranked, set(relevant), k) / k


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    wanted = set(relevant)
    return _relevant_in_top(ranked, wanted, k) / len(wanted) if wanted else 0.0


def f1_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    wanted = set(relevant)
    p, r = precision_at_k(ranked, wanted, k), recall_at_k(ranked, wanted, k)
    return 2 * p * r / (p + r) if p + r else 0.0


def r_precision(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    wanted = set(relevant)
    return precision_at_k(ranked, wanted, len(wanted)) if wanted else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    wanted = set(relevant)
    for rank, doc in enumerate(ranked, start=1):
        if doc in wanted:
            return 1.0 / rank
    return 0.0


def average_precision(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    wanted = set(relevant)
    if not wanted:
        return 0.0
    found, total = 0, 0.0
    for rank, doc in enumerate(ranked, start=1):
        if doc in wanted:
            found += 1
            total += found / rank
    return total / len(wanted)


def ndcg_at_k(ranked: Sequence[str], gains: Mapping[str, float] | Iterable[str], k: int) -> float:
    """`gains`: {doc: graded gain}, or the relevant documents (gain 1 each)."""
    graded = dict(gains) if isinstance(gains, Mapping) else dict.fromkeys(gains, 1.0)
    dcg = sum(graded.get(doc, 0.0) / math.log2(i + 1) for i, doc in enumerate(ranked[:k], start=1))
    ideal = sorted((g for g in graded.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def score(
    ranked: Sequence[str],
    relevant: Iterable[str],
    ks: Iterable[int],
    gains: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Every metric of one question: `<metric>_at_<k>` for each k, then r_precision, rr, ap."""
    wanted = set(relevant)
    result: dict[str, float] = {}
    for k in ks:
        result[f"hit_at_{k}"] = hit_at_k(ranked, wanted, k)
        result[f"precision_at_{k}"] = precision_at_k(ranked, wanted, k)
        result[f"recall_at_{k}"] = recall_at_k(ranked, wanted, k)
        result[f"f1_at_{k}"] = f1_at_k(ranked, wanted, k)
        result[f"ndcg_at_{k}"] = ndcg_at_k(ranked, gains if gains is not None else wanted, k)
    result["r_precision"] = r_precision(ranked, wanted)
    result["rr"] = reciprocal_rank(ranked, wanted)
    result["ap"] = average_precision(ranked, wanted)
    return result


def mean_scores(rows: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Mean of each metric over the questions; `rr` and `ap` become `mrr` and `map`."""
    items = list(rows)
    if not items:
        return {}
    renamed = {"rr": "mrr", "ap": "map"}
    return {
        renamed.get(key, key): sum(float(row[key]) for row in items) / len(items)
        for key in items[0]
    }
