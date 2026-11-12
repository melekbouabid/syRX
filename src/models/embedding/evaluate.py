"""Retrieval metrics of the fine-tuning (step 3.5).

A question is answered by its passage, or by any chunk of the same label section (they share
the question: `relevant_ids` of src/models/embedding/pairs.py); the rank is that of the first
relevant chunk. Recall@k (it is in the first k), MRR@10 (1 / rank, 0 beyond 10) and nDCG@10
(1 / log2(rank + 1), first relevant chunk). The benchmark of the app
(src/benchmark/bench_retrieval.py) is the real comparison; these figures follow the training
on the validation pairs (documents never trained on).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np

KS = (1, 5, 10)


def ranks(
    scores: np.ndarray, ids: Sequence[str], expected: Sequence[str | Collection[str]]
) -> np.ndarray:
    """1-based rank of the first relevant passage of each question (scores: questions x
    corpus); `expected` holds one chunk id or a collection of them per question."""
    column = {chunk_id: i for i, chunk_id in enumerate(ids)}
    best = np.empty(len(expected))
    for row, wanted in enumerate(expected):
        chosen = [wanted] if isinstance(wanted, str) else list(wanted)
        best[row] = max(scores[row, column[c]] for c in chosen if c in column)
    return (scores > best[:, None]).sum(axis=1) + 1


def retrieval_metrics(
    scores: np.ndarray, ids: Sequence[str], expected: Sequence[str | Collection[str]]
) -> dict[str, float]:
    rank = ranks(np.asarray(scores, dtype=np.float64), ids, expected).astype(np.float64)
    out = {f"recall_at_{k}": float(np.mean(rank <= k)) for k in KS}
    out["mrr_at_10"] = float(np.mean(np.where(rank <= 10, 1.0 / rank, 0.0)))
    out["ndcg_at_10"] = float(np.mean(np.where(rank <= 10, 1.0 / np.log2(rank + 1), 0.0)))
    out["questions"] = float(len(rank))
    return out
