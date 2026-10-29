"""Synergy score handling.

Bliss, Loewe, HSA and ZIP are different reference models and are not interchangeable: a pair
can be synergistic under one and antagonistic under another. Each keeps its own column, plus
DrugComb's CSS and ALMANAC's ComboScore. Nothing here converts or blends scores; the project
picks its target score later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SCORES = {
    "synergy_bliss": "Bliss independence excess (DrugComb scale; > 0 means synergy)",
    "synergy_loewe": "Loewe additivity excess (> 0 means synergy)",
    "synergy_hsa": "Highest single agent excess (> 0 means synergy)",
    "synergy_zip": "Zero interaction potency delta score (> 0 means synergy)",
    "css": "Combination sensitivity score (DrugComb): efficacy, not synergy",
    "combo_score": "NCI-ALMANAC ComboScore: sum over doses of expected minus observed growth",
}
SCORE_COLUMNS = list(SCORES)
SYNERGY_SCORES = SCORE_COLUMNS[:4]


AGREEMENT_COLUMNS = [
    "source_a",
    "source_b",
    "score",
    "shared_triplets",
    "pearson_r",
    "median_abs_diff",
    "sign_disagreement_rate",
]
CONFLICT_COLUMNS = [
    "pair_key",
    "cell_line_id",
    "drug_a_name",
    "drug_b_name",
    "cell_line_name",
    "n_records",
    "n_sources",
    "sources",
    "studies",
    *[f"{score}_{stat}" for score in SCORE_COLUMNS for stat in ("n", "min", "max", "range")],
    "max_synergy_range",
    "sign_disagreement",
    "is_conflict",
]


def aggregate_replicates(
    frame: pd.DataFrame, keys: list[str], *, keep_first: list[str] | None = None
) -> pd.DataFrame:
    """Average replicate measurements within `keys`; keep their count and standard deviation.

    `study_id` may be missing (DrugComb's bulk table has none), so nulls form their own group.
    """
    keep_first = keep_first or []
    grouped = frame.groupby(keys, dropna=False, sort=False, observed=True)
    means = grouped[SCORE_COLUMNS].mean()
    spread = grouped[SCORE_COLUMNS].std(ddof=0).add_suffix("_std")
    extras = grouped[keep_first].first() if keep_first else None
    counts = grouped.size().rename("n_replicates").astype("int32")
    parts = [means, spread, counts] + ([extras] if extras is not None else [])
    return pd.concat(parts, axis=1).reset_index()


def flag_outliers(frame: pd.DataFrame, limit: float) -> pd.Series:
    """True where any synergy score is implausibly large. Values are flagged, not clipped."""
    return (frame[SYNERGY_SCORES].abs() > limit).any(axis=1)


def find_conflicts(pairs: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Pairs measured on the same cell line by more than one source or study.

    One row per (pair, cell line) with, for each score, how many records report it and the
    spread between them. `is_conflict` is set when a score's range exceeds `tolerance` or
    its sign differs between records.
    """
    if pairs.empty:
        return pd.DataFrame(columns=CONFLICT_COLUMNS)
    multi = pairs[pairs["n_records"] > 1]
    if multi.empty:
        return pd.DataFrame(columns=CONFLICT_COLUMNS)
    grouped = multi.groupby(["pair_key", "cell_line_id"], sort=False)
    out = grouped.agg(
        drug_a_name=("drug_a_name", "first"),
        drug_b_name=("drug_b_name", "first"),
        cell_line_name=("cell_line_name", "first"),
        n_records=("source", "size"),
        n_sources=("source", "nunique"),
        sources=("source", lambda s: "; ".join(sorted(set(s)))),
        studies=("study_id", lambda s: "; ".join(sorted({str(v) for v in s.dropna()}))),
    )
    worst = pd.Series(0.0, index=out.index)
    sign_flip = pd.Series(False, index=out.index)
    for score in SCORE_COLUMNS:
        stats = grouped[score].agg(["count", "min", "max"])
        comparable = stats["count"] >= 2
        value_range = (stats["max"] - stats["min"]).where(comparable)
        out[f"{score}_n"] = stats["count"].astype("int32")
        out[f"{score}_min"] = stats["min"].where(comparable)
        out[f"{score}_max"] = stats["max"].where(comparable)
        out[f"{score}_range"] = value_range
        if score in SYNERGY_SCORES:
            worst = np.maximum(worst, value_range.fillna(0.0))
            sign_flip |= comparable & (stats["min"] < 0) & (stats["max"] > 0)
    out["max_synergy_range"] = worst
    out["sign_disagreement"] = sign_flip
    out["is_conflict"] = (worst > tolerance) | sign_flip
    return (
        out.reset_index().sort_values("max_synergy_range", ascending=False).reset_index(drop=True)
    )


def source_agreement(pairs: pd.DataFrame) -> pd.DataFrame:
    """How well each pair of sources agrees on the triplets they share, per score."""
    if pairs.empty:
        return pd.DataFrame(columns=AGREEMENT_COLUMNS)
    rows = []
    per_source = (
        pairs.groupby(["pair_key", "cell_line_id", "source"], sort=False)[SYNERGY_SCORES]
        .mean()
        .reset_index()
    )
    sources = sorted(per_source["source"].unique())
    indexed = {
        s: per_source[per_source["source"] == s].set_index(["pair_key", "cell_line_id"])
        for s in sources
    }
    for i, left in enumerate(sources):
        for right in sources[i + 1 :]:
            shared = indexed[left].index.intersection(indexed[right].index)
            if not len(shared):
                continue
            for score in SYNERGY_SCORES:
                a = indexed[left].loc[shared, score]
                b = indexed[right].loc[shared, score]
                both = a.notna() & b.notna()
                if both.sum() < 2:
                    continue
                a, b = a[both], b[both]
                rows.append(
                    {
                        "source_a": left,
                        "source_b": right,
                        "score": score,
                        "shared_triplets": int(both.sum()),
                        "pearson_r": float(a.corr(b)) if a.std() > 0 and b.std() > 0 else None,
                        "median_abs_diff": float((a - b).abs().median()),
                        "sign_disagreement_rate": float(((a > 0) != (b > 0)).mean()),
                    }
                )
    return pd.DataFrame(rows, columns=AGREEMENT_COLUMNS)
