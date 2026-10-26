"""Canonical drug-pair keys and the final synergy table.

A combination has no order: (A, B) and (B, A) are the same experiment. Each pair is stored
with its two drug ids sorted, and `pair_key` is `"<smaller id>|<larger id>"`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from synergyrx_data.cleaning.cell_lines import cell_line_key
from synergyrx_data.cleaning.drugs import name_key, normalize_name
from synergyrx_data.cleaning.scores import SCORE_COLUMNS, aggregate_replicates, flag_outliers
from synergyrx_data.provenance import PROVENANCE_COLUMNS

PAIR_SEPARATOR = "|"
GRAIN = ["pair_key", "drug_a_id", "drug_b_id", "cell_line_id", "source", "study_id"]


def canonical_pair(drug_a: str, drug_b: str) -> tuple[str, str, str]:
    """Return (first, second, pair_key) with the two ids in sorted order."""
    first, second = sorted((str(drug_a), str(drug_b)))
    return first, second, f"{first}{PAIR_SEPARATOR}{second}"


def add_pair_key(frame: pd.DataFrame, a: str = "drug_a_id", b: str = "drug_b_id") -> pd.DataFrame:
    """Sort each row's two ids and add `pair_key`. Vectorised form of `canonical_pair`."""
    left = frame[a].astype(str).to_numpy()
    right = frame[b].astype(str).to_numpy()
    swap = left > right
    out = frame.copy()
    out[a] = np.where(swap, right, left)
    out[b] = np.where(swap, left, right)
    out["pair_key"] = out[a] + PAIR_SEPARATOR + out[b]
    return out


def _map_unique(values: pd.Series, mapping: dict[str, str], key_fn: Any) -> pd.Series:
    """Map raw names to ids through their matching key, computing each key once."""
    uniques = pd.unique(values.dropna())
    lookup = {raw: mapping.get(key_fn(normalize_name(raw))) for raw in uniques}
    return values.map(lookup)


def build_synergy_pairs(
    frames: dict[str, pd.DataFrame],
    drug_name_map: pd.DataFrame,
    cell_name_map: pd.DataFrame,
    drugs: pd.DataFrame,
    cell_lines: pd.DataFrame,
    *,
    outlier_abs: float = 200.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build `synergy_pairs`: one row per pair, cell line, source and study.

    Replicates inside one study are averaged (their count and spread are kept). Records of
    the same pair and cell line from different sources or studies stay on separate rows:
    scores from different studies are never merged.
    """
    drug_ids = dict(zip(drug_name_map["name_key"], drug_name_map["drug_id"], strict=True))
    cell_ids = dict(zip(cell_name_map["key"], cell_name_map["cell_line_id"], strict=True))
    stats: dict[str, Any] = {"input_rows": 0, "self_pairs_dropped": 0, "unmapped_rows_dropped": 0}
    parts = []
    for source, frame in frames.items():
        stats["input_rows"] += len(frame)
        part = frame[[*SCORE_COLUMNS, "study_id", *PROVENANCE_COLUMNS]].copy()
        part["source"] = source
        part["drug_a_id"] = _map_unique(frame["drug_a_name"], drug_ids, name_key)
        part["drug_b_id"] = _map_unique(frame["drug_b_name"], drug_ids, name_key)
        part["cell_line_id"] = _map_unique(frame["cell_line_name"], cell_ids, cell_line_key)
        unmapped = part[["drug_a_id", "drug_b_id", "cell_line_id"]].isna().any(axis=1)
        stats["unmapped_rows_dropped"] += int(unmapped.sum())
        part = part[~unmapped]
        same = part["drug_a_id"] == part["drug_b_id"]
        stats["self_pairs_dropped"] += int(same.sum())
        parts.append(add_pair_key(part[~same]))
    if not parts:
        return pd.DataFrame(), stats
    stacked = pd.concat(parts, ignore_index=True)
    pairs = aggregate_replicates(stacked, GRAIN, keep_first=PROVENANCE_COLUMNS[1:])
    stats["replicates_averaged"] = int(len(stacked) - len(pairs))

    drug_names = dict(zip(drugs["drug_id"], drugs["name"], strict=True))
    cells = cell_lines.set_index("cell_line_id")
    pairs["drug_a_name"] = pairs["drug_a_id"].map(drug_names)
    pairs["drug_b_name"] = pairs["drug_b_id"].map(drug_names)
    pairs["cell_line_name"] = pairs["cell_line_id"].map(cells["name"])
    pairs["depmap_id"] = pairs["cell_line_id"].map(cells["depmap_id"])
    pairs["tissue"] = pairs["cell_line_id"].map(cells["tissue"])
    pairs["score_outlier"] = flag_outliers(pairs, outlier_abs)

    triplet = ["pair_key", "cell_line_id"]
    pairs["n_sources"] = pairs.groupby(triplet)["source"].transform("nunique").astype("int32")
    pairs["n_records"] = pairs.groupby(triplet)["source"].transform("size").astype("int32")
    ordered = [
        "pair_key", "drug_a_id", "drug_b_id", "drug_a_name", "drug_b_name", "cell_line_id",
        "cell_line_name", "depmap_id", "tissue", "source", "study_id", *SCORE_COLUMNS,
        *[f"{c}_std" for c in SCORE_COLUMNS], "n_replicates", "score_outlier", "n_sources",
        "n_records", *PROVENANCE_COLUMNS[1:],
    ]  # fmt: skip
    text = [c for c in ordered if pairs[c].dtype == object or str(pairs[c].dtype) == "str"]
    pairs = pairs[ordered].astype(dict.fromkeys(text, "string"))
    pairs = pairs.sort_values(GRAIN, na_position="first").reset_index(drop=True)
    stats["output_rows"] = len(pairs)
    return pairs, stats
