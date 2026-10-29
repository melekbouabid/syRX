"""Cell-line name harmonisation and mapping to DepMap identifiers.

Sources write the same line in many ways ("MDA-MB-231", "MDAMB231", "MDA-MB-231/ATCC").
Names are reduced to a key of upper-case letters and digits, optionally translated through
an alias table, and matched to DepMap's `StrippedCellLineName`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from synergyrx_data.cleaning.drugs import normalize_name

_NON_ALNUM = re.compile(r"[^0-9A-Z]+")
# Supplier suffixes and annotations that are not part of the line's name.
_SUFFIXES = re.compile(r"(/ATCC|\(TB\)|\s*\[.*?\])$", re.IGNORECASE)

# Spellings that differ from DepMap by more than punctuation. Each target was checked against
# DepMap 24Q4 `StrippedCellLineName`. Extend with `config/cell_line_aliases.csv` (columns:
# alias, canonical) rather than editing this.
BUILTIN_ALIASES: dict[str, str] = {}


def cell_line_key(name: Any) -> str:
    """Upper-case letters and digits of the name, without supplier suffixes."""
    text = _SUFFIXES.sub("", normalize_name(name)).upper()
    return _NON_ALNUM.sub("", text)


@dataclass
class CellLineResolution:
    cell_lines: pd.DataFrame
    name_map: pd.DataFrame
    review_queue: pd.DataFrame


def load_aliases(path: Path) -> dict[str, str]:
    aliases = dict(BUILTIN_ALIASES)
    if path.is_file():
        table = pd.read_csv(path, dtype="string", comment="#").dropna()
        aliases.update(
            {
                cell_line_key(a): cell_line_key(c)
                for a, c in zip(table["alias"], table["canonical"], strict=True)
            }
        )
    return aliases


def collect_cell_line_names(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (source, raw name, tissue) with the number of measurements."""
    parts = []
    for source, frame in frames.items():
        part = pd.DataFrame(
            {
                "source": source,
                "raw_name": frame["cell_line_name"].astype("string"),
                "tissue": frame["tissue"].astype("string") if "tissue" in frame.columns else pd.NA,
            }
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["source", "raw_name", "tissue", "n"])
    stacked = pd.concat(parts, ignore_index=True).dropna(subset=["raw_name"])
    stacked["tissue"] = stacked["tissue"].astype("string")
    return (
        stacked.groupby(["source", "raw_name", "tissue"], dropna=False).size().reset_index(name="n")
    )


class DepMapIndex:
    """Looks a name key up in DepMap, most reliable field first.

    Tiers: `StrippedCellLineName`, then the display name, then the prefix of the CCLE name.
    A later tier is only consulted when the earlier ones have no match, otherwise derived
    lines ("A375 SKIN CJ1", CCLE name `A375_SKIN_CJ1`) would make "A375" look ambiguous.
    """

    def __init__(self, models: pd.DataFrame | None) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self._tiers: list[dict[str, list[str]]] = [{}, {}, {}]
        if models is None or models.empty:
            return
        self.records = {r["depmap_id"]: r for r in models.to_dict("records")}
        for record in self.records.values():
            ccle = record.get("ccle_name")
            keys = (
                cell_line_key(record.get("stripped_name") or ""),
                cell_line_key(record.get("cell_line_name") or ""),
                cell_line_key(str(ccle).split("_", 1)[0]) if isinstance(ccle, str) else "",
            )
            for tier, key in zip(self._tiers, keys, strict=True):
                if key and record["depmap_id"] not in tier.setdefault(key, []):
                    tier[key].append(record["depmap_id"])

    def __bool__(self) -> bool:
        return bool(self.records)

    def find(self, key: str) -> list[str]:
        for tier in self._tiers:
            if key in tier:
                return tier[key]
        return []


def build_cell_line_table(
    names: pd.DataFrame,
    depmap_models: pd.DataFrame | None = None,
    aliases: dict[str, str] | None = None,
) -> CellLineResolution:
    """Map every source cell-line name to one record, with a DepMap id where it can be found."""
    aliases = BUILTIN_ALIASES if aliases is None else aliases
    names = names.copy()
    names["raw_name"] = names["raw_name"].map(normalize_name)
    names["key"] = names["raw_name"].map(cell_line_key)
    names = names[names["key"] != ""]
    index = DepMapIndex(depmap_models)
    records = index.records
    rows = []
    for key, group in names.groupby("key", sort=True):
        key = str(key)
        method, matches = "exact", index.find(key)
        if not matches and aliases.get(key, key) != key:
            method, matches = "alias", index.find(aliases[key])
        canonical_key = aliases.get(key, key) if method == "alias" else key
        review = None
        if not index:
            method, review = "unmapped", "DepMap model table not loaded"
        elif not matches:
            method, review = "unmapped", "no DepMap model with this name"
        elif len(matches) > 1:
            review = f"{len(matches)} DepMap models share this name: {', '.join(matches)}"
        depmap_id = matches[0] if len(matches) == 1 else None
        model = records.get(depmap_id, {}) if depmap_id else {}
        by_name = group.groupby("raw_name")["n"].sum().sort_values(ascending=False)
        tissues = group.dropna(subset=["tissue"]).groupby("tissue")["n"].sum()
        source_tissue = tissues.idxmax() if len(tissues) else None
        rows.append(
            {
                "cell_line_id": depmap_id or f"name:{canonical_key}",
                "name": model.get("cell_line_name") or by_name.index[0],
                "stripped_name": model.get("stripped_name") or canonical_key,
                "depmap_id": depmap_id,
                "tissue": model.get("lineage") or source_tissue,
                "disease": model.get("primary_disease"),
                "source_tissue": source_tissue,
                "source_names": list(by_name.index),
                "sources": sorted(set(group["source"])),
                "match_method": method,
                "mapped": depmap_id is not None,
                "needs_review": review is not None,
                "review_reason": review,
                "n_measurements": int(group["n"].sum()),
                "_key": key,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        empty = pd.DataFrame()
        return CellLineResolution(empty, empty, empty)
    key_to_id = dict(zip(table["_key"], table["cell_line_id"], strict=True))
    name_map = (
        names[["source", "raw_name", "key"]]
        .drop_duplicates()
        .assign(cell_line_id=lambda f: f["key"].map(key_to_id))
        .sort_values(["source", "raw_name"])
        .reset_index(drop=True)
    )
    # Different spellings can land on the same DepMap model: merge them into one record.
    merged = (
        table.drop(columns="_key")
        .groupby("cell_line_id", sort=True)
        .agg(
            name=("name", "first"),
            stripped_name=("stripped_name", "first"),
            depmap_id=("depmap_id", "first"),
            tissue=("tissue", "first"),
            disease=("disease", "first"),
            source_tissue=("source_tissue", "first"),
            source_names=("source_names", lambda v: sorted({n for names in v for n in names})),
            sources=("sources", lambda v: sorted({s for srcs in v for s in srcs})),
            match_method=("match_method", "first"),
            mapped=("mapped", "max"),
            needs_review=("needs_review", "max"),
            review_reason=("review_reason", "first"),
            n_measurements=("n_measurements", "sum"),
        )
        .reset_index()
    )
    text = [
        "cell_line_id", "name", "stripped_name", "depmap_id", "tissue", "disease",
        "source_tissue", "match_method", "review_reason",
    ]  # fmt: skip
    merged = merged.astype(dict.fromkeys(text, "string"))
    review = merged.loc[
        merged["needs_review"],
        ["cell_line_id", "name", "source_names", "sources", "review_reason", "n_measurements"],
    ].sort_values("n_measurements", ascending=False)
    review = review.assign(
        source_names=review["source_names"].map("; ".join), sources=review["sources"].map("; ".join)
    ).reset_index(drop=True)
    return CellLineResolution(merged, name_map, review)
