"""Drug identity resolution.

Each source names drugs its own way ("5-FU", "Fluorouracil", "NSC-19893"). This module
decides which names are the same compound, attaches a standardised structure and public
identifiers, scores how sure it is, and queues everything doubtful for a person to review.
Two names are the same drug when their standardised structures share an InChIKey.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synergyrx_data.cleaning.smiles import connectivity_block, standardize_smiles
from synergyrx_data.provenance import PROVENANCE_COLUMNS

# Unicode hyphens, dashes and the minus sign, all written as a plain hyphen.
_DASHES = re.compile(f"[{chr(0x2010)}-{chr(0x2015)}{chr(0x2212)}]")
_NON_ALNUM = re.compile(r"[^0-9a-z]+")
_STEREO_SIGNS = (
    (re.compile(rf"\(\+\s*/\s*-\)|\({chr(0xB1)}\)"), " rac "),
    (re.compile(r"\(\+\)"), " plus "),
    (re.compile(r"\(-\)"), " minus "),
)

CONFIDENCE = {
    "manual_override": 1.0,
    "source_smiles+pubchem": 1.0,
    "pubchem_name": 0.9,
    "source_smiles": 0.8,
    "pubchem_name_ambiguous": 0.6,
    "source_smiles_conflict": 0.5,
    "unresolved": 0.0,
}


def normalize_name(name: Any) -> str:
    """Trim, unify Unicode forms and dashes, collapse whitespace. Case is preserved."""
    text = unicodedata.normalize("NFKC", str(name))
    return re.sub(r"\s+", " ", _DASHES.sub("-", text)).strip()


def name_key(name: Any) -> str:
    """Matching key: lower-case letters and digits only ("ABT-888" == "abt 888").

    Stereo descriptors written with signs are kept as words first, so that "(+)-JQ1",
    "(-)-JQ1" and "(+/-)-JQ1" stay three different names: they are different compounds.
    """
    text = normalize_name(name).lower()
    for pattern, word in _STEREO_SIGNS:
        text = pattern.sub(word, text)
    return _NON_ALNUM.sub("", text)


@dataclass
class DrugResolution:
    drugs: pd.DataFrame
    name_map: pd.DataFrame
    review_queue: pd.DataFrame


def collect_mentions(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per (source, raw name, SMILES, source id) with how often it appears."""
    parts = []
    for source, frame in frames.items():
        for side in ("a", "b"):
            part = pd.DataFrame(
                {
                    "raw_name": frame[f"drug_{side}_name"],
                    "smiles": frame.get(f"drug_{side}_smiles"),
                    "source_id": frame.get(f"drug_{side}_source_id"),
                }
            )
            for column in PROVENANCE_COLUMNS:
                part[column] = frame[column] if column in frame.columns else source
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["raw_name", "smiles", "source_id", *PROVENANCE_COLUMNS, "n"])
    stacked = pd.concat(parts, ignore_index=True).dropna(subset=["raw_name"])
    keys = ["raw_name", "smiles", "source_id", *PROVENANCE_COLUMNS]
    stacked[keys] = stacked[keys].astype("string")
    return stacked.groupby(keys, dropna=False).size().reset_index(name="n")


def load_overrides(path: Path) -> pd.DataFrame:
    """Manual decisions: CSV with columns `name`, optional `smiles`, `pubchem_cid`, `note`."""
    if not path.is_file():
        return pd.DataFrame(columns=["name", "smiles", "pubchem_cid", "note"])
    return pd.read_csv(path, dtype="string", comment="#")


def _first(values: pd.Series) -> Any:
    values = values.dropna()
    return values.iloc[0] if len(values) else None


def _joined(values: Any) -> str:
    return "; ".join(sorted({str(v) for v in values if pd.notna(v) and str(v)}))


def _plain(value: Any) -> Any:
    """A table cell as plain Python: missing values become None, arrays become lists."""
    if value is None:
        return None
    if isinstance(value, list | tuple | np.ndarray):
        return [str(item) for item in value]
    return None if pd.isna(value) else value


def _pubchem_by_key(pubchem: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    """PubChem answers indexed by name key; a definite hit beats an ambiguous or missing one."""
    if pubchem is None or pubchem.empty:
        return {}
    rank = {"found": 0, "ambiguous": 1, "not_found": 2}
    ordered = pubchem.assign(_rank=pubchem["status"].map(rank)).sort_values("_rank")
    answers: dict[str, dict[str, Any]] = {}
    for record in ordered.to_dict("records"):
        plain = {field: _plain(value) for field, value in record.items()}
        answers.setdefault(name_key(record["query_name"]), plain)
    return answers


def _resolve_one(
    key: str,
    group: pd.DataFrame,
    pubchem: dict[str, Any] | None,
    override: dict[str, Any] | None,
    pubchem_was_run: bool,
) -> dict[str, Any]:
    """Pick the structure for one name key and say how it was chosen."""
    review: list[str] = []
    candidates = group.dropna(subset=["smiles"]).groupby("smiles")["n"].sum()
    standardized = {s: standardize_smiles(s) for s in candidates.index}
    valid = {s: r for s, r in standardized.items() if r.valid}
    invalid_issue = next((r.issue for r in standardized.values() if not r.valid), None)
    if len({r.inchikey for r in valid.values()}) > 1:
        review.append("sources give different structures for this name")
    source_smiles = max(valid, key=lambda s: int(candidates[s])) if valid else None
    has_pubchem = (
        pubchem is not None and pubchem["status"] != "not_found" and bool(pubchem.get("smiles"))
    )
    pubchem_std = standardize_smiles(pubchem["smiles"]) if has_pubchem and pubchem else None

    chosen: str | None
    if override is not None and pd.notna(override.get("smiles")):
        method, chosen = "manual_override", str(override["smiles"])
    elif source_smiles is not None:
        chosen = source_smiles
        if pubchem_std is None or not pubchem_std.valid:
            method = "source_smiles"
        elif connectivity_block(valid[source_smiles].inchikey) == connectivity_block(
            pubchem_std.inchikey
        ):
            method = "source_smiles+pubchem"
        else:
            method = "source_smiles_conflict"
            review.append("source SMILES and PubChem structure disagree")
    elif pubchem_std is not None and pubchem is not None:
        chosen = str(pubchem["smiles"])
        if pubchem["status"] == "ambiguous":
            method = "pubchem_name_ambiguous"
            review.append(f"{pubchem['n_cids']} PubChem compounds match this name")
        else:
            method = "pubchem_name"
    else:
        method, chosen = "unresolved", _first(group["smiles"])
        if invalid_issue:
            review.append(f"invalid SMILES ({invalid_issue})")
        elif pubchem is not None:
            review.append("name not found in PubChem")
        else:
            review.append("no structure" if pubchem_was_run else "PubChem lookup not run yet")

    result = standardize_smiles(chosen)
    if method != "unresolved" and not result.valid:
        review.append(f"invalid SMILES ({result.issue})")
        method = "unresolved"
    cid = None
    if override is not None and pd.notna(override.get("pubchem_cid")):
        cid = int(override["pubchem_cid"])
    elif has_pubchem and pubchem is not None and method != "source_smiles_conflict":
        cid = int(pubchem["pubchem_cid"])
    names = group.groupby("raw_name")["n"].sum().sort_values(ascending=False)
    return {
        "name_key": key,
        "drug_id": result.inchikey if result.valid else f"name:{key}",
        "raw_names": list(names.index),
        "top_name": sorted(names[names == names.max()].index)[0],
        "pubchem_title": pubchem.get("title") if has_pubchem and pubchem and cid else None,
        "pubchem_cid": cid,
        "pubchem_inchikey": pubchem.get("inchikey") if has_pubchem and pubchem else None,
        "pubchem_synonyms": list(pubchem.get("synonyms") or []) if has_pubchem and pubchem else [],
        "smiles_raw": chosen,
        "smiles": result.smiles,
        "inchikey": result.inchikey,
        "smiles_valid": result.valid,
        "smiles_issue": result.issue,
        "smiles_note": result.note,
        "resolution_method": method,
        "resolution_confidence": CONFIDENCE[method],
        "review": review,
        "sources": sorted(set(group["source"].dropna())),
        "source_ids": sorted(set(group["source_id"].dropna())),
        "weight": int(group["n"].sum()),
        **{c: _joined(group[c]) for c in PROVENANCE_COLUMNS if c != "source"},
    }


def _merge_same_structure(rows: list[dict[str, Any]], max_synonyms: int) -> dict[str, Any]:
    """Collapse name keys that resolved to one structure into a single drug."""
    best = max(rows, key=lambda r: (r["resolution_confidence"], r["weight"]))
    raw_names = [
        name for row in sorted(rows, key=lambda r: -r["weight"]) for name in row["raw_names"]
    ]
    title = next((r["pubchem_title"] for r in rows if r["pubchem_title"]), None)
    name = title or best["top_name"]
    synonyms: list[str] = []
    seen = {name.lower()}
    for candidate in [*raw_names, *best["pubchem_synonyms"]]:
        if str(candidate).lower() not in seen:
            seen.add(str(candidate).lower())
            synonyms.append(str(candidate))
    # Every source name is kept; the cap only trims the long tail of PubChem synonyms.
    keep = max(max_synonyms, len(set(raw_names)))
    reasons = sorted({reason for row in rows for reason in row["review"]})
    return {
        "drug_id": best["drug_id"],
        "name": name,
        "synonyms": synonyms[:keep],
        "pubchem_cid": next((r["pubchem_cid"] for r in rows if r["pubchem_cid"]), None),
        "pubchem_inchikey": next(
            (r["pubchem_inchikey"] for r in rows if r["pubchem_inchikey"]), None
        ),
        "inchikey": best["inchikey"],
        "smiles": best["smiles"],
        "smiles_raw": best["smiles_raw"],
        "smiles_valid": best["smiles_valid"],
        "smiles_issue": best["smiles_issue"],
        "smiles_note": best["smiles_note"],
        "resolution_method": best["resolution_method"],
        "resolution_confidence": best["resolution_confidence"],
        "review_reason": "; ".join(reasons) or None,
        "sources": sorted({s for row in rows for s in row["sources"]}),
        "source_ids": sorted({s for row in rows for s in row["source_ids"]}),
        "n_source_names": len(set(raw_names)),
        "n_measurements": sum(row["weight"] for row in rows),
        "source": _joined(s for row in rows for s in row["sources"]),
        **{
            c: _joined(v for row in rows for v in str(row[c]).split("; "))
            for c in PROVENANCE_COLUMNS
            if c != "source"
        },
    }


def _chembl_lookup(chembl: pd.DataFrame | None) -> tuple[dict[str, str], dict[str, str]]:
    if chembl is None or chembl.empty:
        return {}, {}
    by_key = chembl[chembl["query_type"] == "inchikey"]
    by_name = chembl[chembl["query_type"] == "name"]
    return (
        dict(zip(by_key["query"].astype(str), by_key["chembl_id"].astype(str), strict=True)),
        {name_key(q): str(c) for q, c in zip(by_name["query"], by_name["chembl_id"], strict=True)},
    )


def build_drug_table(
    mentions: pd.DataFrame,
    *,
    pubchem: pd.DataFrame | None = None,
    chembl: pd.DataFrame | None = None,
    overrides: pd.DataFrame | None = None,
    review_below: float = 0.7,
    max_synonyms: int = 50,
) -> DrugResolution:
    """Resolve every drug name in `mentions` (see `collect_mentions`) to one drug record."""
    mentions = mentions.copy()
    mentions["raw_name"] = mentions["raw_name"].map(normalize_name)
    mentions["name_key"] = mentions["raw_name"].map(name_key)
    mentions = mentions[mentions["name_key"] != ""]
    pubchem_answers = _pubchem_by_key(pubchem)
    override_rows = (
        {name_key(r["name"]): r for r in overrides.to_dict("records")}
        if overrides is not None and not overrides.empty
        else {}
    )
    resolved = [
        _resolve_one(
            str(key),
            group,
            pubchem_answers.get(str(key)),
            override_rows.get(str(key)),
            bool(pubchem_answers),
        )
        for key, group in mentions.groupby("name_key", sort=True)
    ]
    by_drug: dict[str, list[dict[str, Any]]] = {}
    for row in resolved:
        by_drug.setdefault(row["drug_id"], []).append(row)
    drugs = pd.DataFrame([_merge_same_structure(rows, max_synonyms) for rows in by_drug.values()])
    if drugs.empty:
        empty = pd.DataFrame()
        return DrugResolution(empty, empty, empty)

    chembl_by_key, chembl_by_name = _chembl_lookup(chembl)
    name_keys_by_drug = {d: [r["name_key"] for r in rows] for d, rows in by_drug.items()}

    def chembl_id(row: pd.Series) -> str | None:
        for key in (row["inchikey"], row["pubchem_inchikey"]):
            if key and key in chembl_by_key:
                return chembl_by_key[key]
        return next(
            (chembl_by_name[k] for k in name_keys_by_drug[row["drug_id"]] if k in chembl_by_name),
            None,
        )

    drugs["chembl_id"] = drugs.apply(chembl_id, axis=1)
    drugs["pubchem_cid"] = drugs["pubchem_cid"].astype("Int64")
    drugs["needs_review"] = (drugs["resolution_confidence"] < review_below) | drugs[
        "review_reason"
    ].notna()
    drugs = drugs.sort_values("drug_id").reset_index(drop=True)
    ordered = [
        "drug_id", "name", "synonyms", "pubchem_cid", "chembl_id", "inchikey", "smiles",
        "smiles_raw", "smiles_valid", "smiles_issue", "smiles_note", "resolution_method",
        "resolution_confidence", "needs_review", "review_reason", "sources", "source_ids",
        "n_source_names", "n_measurements", *PROVENANCE_COLUMNS,
    ]  # fmt: skip
    drugs = drugs[ordered]
    # Fixed text types: a column that happens to be all-null must not change type between
    # builds (it would show up as schema drift and break Parquet readers downstream).
    text = [
        "drug_id", "name", "chembl_id", "inchikey", "smiles", "smiles_raw", "smiles_issue",
        "smiles_note", "resolution_method", "review_reason", *PROVENANCE_COLUMNS,
    ]  # fmt: skip
    drugs = drugs.astype(dict.fromkeys(text, "string"))

    name_map = (
        mentions[["source", "raw_name", "name_key"]]
        .drop_duplicates()
        .assign(
            drug_id=lambda f: f["name_key"].map({r["name_key"]: r["drug_id"] for r in resolved})
        )
        .sort_values(["source", "raw_name"])
        .reset_index(drop=True)
    )
    review = drugs.loc[
        drugs["needs_review"],
        [
            "drug_id",
            "name",
            "resolution_method",
            "resolution_confidence",
            "review_reason",
            "smiles_raw",
            "pubchem_cid",
            "sources",
            "n_measurements",
        ],
    ].sort_values("n_measurements", ascending=False)
    review = review.assign(sources=review["sources"].map("; ".join)).reset_index(drop=True)
    return DrugResolution(drugs, name_map, review)
