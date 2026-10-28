"""DDInter interactions restricted to the ingredients of the Tujya pill classes.

Input: the validated DDInter table written by `synergyrx fetch --source ddinter`
(`data/interim/ddinter/interactions.parquet`) and the ingredient names from RxNav. DDInter
names are matched to RxNorm ingredient names case-insensitively, after the aliases of
`config/ingredient_aliases.csv` (DDInter name -> RxNorm name, e.g. "Valproic acid" ->
"valproate") are applied.

DDInter's bulk files give the two drugs and a severity level only, with no mechanism or
management text, so `description` states what DDInter says and nothing more. Each pair is
stored once, alphabetically ordered; when DDInter lists a pair twice with different levels,
the most severe level is kept and `conflict` is true.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

DDINTER_URL = "https://ddinter2.scbdd.com/"
DDINTER_CITATION = (
    "Tian Y, et al. DDInter 2.0: an enhanced drug interaction resource with expanded data "
    "coverage, new interaction types, and improved user interface. Nucleic Acids Res. "
    "2025;53(D1):D1356-62. doi:10.1093/nar/gkae726"
)
SEVERITY_ORDER = {"Major": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}
COLUMNS = [
    "ingredient_a",
    "ingredient_b",
    "pair",
    "severity",
    "description",
    "conflict",
    "ddinter_id_a",
    "ddinter_id_b",
    "source",
    "source_url",
    "license",
]


def _norm(name: object) -> str:
    return " ".join(str(name).lower().split())


def read_aliases(path: Path) -> dict[str, str]:
    """{normalised DDInter name: RxNorm ingredient} from config/ingredient_aliases.csv."""
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype="string")
    return {
        _norm(d): _norm(r)
        for r, d in zip(table["rxnorm_ingredient"], table["ddinter_name"], strict=True)
    }


def ddinter_names(ddinter: pd.DataFrame, aliases: Mapping[str, str] | None = None) -> set[str]:
    aliases = aliases or {}
    names = pd.concat([ddinter["drug_a_name"], ddinter["drug_b_name"]])
    return {aliases.get(_norm(n), _norm(n)) for n in names}


def filter_interactions(
    ddinter: pd.DataFrame,
    ingredients: Iterable[str],
    *,
    both: bool = True,
    aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Interactions where both drugs (or, with both=False, at least one) are in `ingredients`."""
    wanted = {_norm(i) for i in ingredients}
    aliases = aliases or {}

    def name(value: object) -> str:
        return aliases.get(_norm(value), _norm(value))

    frame = pd.DataFrame(
        {
            "a": ddinter["drug_a_name"].map(name),
            "b": ddinter["drug_b_name"].map(name),
            "id_a": ddinter["drug_a_source_id"].astype("string"),
            "id_b": ddinter["drug_b_source_id"].astype("string"),
            "level": ddinter["level"].fillna("Unknown").astype("string").str.strip(),
        }
    )
    in_a, in_b = frame["a"].isin(wanted), frame["b"].isin(wanted)
    frame = frame[(in_a & in_b) if both else (in_a | in_b)]
    frame = frame[frame["a"] != frame["b"]]
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)

    swap = frame["a"] > frame["b"]
    ordered = pd.DataFrame(
        {
            "ingredient_a": frame["a"].where(~swap, frame["b"]),
            "ingredient_b": frame["b"].where(~swap, frame["a"]),
            "ddinter_id_a": frame["id_a"].where(~swap, frame["id_b"]),
            "ddinter_id_b": frame["id_b"].where(~swap, frame["id_a"]),
            "severity": frame["level"],
        }
    )
    ordered["rank"] = ordered["severity"].map(SEVERITY_ORDER).fillna(0)
    levels = ordered.groupby(["ingredient_a", "ingredient_b"])["severity"].nunique()
    best = (
        ordered.sort_values("rank", ascending=False)
        .drop_duplicates(["ingredient_a", "ingredient_b"])
        .drop(columns="rank")
        .sort_values(["ingredient_a", "ingredient_b"])
        .reset_index(drop=True)
    )
    best["conflict"] = [
        bool(levels[(a, b)] > 1)
        for a, b in zip(best["ingredient_a"], best["ingredient_b"], strict=True)
    ]
    best["pair"] = best["ingredient_a"] + " + " + best["ingredient_b"]
    best["description"] = [
        f"DDInter 2.0 lists an interaction between {a} and {b} with severity {s}."
        for a, b, s in zip(
            best["ingredient_a"], best["ingredient_b"], best["severity"], strict=True
        )
    ]
    best["source"] = "DDInter 2.0"
    best["source_url"] = DDINTER_URL
    best["license"] = "CC BY-NC-SA 4.0"
    return best[COLUMNS]
