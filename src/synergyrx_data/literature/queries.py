"""PubMed / Europe PMC query builder for single drugs and drug pairs.

A query is the AND of: one block per drug (name and synonyms in title/abstract, plus MeSH
and substance-name tags), a synergy block and a cancer-context block. Every query that is
sent is appended to a JSON Lines log.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from synergyrx_data.config import Settings
from synergyrx_data.provenance import read_parquet

SYNERGY_TERMS = (
    '"Drug Synergism"[mh]',
    '"Drug Therapy, Combination"[mh]',
    '"Antineoplastic Combined Chemotherapy Protocols"[mh]',
    "synerg*[tiab]",
    "combination*[tiab]",
    "combined[tiab]",
)
CONTEXT_TERMS = (
    "Neoplasms[mh]",
    "cancer*[tiab]",
    "tumor*[tiab]",
    "tumour*[tiab]",
    "carcinoma*[tiab]",
    "leukemia*[tiab]",
)
_EPMC_SYNERGY = "(TITLE_ABS:synerg* OR TITLE_ABS:combination* OR TITLE_ABS:combined)"
_EPMC_CONTEXT = (
    "(TITLE_ABS:cancer* OR TITLE_ABS:tumor* OR TITLE_ABS:tumour* OR TITLE_ABS:carcinoma*)"
)
# Tujya: one query per active ingredient, about interactions and adverse effects.
TOPIC_TERMS = (
    '"Drug Interactions"[mh]',
    "interaction*[tiab]",
    '"adverse effects"[sh]',
    '"Drug-Related Side Effects and Adverse Reactions"[mh]',
    '"adverse effect*"[tiab]',
    '"adverse event*"[tiab]',
    '"adverse reaction*"[tiab]',
    "contraindicat*[tiab]",
)
TOPIC_FILTERS = ("hasabstract[text]", "english[la]")
_EPMC_TOPIC = (
    '(TITLE_ABS:interaction* OR TITLE_ABS:"adverse effect*" OR TITLE_ABS:"adverse event*" '
    'OR TITLE_ABS:"adverse reaction*" OR TITLE_ABS:contraindicat*)'
)
_CAS_NUMBER = re.compile(r"^\d{2,7}-\d{2}-\d$")
_UNSAFE = re.compile(r'["\[\]{}\\]')


@dataclass(frozen=True)
class Query:
    query_id: str
    kind: str  # "drug" or "pair"
    drug_ids: tuple[str, ...]
    drug_names: tuple[str, ...]
    pair_key: str | None
    pubmed: str
    europepmc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def usable_terms(name: str, synonyms: list[str], max_terms: int = 6) -> list[str]:
    """Search terms for one drug: its name first, then short, unambiguous synonyms."""
    terms: list[str] = []
    seen: set[str] = set()
    for position, raw in enumerate([name, *synonyms]):
        term = _UNSAFE.sub(" ", str(raw)).strip()
        term = re.sub(r"\s+", " ", term)
        key = term.lower()
        if not term or key in seen:
            continue
        is_name = position == 0
        too_vague = len(term) < 3 or term.isdigit()
        looks_like_registry = bool(_CAS_NUMBER.match(term)) or len(term) > 60
        if not is_name and (too_vague or looks_like_registry):
            continue
        seen.add(key)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    return terms


def drug_block(name: str, synonyms: list[str], max_terms: int = 6) -> str:
    terms = usable_terms(name, synonyms, max_terms)
    parts = [f'"{t}"[tiab]' for t in terms]
    primary = terms[0] if terms else name
    parts += [f'"{primary}"[mh]', f'"{primary}"[nm]']
    return "(" + " OR ".join(parts) + ")"


def _epmc_block(name: str, synonyms: list[str], max_terms: int = 6) -> str:
    terms = usable_terms(name, synonyms, max_terms)
    return "(" + " OR ".join(f'TITLE_ABS:"{t}"' for t in terms) + ")"


def _query_id(kind: str, drug_ids: tuple[str, ...]) -> str:
    return hashlib.sha1("|".join((kind, *drug_ids)).encode(), usedforsecurity=False).hexdigest()[
        :12
    ]


def build_query(
    drugs: list[tuple[str, str, list[str]]],
    *,
    pair_key: str | None = None,
    synergy_terms: tuple[str, ...] = SYNERGY_TERMS,
    context_terms: tuple[str, ...] = CONTEXT_TERMS,
    max_terms: int = 6,
) -> Query:
    """Build one query from `(drug_id, name, synonyms)` tuples (one drug, or a pair)."""
    kind = "pair" if len(drugs) == 2 else "drug"
    blocks = [drug_block(name, synonyms, max_terms) for _, name, synonyms in drugs]
    pubmed = " AND ".join(
        [*blocks, "(" + " OR ".join(synergy_terms) + ")", "(" + " OR ".join(context_terms) + ")"]
    )
    epmc_blocks = [_epmc_block(name, synonyms, max_terms) for _, name, synonyms in drugs]
    europepmc = " AND ".join([*epmc_blocks, _EPMC_SYNERGY, _EPMC_CONTEXT, "SRC:MED"])
    ids = tuple(d[0] for d in drugs)
    return Query(
        query_id=_query_id(kind, ids),
        kind=kind,
        drug_ids=ids,
        drug_names=tuple(d[1] for d in drugs),
        pair_key=pair_key,
        pubmed=pubmed,
        europepmc=europepmc,
    )


def build_queries(
    drugs: pd.DataFrame,
    pairs: pd.DataFrame,
    *,
    mode: str = "pairs",
    max_queries: int = 25,
    max_terms: int = 6,
) -> list[Query]:
    """Queries for the most-measured pairs (or drugs), so the first hits matter most."""
    if drugs.empty or pairs.empty:
        return []
    info = {
        row.drug_id: (
            row.drug_id,
            row.name,
            # Read back from Parquet the list is an array, so it cannot be tested with `or`.
            [str(s) for s in (row.synonyms if row.synonyms is not None else [])],
        )
        for row in drugs[["drug_id", "name", "synonyms"]].itertuples(index=False)
    }
    queries: list[Query] = []
    if mode == "pairs":
        ranked = (
            pairs.groupby(["pair_key", "drug_a_id", "drug_b_id"])["cell_line_id"]
            .nunique()
            .reset_index(name="n")
            .sort_values(["n", "pair_key"], ascending=[False, True])
        )
        for row in ranked.head(max_queries).itertuples(index=False):
            if row.drug_a_id in info and row.drug_b_id in info:
                queries.append(
                    build_query(
                        [info[row.drug_a_id], info[row.drug_b_id]],
                        pair_key=row.pair_key,
                        max_terms=max_terms,
                    )
                )
    elif mode == "drugs":
        counts = pd.concat([pairs["drug_a_id"], pairs["drug_b_id"]]).value_counts()
        ranked_ids = sorted(counts.index, key=lambda d: (-int(counts[d]), str(d)))
        for drug_id in ranked_ids[:max_queries]:
            if drug_id in info:
                queries.append(build_query([info[drug_id]], max_terms=max_terms))
    else:
        raise ValueError(f"unknown query mode {mode!r} (use 'pairs' or 'drugs')")
    return queries


def build_topic_query(name: str, synonyms: list[str], *, max_terms: int = 6) -> Query:
    """One ingredient AND (interactions OR adverse effects), English abstracts only."""
    block = drug_block(name, synonyms, max_terms)
    topic = "(" + " OR ".join(TOPIC_TERMS) + ")"
    pubmed = " AND ".join([block, topic, *TOPIC_FILTERS])
    europepmc = " AND ".join([_epmc_block(name, synonyms, max_terms), _EPMC_TOPIC, "SRC:MED"])
    ids = (f"ingredient:{name}",)
    return Query(
        query_id=_query_id("ingredient", ids),
        kind="ingredient",
        drug_ids=ids,
        drug_names=(name,),
        pair_key=None,
        pubmed=pubmed,
        europepmc=europepmc,
    )


def read_ingredients(path: Path) -> list[str]:
    """Distinct active ingredients of `config/pill_classes.csv` (column `active_ingredients`)."""
    if not path.is_file():
        return []
    frame = pd.read_csv(path, dtype="string", usecols=["active_ingredients"])
    names = {
        part.strip().lower()
        for value in frame["active_ingredients"].dropna()
        for part in str(value).split(";")
        if part.strip()
    }
    return sorted(names)


def build_ingredient_queries(
    names: list[str],
    aliases: dict[str, list[str]] | None = None,
    *,
    max_queries: int = 100,
    max_terms: int = 6,
) -> list[Query]:
    aliases = aliases or {}
    return [
        build_topic_query(name, aliases.get(name, []), max_terms=max_terms)
        for name in names[:max_queries]
    ]


def load_query_inputs(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The processed drug and pair tables, or empty frames before `clean` has run."""
    drugs = read_parquet(settings.parquet_root / "drugs.parquet")
    pairs = read_parquet(
        settings.parquet_root / "synergy_pairs.parquet",
        ["pair_key", "drug_a_id", "drug_b_id", "cell_line_id"],
    )
    return (
        drugs if drugs is not None else pd.DataFrame(),
        pairs if pairs is not None else pd.DataFrame(),
    )


def append_query_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
