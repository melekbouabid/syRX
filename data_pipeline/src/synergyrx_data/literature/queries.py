"""PubMed query builder for the active ingredients of the Tujya pill classes.

A query is the AND of one block per ingredient (name and synonyms in title/abstract, plus
MeSH and substance-name tags), a topic block (interactions, adverse effects) and filters.
Every query that is sent is appended to a JSON Lines log.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

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
    kind: str  # "ingredient" (SynergyRx also wrote "drug" and "pair")
    drug_ids: tuple[str, ...]
    drug_names: tuple[str, ...]
    pair_key: str | None  # always None for ingredients; kept so query logs keep one schema
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


def append_query_log(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
