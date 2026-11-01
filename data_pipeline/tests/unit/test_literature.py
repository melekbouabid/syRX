"""PubMed query builder for the Tujya ingredients and PubMed article classification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synergyrx_data.literature.abstracts import classify_article_type
from synergyrx_data.literature.queries import (
    append_query_log,
    build_ingredient_queries,
    build_topic_query,
    usable_terms,
)

# ----------------------------------------------------------------- queries


def test_search_terms_drop_registry_numbers_and_vague_synonyms() -> None:
    synonyms = ["5-FU", "5-fu", "51-21-8", "FU", "12345", "x" * 80, 'Fluoro"uracil[x]']
    terms = usable_terms("Fluorouracil", synonyms)
    assert terms == ["Fluorouracil", "5-FU", "Fluoro uracil x"]


def test_ingredient_query_structure_for_pubmed_and_europepmc() -> None:
    query = build_topic_query("Fluorouracil", ["5-FU"])
    assert query.kind == "ingredient" and query.pair_key is None
    blocks = query.pubmed.split(" AND ")
    assert len(blocks) == 4  # ingredient, topic, has an abstract, English
    assert (
        blocks[0]
        == '("Fluorouracil"[tiab] OR "5-FU"[tiab] OR "Fluorouracil"[mh] OR "Fluorouracil"[nm])'
    )
    assert '"Drug Interactions"[mh]' in blocks[1] and '"adverse event*"[tiab]' in blocks[1]
    assert blocks[2:] == ["hasabstract[text]", "english[la]"]
    assert query.europepmc.startswith('(TITLE_ABS:"Fluorouracil" OR TITLE_ABS:"5-FU") AND ')
    assert query.europepmc.endswith("SRC:MED")
    # The id only depends on the ingredient, so a re-run finds its previous state.
    assert build_topic_query("Fluorouracil", []).query_id == query.query_id
    assert build_topic_query("Veliparib", []).query_id != query.query_id


def test_ingredient_queries_keep_the_order_the_bound_and_the_aliases() -> None:
    queries = build_ingredient_queries(
        ["fluorouracil", "veliparib", "warfarin"], {"veliparib": ["ABT-888"]}, max_queries=2
    )
    assert [q.drug_names for q in queries] == [("fluorouracil",), ("veliparib",)]
    assert '"ABT-888"[tiab]' in queries[1].pubmed and "ABT-888" not in queries[0].pubmed
    assert build_ingredient_queries([]) == []


def test_every_query_is_logged(tmp_path: Path) -> None:
    log = tmp_path / "interim" / "pubmed" / "query_log.jsonl"
    append_query_log(log, {"query_id": "a", "query": "x[tiab]", "count": 3})
    append_query_log(log, {"query_id": "b", "query": "y[tiab]", "count": 0})
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [entry["query_id"] for entry in lines] == ["a", "b"]


# ---------------------------------------------------------------- articles


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        (["Journal Article"], "research"),
        (["Journal Article", "Review"], "review"),
        (["Systematic Review", "Review"], "meta_analysis"),
        (["Randomized Controlled Trial"], "clinical_trial"),
        (["Retraction of Publication"], "retraction_notice"),
        (["Letter", "Comment"], "commentary"),
        ([], "research"),
    ],
)
def test_article_type_classification(types: list[str], expected: str) -> None:
    assert classify_article_type(types) == expected
