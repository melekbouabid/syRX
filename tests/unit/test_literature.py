"""Query builder, sentence segmentation, mention detection and the annotation sample."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from synergyrx_data.literature.abstracts import (
    articles_frame,
    build_abstract_store,
    classify_article_type,
)
from synergyrx_data.literature.annotation_export import (
    export_annotation_sample,
    stratified_sample,
)
from synergyrx_data.literature.mentions import SynonymIndex, find_candidate_sentences
from synergyrx_data.literature.queries import (
    append_query_log,
    build_queries,
    build_query,
    usable_terms,
)
from synergyrx_data.literature.sentences import split_sentences
from synergyrx_data.provenance import PROVENANCE_COLUMNS

PROVENANCE = {
    "source": "pubmed",
    "source_url": "https://pubmed.ncbi.nlm.nih.gov/1/",
    "retrieved_at": "2026-01-01T00:00:00Z",
    "source_version": "current",
    "license": "NLM",
}
DRUGS = pd.DataFrame(
    {
        "drug_id": ["FU", "VELI", "DASA", "GOLD"],
        "name": ["Fluorouracil", "Veliparib", "Dasatinib", "Auranofin"],
        "synonyms": [
            ["5-FU", "5-fluorouracil", "51-21-8", "Adrucil"],
            ["ABT-888", "ABT 888"],
            ["BMS-354825", "Sprycel"],
            ["gold", "Ridaura", "SKF"],
        ],
    }
)

# ---------------------------------------------------------------- sentences


def test_sentences_respect_abbreviations_numbers_and_initials() -> None:
    text = (
        "Veliparib (ABT-888) was combined with 5-FU, e.g. at 0.5 mg/kg vs. placebo. "
        "As shown by Smith et al. in Fig. 2, the effect was synergistic (p < 0.01). "
        "J. Doe confirmed this! Was it reproducible? Yes."
    )
    sentences = [s.text for s in split_sentences(text)]
    assert sentences == [
        "Veliparib (ABT-888) was combined with 5-FU, e.g. at 0.5 mg/kg vs. placebo.",
        "As shown by Smith et al. in Fig. 2, the effect was synergistic (p < 0.01).",
        "J. Doe confirmed this!",
        "Was it reproducible?",
        "Yes.",
    ]


def test_sentence_offsets_point_back_into_the_text_and_sections_split() -> None:
    text = "BACKGROUND: First one. Second one.\nRESULTS: Third one."
    sentences = split_sentences(text)
    assert [s.text for s in sentences] == [
        "BACKGROUND: First one.",
        "Second one.",
        "RESULTS: Third one.",
    ]
    assert all(text[s.start : s.end] == s.text for s in sentences)
    assert [s.index for s in sentences] == [0, 1, 2]
    assert split_sentences(None) == [] and split_sentences("   ") == []


# ------------------------------------------------------------------ queries


def test_search_terms_drop_registry_numbers_and_vague_synonyms() -> None:
    synonyms = ["5-FU", "5-fu", "51-21-8", "FU", "12345", "x" * 80, 'Fluoro"uracil[x]']
    terms = usable_terms("Fluorouracil", synonyms)
    assert terms == ["Fluorouracil", "5-FU", "Fluoro uracil x"]


def test_pair_query_structure_for_pubmed_and_europepmc() -> None:
    query = build_query(
        [("FU", "Fluorouracil", ["5-FU"]), ("VELI", "Veliparib", ["ABT-888"])], pair_key="FU|VELI"
    )
    assert query.kind == "pair" and query.pair_key == "FU|VELI"
    blocks = query.pubmed.split(" AND ")
    assert len(blocks) == 4  # drug A, drug B, synergy, cancer context
    assert (
        blocks[0]
        == '("Fluorouracil"[tiab] OR "5-FU"[tiab] OR "Fluorouracil"[mh] OR "Fluorouracil"[nm])'
    )
    assert '"ABT-888"[tiab]' in blocks[1]
    assert '"Drug Synergism"[mh]' in blocks[2] and "synerg*[tiab]" in blocks[2]
    assert "Neoplasms[mh]" in blocks[3]
    assert query.europepmc.startswith('(TITLE_ABS:"Fluorouracil" OR TITLE_ABS:"5-FU") AND ')
    assert query.europepmc.endswith("SRC:MED")
    # The id only depends on the drugs, so a re-run finds its previous state.
    assert build_query([("FU", "x", []), ("VELI", "y", [])]).query_id == query.query_id
    assert build_query([("FU", "Fluorouracil", [])]).kind == "drug"


def test_queries_start_with_the_most_measured_pairs() -> None:
    pairs = pd.DataFrame(
        {
            "pair_key": ["FU|VELI"] * 3 + ["DASA|FU"],
            "drug_a_id": ["FU"] * 3 + ["DASA"],
            "drug_b_id": ["VELI"] * 3 + ["FU"],
            "cell_line_id": ["c1", "c2", "c3", "c1"],
        }
    )
    queries = build_queries(DRUGS, pairs, mode="pairs", max_queries=5)
    assert [q.pair_key for q in queries] == ["FU|VELI", "DASA|FU"]
    assert [q.drug_names[0] for q in build_queries(DRUGS, pairs, mode="drugs", max_queries=1)] == [
        "Fluorouracil"
    ]
    assert build_queries(DRUGS, pairs.iloc[:0]) == []
    with pytest.raises(ValueError, match="unknown query mode"):
        build_queries(DRUGS, pairs, mode="everything")


def test_every_query_is_logged(tmp_path: Path) -> None:
    log = tmp_path / "interim" / "pubmed" / "query_log.jsonl"
    append_query_log(log, {"query_id": "a", "query": "x[tiab]", "count": 3})
    append_query_log(log, {"query_id": "b", "query": "y[tiab]", "count": 0})
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert [entry["query_id"] for entry in lines] == ["a", "b"]


# ----------------------------------------------------------------- mentions


def test_synonym_index_matches_spelling_variants_offline() -> None:
    index = SynonymIndex.from_drugs(DRUGS)
    text = "abt888 plus 5-fluorouracil, or ABT 888 with Adrucil; veliparib-treated cells"
    found = index.find(text)
    assert [(drug, surface) for drug, surface, _, _ in found] == [
        ("VELI", "abt888"),
        ("FU", "5-fluorouracil"),
        ("VELI", "ABT 888"),
        ("FU", "Adrucil"),
        ("VELI", "veliparib"),
    ]
    assert all(text[start:end] == surface for _, surface, start, end in found)


def test_ambiguous_and_too_short_synonyms_are_not_matched() -> None:
    index = SynonymIndex.from_drugs(DRUGS)
    assert index.find("The gold standard of care is SKF.") == []  # "gold" is a word, SKF too short
    assert index.dropped_ambiguous >= 1 and index.dropped_short >= 1
    # A term claimed by two different drugs identifies neither.
    clash = pd.DataFrame(
        {
            "drug_id": ["A", "B"],
            "name": ["Alphamab", "Betamab"],
            "synonyms": [["XY-100"], ["XY-100"]],
        }
    )
    assert SynonymIndex.from_drugs(clash).find("XY-100 was given") == []


def abstracts_table() -> pd.DataFrame:
    records = [
        {
            "pmid": "1",
            "abstract": "Veliparib and 5-FU were synergistic in A2058 cells. Dasatinib alone did "
            "nothing. We also tested ABT-888, Sprycel and fluorouracil together.",
            "year": 2019,
            "article_type": "research",
            "language": "eng",
            "is_retracted": False,
        },
        {
            "pmid": "2",
            "abstract": None,
            "year": 2015,
            "article_type": "review",
            "language": "chi",
            "is_retracted": True,
        },
    ]
    return articles_frame([{**r, **PROVENANCE} for r in records])


def test_candidate_sentences_need_two_different_drugs() -> None:
    index = SynonymIndex.from_drugs(DRUGS)
    candidates, stats = find_candidate_sentences(abstracts_table(), index)
    assert stats == {
        "abstracts": 1,
        "sentences": 3,
        "sentences_with_a_drug": 3,
        "candidate_sentences": 2,
    }
    first, second = candidates.iloc[0], candidates.iloc[1]
    assert first.sentence == "Veliparib and 5-FU were synergistic in A2058 cells."
    assert list(first.drug_ids) == ["FU", "VELI"] and list(first.pair_keys) == ["FU|VELI"]
    assert first.has_synergy_cue and list(first.cue_terms) == ["synergistic"]
    assert list(first.drug_names) == ["Fluorouracil", "Veliparib"]
    assert second.n_drugs == 3 and len(second.pair_keys) == 3
    assert not second.has_synergy_cue
    assert all(c in candidates.columns for c in PROVENANCE_COLUMNS)
    assert set(candidates["pmid"]) == {"1"}


# ------------------------------------------------------------ abstract store


def test_abstract_store_merges_open_access_and_fills_missing_abstracts() -> None:
    older = {**PROVENANCE, "retrieved_at": "2025-01-01T00:00:00Z"}
    articles = pd.concat(
        [
            abstracts_table(),
            articles_frame(
                [{"pmid": "1", "abstract": "Old copy.", "is_retracted": False, **older}]
            ),
        ],
        ignore_index=True,
    )
    europepmc = pd.DataFrame(
        {
            "pmid": ["1", "2"],
            "pmcid": ["PMC1", None],
            "abstract": ["ignored: PubMed has one", "Abstract only Europe PMC has. Two sentences."],
            "is_open_access": [True, False],
            "in_epmc": [True, True],
        }
    )
    store = build_abstract_store(articles, europepmc).set_index("pmid")
    assert len(store) == 2  # the duplicate PMID collapsed onto its newest copy
    assert store.loc["1", "abstract"].startswith("Veliparib and 5-FU")
    assert store.loc["1", "abstract_source"] == "pubmed" and store.loc["1", "n_sentences"] == 3
    assert bool(store.loc["1", "is_open_access"]) and store.loc["1", "pmcid"] == "PMC1"
    assert store.loc["2", "abstract_source"] == "europepmc" and store.loc["2", "n_sentences"] == 2
    assert bool(store.loc["2", "has_abstract"]) and bool(store.loc["2", "is_retracted"])
    assert store.loc["2", "pubmed_url"] == "https://pubmed.ncbi.nlm.nih.gov/2/"
    assert list(store.columns[-5:]) == PROVENANCE_COLUMNS


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


# -------------------------------------------------------- annotation sample


def many_candidates(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pmid": [str(1000 + i // 4) for i in range(n)],  # four sentences per abstract
            "sentence_index": [i % 4 for i in range(n)],
            "sentence": [f"Sentence {i} about two drugs." for i in range(n)],
            "n_drugs": [2 if i % 5 else 3 for i in range(n)],
            "drug_names": [["Fluorouracil", "Veliparib"]] * n,
            "has_synergy_cue": [i % 3 == 0 for i in range(n)],
            "cue_terms": [["synergistic"] if i % 3 == 0 else [] for i in range(n)],
            "article_type": ["review" if i % 7 == 0 else "research" for i in range(n)],
            "year": [2000 + i % 25 for i in range(n)],
            "is_retracted": [False] * n,
        }
    )


def test_annotation_sample_is_stratified_bounded_and_reproducible() -> None:
    candidates = many_candidates(4000)
    sample = stratified_sample(candidates, size=300, max_per_pmid=3, seed=7)
    assert len(sample) == 300
    assert (
        sample["sample_id"].is_unique and sample[["pmid", "sentence_index"]].duplicated().sum() == 0
    )
    assert sample.groupby("pmid").size().max() <= 3
    # Every stratum present in the pool is represented, including the rare ones.
    pool_strata = {
        f"{'cue' if c else 'nocue'}/{'2drugs' if n == 2 else '3plus'}/"
        f"{'review' if t == 'review' else 'primary'}"
        for c, n, t in zip(
            candidates["has_synergy_cue"],
            candidates["n_drugs"],
            candidates["article_type"],
            strict=True,
        )
    }
    assert set(sample["stratum"]) == pool_strata
    assert sample["stratum"].value_counts().min() >= 5
    assert (sample["label"] == "").all()  # labels are for a person to fill in
    assert stratified_sample(candidates, size=300, max_per_pmid=3, seed=7).equals(sample)
    assert not stratified_sample(candidates, size=300, max_per_pmid=3, seed=8).equals(sample)


def test_annotation_sample_is_never_padded(tmp_path: Path) -> None:
    few = many_candidates(40)
    sample = stratified_sample(few, size=300, max_per_pmid=3)
    assert len(sample) == 30  # ten abstracts, at most three sentences each: no repeats added
    paths = export_annotation_sample(
        sample, csv_dir=tmp_path / "csv", json_dir=tmp_path / "json", guide_dir=tmp_path / "docs"
    )
    assert [p.parent.name for p in paths.values()] == ["csv", "json", "docs"]
    csv = pd.read_csv(paths["csv"])
    assert len(csv) == 30 and {"label", "annotator", "comment", "pubmed_url"} <= set(csv.columns)
    lines = paths["jsonl"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 30 and json.loads(lines[0])["sample_id"] == "S0001"
    assert "no_relation" in paths["guide"].read_text(encoding="utf-8")
    assert stratified_sample(few.iloc[:0]).empty
