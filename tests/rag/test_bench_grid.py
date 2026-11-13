"""Retrieval grid of step 3.3: corpus variants, relevance of their chunks, metrics, selection
of the agent's setting (src/benchmark/rechunk.py, src/benchmark/bench_grid.py)."""

from __future__ import annotations

import asyncio
import json
import math
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.benchmark import bench_grid as G
from src.benchmark import rechunk as R
from src.data.corpus import corpus

SENTENCES = [f"Sentence number {i} of the warnings about warfarin and aspirin." for i in range(12)]


def labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ingredient": "warfarin",
                "label_id": "L1",
                "set_id": "S1",
                "label_version": 3,
                "effective_time": "20260101",
                "product_type": "HUMAN PRESCRIPTION DRUG",
                "brand_name": "Coumadin",
                "generic_name": "warfarin",
                "manufacturer": "M",
                "substances": "warfarin",
                "label_url": "https://example.org/l1",
                "section": "warnings",
                "text": " ".join(SENTENCES),
                "source": "openfda",
                "source_url": "https://example.org",
                "retrieved_at": "2026-10-28",
                "source_version": "1",
                "license": "CC0",
            }
        ]
    )


def articles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pmid": "123",
                "title": "A case",
                "abstract": "RESULTS: Bleeding occurred with warfarin and aspirin together.",
                "journal": "J",
                "year": 2020,
                "language": "eng",
                "article_type": "case_report",
                "is_retracted": False,
                "retrieved_at": "2026-10-28",
                "license": "abstract",
            }
        ]
    )


def served(max_chars: int = 200) -> pd.DataFrame:
    names = {"warfarin", "aspirin"}
    original = corpus.split_text
    try:
        cut = partial(original, max_chars=max_chars, overlap=1)
        corpus.split_text = cut  # type: ignore[assignment]
        rows = corpus.label_chunks(labels(), names)
        rows += corpus.abstract_chunks(articles(), {"123": ["warfarin"]}, names)
    finally:
        corpus.split_text = original
    pair = pd.DataFrame(
        [
            {
                "ingredient_a": "aspirin",
                "ingredient_b": "warfarin",
                "severity": "Major",
                "ddinter_id_a": "A",
                "ddinter_id_b": "B",
                "source_url": "u",
                "license": "l",
            }
        ]
    )
    rows += corpus.interaction_chunks(pair, "2026")
    return pd.DataFrame(rows, columns=corpus.CHUNK_COLUMNS)


def test_variant_with_the_served_settings_gives_back_the_served_corpus() -> None:
    base = served(1500)
    again = R.variant(base, labels(), articles(), 1500, 1)
    assert R.same_corpus(base, again)
    smaller = R.variant(base, labels(), articles(), 200, 1)
    assert len(smaller) > len(base)
    assert set(smaller.loc[smaller["doc_type"] == "interaction", "chunk_id"]) == set(
        base.loc[base["doc_type"] == "interaction", "chunk_id"]
    )


def test_relevance_maps_an_expected_chunk_to_the_pieces_cut_from_it() -> None:
    base = served(1500)
    whole = base[base["doc_type"] == "label"]["chunk_id"].iloc[0]
    question = {"type": "warnings", "ingredients": ["warfarin"], "expected_chunks": [whole]}
    assert R.Relevance(base, base).expected(question) == {whole}
    small = R.variant(base, labels(), articles(), 200, 1)
    pieces = set(small.loc[small["doc_type"] == "label", "chunk_id"])
    assert R.Relevance(base, small).expected(question) == pieces  # every piece is inside it


def test_a_neighbour_sharing_only_the_overlap_sentence_is_not_relevant() -> None:
    base = served(200)
    first = base[base["doc_type"] == "label"].iloc[0]
    question = {
        "type": "warnings",
        "ingredients": ["warfarin"],
        "expected_chunks": [first.chunk_id],
    }
    assert R.Relevance(base, base).expected(question) == {first.chunk_id}
    a, b = R.sentences(SENTENCES[0] + " " + SENTENCES[1]), R.sentences(SENTENCES[1])
    assert R.shared(a, b) == 1.0 and R.shared(a, R.sentences(SENTENCES[5])) == 0.0


def test_an_interaction_piece_must_name_the_other_ingredient() -> None:
    piece = R.Piece("c", "d", "warnings", "label", ("warfarin",), (), frozenset())
    question = {"type": "interaction", "ingredients": ["warfarin", "aspirin"]}
    assert not R.Relevance._names_other(question, piece)
    named = R.Piece("c", "d", "warnings", "label", ("warfarin",), ("aspirin",), frozenset())
    assert R.Relevance._names_other(question, named)
    assert R.Relevance._names_other({"type": "warnings"}, piece)


def test_scores_recall_mrr_and_ndcg() -> None:
    s = G.score(["x", "a", "y", "b"], {"a", "b"})
    assert s["recall_at_1"] == 0 and s["recall_at_3"] == 1 and s["mrr"] == 0.5
    ideal = 1 + 1 / math.log2(3)
    assert s["ndcg_at_10"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)
    assert s["coverage_at_10"] == 1.0
    assert G.score(["a"], {"a"})["ndcg_at_10"] == 1.0
    assert G.score(["x"], {"a"})["mrr"] == 0.0


def test_config_names_and_collections() -> None:
    served_config = G.Config("ollama:bge-m3", mode="hybrid", rewrite=True)
    assert served_config.name == "bge-m3-hybrid-rewrite" and served_config.collection == ""
    chunked = G.Config("ollama:bge-m3", chunk="800/1", mode="dense")
    assert chunked.name == "bge-m3-dense-c800-o1"
    assert chunked.collection == "tujya_bench_bge-m3-c800-o1"
    reranked = G.Config(
        "ollama:bge-m3", mode="hybrid", rerank="Xenova/ms-marco-MiniLM-L-6-v2", fetch_k=20
    )
    assert reranked.name == "bge-m3-hybrid-rerank-minilm-l-6-v2-f20"
    assert G.Config("st:multilingual-e5-base").collection == "tujya_bench_multilingual-e5-base"


def test_the_grid_file_is_valid_and_names_are_unique() -> None:
    selection, groups = G.load_grid()
    names = [c.name for items in groups.values() for c in items]
    assert len(names) == len(set(names))
    assert selection["metric"] == "mrr" and selection["latency_budget_ms"] > 0
    assert {"bge-m3", "rerankers", "chunking", "bge-m3-finetuned"} <= set(groups)
    assert all(c.mode in ("dense", "hybrid", "bm25") for items in groups.values() for c in items)
    assert all(bool(c.rerank) == bool(c.fetch_k) for items in groups.values() for c in items)


def test_a_reranked_configuration_reading_fewer_than_k_candidates_is_refused(
    tmp_path: Path,
) -> None:
    grid = tmp_path / "grid.yaml"
    grid.write_text(
        "selection: {metric: mrr}\ngroups:\n  g:\n"
        "    - {embedding: 'ollama:bge-m3', rerank: BAAI/bge-reranker-base, fetch_k: 5}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fetch_k 5"):
        G.load_grid(grid)
    grid.write_text(grid.read_text(encoding="utf-8").replace("fetch_k: 5", "fetch_k: 10"))
    assert G.load_grid(grid)[1]["g"][0].fetch_k == 10


def test_selection_keeps_the_best_metric_within_the_latency_budget() -> None:
    results = {
        "slow-best": {"mrr": 0.9, "ms_per_question": 15000},
        "fast-good": {"mrr": 0.86, "ms_per_question": 900, "latency_d96_ms": 1100},
        "fast-ok": {"mrr": 0.83, "ms_per_question": 300},
        "measured-slow": {"mrr": 0.88, "ms_per_question": 900, "latency_d96_ms": 2500},
    }
    assert G.choose(results, {"metric": "mrr", "latency_budget_ms": 2000}) == [
        "fast-good",
        "fast-ok",
    ]


def test_another_chunking_is_chosen_only_when_its_gain_is_significant() -> None:
    def record(mrr: float, chars: int) -> dict[str, float]:
        return {
            "mrr": mrr,
            "ms_per_question": 500,
            "chunk_max_chars": chars,
            "chunk_overlap_sentences": 1,
        }

    results = {"served": record(0.82, 1500), "c800": record(0.87, 800)}
    selection = {"metric": "mrr", "latency_budget_ms": 2000, "rebuild_needs_significance": True}
    assert G.choose(results, selection) == ["served"]
    assert G.choose(results, selection, {"c800": False}) == ["served"]
    assert G.choose(results, selection, {"c800": True}) == ["c800", "served"]
    assert G.choose(results, {**selection, "rebuild_needs_significance": False}) == [
        "c800",
        "served",
    ]


def test_paired_bootstrap_interval() -> None:
    better = [1.0] * 40 + [0.5] * 10
    worse = [0.5] * 40 + [0.5] * 10
    diff, low, high = G.paired_bootstrap(better, worse)
    assert diff == pytest.approx(0.4) and 0 < low <= diff <= high
    same = [1.0, 0.0] * 25
    diff, low, high = G.paired_bootstrap(same, list(reversed(same)))
    assert diff == 0 and low < 0 < high
    with pytest.raises(ValueError):
        G.paired_bootstrap([1.0], [1.0, 0.0])


def test_precomputed_embedder_returns_chunk_and_query_vectors(tmp_path: Path) -> None:
    path = tmp_path / "m.npz"
    np.savez(
        path,
        chunk_ids=np.array(["c1", "c2"]),
        chunks=np.eye(2, dtype=np.float32),
        queries=np.array(["q?"]),
        query_vectors=np.array([[0.6, 0.8]], dtype=np.float32),
        meta=np.array(json.dumps({"slug": "m", "dim": 2, "query_cpu_ms": 5.0})),
    )
    embedder = G.PrecomputedEmbedder(path, {"c1": "T\none", "c2": "T\ntwo"})
    assert embedder.dim == 2 and embedder.model == "m"
    vectors = asyncio.run(embedder.embed(["T\ntwo", "q?"]))
    assert vectors == [[0.0, 1.0], [pytest.approx(0.6), pytest.approx(0.8)]]
    with pytest.raises(KeyError):
        asyncio.run(embedder.embed(["unknown"]))


def test_rows_of_matches_the_indexer_reading() -> None:
    rows = G.rows_of(served(1500))
    assert all(isinstance(r["ingredients"], list) for r in rows)
    assert all(isinstance(r["chunk_index"], int) and isinstance(r["date"], str) for r in rows)


def test_only_an_embedding_the_api_computes_can_be_chosen() -> None:
    results = {
        "e5": {"mrr": 0.9, "ms_per_question": 100, "embedding_backend": "st"},
        "bge": {"mrr": 0.8, "ms_per_question": 100, "embedding_backend": "ollama"},
    }
    selection = {"metric": "mrr", "latency_budget_ms": 2000}
    assert G.choose(results, selection) == ["e5", "bge"]
    assert G.choose(results, {**selection, "servable_only": True}) == ["bge"]


def test_agent_settings_of_a_configuration() -> None:
    config = G.Config("ollama:bge-m3", mode="hybrid", rewrite=True, rerank="m/r", fetch_k=8)
    assert G.agent_settings(config.params()) == {
        "embedding_model": "bge-m3",
        "rag_search_mode": "hybrid",
        "rag_query_rewrite": True,
        "rag_rerank": True,
        "rag_rerank_model": "m/r",
        "rag_fetch_k": 8,
    }
    assert G.agent_settings(G.Config("ollama:bge-m3").params())["rag_rerank"] is False
    with pytest.raises(ValueError):
        G.agent_settings(G.Config("st:e5").params())


@pytest.mark.skipif(not G.CHAMPION.is_file(), reason="no registered champion yet")
def test_the_agent_defaults_are_the_registered_champion() -> None:
    from app.config import Settings

    champion = json.loads(G.CHAMPION.read_text(encoding="utf-8"))
    fields = Settings.model_fields
    for key, value in champion["agent_settings"].items():
        assert fields[key].default == value, key
