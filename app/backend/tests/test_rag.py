"""RAG retrieval on SQLite with fake embeddings: index, BM25, dense, hybrid, filters."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.rag.embeddings import FakeEmbedder
from app.rag.index import index_chunks
from app.rag.retrieval import BM25, Retriever, RetrieverHolder, rrf, tokenize
from app.rag.store import ChunkStore, Filters
from tests.rag_fakes import CORPUS, DIM, build_index


@pytest.fixture
def store(tmp_path: Path) -> ChunkStore:
    return asyncio.run(build_index(f"sqlite:///{(tmp_path / 'rag.db').as_posix()}"))


def test_index_is_idempotent_and_follows_the_corpus(store: ChunkStore) -> None:
    embedder = FakeEmbedder(DIM)
    rows = [dict(r) for r in CORPUS]
    again = asyncio.run(index_chunks(store, embedder, rows))
    assert again["embedded"] == 0 and again["unchanged"] == len(CORPUS) and embedder.calls == 0
    first_version = store.get_meta()["indexed_at"]

    rows[0] = {**rows[0], "text": rows[0]["text"] + " Updated."}
    changed = asyncio.run(index_chunks(store, embedder, rows[:-1]))
    assert (changed["embedded"], changed["deleted"]) == (1, 1)
    assert len(store.hashes()) == len(CORPUS) - 1
    meta = store.get_meta()
    assert meta["model"] == "fake-hash" and meta["dim"] == str(DIM) and meta["chunks"] == "5"
    assert meta["indexed_at"] >= first_version


def test_another_embedding_model_rebuilds_the_table(store: ChunkStore) -> None:
    other = FakeEmbedder(DIM)
    other.model = "fake-other"
    stats = asyncio.run(index_chunks(store, other, [dict(r) for r in CORPUS]))
    assert stats["embedded"] == len(CORPUS)


def test_tokenize_and_bm25() -> None:
    assert tokenize("Les interactions du Fluconazole avec la warfarine") == [
        "interaction",
        "fluconazole",
        "warfarine",
    ]
    bm25 = BM25(["warfarin bleeding", "fluconazole warfarin", "losartan dizziness"])
    scores = bm25.scores("warfarin fluconazole")
    assert scores.argmax() == 1 and scores[2] == 0
    assert rrf([["a", "b"], ["b", "c"]])[0][0] == "b"


def test_three_modes_and_filters(store: ChunkStore) -> None:
    retriever = Retriever(store, FakeEmbedder(DIM))
    query = "fluconazole warfarin prothrombin"
    for mode in ("bm25", "dense", "hybrid"):
        hits = asyncio.run(retriever.search(query, k=3, mode=mode))  # type: ignore[arg-type]
        assert hits and hits[0].chunk.chunk_id in {"flu-di", "ddi-fw"}, mode
        assert all(not h.chunk.is_retracted for h in hits)
    hybrid = asyncio.run(retriever.search(query, k=6))
    assert any(h.bm25_rank and h.dense_rank for h in hybrid)

    only_labels = Filters(doc_types=("label",), ingredients=("warfarin",))
    hits = asyncio.run(retriever.search("warfarin", k=10, filters=only_labels))
    # "about or mentioning warfarin": the fluconazole label mentions it
    assert {h.chunk.chunk_id for h in hits} == {"war-ci", "flu-di"}
    with_retracted = Filters(ingredients=("warfarin",), exclude_retracted=False)
    ids = {
        h.chunk.chunk_id
        for h in asyncio.run(retriever.search("warfarin", k=10, filters=with_retracted))
    }
    assert "pm-retracted" in ids
    sections = Filters(sections=("contraindications",))
    assert [
        h.chunk.chunk_id for h in asyncio.run(retriever.search("pregnancy", filters=sections))
    ] == ["war-ci"]


def test_holder_reloads_after_reindexing(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'rag.db').as_posix()}"
    from app.db import make_engine

    empty = ChunkStore(make_engine(url), DIM)
    holder = RetrieverHolder(empty, FakeEmbedder(DIM))
    assert holder.get() is None  # no index yet
    asyncio.run(index_chunks(empty, FakeEmbedder(DIM), [dict(r) for r in CORPUS[:2]]))
    first = holder.get()
    assert first is not None and len(first.chunks) == 2
    assert holder.get() is first
    import time

    time.sleep(1.1)  # indexed_at has a one-second resolution
    asyncio.run(index_chunks(empty, FakeEmbedder(DIM), [dict(r) for r in CORPUS]))
    second = holder.get()
    assert second is not first and second is not None and len(second.chunks) == len(CORPUS)
