"""The chunk index in Qdrant (D113): idempotent indexing, payload, filters, copy from SQL.

Runs on the client's local in-memory mode (same API as the server, no network). The same
tests run against a real server when TUJYA_TEST_QDRANT_URL is set (the CI job `src` has a
Qdrant service); they use their own collection, so the real index is never touched.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.db import make_engine
from src.rag.embeddings import FakeEmbedder
from src.rag.index import copy_index, index_chunks
from src.rag.qdrant_store import DENSE, QdrantStore, point_id, to_filter
from src.rag.retrieval import Retriever, RetrieverHolder
from src.rag.store import METADATA_COLUMNS, ChunkStore, Filters
from tujya_testing.rag_fakes import CORPUS, DIM, build_index

SERVER = os.environ.get("TUJYA_TEST_QDRANT_URL")
TARGETS = [":memory:"] + ([SERVER] if SERVER else [])


@pytest.fixture(params=TARGETS, ids=lambda t: "memory" if t == ":memory:" else "server")
def empty(request: pytest.FixtureRequest) -> Iterator[QdrantStore]:
    store = QdrantStore.connect(request.param, DIM, "tujya_test_chunks")
    store.drop()
    yield store
    store.drop()


@pytest.fixture
def store(empty: QdrantStore) -> QdrantStore:
    asyncio.run(index_chunks(empty, FakeEmbedder(DIM), [dict(r) for r in CORPUS]))
    return empty


def test_collection_payload_and_indexes(store: QdrantStore) -> None:
    info = store.info()
    assert info["points"] == len(CORPUS)
    if store.url != ":memory:":  # the local mode has no payload indexes
        indexed = set(info["payload_indexes"])
        assert {"doc_type", "ingredients", "is_retracted", "language"} <= indexed
    (record,) = store.client.retrieve(
        store.collection, [point_id("war-ci")], with_payload=True, with_vectors=True
    )
    payload = record.payload or {}
    assert set(METADATA_COLUMNS) <= set(payload)
    assert payload["chunk_id"] == "war-ci" and payload["embedding_model"] == "fake-hash"
    assert re.fullmatch(r"[0-9a-f]{64}", payload["content_hash"])  # SHA-256
    assert isinstance(record.vector, dict) and len(record.vector[DENSE]) == DIM
    meta = store.get_meta()
    assert (meta["model"], meta["dim"], meta["chunks"]) == ("fake-hash", str(DIM), "6")


def test_index_is_idempotent_and_follows_the_corpus(store: QdrantStore) -> None:
    embedder = FakeEmbedder(DIM)
    rows = [dict(r) for r in CORPUS]
    again = asyncio.run(index_chunks(store, embedder, rows))
    assert again["embedded"] == 0 and again["unchanged"] == len(CORPUS) and embedder.calls == 0
    first_version = store.get_meta()["indexed_at"]

    rows[0] = {**rows[0], "text": rows[0]["text"] + " Updated."}
    changed = asyncio.run(index_chunks(store, embedder, rows[:-1]))
    assert (changed["embedded"], changed["deleted"]) == (1, 1)
    assert len(store.hashes()) == len(CORPUS) - 1 == store.info()["points"]
    assert store.get_meta()["chunks"] == "5" and store.get_meta()["indexed_at"] >= first_version


def test_another_embedding_model_rebuilds_the_collection(store: QdrantStore) -> None:
    other = FakeEmbedder(DIM)
    other.model = "fake-other"
    stats = asyncio.run(index_chunks(store, other, [dict(r) for r in CORPUS]))
    assert stats["embedded"] == len(CORPUS) and store.get_meta()["model"] == "fake-other"


def test_dense_search_and_filters_in_qdrant(store: QdrantStore) -> None:
    retriever = Retriever(store, FakeEmbedder(DIM))
    assert retriever.matrix is None  # dense search runs in Qdrant
    query = "fluconazole warfarin prothrombin"
    for mode in ("bm25", "dense", "hybrid"):
        hits = asyncio.run(retriever.search(query, k=3, mode=mode))  # type: ignore[arg-type]
        assert hits and hits[0].chunk.chunk_id in {"flu-di", "ddi-fw"}, mode
        assert all(not h.chunk.is_retracted for h in hits)

    only_labels = Filters(doc_types=("label",), ingredients=("warfarin",))
    hits = asyncio.run(retriever.search("warfarin", k=10, mode="dense", filters=only_labels))
    # "about or mentioning warfarin": the fluconazole label mentions it
    assert {h.chunk.chunk_id for h in hits} == {"war-ci", "flu-di"}
    with_retracted = Filters(ingredients=("warfarin",), exclude_retracted=False)
    ids = store.dense_search([0.1] * DIM, with_retracted, 10)
    assert "pm-retracted" in ids
    assert "pm-retracted" not in store.dense_search([0.1] * DIM, Filters(), 10)
    sections = Filters(sections=("contraindications",))
    assert store.dense_search([0.1] * DIM, sections, 10) == ["war-ci"]


def test_filter_translation() -> None:
    assert to_filter(Filters(exclude_retracted=False)) is None
    built = to_filter(Filters(ingredients=("warfarin",), doc_types=("label",)))
    assert built is not None and built.must is not None and len(built.must) == 3


def test_same_rankings_as_the_sql_store(store: QdrantStore, tmp_path: Path) -> None:
    sql = asyncio.run(build_index(f"sqlite:///{(tmp_path / 'rag.db').as_posix()}"))
    in_sql, in_qdrant = Retriever(sql, FakeEmbedder(DIM)), Retriever(store, FakeEmbedder(DIM))
    filters = Filters(ingredients=("warfarin", "fluconazole"))
    for query in ("fluconazole warfarin", "pregnancy", "bleeding risk cohort"):
        expected = asyncio.run(in_sql.search(query, k=5, mode="dense", filters=filters))
        found = asyncio.run(in_qdrant.search(query, k=5, mode="dense", filters=filters))
        assert [h.chunk.chunk_id for h in found] == [h.chunk.chunk_id for h in expected]


def test_copy_from_sql_embeds_nothing(empty: QdrantStore, tmp_path: Path) -> None:
    sql = asyncio.run(build_index(f"sqlite:///{(tmp_path / 'rag.db').as_posix()}"))
    stats = copy_index(sql, empty)
    assert stats["copied"] == len(CORPUS) and stats["deleted"] == 0
    assert empty.hashes() == sql.hashes()
    ids, vectors = empty.load_embeddings()
    sql_ids, sql_vectors = sql.load_embeddings()
    assert ids == sql_ids and abs(vectors - sql_vectors).max() < 1e-6
    meta = empty.get_meta()
    assert meta["model"] == "fake-hash" and meta["copied_from"] == "sqlite"
    embedder = FakeEmbedder(DIM)
    again = asyncio.run(index_chunks(empty, embedder, [dict(r) for r in CORPUS]))
    assert again["embedded"] == 0 and embedder.calls == 0


def test_copy_needs_an_index_and_the_same_dimension(empty: QdrantStore, tmp_path: Path) -> None:
    blank = ChunkStore(make_engine(f"sqlite:///{(tmp_path / 'blank.db').as_posix()}"), DIM)
    with pytest.raises(ValueError, match="no index"):
        copy_index(blank, empty)
    sql = asyncio.run(build_index(f"sqlite:///{(tmp_path / 'rag.db').as_posix()}"))
    with pytest.raises(ValueError, match="dimension"):
        copy_index(sql, QdrantStore(empty.client, DIM * 2, "tujya_test_other"))


def test_holder_reloads_after_reindexing(empty: QdrantStore) -> None:
    holder = RetrieverHolder(empty, FakeEmbedder(DIM))
    assert holder.get() is None  # no collection yet
    asyncio.run(index_chunks(empty, FakeEmbedder(DIM), [dict(r) for r in CORPUS[:2]]))
    first = holder.get()
    assert first is not None and len(first.chunks) == 2 and holder.get() is first
    time.sleep(1.1)  # indexed_at has a one-second resolution
    asyncio.run(index_chunks(empty, FakeEmbedder(DIM), [dict(r) for r in CORPUS]))
    second = holder.get()
    assert second is not first and second is not None and len(second.chunks) == len(CORPUS)
