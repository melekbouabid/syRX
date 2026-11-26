"""Step 5 updates, src/rag/vector_store.py: add, update and delete one document of the index.

Qdrant in memory with the offline embedder and BM25 encoder; the SQL store (SQLite) as well.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.db import make_engine
from src.rag.embedder import FakeEmbedder, TokenBM25
from src.rag.retrieval import Retriever
from src.rag.vector_store import (
    ChunkStore,
    QdrantStore,
    delete_document,
    document_rows,
    index_chunks,
    main,
    point_id,
    sample_collection,
    write_document,
)
from tests.support.rag_fakes import CORPUS, DIM

LONG = " ".join(
    f"Sentence {i} says that drug zorbatide may raise the level of drug quellomab."
    for i in range(80)
)
DOCUMENT = {
    "doc_id": "note:zorbatide",
    "doc_type": "label",
    "title": "Zorbatide note: Drug interactions",
    "section": "drug_interactions",
    "text": LONG,
    "source_name": "Test source",
    "url": "https://example.org/zorbatide",
    "date": "2026-11-26",
    "version": "1",
    "license": "CC0",
    "ingredients": ["Zorbatide"],
    "mentioned_ingredients": ["quellomab", "zorbatide"],
}
UPDATED = {**DOCUMENT, "version": "2", "text": "Zorbatide is now known to lower quellomab levels."}


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    qdrant = QdrantStore.connect(":memory:", DIM, "tujya_test_updates")
    encoder = TokenBM25()
    asyncio.run(index_chunks(qdrant, FakeEmbedder(DIM), [dict(r) for r in CORPUS], sparse=encoder))
    yield qdrant
    qdrant.drop()


def run(coro: Any) -> Any:
    return asyncio.run(coro)


def point_ids(store: QdrantStore) -> set[str]:
    return {str(r.id) for r in store._scroll(["chunk_id"])}


def test_document_rows_are_cut_like_the_corpus_with_stable_ids() -> None:
    rows = document_rows(DOCUMENT)
    assert len(rows) > 1  # 80 sentences are longer than one chunk
    assert all(len(r["text"]) <= 1500 for r in rows)
    assert [r["chunk_index"] for r in rows] == list(range(len(rows)))
    assert rows[0]["ingredients"] == ["zorbatide"]
    assert rows[0]["mentioned_ingredients"] == ["quellomab"]  # its own ingredient is not repeated
    assert [r["chunk_id"] for r in document_rows(DOCUMENT)] == [r["chunk_id"] for r in rows]
    assert len({r["chunk_id"] for r in rows}) == len(rows)


def test_document_rows_refuse_a_dosage_section_or_missing_fields() -> None:
    with pytest.raises(ValueError, match="dosage"):
        document_rows({**DOCUMENT, "section": "dosage_and_administration"})
    with pytest.raises(ValueError, match="title"):
        document_rows({**DOCUMENT, "title": " "})


def test_add_writes_every_chunk_and_search_finds_it(store: QdrantStore) -> None:
    before = store.info()["points"]
    rows = document_rows(DOCUMENT)
    stats = run(
        write_document(store, FakeEmbedder(DIM), rows, sparse=TokenBM25(), must_exist=False)
    )
    assert stats == {
        "doc_id": "note:zorbatide",
        "chunks": len(rows),
        "embedded": len(rows),
        "deleted": 0,
        "unchanged": 0,
    }
    assert store.info()["points"] == before + len(rows)
    assert store.get_meta()["chunks"] == str(before + len(rows))
    retriever = Retriever(store, FakeEmbedder(DIM), sparse=TokenBM25())
    for mode in ("dense", "bm25", "hybrid"):
        hits = run(retriever.search("zorbatide quellomab", k=3, mode=mode))
        assert hits and hits[0].chunk.doc_id == "note:zorbatide", mode


def test_adding_twice_is_refused_and_a_rerun_never_duplicates(store: QdrantStore) -> None:
    rows = document_rows(DOCUMENT)
    run(write_document(store, FakeEmbedder(DIM), rows, sparse=TokenBM25(), must_exist=False))
    ids, meta = point_ids(store), store.get_meta()
    with pytest.raises(ValueError, match="already indexed"):
        run(write_document(store, FakeEmbedder(DIM), rows, sparse=TokenBM25(), must_exist=False))
    embedder = FakeEmbedder(DIM)
    stats = run(write_document(store, embedder, rows, sparse=TokenBM25()))  # same text again
    assert stats["embedded"] == 0 and stats["deleted"] == 0 and embedder.calls == 0
    assert point_ids(store) == ids
    assert store.get_meta()["indexed_at"] == meta["indexed_at"]  # nothing changed
    assert {point_id(r["chunk_id"]) for r in rows} <= ids  # deterministic point ids


def test_update_replaces_the_text_and_removes_the_old_chunks(store: QdrantStore) -> None:
    old = document_rows(DOCUMENT)
    run(write_document(store, FakeEmbedder(DIM), old, sparse=TokenBM25(), must_exist=False))
    others = point_ids(store) - {point_id(r["chunk_id"]) for r in old}
    new = document_rows(UPDATED)
    stats = run(write_document(store, FakeEmbedder(DIM), new, sparse=TokenBM25(), must_exist=True))
    assert stats["chunks"] == 1 and stats["embedded"] == 1 and stats["deleted"] == len(old) - 1
    assert store.document_hashes("note:zorbatide").keys() == {new[0]["chunk_id"]}
    (chunk,) = [c for c in store.load_chunks() if c.doc_id == "note:zorbatide"]
    assert chunk.text == UPDATED["text"] and chunk.version == "2"
    assert point_ids(store) == others | {point_id(new[0]["chunk_id"])}  # the rest is untouched
    hits = run(Retriever(store, FakeEmbedder(DIM), sparse=TokenBM25()).search("lower quellomab"))
    assert hits[0].chunk.text == UPDATED["text"]


def test_update_of_an_unknown_document_is_refused(store: QdrantStore) -> None:
    with pytest.raises(LookupError, match="use add"):
        rows = document_rows(DOCUMENT)
        run(write_document(store, FakeEmbedder(DIM), rows, sparse=TokenBM25(), must_exist=True))


def test_delete_removes_the_document_only(store: QdrantStore) -> None:
    before = point_ids(store)
    rows = document_rows(DOCUMENT)
    run(write_document(store, FakeEmbedder(DIM), rows, sparse=TokenBM25(), must_exist=False))
    assert delete_document(store, "note:zorbatide") == {
        "doc_id": "note:zorbatide",
        "deleted": len(rows),
    }
    assert point_ids(store) == before
    assert store.document_hashes("note:zorbatide") == {}
    assert delete_document(store, "note:zorbatide")["deleted"] == 0


def test_another_model_or_sparse_encoder_is_refused(store: QdrantStore) -> None:
    rows = document_rows(DOCUMENT)

    class OtherEmbedder(FakeEmbedder):
        model = "other-model"

    with pytest.raises(ValueError, match="fake-hash"):
        run(write_document(store, OtherEmbedder(DIM), rows, sparse=TokenBM25()))
    with pytest.raises(ValueError, match="sparse"):
        run(write_document(store, FakeEmbedder(DIM), rows, sparse=None))


def test_the_sql_store_supports_the_same_updates(tmp_path: Path) -> None:
    sql = ChunkStore(make_engine(f"sqlite:///{tmp_path / 'chunks.db'}"), DIM)
    run(index_chunks(sql, FakeEmbedder(DIM), [dict(r) for r in CORPUS]))
    total = len(sql.hashes())
    old = document_rows(DOCUMENT)
    run(write_document(sql, FakeEmbedder(DIM), old, must_exist=False))
    assert len(sql.hashes()) == total + len(old)
    run(write_document(sql, FakeEmbedder(DIM), document_rows(UPDATED), must_exist=True))
    assert len(sql.hashes()) == total + 1
    delete_document(sql, "note:zorbatide")
    assert len(sql.hashes()) == total


def test_sample_copies_points_with_their_vectors(store: QdrantStore) -> None:
    trial = QdrantStore(store.client, DIM, "tujya_test_sample", ":memory:")
    stats = sample_collection(store, trial, 3)
    assert stats["copied"] == 3 and trial.info()["points"] == 3
    assert trial.get_meta()["sparse_model"] == store.get_meta()["sparse_model"]
    assert store.info()["points"] == len(CORPUS)  # the source is only read
    (record,) = trial.client.scroll(trial.collection, limit=1, with_vectors=True)[0]
    assert isinstance(record.vector, dict) and {"dense", "bm25"} <= set(record.vector)
    with pytest.raises(ValueError, match="another collection"):
        sample_collection(store, store, 3)


@pytest.mark.parametrize("command", ["add", "update", "delete", "sample", "drop"])
def test_the_command_line_never_writes_a_served_collection_by_default(
    command: str, tmp_path: Path
) -> None:
    served = Settings().qdrant_collection
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(DOCUMENT), encoding="utf-8")
    argv = {
        "add": ["add", "--document", str(path), "--collection", served],
        "update": ["update", "--document", str(path), "--collection", served],
        "delete": ["delete", "--doc-id", "x", "--collection", served],
        "sample": ["sample", "--collection", "tujya_test_x", "--to", served],
        "drop": ["drop", "--collection", served],
    }[command]
    with pytest.raises(SystemExit, match="served collection"):
        main([*argv, "--qdrant-url", "http://127.0.0.1:9"])  # never reached


def test_the_command_line_refuses_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trial = QdrantStore.connect(":memory:", DIM, "tujya_test_cli")
    asyncio.run(index_chunks(trial, FakeEmbedder(DIM), [dict(r) for r in CORPUS]))
    monkeypatch.setattr(QdrantStore, "connect", classmethod(lambda cls, *a, **k: trial))
    monkeypatch.setattr("src.rag.vector_store.OllamaEmbedder", lambda *a, **k: FakeEmbedder(DIM))
    path = tmp_path / "doc.json"
    path.write_text(json.dumps(DOCUMENT), encoding="utf-8")
    argv = ["--document", str(path), "--collection", "tujya_test_cli", "--no-sparse"]
    main(["add", *argv])
    with pytest.raises(SystemExit, match="add refused: note:zorbatide is already indexed"):
        main(["add", *argv])
    main(["update", *argv])  # same text: nothing to do, no error
    main(["delete", "--doc-id", "note:zorbatide", "--collection", "tujya_test_cli"])
    with pytest.raises(SystemExit, match="update refused: note:zorbatide is not indexed"):
        main(["update", *argv])
