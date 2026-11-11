"""Step 3.2: BM25 sparse vectors and hybrid search in Qdrant, reranking, query rewriting.

Fakes only (no download, no network): `TokenBM25` as the sparse encoder, a word-overlap
reranker, the fake chat model. Runs on Qdrant's in-memory mode, and on a real server when
TUJYA_TEST_QDRANT_URL is set (own collection, like test_rag_qdrant.py).
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Iterator

import pytest
from langchain_core.messages import AIMessage

from src.rag.embeddings import FakeEmbedder
from src.rag.index import index_chunks
from src.rag.qdrant_store import DENSE, QdrantStore
from src.rag.retrieval import BM25, Retriever, rrf, tokenize
from src.rag.rewrite import QueryRewriter, clean
from src.rag.sparse import TokenBM25
from src.rag.store import METADATA_COLUMNS, Filters
from tujya_testing.rag_fakes import CORPUS, DIM, FakeChatModel

SERVER = os.environ.get("TUJYA_TEST_QDRANT_URL")
TARGETS = [":memory:"] + ([SERVER] if SERVER else [])
QUERIES = ("fluconazole warfarin prothrombin", "bleeding risk cohort", "pregnancy")


def texts() -> list[str]:
    return [f"{r['title']}\n{r['text']}" for r in CORPUS]


def encoder() -> TokenBM25:
    """Average length of the test corpus, so Qdrant's BM25 equals the in-memory one."""
    lengths = [len(tokenize(t)) for t in texts()]
    return TokenBM25(avg_len=sum(lengths) / len(lengths))


class WordReranker:
    """Score = number of query words in the text: deterministic, and different from BM25."""

    name = "fake-reranker"

    def __init__(self) -> None:
        self.calls: list[int] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.calls.append(len(texts))
        words = set(query.lower().split())
        return [float(sum(w in t.lower() for w in words)) for t in texts]


@pytest.fixture(params=TARGETS, ids=lambda t: "memory" if t == ":memory:" else "server")
def empty(request: pytest.FixtureRequest) -> Iterator[QdrantStore]:
    store = QdrantStore.connect(request.param, DIM, "tujya_test_hybrid")
    store.drop()
    yield store
    store.drop()


@pytest.fixture
def store(empty: QdrantStore) -> QdrantStore:
    rows = [dict(r) for r in CORPUS]
    asyncio.run(index_chunks(empty, FakeEmbedder(DIM), rows, sparse=encoder()))
    return empty


def test_token_bm25_weights() -> None:
    bm25 = TokenBM25(k1=1.2, b=0.75, avg_len=4)
    (doc,) = bm25.documents(["warfarin warfarin bleeding"])  # 3 terms, warfarin twice
    norm = 1.2 * (1 - 0.75 + 0.75 * 3 / 4)
    weights = dict(zip(doc.indices, doc.values, strict=True))
    query = bm25.query("Warfarin and bleeding?")
    assert sorted(query.values) == [1.0, 1.0] and set(query.indices) == set(doc.indices)
    warfarin = bm25.query("warfarin").indices[0]
    assert math.isclose(weights[warfarin], 2 * 2.2 / (2 + norm), rel_tol=1e-6)
    assert bm25.query("the and of").indices == []  # stop words only


def test_index_writes_sparse_vectors_once(empty: QdrantStore) -> None:
    rows = [dict(r) for r in CORPUS]
    first = asyncio.run(index_chunks(empty, FakeEmbedder(DIM), rows, sparse=encoder()))
    assert empty.has_sparse() and empty.get_meta()["sparse_model"] == encoder().name
    assert first["embedded"] == len(CORPUS) and first["sparse_recomputed"] == 0
    again = asyncio.run(index_chunks(empty, FakeEmbedder(DIM), rows, sparse=encoder()))
    assert again["embedded"] == 0 and again["sparse_recomputed"] == 0
    other = TokenBM25(k1=1.5)  # another encoder: sparse vectors only, nothing embedded
    changed = asyncio.run(index_chunks(empty, FakeEmbedder(DIM), rows, sparse=other))
    assert changed["embedded"] == 0 and changed["sparse_recomputed"] == len(CORPUS)
    assert empty.get_meta()["sparse_model"] == other.name


def test_old_collection_is_migrated_without_embedding(empty: QdrantStore) -> None:
    from qdrant_client import models as qm

    rows = [dict(r) for r in CORPUS]
    asyncio.run(index_chunks(empty, FakeEmbedder(DIM), rows))  # dense only
    # A collection made before step 3.2: no sparse vector in its configuration.
    meta = empty.get_meta()
    records = empty.client.scroll(empty.collection, limit=100, with_vectors=True)[0]
    empty.drop()
    empty.client.create_collection(
        empty.collection,
        vectors_config={DENSE: qm.VectorParams(size=DIM, distance=qm.Distance.COSINE)},
    )
    empty.client.upsert(
        empty.collection,
        points=[qm.PointStruct(id=r.id, vector=r.vector, payload=r.payload) for r in records],
    )
    empty.client.update_collection(empty.collection, metadata=meta)
    assert not empty.has_sparse()

    stats = asyncio.run(index_chunks(empty, FakeEmbedder(DIM), rows, sparse=encoder()))
    assert stats["migrated"] == len(CORPUS) and stats["embedded"] == 0
    assert stats["sparse_recomputed"] == len(CORPUS)
    assert empty.has_sparse() and empty.get_meta()["indexed_at"]
    hits = asyncio.run(
        Retriever(empty, FakeEmbedder(DIM), sparse=encoder()).search("warfarin", mode="bm25")
    )
    assert hits


def test_sparse_search_in_qdrant_ranks_like_the_in_memory_bm25(store: QdrantStore) -> None:
    native = Retriever(store, FakeEmbedder(DIM), sparse=encoder())
    memory = BM25(texts())
    ids = [r["chunk_id"] for r in CORPUS]
    assert native.lexical_kind.startswith("qdrant-sparse")
    for query in QUERIES:
        scores = memory.scores(query)
        expected = {ids[i]: round(float(scores[i]), 4) for i in range(len(ids)) if scores[i] > 0}
        found = native.bm25_ranking(query, Filters(exclude_retracted=False), 50)
        assert set(found) == set(expected), query
        ranked = [expected[cid] for cid in found]  # same scores, so the same order (ties aside)
        assert ranked == sorted(ranked, reverse=True), query


def test_hybrid_is_fused_by_qdrant_like_our_rrf(store: QdrantStore) -> None:
    retriever = Retriever(store, FakeEmbedder(DIM), sparse=encoder())
    for query in QUERIES:
        hits = asyncio.run(retriever.search(query, k=5, mode="hybrid"))
        lexical = retriever.bm25_ranking(query, Filters(), 50)
        dense = asyncio.run(retriever.dense_ranking(query, Filters(), 50))
        expected = rrf([lexical, dense])[:5]
        assert [round(h.score, 6) for h in hits] == [round(s, 6) for _, s in expected], query
        fused = dict(rrf([lexical, dense]))  # same chunk scores; ties may be ordered apart
        assert all(round(fused[h.chunk.chunk_id], 6) == round(h.score, 6) for h in hits), query


def test_filters_apply_to_sparse_and_hybrid(store: QdrantStore) -> None:
    retriever = Retriever(store, FakeEmbedder(DIM), sparse=encoder())
    labels = Filters(doc_types=("label",), ingredients=("warfarin",))
    for mode in ("bm25", "hybrid"):
        hits = asyncio.run(retriever.search("warfarin", k=10, mode=mode, filters=labels))
        assert hits and {h.chunk.chunk_id for h in hits} <= {"war-ci", "flu-di"}, mode
    for mode in ("bm25", "hybrid"):
        ids = {h.chunk.chunk_id for h in asyncio.run(retriever.search("cohort", mode=mode, k=20))}
        assert "pm-retracted" not in ids, mode


def test_no_sparse_vectors_of_this_encoder_means_bm25_in_memory(empty: QdrantStore) -> None:
    rows = [dict(r) for r in CORPUS]
    asyncio.run(index_chunks(empty, FakeEmbedder(DIM), rows))  # no sparse vectors
    retriever = Retriever(empty, FakeEmbedder(DIM), sparse=encoder())
    assert retriever.sparse is None and retriever.lexical_kind == "memory"
    hits = asyncio.run(retriever.search("fluconazole warfarin", k=3, mode="hybrid"))
    assert hits and hits[0].bm25_rank is not None  # fused here, with both ranks


def test_rerank_reads_fetch_k_candidates_and_keeps_k(store: QdrantStore) -> None:
    reranker = WordReranker()
    retriever = Retriever(store, FakeEmbedder(DIM), sparse=encoder(), reranker=reranker)
    query = "prothrombin time monitoring"
    first = asyncio.run(retriever.search(query, k=6, mode="dense", filters=Filters()))
    hits = asyncio.run(retriever.search(query, k=2, mode="dense", rerank=True, fetch_k=6))
    assert reranker.calls == [len(first)] and len(hits) == 2
    best = max(reranker.score(query, [h.chunk.search_text for h in first]))
    assert hits[0].rerank_score == best and hits[0].score == best
    assert hits[0].rerank_score is not None and hits[1].rerank_score is not None
    assert hits[0].rerank_score >= hits[1].rerank_score
    ranks = {h.chunk.chunk_id: i for i, h in enumerate(first, start=1)}
    assert all(h.retrieval_rank == ranks[h.chunk.chunk_id] for h in hits)
    # Without reranking (or without a reranker) the order is the retrieval order.
    plain = asyncio.run(retriever.search(query, k=2, mode="dense"))
    assert [h.chunk.chunk_id for h in plain] == [h.chunk.chunk_id for h in first[:2]]
    bare = Retriever(store, FakeEmbedder(DIM))
    unranked = asyncio.run(bare.search(query, k=2, mode="dense", rerank=True))
    assert all(h.rerank_score is None for h in unranked)


def test_rerank_keeps_the_retrieval_order_on_ties(store: QdrantStore) -> None:
    class Flat:
        name = "flat"

        def score(self, query: str, texts: list[str]) -> list[float]:
            return [0.0] * len(texts)

    retriever = Retriever(store, FakeEmbedder(DIM), reranker=Flat())
    first = asyncio.run(retriever.search("warfarin", k=4, mode="dense"))
    hits = asyncio.run(retriever.search("warfarin", k=4, mode="dense", rerank=True, fetch_k=4))
    assert [h.chunk.chunk_id for h in hits] == [h.chunk.chunk_id for h in first]


def test_clean_rewritten_query() -> None:
    assert clean('Query: "warfarin fluconazole bleeding"\nextra') == "warfarin fluconazole bleeding"
    assert clean("  \n search query: ibuprofen pregnancy ") == "ibuprofen pregnancy"
    assert clean("") == ""


def test_query_rewriter() -> None:
    llm = FakeChatModel(answer="warfarin fluconazole interaction bleeding")
    question = "La warfarine interagit-elle avec le fluconazole ?"
    assert asyncio.run(QueryRewriter(llm).rewrite(question)) == llm.answer
    system = str(llm.prompts[0][0].content)
    assert "Do not answer" in system and llm.prompts[0][1].content == question
    too_long = FakeChatModel(answer="word " * 100)
    assert asyncio.run(QueryRewriter(too_long).rewrite(question)) == question
    empty = FakeChatModel(answer="   ")
    assert asyncio.run(QueryRewriter(empty).rewrite(question)) == question


def test_query_rewriter_keeps_the_question_when_the_llm_fails() -> None:
    class Down(FakeChatModel):
        def _reply(self, messages: list) -> AIMessage:  # type: ignore[override]
            raise RuntimeError("ollama down")

    question = "warfarin and aspirin"
    assert asyncio.run(QueryRewriter(Down()).rewrite(question)) == question


def test_payload_keeps_every_metadata_field_with_sparse(store: QdrantStore) -> None:
    (record,) = store.client.scroll(store.collection, limit=1, with_vectors=True)[0]
    assert set(METADATA_COLUMNS) <= set(record.payload or {})
    assert isinstance(record.vector, dict) and set(record.vector) == {"dense", "bm25"}
