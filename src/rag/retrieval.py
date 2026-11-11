"""Hybrid retrieval: BM25 (lexical) + embeddings (dense), fused by reciprocal rank fusion,
then optionally reranked by a cross-encoder.

- BM25 (k1 = 1.2, b = 0.75) over title + text. In Qdrant (step 3.2, D116): sparse vectors
  `Qdrant/bm25` with the IDF modifier (src/rag/sparse.py). Otherwise (SQL store) in memory:
  the corpus is ~10^4 chunks, so scoring every chunk takes a few milliseconds; tokens are
  lower-cased, accents removed, English and French stop words dropped, plural "s" stripped.
- Dense: cosine similarity of bge-m3 vectors, in Qdrant (D113), PostgreSQL (pgvector) or
  numpy (SQLite, tests).
- Hybrid: RRF, score = sum over the two lists of 1 / (60 + rank) (Cormack et al., 2009),
  computed by Qdrant (prefetch of both lists, then fusion) or here for the SQL store.
- Reranking (D117, src/rag/rerank.py): the first `fetch_k` results are scored by a
  cross-encoder (BAAI/bge-reranker-base) and the best `k` are kept.
Metadata filters (ingredient, document type, section, retracted) apply to all of them.
"""

from __future__ import annotations

import asyncio
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, replace
from functools import cached_property
from typing import Literal

import numpy as np

from src.rag.documents import Catalog
from src.rag.embeddings import Embedder, fold
from src.rag.rerank import Reranker
from src.rag.sparse import SparseEncoder
from src.rag.store import Chunk, Filters, VectorStore

Mode = Literal["bm25", "dense", "hybrid"]
RRF_K = 60
CANDIDATES = 50
FETCH_K = 20  # candidates read by the reranker

STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "may",
        "might",
        "more",
        "most",
        "no",
        "not",
        "of",
        "on",
        "or",
        "should",
        "so",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "which",
        "while",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "about",
        "any",
        "between",
        "both",
        "each",
        "other",
        "also",
        "only",
        "same",
        "use",
        "used",
        "au",
        "aux",
        "avec",
        "ce",
        "ces",
        "dans",
        "de",
        "des",
        "du",
        "elle",
        "en",
        "est",
        "et",
        "eux",
        "il",
        "ils",
        "je",
        "la",
        "le",
        "les",
        "leur",
        "lui",
        "ma",
        "mais",
        "me",
        "meme",
        "mes",
        "moi",
        "mon",
        "ne",
        "nos",
        "notre",
        "nous",
        "ou",
        "par",
        "pas",
        "pour",
        "qu",
        "que",
        "qui",
        "sa",
        "se",
        "ses",
        "son",
        "sur",
        "ta",
        "te",
        "tes",
        "toi",
        "ton",
        "tu",
        "un",
        "une",
        "vos",
        "votre",
        "vous",
        "c",
        "d",
        "j",
        "l",
        "m",
        "n",
        "s",
        "t",
        "y",
        "est",
        "sont",
        "peut",
        "quel",
        "quelle",
        "quels",
        "quelles",
        "quoi",
        "comment",
        "entre",
        "avec",
        "sans",
        "plus",
        "tres",
        "faut",
        "dois",
        "puis",
    ]
)
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN.findall(fold(text)):
        if token in STOP_WORDS or len(token) < 2:
            continue
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.n = len(documents)
        lengths = np.zeros(self.n, dtype=np.float32)
        postings: dict[str, tuple[list[int], list[int]]] = {}
        for index, document in enumerate(documents):
            counts = Counter(tokenize(document))
            lengths[index] = sum(counts.values())
            for term, tf in counts.items():
                docs, tfs = postings.setdefault(term, ([], []))
                docs.append(index)
                tfs.append(tf)
        self.lengths = lengths
        self.avg_length = float(lengths.mean()) if self.n else 0.0
        self.postings = {
            term: (np.array(docs, dtype=np.int32), np.array(tfs, dtype=np.float32))
            for term, (docs, tfs) in postings.items()
        }

    def idf(self, term: str) -> float:
        df = len(self.postings[term][0])
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        if not self.n:
            return scores
        norm = self.k1 * (1 - self.b + self.b * self.lengths / max(self.avg_length, 1e-9))
        for term in set(tokenize(query)):
            if term not in self.postings:
                continue
            docs, tfs = self.postings[term]
            scores[docs] += self.idf(term) * tfs * (self.k1 + 1) / (tfs + norm[docs])
        return scores


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None
    # Set by the reranker: its score, and the rank the chunk had before reranking.
    rerank_score: float | None = None
    retrieval_rank: int | None = None


def rrf(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        sparse: SparseEncoder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.store, self.embedder = store, embedder
        # BM25 in the store when it keeps sparse vectors made by this encoder (Qdrant), in
        # memory otherwise.
        indexed = store.get_meta().get("sparse_model") if sparse is not None else None
        self.sparse = sparse if sparse is not None and indexed == sparse.name else None
        self.reranker = reranker
        self.chunks = store.load_chunks()
        self.by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.position = {chunk.chunk_id: i for i, chunk in enumerate(self.chunks)}
        self.matrix: np.ndarray | None = None
        if not store.native_dense:
            ids, matrix = store.load_embeddings()
            order = [self.position[i] for i in ids]
            self.matrix = np.zeros((len(self.chunks), store.dim), dtype=np.float32)
            self.matrix[order] = matrix

    @cached_property
    def bm25(self) -> BM25:
        """In-memory BM25, built on first use (stores without sparse vectors)."""
        return BM25([chunk.search_text for chunk in self.chunks])

    @property
    def lexical_kind(self) -> str:
        """Where BM25 runs: in Qdrant (sparse vectors of this encoder) or in memory."""
        return f"qdrant-sparse:{self.sparse.name}" if self.sparse else "memory"

    @cached_property
    def catalog(self) -> Catalog:
        """The indexed documents (Documents page), built on first use, once per index."""
        return Catalog(self.chunks, self.store.hashes())

    def _mask(self, filters: Filters) -> np.ndarray:
        return np.array([filters.accepts(chunk) for chunk in self.chunks], dtype=bool)

    def bm25_ranking(self, query: str, filters: Filters, n: int) -> list[str]:
        if self.sparse is not None:
            return self.store.sparse_search(self.sparse.query(query), filters, n)
        scores = self.bm25.scores(query)
        mask = self._mask(filters) & (scores > 0)
        order = np.argsort(-scores, kind="stable")
        return [self.chunks[i].chunk_id for i in order if mask[i]][:n]

    async def dense_ranking(self, query: str, filters: Filters, n: int) -> list[str]:
        (vector,) = await self.embedder.embed([query])
        if self.matrix is None:
            return self.store.dense_search(vector, filters, n)
        query_vector = np.asarray(vector, dtype=np.float32)
        norms = np.linalg.norm(self.matrix, axis=1) * (np.linalg.norm(query_vector) or 1.0)
        similarity = self.matrix @ query_vector / np.where(norms == 0, 1.0, norms)
        similarity[~self._mask(filters)] = -np.inf
        order = np.argsort(-similarity, kind="stable")
        return [self.chunks[i].chunk_id for i in order[:n] if np.isfinite(similarity[i])]

    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        mode: Mode = "hybrid",
        filters: Filters | None = None,
        candidates: int = CANDIDATES,
        rerank: bool = False,
        fetch_k: int = FETCH_K,
    ) -> list[Hit]:
        """The `k` best chunks. With `rerank` (and a reranker), the first `fetch_k` (at
        least `k`) are read by the cross-encoder, which keeps the `k` it ranks best."""
        filters = filters or Filters()
        rerank = rerank and self.reranker is not None
        n = max(k, fetch_k) if rerank else k
        if mode == "hybrid" and self.sparse is not None:
            (vector,) = await self.embedder.embed([query])
            native = self.store.hybrid_search(
                vector, self.sparse.query(query), filters, n, candidates
            )
            hits = [Hit(self.by_id[cid], score) for cid, score in native]
        else:
            hits = await self._fused(query, mode, filters, candidates, n)
        return (await self._rerank(query, hits))[:k] if rerank else hits[:k]

    async def _fused(
        self, query: str, mode: Mode, filters: Filters, candidates: int, n: int
    ) -> list[Hit]:
        lexical = self.bm25_ranking(query, filters, candidates) if mode != "dense" else []
        dense = await self.dense_ranking(query, filters, candidates) if mode != "bm25" else []
        lexical_rank = {cid: r for r, cid in enumerate(lexical, start=1)}
        dense_rank = {cid: r for r, cid in enumerate(dense, start=1)}
        if mode == "bm25":
            fused = [(cid, 1.0 / (RRF_K + r)) for cid, r in lexical_rank.items()]
        elif mode == "dense":
            fused = [(cid, 1.0 / (RRF_K + r)) for cid, r in dense_rank.items()]
        else:
            fused = rrf([lexical, dense])
        return [
            Hit(self.by_id[cid], score, lexical_rank.get(cid), dense_rank.get(cid))
            for cid, score in fused[:n]
        ]

    async def _rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        assert self.reranker is not None
        texts = [hit.chunk.search_text for hit in hits]
        scores = await asyncio.to_thread(self.reranker.score, query, texts)
        order = sorted(range(len(hits)), key=lambda i: (-scores[i], i))  # ties: first rank
        return [
            replace(hits[i], score=scores[i], rerank_score=scores[i], retrieval_rank=i + 1)
            for i in order
        ]


class RetrieverHolder:
    """Builds the retriever on first use and again whenever the index is rebuilt."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        *,
        sparse: SparseEncoder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.store, self.embedder = store, embedder
        self.sparse, self.reranker = sparse, reranker
        self._retriever: Retriever | None = None
        self._version: str | None = None
        self._lock = threading.Lock()

    def get(self) -> Retriever | None:
        """None when the index does not exist yet (`python -m src.rag.index`)."""
        with self._lock:
            if not self.store.exists():
                return None
            version = self.store.get_meta().get("indexed_at")
            if self._retriever is None or version != self._version:
                self._retriever = Retriever(
                    self.store, self.embedder, sparse=self.sparse, reranker=self.reranker
                )
                self._version = version
            return self._retriever
