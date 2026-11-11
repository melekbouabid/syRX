"""Sparse (BM25) vectors stored in Qdrant next to the dense ones (step 3.2, D116).

Qdrant's BM25 is split in two: the encoder gives each term of a chunk its saturated term
frequency, tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avg_len)), and the collection's
sparse vector `bm25` has the IDF modifier, so Qdrant multiplies by the inverse document
frequency it keeps up to date itself. A query term weighs 1: the score of a chunk is the
BM25 sum over the query terms it contains.

- `FastEmbedBM25`: the `Qdrant/bm25` model of fastembed (what LangChain's FastEmbedSparse
  wraps, as in the course): English stop words and Snowball stemmer, terms hashed with
  murmur3. Downloaded once from Hugging Face (a few kB, no account) into
  `TUJYA_RAG_MODEL_CACHE_DIR`.
- `TokenBM25`: the same formula on the tokenizer of the in-memory BM25 (src/rag/retrieval.py),
  terms hashed with CRC-32; no download, used by the tests.
"""

from __future__ import annotations

import zlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class SparseEncoder(Protocol):
    @property
    def name(self) -> str:
        """Recorded in the index metadata: another encoder recomputes every sparse vector."""
        ...

    def documents(self, texts: list[str]) -> list[SparseVector]: ...

    def query(self, text: str) -> SparseVector: ...


class FastEmbedBM25:
    def __init__(self, model: str = "Qdrant/bm25", cache_dir: Path | None = None) -> None:
        self.model, self.cache_dir = model, cache_dir
        self._encoder: Any = None

    @property
    def name(self) -> str:
        return self.model

    def _load(self) -> Any:
        if self._encoder is None:
            from fastembed import SparseTextEmbedding

            cache = str(self.cache_dir) if self.cache_dir else None
            self._encoder = SparseTextEmbedding(self.model, cache_dir=cache)
        return self._encoder

    def documents(self, texts: list[str]) -> list[SparseVector]:
        return [_vector(e) for e in self._load().embed(texts)]

    def query(self, text: str) -> SparseVector:
        (embedding,) = list(self._load().query_embed(text))
        return _vector(embedding)


def _vector(embedding: Any) -> SparseVector:
    return SparseVector([int(i) for i in embedding.indices], [float(v) for v in embedding.values])


class TokenBM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75, avg_len: float = 256.0) -> None:
        self.k1, self.b, self.avg_len = k1, b, avg_len

    @property
    def name(self) -> str:
        return f"token-bm25-k1={self.k1}-b={self.b}"

    @staticmethod
    def _terms(text: str) -> list[str]:
        from src.rag.retrieval import tokenize  # retrieval imports this module

        return tokenize(text)

    @staticmethod
    def _index(term: str) -> int:
        return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF

    def _merge(self, weights: Iterable[tuple[str, float]]) -> SparseVector:
        merged: dict[int, float] = {}
        for term, weight in weights:
            index = self._index(term)
            merged[index] = merged.get(index, 0.0) + weight
        indices = sorted(merged)
        return SparseVector(indices, [merged[i] for i in indices])

    def documents(self, texts: list[str]) -> list[SparseVector]:
        vectors = []
        for text in texts:
            counts = Counter(self._terms(text))
            length = sum(counts.values())
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            vectors.append(
                self._merge((t, tf * (self.k1 + 1) / (tf + norm)) for t, tf in counts.items())
            )
        return vectors

    def query(self, text: str) -> SparseVector:
        return self._merge((term, 1.0) for term in set(self._terms(text)))
