"""Step 4 — Embedder: each chunk becomes a dense bge-m3 vector and a sparse BM25 vector.

Dense: BAAI/bge-m3 served by Ollama (`/api/embed`, 1024 dimensions, L2-normalised by Ollama),
multilingual, so a French question finds English label text. Sparse: fastembed `Qdrant/bm25`,
stored in Qdrant next to the dense vector (the IDF is applied by Qdrant). The tests use the
offline `FakeEmbedder` and `TokenBM25` (same formulas, no download).
"""

from __future__ import annotations

import hashlib
import math
import re
import zlib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from src.rag.chunking import fold

# --- Dense text embeddings: bge-m3 served by Ollama, a deterministic fake in tests -----------


class Embedder(Protocol):
    model: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class EmbeddingError(RuntimeError):
    """The embedding service is unreachable or answered something unusable."""


class OllamaEmbedder:
    """POST /api/embed on the Ollama server (docker compose service `ollama`)."""

    def __init__(
        self,
        base_url: str,
        model: str = "bge-m3",
        dim: int = 1024,
        *,
        num_ctx: int | None = None,
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url, self.model, self.dim = base_url.rstrip("/"), model, dim
        self.timeout, self.transport = timeout, transport
        self.options = {"num_ctx": num_ctx} if num_ctx else {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        body = {
            "model": self.model,
            "input": texts,
            "truncate": True,
            **({"options": self.options} if self.options else {}),
        }
        for attempt in (1, 2):  # once more if Ollama was busy swapping models on the GPU
            try:
                async with httpx.AsyncClient(
                    timeout=self.timeout, transport=self.transport
                ) as client:
                    response = await client.post(f"{self.base_url}/api/embed", json=body)
                    response.raise_for_status()
                break
            except httpx.HTTPError as exc:
                if attempt == 2:
                    raise EmbeddingError(
                        f"Ollama embeddings failed: {type(exc).__name__} {exc}"
                    ) from exc
        vectors: list[list[float]] = response.json().get("embeddings", [])
        if len(vectors) != len(texts) or any(len(v) != self.dim for v in vectors):
            raise EmbeddingError(f"unexpected embedding shape from {self.model}")
        return vectors


_WORD = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Hashed bag of words and character trigrams, L2-normalised. Deterministic, offline.

    Texts sharing words get similar vectors, which is enough to test the retrieval code
    without a model (CI never downloads embeddings).
    """

    model = "fake-hash"

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim
        self.calls = 0

    def vector(self, text: str) -> list[float]:
        values = [0.0] * self.dim
        for word in _WORD.findall(fold(text)):
            features = [word] + [word[i : i + 3] for i in range(max(0, len(word) - 2))]
            for feature in features:
                digest = hashlib.md5(feature.encode(), usedforsecurity=False).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                values[index] += 1.0 if digest[4] % 2 else -1.0
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [self.vector(t) for t in texts]


async def pull_models(base_url: str, models: list[str]) -> None:
    """Pull the Ollama models that are missing (first start of the stack only)."""
    async with httpx.AsyncClient(timeout=3600) as client:
        tags = (await client.get(f"{base_url}/api/tags")).json().get("models", [])
        present = {m["name"] for m in tags} | {m["name"].split(":")[0] for m in tags}
        for model in models:
            if model in present or f"{model}:latest" in present:
                continue
            print(f"pulling {model} (first start only)...", flush=True)
            response = await client.post(
                f"{base_url}/api/pull", json={"model": model, "stream": False}
            )
            response.raise_for_status()


# --- Sparse BM25 vectors, stored in Qdrant next to the dense ones ------------------------------
# `FastEmbedBM25` (fastembed `Qdrant/bm25`) in the app; `TokenBM25` (same formula, no download)
# in the tests.


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
