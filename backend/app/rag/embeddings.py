"""Text embeddings: bge-m3 served by Ollama in production, a deterministic fake in tests.

bge-m3 is multilingual (100+ languages), so a French question finds English label text.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from typing import Protocol

import httpx


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
        timeout: float = 120.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url, self.model, self.dim = base_url.rstrip("/"), model, dim
        self.timeout, self.transport = timeout, transport

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                response = await client.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": texts, "truncate": True},
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Ollama embeddings failed: {exc}") from exc
        vectors: list[list[float]] = response.json().get("embeddings", [])
        if len(vectors) != len(texts) or any(len(v) != self.dim for v in vectors):
            raise EmbeddingError(f"unexpected embedding shape from {self.model}")
        return vectors


_WORD = re.compile(r"[a-z0-9]+")


def fold(text: str) -> str:
    """Lower case without accents: "Warfarine sûre" -> "warfarine sure"."""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


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
