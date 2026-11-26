"""Query path — Reranking: a cross-encoder reorders the retrieved candidates (measured, not served).

BAAI/bge-reranker-base (ONNX via fastembed) by default. Retrieval returns `fetch_k`
candidates; the cross-encoder keeps the `top_n` best. Off in the served setting
(`TUJYA_RAG_RERANK`): too slow on the CPU for the latency budget (docs/module2/results.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class Reranker(Protocol):
    @property
    def name(self) -> str: ...

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text, higher is more relevant (same order as `texts`)."""
        ...


class CrossEncoderReranker:
    def __init__(
        self,
        model: str = "BAAI/bge-reranker-base",
        cache_dir: Path | None = None,
        threads: int | None = None,
    ) -> None:
        self.model, self.cache_dir, self.threads = model, cache_dir, threads
        self._encoder: Any = None

    @property
    def name(self) -> str:
        return self.model

    def _load(self) -> Any:
        if self._encoder is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            cache = str(self.cache_dir) if self.cache_dir else None
            self._encoder = TextCrossEncoder(self.model, cache_dir=cache, threads=self.threads)
        return self._encoder

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._load().rerank(query, texts, batch_size=len(texts))]
