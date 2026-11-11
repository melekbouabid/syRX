"""Cross-encoder reranking of the retrieved candidates (step 3.2, remark 13, D117).

Retrieval (dense, BM25 or hybrid) returns `fetch_k` candidates; a cross-encoder reads the
question and each candidate together and gives a relevance score; the `top_n` best are
kept. More precise than comparing two separately computed vectors, too slow to run on
the whole corpus: hence the two stages.

`CrossEncoderReranker`: BAAI/bge-reranker-base (XLM-RoBERTa base, multilingual, the model
of the course's lab) in ONNX through fastembed, so the API needs no PyTorch. Downloaded
once from Hugging Face (1.1 GB, no account) into `TUJYA_RAG_MODEL_CACHE_DIR`. Each
candidate is read as title + text, like the embeddings.
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
