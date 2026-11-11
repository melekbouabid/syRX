"""Opens the chunk store the settings name: Qdrant (default, D113) or the SQL table, and
the retriever with its BM25 sparse encoder and reranker (step 3.2)."""

from __future__ import annotations

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from app.config import Settings
from src.rag.embeddings import Embedder
from src.rag.rerank import CrossEncoderReranker
from src.rag.retrieval import RetrieverHolder
from src.rag.sparse import FastEmbedBM25
from src.rag.store import ChunkStore, VectorStore

# A store that cannot be reached (Qdrant or PostgreSQL down): the API answers 503.
STORE_ERRORS = (ResponseHandlingException, UnexpectedResponse, OperationalError, OSError)


def open_store(settings: Settings, dim: int, pharma_engine: Engine | None = None) -> VectorStore:
    if settings.vector_store == "sql":
        if pharma_engine is None:
            from app.db import make_engine

            pharma_engine = make_engine(settings.pharma_database_url)
        return ChunkStore(pharma_engine, dim)
    from src.rag.qdrant_store import QdrantStore

    return QdrantStore.connect(settings.qdrant_url, dim, settings.qdrant_collection)


def retriever_holder(settings: Settings, store: VectorStore, embedder: Embedder) -> RetrieverHolder:
    """BM25 runs in Qdrant when the index has the sparse vectors of `rag_sparse_model`; the
    reranker is loaded on its first use only (`rag_rerank`, or a benchmark asking for it)."""
    cache = settings.rag_model_cache_dir
    return RetrieverHolder(
        store,
        embedder,
        sparse=FastEmbedBM25(settings.rag_sparse_model, cache_dir=cache),
        reranker=CrossEncoderReranker(settings.rag_rerank_model, cache_dir=cache),
    )
