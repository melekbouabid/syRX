"""Opens the chunk store the settings name: Qdrant (default, D113) or the SQL table."""

from __future__ import annotations

from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from app.config import Settings
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
