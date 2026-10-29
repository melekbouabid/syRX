"""The same retrieval on PostgreSQL + pgvector (dense search in SQL, array filters).

Runs when TUJYA_TEST_PG_URL points to a PostgreSQL server with the pgvector extension
(the CI backend job has one; locally the compose `postgres`). Uses its own schema, so the
real index is never touched.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterator

import pytest
from sqlalchemy import MetaData, text

from app.db import make_engine
from app.rag.embeddings import FakeEmbedder
from app.rag.index import index_chunks
from app.rag.retrieval import Retriever
from app.rag.store import ChunkStore, Filters
from tests.rag_fakes import CORPUS, DIM

URL = os.environ.get("TUJYA_TEST_PG_URL")
pytestmark = pytest.mark.skipif(not URL, reason="TUJYA_TEST_PG_URL not set")
SCHEMA = "tujya_test_rag"


@pytest.fixture
def store() -> Iterator[ChunkStore]:
    assert URL
    engine = make_engine(URL)
    store = ChunkStore(engine, DIM, MetaData(schema=SCHEMA))
    store.drop() if store.exists() else None
    asyncio.run(index_chunks(store, FakeEmbedder(DIM), [dict(r) for r in CORPUS]))
    yield store
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{SCHEMA}" CASCADE'))
    engine.dispose()


def test_pgvector_column_and_sql_dense_search(store: ChunkStore) -> None:
    with store.engine.connect() as conn:
        kind = conn.execute(
            text(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = :t ::regclass AND attname = 'embedding'"
            ),
            {"t": f"{SCHEMA}.rag_chunks"},
        ).scalar()
    assert kind == f"vector({DIM})"
    retriever = Retriever(store, FakeEmbedder(DIM))
    assert retriever.matrix is None  # dense search runs in SQL
    hits = asyncio.run(retriever.search("fluconazole warfarin prothrombin", k=3, mode="dense"))
    assert hits[0].chunk.chunk_id in {"flu-di", "ddi-fw"}
    filters = Filters(doc_types=("label",), ingredients=("warfarin",))
    ids = {
        h.chunk.chunk_id for h in asyncio.run(retriever.search("warfarin", k=10, filters=filters))
    }
    assert ids == {"war-ci", "flu-di"}  # text[] overlap on ingredients or mentions
    retracted = asyncio.run(retriever.search("bleeding cohort", k=10, mode="dense"))
    assert all(not h.chunk.is_retracted for h in retracted)


def test_reindexing_postgres_is_idempotent(store: ChunkStore) -> None:
    stats = asyncio.run(index_chunks(store, FakeEmbedder(DIM), [dict(r) for r in CORPUS]))
    assert stats["embedded"] == 0 and stats["unchanged"] == len(CORPUS)
