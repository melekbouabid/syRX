"""Step 5 — Vector store: the chunks, their metadata and their vectors in Qdrant.

One point per chunk in the collection `tujya_chunks_v2` (`TUJYA_QDRANT_COLLECTION`): named
vectors `dense` (bge-m3, HNSW, cosine) and `bm25` (sparse, IDF modifier), the chunk metadata
as payload (keyword indexes on the filtered fields). The SQL store (pgvector on PostgreSQL,
numpy on SQLite, `TUJYA_VECTOR_STORE=sql`) serves the tests and the pgvector baseline.

Indexing is idempotent: a chunk is embedded again only when its content hash changes.
Run: `python -m src.rag.vector_store index [--pull] [--chunks PATH] [--copy-from-sql]`.
One document at a time (same chunking, deterministic point ids, no duplicate on a re-run):
`python -m src.rag.vector_store add|update --document doc.json`, `delete|show --doc-id ID`;
trials on a copy: `sample --to COLLECTION --n 300`, `drop --collection COLLECTION`. Writing
to a served collection needs `--allow-served`; a served collection is never dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pyarrow.parquet as pq
from pgvector.sqlalchemy import Vector
from qdrant_client import QdrantClient
from qdrant_client import models as qm
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Engine,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    delete,
    func,
    insert,
    inspect,
    not_,
    or_,
    select,
    text,
    type_coerce,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.engine import Connection
from sqlalchemy.exc import OperationalError

from app.config import Settings
from src.rag import chunking
from src.rag.embedder import (
    Embedder,
    FastEmbedBM25,
    OllamaEmbedder,
    SparseEncoder,
    SparseVector,
    pull_models,
)

# --- The chunk, its metadata and the filters ---------------------------------------------------

TABLE = "rag_chunks"
META_TABLE = "rag_index_meta"
METADATA_COLUMNS = (
    "chunk_id",
    "doc_id",
    "doc_type",
    "ingredients",
    "mentioned_ingredients",
    "section",
    "chunk_index",
    "title",
    "text",
    "source_name",
    "url",
    "date",
    "version",
    "license",
    "article_type",
    "is_retracted",
    "language",
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str  # label | abstract | interaction
    ingredients: list[str]
    mentioned_ingredients: list[str]
    section: str
    chunk_index: int
    title: str
    text: str
    source_name: str
    url: str
    date: str
    version: str
    license: str
    article_type: str | None
    is_retracted: bool
    language: str

    @property
    def search_text(self) -> str:
        """What is embedded and matched by BM25: the title gives each chunk its context."""
        return f"{self.title}\n{self.text}"


@dataclass(frozen=True)
class Filters:
    """Metadata filters. `ingredients` matches a chunk about OR mentioning one of them."""

    ingredients: tuple[str, ...] = ()
    doc_types: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    exclude_retracted: bool = True

    def accepts(self, chunk: Chunk) -> bool:
        if self.exclude_retracted and chunk.is_retracted:
            return False
        if self.doc_types and chunk.doc_type not in self.doc_types:
            return False
        if self.sections and chunk.section not in self.sections:
            return False
        if self.ingredients:
            names = set(chunk.ingredients) | set(chunk.mentioned_ingredients)
            if not names & set(self.ingredients):
                return False
        return True


class VectorStore(Protocol):
    """Chunks, their embeddings and the index metadata (model, dim, chunks, indexed_at)."""

    dim: int

    @property
    def kind(self) -> str: ...

    @property
    def native_dense(self) -> bool:
        """True when `dense_search` runs in the store; False: the retriever ranks in numpy."""
        ...

    def exists(self) -> bool: ...

    def create(self) -> None: ...

    def drop(self) -> None: ...

    def hashes(self) -> dict[str, str]: ...

    def document_hashes(self, doc_id: str) -> dict[str, str]:
        """chunk_id -> content_hash of the chunks of one document (empty when not indexed)."""
        ...

    def get_meta(self) -> dict[str, str]: ...

    def write(self, records: list[dict[str, Any]]) -> None:
        """Insert or replace chunks: metadata columns, `content_hash` and `embedding`."""
        ...

    def finish(self, stale: list[str], meta: dict[str, str]) -> None:
        """Delete the chunks gone from the corpus and record the index metadata."""
        ...

    def load_chunks(self) -> list[Chunk]: ...

    def load_embeddings(self) -> tuple[list[str], np.ndarray]: ...

    def dense_search(self, vector: list[float], filters: Filters, k: int) -> list[str]: ...

    def has_sparse(self) -> bool:
        """True when the store keeps BM25 sparse vectors; otherwise BM25 runs in memory."""
        ...

    def add_sparse(self) -> int:
        """Give an existing index the sparse vector; returns the number of points migrated."""
        ...

    def set_sparse(self, vectors: dict[str, SparseVector]) -> None: ...

    def sparse_search(self, vector: SparseVector, filters: Filters, k: int) -> list[str]: ...

    def hybrid_search(
        self, dense: list[float], sparse: SparseVector, filters: Filters, k: int, candidates: int
    ) -> list[tuple[str, float]]: ...

    def info(self) -> dict[str, Any]:
        """Collection (or table) name, number of points, status, indexed payload fields."""
        ...


# --- The SQL store: pgvector on PostgreSQL, numpy on SQLite ------------------------------------


def chunk_table(metadata: MetaData, dim: int) -> Table:
    names = JSON().with_variant(ARRAY(String), "postgresql")
    return Table(
        TABLE,
        metadata,
        Column("chunk_id", String(32), primary_key=True),
        Column("doc_id", String(120), nullable=False, index=True),
        Column("doc_type", String(20), nullable=False, index=True),
        Column("ingredients", names, nullable=False),
        Column("mentioned_ingredients", names, nullable=False),
        Column("section", String(80), nullable=False),
        Column("chunk_index", Integer, nullable=False),
        Column("title", Text, nullable=False),
        Column("text", Text, nullable=False),
        Column("source_name", String(200), nullable=False),
        Column("url", Text, nullable=False),
        Column("date", String(20), nullable=False),
        Column("version", String(200), nullable=False),
        Column("license", String(200), nullable=False),
        Column("article_type", String(40)),
        Column("is_retracted", Boolean, nullable=False),
        Column("language", String(8), nullable=False),
        Column("content_hash", String(64), nullable=False),
        Column("embedding", JSON().with_variant(Vector(dim), "postgresql"), nullable=False),
    )


def meta_table(metadata: MetaData) -> Table:
    return Table(
        META_TABLE,
        metadata,
        Column("key", String(40), primary_key=True),
        Column("value", Text, nullable=False),
    )


@dataclass
class ChunkStore:
    engine: Engine
    dim: int
    metadata: MetaData = field(default_factory=MetaData)

    def __post_init__(self) -> None:
        self.table = chunk_table(self.metadata, self.dim)
        self.meta = meta_table(self.metadata)

    @property
    def is_postgres(self) -> bool:
        return self.engine.dialect.name == "postgresql"

    @property
    def kind(self) -> str:
        return "pgvector" if self.is_postgres else "sqlite"

    @property
    def native_dense(self) -> bool:
        return self.is_postgres

    def has_sparse(self) -> bool:
        return False

    def add_sparse(self) -> int:
        return 0

    def set_sparse(self, vectors: dict[str, SparseVector]) -> None:
        raise NotImplementedError("the SQL store has no sparse vectors")

    def sparse_search(self, vector: SparseVector, filters: Filters, k: int) -> list[str]:
        raise NotImplementedError("the SQL store has no sparse vectors: BM25 runs in memory")

    def hybrid_search(
        self, dense: list[float], sparse: SparseVector, filters: Filters, k: int, candidates: int
    ) -> list[tuple[str, float]]:
        raise NotImplementedError("the SQL store has no sparse vectors: RRF runs in memory")

    def create(self) -> None:
        if self.is_postgres:
            with self.engine.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                if self.metadata.schema:  # tests use their own schema
                    conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self.metadata.schema}"'))
        self.metadata.create_all(self.engine)

    def drop(self) -> None:
        self.metadata.drop_all(self.engine)

    def exists(self) -> bool:
        return inspect(self.engine).has_table(TABLE, schema=self.metadata.schema)

    # ------------------------------------------------------------- writing

    def hashes(self) -> dict[str, str]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(self.table.c.chunk_id, self.table.c.content_hash))
            return {row.chunk_id: row.content_hash for row in rows}

    def document_hashes(self, doc_id: str) -> dict[str, str]:
        if not self.exists():
            return {}
        t = self.table
        with self.engine.connect() as conn:
            rows = conn.execute(select(t.c.chunk_id, t.c.content_hash).where(t.c.doc_id == doc_id))
            return {row.chunk_id: row.content_hash for row in rows}

    def get_meta(self) -> dict[str, str]:
        if not inspect(self.engine).has_table(META_TABLE, schema=self.metadata.schema):
            return {}
        with self.engine.connect() as conn:
            return {row.key: row.value for row in conn.execute(select(self.meta))}

    def set_meta(self, conn: Connection, values: dict[str, str]) -> None:
        conn.execute(delete(self.meta).where(self.meta.c.key.in_(list(values))))
        conn.execute(insert(self.meta), [{"key": k, "value": v} for k, v in values.items()])

    def upsert(self, conn: Connection, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        ids = [r["chunk_id"] for r in rows]
        conn.execute(delete(self.table).where(self.table.c.chunk_id.in_(ids)))
        conn.execute(insert(self.table), rows)

    def delete_ids(self, conn: Connection, ids: list[str]) -> None:
        for start in range(0, len(ids), 1000):
            batch = ids[start : start + 1000]
            conn.execute(delete(self.table).where(self.table.c.chunk_id.in_(batch)))

    def write(self, records: list[dict[str, Any]]) -> None:
        columns = set(self.table.c.keys())
        with self.engine.begin() as conn:
            self.upsert(conn, [{k: v for k, v in r.items() if k in columns} for r in records])

    def finish(self, stale: list[str], meta: dict[str, str]) -> None:
        with self.engine.begin() as conn:
            if stale:
                self.delete_ids(conn, stale)
            self.set_meta(conn, meta)

    # ------------------------------------------------------------- reading

    def load_chunks(self) -> list[Chunk]:
        columns = [self.table.c[name] for name in METADATA_COLUMNS]
        with self.engine.connect() as conn:
            rows = conn.execute(select(*columns).order_by(self.table.c.chunk_id)).mappings()
            return [
                Chunk(
                    **{
                        **row,
                        "ingredients": list(row["ingredients"] or []),
                        "mentioned_ingredients": list(row["mentioned_ingredients"] or []),
                    }
                )
                for row in rows
            ]

    def load_embeddings(self) -> tuple[list[str], np.ndarray]:
        t = self.table
        with self.engine.connect() as conn:
            rows = conn.execute(select(t.c.chunk_id, t.c.embedding).order_by(t.c.chunk_id)).all()
        ids = [row.chunk_id for row in rows]
        matrix = np.array([list(row.embedding) for row in rows], dtype=np.float32)
        return ids, matrix.reshape(len(ids), self.dim)

    def dense_search(self, vector: list[float], filters: Filters, k: int) -> list[str]:
        return self.dense_search_sql(vector, filters, k)

    def info(self) -> dict[str, Any]:
        with self.engine.connect() as conn:
            points = conn.execute(select(func.count()).select_from(self.table)).scalar_one()
        indexed = [c.name for c in self.table.c if c.index]
        return {"collection": TABLE, "points": points, "status": "ok", "payload_indexes": indexed}

    def dense_search_sql(self, vector: list[float], filters: Filters, k: int) -> list[str]:
        """Nearest chunks by cosine distance in PostgreSQL, filters in the WHERE clause."""
        t = self.table
        conditions = []
        if filters.exclude_retracted:
            conditions.append(not_(t.c.is_retracted))
        if filters.doc_types:
            conditions.append(t.c.doc_type.in_(filters.doc_types))
        if filters.sections:
            conditions.append(t.c.section.in_(filters.sections))
        if filters.ingredients:
            wanted = list(filters.ingredients)
            about = type_coerce(t.c.ingredients, ARRAY(String))
            mentioned = type_coerce(t.c.mentioned_ingredients, ARRAY(String))
            conditions.append(or_(about.overlap(wanted), mentioned.overlap(wanted)))
        # The columns are declared JSON-with-a-PostgreSQL-variant; the pgvector operators
        # need the PostgreSQL types.
        distance = type_coerce(t.c.embedding, Vector(self.dim)).cosine_distance(vector)
        query = select(t.c.chunk_id).where(*conditions).order_by(distance).limit(k)
        with self.engine.connect() as conn:
            return [row.chunk_id for row in conn.execute(query)]


# --- The Qdrant store: one point per chunk, its metadata as payload -----------------------------
# Named vectors `dense` (cosine) and `bm25` (sparse, IDF modifier); hybrid search fuses both (RRF).

COLLECTION = "tujya_chunks"
DENSE = "dense"
SPARSE = "bm25"
RRF_K = 60
POINT_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/melekbouabid/syRX/rag")
KEYWORD_FIELDS = (
    "chunk_id",
    "doc_id",
    "doc_type",
    "ingredients",
    "mentioned_ingredients",
    "section",
    "language",
    "source_name",
)
PAGE = 512


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def to_filter(filters: Filters) -> qm.Filter | None:
    """The metadata filters as a Qdrant filter (same rules as `Filters.accepts`)."""
    must: list[qm.Condition] = []
    if filters.exclude_retracted:
        must.append(qm.FieldCondition(key="is_retracted", match=qm.MatchValue(value=False)))
    if filters.doc_types:
        must.append(qm.FieldCondition(key="doc_type", match=qm.MatchAny(any=[*filters.doc_types])))
    if filters.sections:
        must.append(qm.FieldCondition(key="section", match=qm.MatchAny(any=[*filters.sections])))
    if filters.ingredients:
        wanted = [*filters.ingredients]
        must.append(
            qm.Filter(
                should=[
                    qm.FieldCondition(key="ingredients", match=qm.MatchAny(any=wanted)),
                    qm.FieldCondition(key="mentioned_ingredients", match=qm.MatchAny(any=wanted)),
                ]
            )
        )
    return qm.Filter(must=must) if must else None


def to_chunk(payload: dict[str, Any]) -> Chunk:
    values: dict[str, Any] = {name: payload.get(name) for name in METADATA_COLUMNS}
    values["ingredients"] = list(values["ingredients"] or [])
    values["mentioned_ingredients"] = list(values["mentioned_ingredients"] or [])
    return Chunk(**values)


@dataclass
class QdrantStore:
    client: QdrantClient
    dim: int
    collection: str = COLLECTION
    url: str = ""
    _meta: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def connect(
        cls, url: str, dim: int, collection: str = COLLECTION, *, timeout: int = 30
    ) -> QdrantStore:
        """`url` http://host:6333 for a server, ":memory:" for the local in-memory mode."""
        if url == ":memory:":
            return cls(QdrantClient(location=":memory:"), dim, collection, url)
        # No version check at construction (a network call): docker-compose.yml pins the
        # server to the client's version.
        client = QdrantClient(url=url, timeout=timeout, check_compatibility=False)
        return cls(client, dim, collection, url)

    @property
    def kind(self) -> str:
        return "qdrant"

    @property
    def native_dense(self) -> bool:
        return True

    def exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection))

    def create(self) -> None:
        if self.exists():
            return
        self.client.create_collection(
            self.collection,
            vectors_config={DENSE: qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE)},
            sparse_vectors_config={SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF)},
        )
        if self.url == ":memory:":
            return  # the local mode has no payload indexes (it filters by scanning)
        for name in KEYWORD_FIELDS:
            self.client.create_payload_index(
                self.collection, name, field_schema=qm.PayloadSchemaType.KEYWORD
            )
        self.client.create_payload_index(
            self.collection, "is_retracted", field_schema=qm.PayloadSchemaType.BOOL
        )

    def drop(self) -> None:
        if self.exists():
            self.client.delete_collection(self.collection)

    # ------------------------------------------------------------- writing

    def _scroll(self, payload: list[str] | bool, vectors: bool = False) -> list[qm.Record]:
        records: list[qm.Record] = []
        offset: Any = None
        while True:
            page, offset = self.client.scroll(
                self.collection,
                limit=PAGE,
                offset=offset,
                with_payload=payload,
                with_vectors=[DENSE] if vectors else False,
            )
            records.extend(page)
            if offset is None:
                return records

    def hashes(self) -> dict[str, str]:
        if not self.exists():
            return {}
        return {
            str(r.payload["chunk_id"]): str(r.payload["content_hash"])
            for r in self._scroll(["chunk_id", "content_hash"])
            if r.payload
        }

    def document_hashes(self, doc_id: str) -> dict[str, str]:
        if not self.exists():
            return {}
        condition = qm.Filter(
            must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
        )
        found: dict[str, str] = {}
        offset: Any = None
        while True:
            page, offset = self.client.scroll(
                self.collection,
                scroll_filter=condition,
                limit=PAGE,
                offset=offset,
                with_payload=["chunk_id", "content_hash"],
            )
            for record in page:
                if record.payload:
                    payload = record.payload
                    found[str(payload["chunk_id"])] = str(payload["content_hash"])
            if offset is None:
                return found

    def get_meta(self) -> dict[str, str]:
        if not self.exists():
            return {}
        metadata = self.client.get_collection(self.collection).config.metadata or {}
        return {str(k): str(v) for k, v in metadata.items()}

    def has_sparse(self) -> bool:
        """False for a collection without sparse vectors: `add_sparse` migrates it."""
        if not self.exists():
            return False
        params = self.client.get_collection(self.collection).config.params
        return SPARSE in (params.sparse_vectors or {})

    def add_sparse(self) -> int:
        """Recreate a collection without the sparse vector, keeping every point (payload and
        dense vector, nothing is embedded again) and the index metadata. Qdrant cannot add a
        vector to an existing collection. Returns the number of points copied."""
        if not self.exists() or self.has_sparse():
            return 0
        meta = self.get_meta()
        records = self._scroll(True, vectors=True)
        self.drop()
        self.create()
        for start in range(0, len(records), PAGE):
            batch = records[start : start + PAGE]
            points = [
                qm.PointStruct(id=r.id, vector={DENSE: _dense(r)}, payload=r.payload) for r in batch
            ]
            self.client.upsert(self.collection, points=points, wait=True)
        self.client.update_collection(self.collection, metadata=meta)
        return len(records)

    def set_sparse(self, vectors: dict[str, SparseVector]) -> None:
        """Replace the sparse vectors of existing points (chunk id -> vector)."""
        items = list(vectors.items())
        for start in range(0, len(items), PAGE):
            points = [
                qm.PointVectors(id=point_id(cid), vector={SPARSE: _sparse(v)})
                for cid, v in items[start : start + PAGE]
            ]
            self.client.update_vectors(self.collection, points=points, wait=True)

    def write(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        points = [
            qm.PointStruct(
                id=point_id(r["chunk_id"]),
                vector={
                    DENSE: list(map(float, r["embedding"])),
                    **({SPARSE: _sparse(r["sparse"])} if r.get("sparse") else {}),
                },
                payload={
                    **{k: r.get(k) for k in METADATA_COLUMNS},
                    "content_hash": r["content_hash"],
                    "embedding_model": r.get("embedding_model", ""),
                },
            )
            for r in records
        ]
        self.client.upsert(self.collection, points=points, wait=True)

    def finish(self, stale: list[str], meta: dict[str, str]) -> None:
        for start in range(0, len(stale), 1000):
            batch = [point_id(cid) for cid in stale[start : start + 1000]]
            self.client.delete(
                self.collection, points_selector=qm.PointIdsList(points=batch), wait=True
            )
        self.client.update_collection(self.collection, metadata={**self.get_meta(), **meta})

    # ------------------------------------------------------------- reading

    def load_chunks(self) -> list[Chunk]:
        records = self._scroll([*METADATA_COLUMNS])
        chunks = [to_chunk(r.payload) for r in records if r.payload]
        return sorted(chunks, key=lambda c: c.chunk_id)

    def load_embeddings(self) -> tuple[list[str], np.ndarray]:
        records = sorted(
            self._scroll(["chunk_id"], vectors=True),
            key=lambda r: str((r.payload or {})["chunk_id"]),
        )
        ids = [str((r.payload or {})["chunk_id"]) for r in records]
        vectors = [r.vector[DENSE] for r in records if isinstance(r.vector, dict)]
        return ids, np.array(vectors, dtype=np.float32).reshape(len(ids), self.dim)

    def dense_search(self, vector: list[float], filters: Filters, k: int) -> list[str]:
        response = self.client.query_points(
            self.collection,
            query=vector,
            using=DENSE,
            query_filter=to_filter(filters),
            limit=k,
            with_payload=["chunk_id"],
        )
        return [str((p.payload or {})["chunk_id"]) for p in response.points]

    def sparse_search(self, vector: SparseVector, filters: Filters, k: int) -> list[str]:
        if not vector.indices:
            return []
        response = self.client.query_points(
            self.collection,
            query=_sparse(vector),
            using=SPARSE,
            query_filter=to_filter(filters),
            limit=k,
            with_payload=["chunk_id"],
        )
        return [str((p.payload or {})["chunk_id"]) for p in response.points]

    def hybrid_search(
        self,
        dense: list[float],
        sparse: SparseVector,
        filters: Filters,
        k: int,
        candidates: int,
    ) -> list[tuple[str, float]]:
        """Dense and BM25 candidates, each filtered, fused by Qdrant with RRF (k = 60).

        Qdrant orders tied RRF scores differently from one call to the next, so every fused
        candidate is fetched and sorted by score, then chunk_id (as `retrieval.rrf`), before
        the cut to `k`: the same query always gives the same ranking.
        """
        condition = to_filter(filters)
        prefetch = [qm.Prefetch(query=dense, using=DENSE, filter=condition, limit=candidates)]
        if sparse.indices:
            prefetch.append(
                qm.Prefetch(query=_sparse(sparse), using=SPARSE, filter=condition, limit=candidates)
            )
        response = self.client.query_points(
            self.collection,
            prefetch=prefetch,
            # Qdrant scores 1 / (k + position), positions from 0: k = 61 there is exactly
            # 1 / (60 + rank) with ranks from 1, the fusion of the SQL store (retrieval.rrf).
            query=qm.RrfQuery(rrf=qm.Rrf(k=RRF_K + 1)),
            limit=max(k, len(prefetch) * candidates),  # every fused candidate: ties at the cut
            with_payload=["chunk_id"],
        )
        fused = [(str((p.payload or {})["chunk_id"]), float(p.score)) for p in response.points]
        return sorted(fused, key=lambda item: (-item[1], item[0]))[:k]

    def info(self) -> dict[str, Any]:
        """What the Documents page shows about the collection."""
        details = self.client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "points": details.points_count or 0,
            "status": str(details.status.value if details.status else ""),
            "payload_indexes": sorted((details.payload_schema or {}).keys()),
        }


def _sparse(vector: SparseVector) -> qm.SparseVector:
    return qm.SparseVector(indices=vector.indices, values=vector.values)


def _dense(record: qm.Record) -> list[float]:
    vector = record.vector
    assert isinstance(vector, dict), "named vectors expected"
    return list(map(float, vector[DENSE]))  # type: ignore[arg-type]


# --- The store the settings name: Qdrant (default) or the SQL table -----------------------------

# A store that cannot be reached (Qdrant or PostgreSQL down): the API answers 503.
STORE_ERRORS = (ResponseHandlingException, UnexpectedResponse, OperationalError, OSError)


def open_store(settings: Settings, dim: int, pharma_engine: Engine | None = None) -> VectorStore:
    if settings.vector_store == "sql":
        if pharma_engine is None:
            from app.db import make_engine

            pharma_engine = make_engine(settings.pharma_database_url)
        return ChunkStore(pharma_engine, dim)
    return QdrantStore.connect(settings.qdrant_url, dim, settings.qdrant_collection)


# --- Indexing the corpus (dense + sparse vectors), idempotent -----------------------------------


def read_chunks(path: Path) -> list[dict[str, Any]]:
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        row["ingredients"] = list(row.get("ingredients") or [])
        row["mentioned_ingredients"] = list(row.get("mentioned_ingredients") or [])
        for key in ("date", "version", "license", "url", "language", "section"):
            row[key] = str(row.get(key) or "")
        row["is_retracted"] = bool(row.get("is_retracted"))
    return rows


def content_hash(row: dict[str, Any], model: str) -> str:
    payload = json.dumps(
        {"model": model, **{k: row.get(k) for k in METADATA_COLUMNS}}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def index_chunks(
    store: VectorStore,
    embedder: Embedder,
    rows: list[dict[str, Any]],
    *,
    batch_size: int = 32,
    log: Any = None,
    sparse: SparseEncoder | None = None,
) -> dict[str, float]:
    sparse = sparse if store.kind == "qdrant" else None  # sparse vectors live in Qdrant only
    meta = store.get_meta() if store.exists() else {}
    if meta and (meta.get("model") != embedder.model or meta.get("dim") != str(store.dim)):
        store.drop()  # another embedding model: every vector must be recomputed
        meta = {}
    migrated = store.add_sparse() if sparse is not None else 0
    store.create()
    existing = store.hashes()
    wanted = {row["chunk_id"]: content_hash(row, embedder.model) for row in rows}
    todo = [row for row in rows if existing.get(row["chunk_id"]) != wanted[row["chunk_id"]]]
    stale = sorted(set(existing) - set(wanted))
    started = time.perf_counter()
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        texts = [f"{r['title']}\n{r['text']}" for r in batch]
        vectors = await embedder.embed(texts)
        sparse_vectors = sparse.documents(texts) if sparse is not None else [None] * len(batch)
        store.write(
            [
                {
                    **{k: r.get(k) for k in METADATA_COLUMNS},
                    "content_hash": wanted[r["chunk_id"]],
                    "embedding_model": embedder.model,
                    "embedding": vector,
                    "sparse": sparse_vector,
                }
                for r, vector, sparse_vector in zip(batch, vectors, sparse_vectors, strict=True)
            ]
        )
        if log and (start // batch_size) % 20 == 0:
            done = start + len(batch)
            log(f"embedded {done}/{len(todo)} ({time.perf_counter() - started:.0f} s)")
    values = {"model": embedder.model, "dim": str(store.dim), "chunks": str(len(rows))}
    resparsed = 0
    if sparse is not None:
        if meta.get("sparse_model") != sparse.name:  # new encoder, or migrated collection
            embedded = {r["chunk_id"] for r in todo}
            rest = [r for r in rows if r["chunk_id"] not in embedded]
            for start in range(0, len(rest), 256):
                batch = rest[start : start + 256]
                encoded = sparse.documents([f"{r['title']}\n{r['text']}" for r in batch])
                store.set_sparse({r["chunk_id"]: v for r, v in zip(batch, encoded, strict=True)})
            resparsed = len(rest)
        values["sparse_model"] = sparse.name
    if todo or stale or resparsed or "indexed_at" not in meta:
        values["indexed_at"] = now()
    store.finish(stale, values)
    return {
        "total": len(rows),
        "embedded": len(todo),
        "deleted": len(stale),
        "unchanged": len(rows) - len(todo),
        "sparse_recomputed": resparsed,
        "migrated": migrated,
        "seconds": round(time.perf_counter() - started, 1),
    }


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def copy_index(
    source: VectorStore, target: VectorStore, *, batch_size: int = 256
) -> dict[str, float]:
    """Copy chunks, hashes and vectors from one store to another, without embedding anything."""
    meta = source.get_meta()
    if not meta:
        raise ValueError("the source store has no index")
    if target.dim != source.dim:
        raise ValueError(f"dimension {source.dim} -> {target.dim}")
    target_meta = target.get_meta() if target.exists() else {}
    if target_meta and target_meta.get("model") != meta.get("model"):
        target.drop()
    target.create()
    started = time.perf_counter()
    chunks = source.load_chunks()
    ids, matrix = source.load_embeddings()
    position = {cid: i for i, cid in enumerate(ids)}
    hashes = source.hashes()
    stale = sorted(set(target.hashes()) - set(hashes))
    for start in range(0, len(chunks), batch_size):
        target.write(
            [
                {
                    **{k: getattr(c, k) for k in METADATA_COLUMNS},
                    "content_hash": hashes[c.chunk_id],
                    "embedding_model": meta.get("model", ""),
                    "embedding": matrix[position[c.chunk_id]].tolist(),
                }
                for c in chunks[start : start + batch_size]
            ]
        )
    # Copied points have no sparse vector: the next `index_chunks` computes them.
    target.finish(
        stale, {**meta, "indexed_at": now(), "copied_from": source.kind, "sparse_model": ""}
    )
    return {
        "copied": len(chunks),
        "deleted": len(stale),
        "seconds": round(time.perf_counter() - started, 1),
    }


# --- One document at a time: add, update, delete -----------------------------------------------
# A document is one JSON object. Its chunk ids come from its doc_id, section and chunk index, so
# the points of a document are always the same: adding it again overwrites, never duplicates.

DOCUMENT_FIELDS = ("doc_id", "doc_type", "title", "text", "source_name")


def document_rows(document: dict[str, Any]) -> list[dict[str, Any]]:
    """One document -> its chunk rows (the columns `read_chunks` gives).

    Required: doc_id, doc_type, title, text, source_name. Optional: section (default
    "document"), url, date, version, license, language (default "en"), ingredients,
    mentioned_ingredients, article_type, is_retracted. The text is cut like the corpus
    (`chunking.split_text`); a dosage section is refused, as in the corpus.
    """
    missing = [name for name in DOCUMENT_FIELDS if not str(document.get(name) or "").strip()]
    if missing:
        raise ValueError(f"document without {', '.join(missing)}")
    section = str(document.get("section") or "document")
    if chunking.FORBIDDEN_SECTION.search(section):
        raise ValueError(f"dosage section must never be indexed: {section}")

    def names(key: str) -> list[str]:
        return sorted({str(n).strip().lower() for n in document.get(key) or [] if str(n).strip()})

    ingredients = names("ingredients")
    mentioned = [n for n in names("mentioned_ingredients") if n not in ingredients]
    doc_id = str(document["doc_id"])
    rows = [
        {
            "chunk_id": chunking.chunk_id("document", doc_id, section, index),
            "doc_id": doc_id,
            "doc_type": str(document["doc_type"]),
            "ingredients": ingredients,
            "mentioned_ingredients": mentioned,
            "section": section,
            "chunk_index": index,
            "title": str(document["title"]),
            "text": piece,
            "source_name": str(document["source_name"]),
            "url": str(document.get("url") or ""),
            "date": str(document.get("date") or ""),
            "version": str(document.get("version") or ""),
            "license": str(document.get("license") or ""),
            "article_type": document.get("article_type"),
            "is_retracted": bool(document.get("is_retracted")),
            "language": str(document.get("language") or "en"),
        }
        for index, piece in enumerate(chunking.split_text(str(document["text"])))
    ]
    if not rows:
        raise ValueError("document without text")
    return rows


async def write_document(
    store: VectorStore,
    embedder: Embedder,
    rows: list[dict[str, Any]],
    *,
    sparse: SparseEncoder | None = None,
    must_exist: bool | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Upsert the chunks of one document and delete its chunks the new version no longer has.

    `must_exist`: True (update) refuses a document that is not indexed, False (add) one that is,
    None accepts both. Only the chunks whose content hash changed are embedded again; an
    unchanged document changes nothing. The index must use the same embedding model (and, when
    it keeps sparse vectors, the same sparse encoder).
    """
    doc_ids = {row["doc_id"] for row in rows}
    if len(doc_ids) != 1:
        raise ValueError(f"the chunks of exactly one document are expected, got {sorted(doc_ids)}")
    (doc_id,) = doc_ids
    meta = store.get_meta() if store.exists() else {}
    if meta.get("model", embedder.model) != embedder.model:
        raise ValueError(f"the index uses {meta['model']}, not {embedder.model}")
    if meta.get("dim", str(store.dim)) != str(store.dim):
        raise ValueError(f"the index has {meta['dim']} dimensions, not {store.dim}")
    sparse = sparse if store.kind == "qdrant" else None  # sparse vectors live in Qdrant only
    indexed_sparse = meta.get("sparse_model", "")
    if indexed_sparse and (sparse is None or sparse.name != indexed_sparse):
        raise ValueError(f"the index keeps {indexed_sparse} sparse vectors: use that encoder")
    if meta and not indexed_sparse:
        sparse = None  # a dense-only index stays dense-only
    store.create()
    existing = store.document_hashes(doc_id)
    if must_exist is True and not existing:
        raise LookupError(f"{doc_id} is not indexed: use add")
    if must_exist is False and existing:
        raise ValueError(f"{doc_id} is already indexed: use update")
    wanted = {row["chunk_id"]: content_hash(row, embedder.model) for row in rows}
    todo = [row for row in rows if existing.get(row["chunk_id"]) != wanted[row["chunk_id"]]]
    stale = sorted(set(existing) - set(wanted))
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        texts = [f"{r['title']}\n{r['text']}" for r in batch]
        vectors = await embedder.embed(texts)
        sparse_vectors = sparse.documents(texts) if sparse is not None else [None] * len(batch)
        store.write(
            [
                {
                    **{k: r.get(k) for k in METADATA_COLUMNS},
                    "content_hash": wanted[r["chunk_id"]],
                    "embedding_model": embedder.model,
                    "embedding": vector,
                    "sparse": sparse_vector,
                }
                for r, vector, sparse_vector in zip(batch, vectors, sparse_vectors, strict=True)
            ]
        )
    if todo or stale:
        values = {"indexed_at": now()}  # the API's retriever reloads on a new indexed_at
        if not meta:
            values.update({"model": embedder.model, "dim": str(store.dim)})
            if sparse is not None:
                values["sparse_model"] = sparse.name
        store.finish(stale, values)
        store.finish([], {"chunks": str(store.info()["points"])})
    return {
        "doc_id": doc_id,
        "chunks": len(rows),
        "embedded": len(todo),
        "deleted": len(stale),
        "unchanged": len(rows) - len(todo),
    }


def delete_document(store: VectorStore, doc_id: str) -> dict[str, Any]:
    """Delete every chunk of one document; 0 deleted when it is not indexed."""
    existing = store.document_hashes(doc_id)
    if existing:
        store.finish(sorted(existing), {"indexed_at": now()})
        store.finish([], {"chunks": str(store.info()["points"])})
    return {"doc_id": doc_id, "deleted": len(existing)}


def sample_collection(source: QdrantStore, target: QdrantStore, n: int) -> dict[str, Any]:
    """Copy `n` points of `source` (payload, dense and sparse vectors, index metadata) into a new
    `target` collection, for trials. The source is only read."""
    if target.collection == source.collection:
        raise ValueError("the sample must go to another collection")
    if not source.exists():
        raise ValueError(f"{source.collection} does not exist")
    if target.dim != source.dim:
        raise ValueError(f"dimension {source.dim} -> {target.dim}")
    records, _ = source.client.scroll(
        source.collection, limit=n, with_payload=True, with_vectors=True
    )
    target.drop()
    target.create()
    points = [qm.PointStruct(id=r.id, vector=r.vector or {}, payload=r.payload) for r in records]
    for start in range(0, len(points), PAGE):
        target.client.upsert(target.collection, points=points[start : start + PAGE], wait=True)
    meta = {
        **source.get_meta(),
        "chunks": str(len(points)),
        "indexed_at": now(),
        "copied_from": source.collection,
    }
    target.client.update_collection(target.collection, metadata=meta)
    return {"collection": target.collection, "copied": len(points), "from": source.collection}


# --- Command line ------------------------------------------------------------------------------


def run_index(args: argparse.Namespace, settings: Settings) -> None:
    target = settings.model_copy(
        update={
            "vector_store": args.store,
            "qdrant_url": args.qdrant_url,
            "pharma_database_url": args.database_url,
        }
    )
    store = open_store(target, args.dim)
    if args.copy_from_sql:
        if args.store != "qdrant":
            sys.exit("--copy-from-sql copies into Qdrant: use it with --store qdrant")
        from app.db import make_engine

        source = ChunkStore(make_engine(args.database_url), args.dim)
        print(json.dumps(copy_index(source, store)))
        return
    if not args.chunks.exists():  # a Parquet file (v1) or a partitioned folder (v2, Spark)
        sys.exit(f"{args.chunks} not found: run `dvc repro spark_rag` or `dvc pull`")
    if args.pull:
        asyncio.run(pull_models(args.ollama_url, [args.model, settings.llm_model]))
    rows = read_chunks(args.chunks)
    embedder = OllamaEmbedder(
        args.ollama_url, args.model, args.dim, num_ctx=settings.embedding_num_ctx
    )
    sparse = (
        None
        if args.no_sparse
        else FastEmbedBM25(args.sparse_model, cache_dir=settings.rag_model_cache_dir)
    )
    stats = asyncio.run(
        index_chunks(store, embedder, rows, batch_size=args.batch_size, log=print, sparse=sparse)
    )
    print(json.dumps(stats))


def protected(settings: Settings) -> set[str]:
    """The served collections: changed only with --allow-served, never dropped."""
    return {settings.qdrant_collection, COLLECTION, "tujya_chunks_v2"}


def run_document(args: argparse.Namespace, settings: Settings) -> None:
    """add | update | delete | show one document, sample | drop a trial collection."""
    if args.command == "drop":
        if args.collection in protected(settings):
            sys.exit(f"{args.collection} is a served collection: it is never dropped here")
    elif args.command in ("add", "update", "delete", "sample"):
        target = args.to if args.command == "sample" else args.collection
        if target in protected(settings) and not args.allow_served:
            sys.exit(f"{target} is a served collection: pass --allow-served to change it")
    connect = QdrantStore.connect
    if args.command == "sample":
        source = connect(args.qdrant_url, args.dim, args.collection, timeout=120)
        target_store = connect(args.qdrant_url, args.dim, args.to, timeout=120)
        print(json.dumps(sample_collection(source, target_store, args.n)))
        return
    store = connect(args.qdrant_url, args.dim, args.collection, timeout=120)
    if args.command == "drop":
        store.drop()
        print(json.dumps({"dropped": args.collection}))
    elif args.command == "delete":
        print(json.dumps(delete_document(store, args.doc_id)))
    elif args.command == "show":
        chunks = [c for c in store.load_chunks() if c.doc_id == args.doc_id]
        chunks.sort(key=lambda c: c.chunk_index)
        print(json.dumps({"doc_id": args.doc_id, "points": store.info()["points"]}))
        for c in chunks:
            print(f"[{c.chunk_index}] {c.chunk_id} {c.section} | {c.title}\n    {c.text}")
        if not chunks:
            print("    (not indexed)")
    else:
        document = json.loads(args.document.read_text(encoding="utf-8"))
        embedder = OllamaEmbedder(
            args.ollama_url, args.model, args.dim, num_ctx=settings.embedding_num_ctx
        )
        sparse = (
            None
            if args.no_sparse
            else FastEmbedBM25(args.sparse_model, cache_dir=settings.rag_model_cache_dir)
        )
        try:
            stats = asyncio.run(
                write_document(
                    store,
                    embedder,
                    document_rows(document),
                    sparse=sparse,
                    must_exist=args.command == "update",
                )
            )
        except (ValueError, LookupError) as refused:  # already indexed, unknown, wrong model
            sys.exit(f"{args.command} refused: {refused}")
        print(json.dumps(stats))


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(prog="python -m src.rag.vector_store")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("add", "add one document (JSON), refused when it is already indexed"),
        ("update", "replace one indexed document (JSON): new chunks in, old chunks out"),
        ("delete", "delete every chunk of one document"),
        ("show", "print the chunks of one document"),
        ("sample", "copy N points of --collection into the trial collection --to"),
        ("drop", "drop a trial collection (never a served one)"),
    ):
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("--collection", default=settings.qdrant_collection)
        sub.add_argument("--qdrant-url", default=settings.qdrant_url)
        sub.add_argument("--dim", type=int, default=settings.embedding_dim)
        if name in ("add", "update", "delete", "sample"):
            sub.add_argument("--allow-served", action="store_true")
        if name in ("add", "update"):
            sub.add_argument("--document", type=Path, required=True, help="a JSON object")
            sub.add_argument("--ollama-url", default=settings.ollama_url)
            sub.add_argument("--model", default=settings.embedding_model)
            sub.add_argument("--sparse-model", default=settings.rag_sparse_model)
            sub.add_argument("--no-sparse", action="store_true", help="dense vectors only")
        if name in ("delete", "show"):
            sub.add_argument("--doc-id", required=True)
        if name == "sample":
            sub.add_argument("--to", required=True)
            sub.add_argument("--n", type=int, default=300)
    index = commands.add_parser("index", help="index the corpus (idempotent)")
    index.add_argument("--chunks", type=Path, default=settings.rag_chunks_file)
    index.add_argument("--store", choices=["qdrant", "sql"], default=settings.vector_store)
    index.add_argument("--qdrant-url", default=settings.qdrant_url)
    index.add_argument("--database-url", default=settings.pharma_database_url)
    index.add_argument("--ollama-url", default=settings.ollama_url)
    index.add_argument("--model", default=settings.embedding_model)
    index.add_argument("--dim", type=int, default=settings.embedding_dim)
    index.add_argument("--batch-size", type=int, default=32)
    index.add_argument("--pull", action="store_true", help="pull missing Ollama models first")
    index.add_argument("--sparse-model", default=settings.rag_sparse_model)
    index.add_argument("--no-sparse", action="store_true", help="dense vectors only")
    index.add_argument(
        "--copy-from-sql",
        action="store_true",
        help="copy the SQL table rag_chunks (--database-url) into Qdrant, without embedding",
    )
    args = parser.parse_args(argv)
    if args.command == "index":
        run_index(args, settings)
    else:
        run_document(args, settings)


if __name__ == "__main__":
    main()
