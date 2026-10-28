"""The chunk table: text, metadata and embedding of every corpus chunk.

PostgreSQL (docker compose): `embedding` is a pgvector `vector(dim)` and the ingredient
lists are `text[]`; dense search runs in SQL (`<=>`, cosine distance) with the metadata
filters in the WHERE clause. Exact search on purpose: at ~10^4 chunks it takes a few
milliseconds and its recall is exact; an HNSW index is only worth it from ~10^5 chunks
(docs/RAG.md). SQLite (tests): the same columns as JSON, dense search in numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    delete,
    insert,
    inspect,
    not_,
    or_,
    select,
    text,
    type_coerce,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.engine import Connection, Engine

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
