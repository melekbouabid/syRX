"""The chunk index in Qdrant: one point per chunk, its metadata as payload (D113).

docker compose runs the server (`qdrant`, http://127.0.0.1:6333, dashboard on /dashboard);
the tests use the client's local in-memory mode (`QdrantClient(":memory:")`), same API.

- Collection `tujya_chunks` (TUJYA_QDRANT_COLLECTION); named vector `dense` (cosine, the
  embedding model's dimension).
- Point id: UUID v5 of the chunk id (Qdrant ids are integers or UUIDs); the chunk id stays
  in the payload.
- Payload: every metadata column of the chunk (source, document type, section, language,
  date, ingredients, URL, licence...), `content_hash` (SHA-256 of text + metadata + model,
  so an unchanged chunk is never embedded again) and `embedding_model`.
- Payload indexes on the filtered fields, so filtered search stays exact and fast.
- Index metadata (model, dim, chunks, indexed_at) in the collection's own `metadata`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

from src.rag.store import METADATA_COLUMNS, Chunk, Filters

COLLECTION = "tujya_chunks"
DENSE = "dense"
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

    def get_meta(self) -> dict[str, str]:
        if not self.exists():
            return {}
        metadata = self.client.get_collection(self.collection).config.metadata or {}
        return {str(k): str(v) for k, v in metadata.items()}

    def write(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        points = [
            qm.PointStruct(
                id=point_id(r["chunk_id"]),
                vector={DENSE: list(map(float, r["embedding"]))},
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

    def info(self) -> dict[str, Any]:
        """What the Documents page shows about the collection."""
        details = self.client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "points": details.points_count or 0,
            "status": str(details.status.value if details.status else ""),
            "payload_indexes": sorted((details.payload_schema or {}).keys()),
        }
