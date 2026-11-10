"""Index the RAG corpus: data/processed/rag/corpus/chunks.parquet -> Qdrant collection
tujya_chunks (or the SQL table rag_chunks with `--store sql`).

Idempotent: each chunk keeps a SHA-256 hash of its text, metadata and embedding model, so a
second run embeds nothing, a changed chunk is re-embedded, and a chunk gone from the corpus
is deleted. Changing the embedding model (or its dimension) rebuilds the collection.

    python -m src.rag.index          # settings from TUJYA_* env
    python -m src.rag.index --pull   # also pull the Ollama models first
    python -m src.rag.index --store sql --database-url postgresql+psycopg://...
    python -m src.rag.index --copy-from-sql   # move the pgvector vectors to Qdrant, no embedding
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq

from app.config import Settings
from app.db import make_engine
from src.rag.embeddings import Embedder, OllamaEmbedder
from src.rag.store import METADATA_COLUMNS, ChunkStore, VectorStore
from src.rag.stores import open_store


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
) -> dict[str, float]:
    meta = store.get_meta() if store.exists() else {}
    if meta and (meta.get("model") != embedder.model or meta.get("dim") != str(store.dim)):
        store.drop()  # another embedding model: every vector must be recomputed
    store.create()
    existing = store.hashes()
    wanted = {row["chunk_id"]: content_hash(row, embedder.model) for row in rows}
    todo = [row for row in rows if existing.get(row["chunk_id"]) != wanted[row["chunk_id"]]]
    stale = sorted(set(existing) - set(wanted))
    started = time.perf_counter()
    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        vectors = await embedder.embed([f"{r['title']}\n{r['text']}" for r in batch])
        store.write(
            [
                {
                    **{k: r.get(k) for k in METADATA_COLUMNS},
                    "content_hash": wanted[r["chunk_id"]],
                    "embedding_model": embedder.model,
                    "embedding": vector,
                }
                for r, vector in zip(batch, vectors, strict=True)
            ]
        )
        if log and (start // batch_size) % 20 == 0:
            done = start + len(batch)
            log(f"embedded {done}/{len(todo)} ({time.perf_counter() - started:.0f} s)")
    values = {"model": embedder.model, "dim": str(store.dim), "chunks": str(len(rows))}
    if todo or stale or "indexed_at" not in meta:
        values["indexed_at"] = now()
    store.finish(stale, values)
    return {
        "total": len(rows),
        "embedded": len(todo),
        "deleted": len(stale),
        "unchanged": len(rows) - len(todo),
        "seconds": round(time.perf_counter() - started, 1),
    }


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def copy_index(
    source: VectorStore, target: VectorStore, *, batch_size: int = 256
) -> dict[str, float]:
    """Copy chunks, hashes and vectors from one store to another, without embedding anything.

    Used once to move the pgvector index to Qdrant (D113): same vectors, so the benchmark
    compares the stores and nothing else. A later `index_chunks` on the target finds every
    hash unchanged.
    """
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
    target.finish(stale, {**meta, "indexed_at": now(), "copied_from": source.kind})
    return {
        "copied": len(chunks),
        "deleted": len(stale),
        "seconds": round(time.perf_counter() - started, 1),
    }


async def pull_models(base_url: str, models: list[str]) -> None:
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


def main(argv: list[str] | None = None) -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(prog="python -m src.rag.index")
    parser.add_argument("--chunks", type=Path, default=settings.rag_chunks_file)
    parser.add_argument("--store", choices=["qdrant", "sql"], default=settings.vector_store)
    parser.add_argument("--qdrant-url", default=settings.qdrant_url)
    parser.add_argument("--database-url", default=settings.pharma_database_url)
    parser.add_argument("--ollama-url", default=settings.ollama_url)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--dim", type=int, default=settings.embedding_dim)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pull", action="store_true", help="pull missing Ollama models first")
    parser.add_argument(
        "--copy-from-sql",
        action="store_true",
        help="copy the SQL table rag_chunks (--database-url) into Qdrant, without embedding",
    )
    args = parser.parse_args(argv)
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
        source = ChunkStore(make_engine(args.database_url), args.dim)
        print(json.dumps(copy_index(source, store)))
        return
    if not args.chunks.is_file():
        sys.exit(f"{args.chunks} not found: run `python -m src.data.corpus corpus` or dvc pull")
    if args.pull:
        asyncio.run(pull_models(args.ollama_url, [args.model, settings.llm_model]))
    rows = read_chunks(args.chunks)
    embedder = OllamaEmbedder(
        args.ollama_url, args.model, args.dim, num_ctx=settings.embedding_num_ctx
    )
    stats = asyncio.run(index_chunks(store, embedder, rows, batch_size=args.batch_size, log=print))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
