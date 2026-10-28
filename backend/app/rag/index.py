"""Index the RAG corpus: data/rag/corpus/chunks.parquet -> table rag_chunks (pgvector).

Idempotent: each row keeps a hash of its text, metadata and embedding model, so a second
run embeds nothing, a changed chunk is re-embedded, and a chunk gone from the corpus is
deleted. Changing the embedding model (or its dimension) rebuilds the table.

    python -m app.rag.index                               # settings from SYNERGYRX_* env
    python -m app.rag.index --database-url postgresql+psycopg://tujya:tujya@localhost:5432/tujya
    python -m app.rag.index --pull                        # also pull the Ollama models first
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
from app.rag.embeddings import Embedder, OllamaEmbedder
from app.rag.store import METADATA_COLUMNS, ChunkStore


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
    store: ChunkStore,
    embedder: Embedder,
    rows: list[dict[str, Any]],
    *,
    batch_size: int = 32,
    log: Any = None,
) -> dict[str, int]:
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
        records = [
            {
                **{k: r.get(k) for k in METADATA_COLUMNS},
                "content_hash": wanted[r["chunk_id"]],
                "embedding": vector,
            }
            for r, vector in zip(batch, vectors, strict=True)
        ]
        with store.engine.begin() as conn:
            store.upsert(conn, records)
        if log and (start // batch_size) % 20 == 0:
            done = start + len(batch)
            log(f"embedded {done}/{len(todo)} ({time.perf_counter() - started:.0f} s)")
    changed = bool(todo or stale)
    with store.engine.begin() as conn:
        if stale:
            store.delete_ids(conn, stale)
        values = {"model": embedder.model, "dim": str(store.dim), "chunks": str(len(rows))}
        if changed or "indexed_at" not in meta:
            values["indexed_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        store.set_meta(conn, values)
    return {
        "total": len(rows),
        "embedded": len(todo),
        "deleted": len(stale),
        "unchanged": len(rows) - len(todo),
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
    parser = argparse.ArgumentParser(prog="python -m app.rag.index")
    parser.add_argument("--chunks", type=Path, default=settings.rag_chunks_file)
    parser.add_argument("--database-url", default=settings.pharma_database_url)
    parser.add_argument("--ollama-url", default=settings.ollama_url)
    parser.add_argument("--model", default=settings.embedding_model)
    parser.add_argument("--dim", type=int, default=settings.embedding_dim)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pull", action="store_true", help="pull missing Ollama models first")
    args = parser.parse_args(argv)
    if not args.chunks.is_file():
        sys.exit(
            f"{args.chunks} not found: run `python -m synergyrx_data.tujya corpus` or dvc pull"
        )
    if args.pull:
        asyncio.run(pull_models(args.ollama_url, [args.model, settings.llm_model]))
    rows = read_chunks(args.chunks)
    store = ChunkStore(make_engine(args.database_url), args.dim)
    embedder = OllamaEmbedder(
        args.ollama_url, args.model, args.dim, num_ctx=settings.embedding_num_ctx
    )
    stats = asyncio.run(index_chunks(store, embedder, rows, batch_size=args.batch_size, log=print))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
