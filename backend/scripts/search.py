"""Search the RAG index from the command line and time each mode (real bge-m3 embeddings).

cd backend
python scripts/search.py "fluconazole warfarin bleeding" --ingredient warfarin --k 5
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.db import make_engine
from app.rag.embeddings import OllamaEmbedder
from app.rag.retrieval import Retriever
from app.rag.store import ChunkStore, Filters


async def run(args: argparse.Namespace) -> None:
    settings = Settings()
    embedder = OllamaEmbedder(
        settings.ollama_url,
        settings.embedding_model,
        settings.embedding_dim,
        num_ctx=settings.embedding_num_ctx,
    )
    started = time.perf_counter()
    store = ChunkStore(make_engine(settings.pharma_database_url), embedder.dim)
    retriever = Retriever(store, embedder)
    print(f"loaded {len(retriever.chunks)} chunks in {time.perf_counter() - started:.2f} s")
    filters = Filters(
        ingredients=tuple(args.ingredient),
        doc_types=tuple(args.doc_type),
        exclude_retracted=not args.include_retracted,
    )
    await retriever.search(args.query, k=1, mode="dense")  # warm the embedding model
    for mode in ("bm25", "dense", "hybrid"):
        started = time.perf_counter()
        hits = await retriever.search(args.query, k=args.k, mode=mode, filters=filters)
        print(f"\n== {mode} ({(time.perf_counter() - started) * 1000:.0f} ms)")
        for hit in hits:
            print(
                f"  {hit.score:.4f} bm25={hit.bm25_rank} dense={hit.dense_rank} {hit.chunk.title}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--ingredient", action="append", default=[])
    parser.add_argument("--doc-type", action="append", default=[])
    parser.add_argument("--include-retracted", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
