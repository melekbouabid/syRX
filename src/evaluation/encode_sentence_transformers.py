"""Encode a corpus variant and the benchmark queries with a sentence-transformers model (step 3.3).

The served embeddings (bge-m3) come from Ollama. The other models of the retrieval grid
(multilingual-e5-base, all-MiniLM-L6-v2, and the bge-m3 fine-tuned on Kaggle in step 3.5, an
export of sentence-transformers weights) have no Ollama build: this script encodes the chunks
and every query of the benchmark once, on the GPU of the training venv, and writes them to
`data/processed/rag/bench/vectors/<slug>.npz`. `bench_grid.py` (project venv) then loads the
vectors into a Qdrant collection of their own and searches it exactly like the served index.

    # training venv (torch, sentence-transformers); stop the Ollama models first (4 GB GPU)
    .venv-train/Scripts/python -m src.benchmark.encode_st --model intfloat/multilingual-e5-base
    .venv-train/Scripts/python -m src.benchmark.encode_st --model BAAI/bge-m3 --slug bge-m3-st
    .venv-train/Scripts/python -m src.benchmark.encode_st \
        --model runs/kaggle/finetune_bge-m3/model --slug bge-m3-finetuned

Queries: the questions and their rewritten English queries (`queries.json`, written by
`bench_grid.py --rewrites`). The latency of one query on the CPU (the API has no GPU) is
measured here too, median of 3 interleaved passes over 20 questions (protocol of D96), and
stored with the vectors.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmark.rechunk import load_variant

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "data" / "processed" / "rag" / "bench"
QUERIES = BENCH / "queries.json"
VECTORS = BENCH / "vectors"
# Prefixes the model was trained with (E5 needs them; bge-m3 and MiniLM use none).
PREFIXES = {"e5": ("query: ", "passage: ")}


def prefixes(model: str) -> tuple[str, str]:
    for key, value in PREFIXES.items():
        if key in Path(model).name.lower():
            return value
    return "", ""


def slug_of(model: str) -> str:
    return Path(model).name.lower().replace("_", "-")


def chunk_texts(frame: Any) -> tuple[list[str], list[str]]:
    """Ids and texts as the index embeds them (title, new line, text)."""
    ids = [str(c) for c in frame["chunk_id"]]
    texts = [f"{t}\n{x}" for t, x in zip(frame["title"], frame["text"], strict=True)]
    return ids, texts


def cpu_latency_ms(model_name: str, queries: list[str], rounds: int = 3) -> float:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    model.encode(queries[:2])
    medians = []
    for _ in range(rounds):
        times = []
        for query in queries:
            started = time.perf_counter()
            model.encode([query], normalize_embeddings=True)
            times.append((time.perf_counter() - started) * 1000)
        medians.append(statistics.median(times))
    return round(statistics.median(medians), 1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.benchmark.encode_st")
    parser.add_argument("--model", required=True, help="Hugging Face name or local folder")
    parser.add_argument("--slug", default="", help="name of the output (default: model name)")
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args(argv)
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = args.max_seq_length
    if device == "cuda":
        model.half()
    query_prefix, passage_prefix = prefixes(args.model)
    frame = load_variant(args.max_chars, args.overlap)
    ids, texts = chunk_texts(frame)
    started = time.perf_counter()
    chunks = model.encode(
        [passage_prefix + t for t in texts],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    seconds = time.perf_counter() - started
    queries: list[str] = sorted(set(json.loads(QUERIES.read_text(encoding="utf-8"))["all"]))
    encoded = model.encode([query_prefix + q for q in queries], normalize_embeddings=True)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    latency = cpu_latency_ms(args.model, [query_prefix + q for q in queries[:20]])
    slug = args.slug or slug_of(args.model)
    if (args.max_chars, args.overlap) != (1500, 1):
        slug += f"-c{args.max_chars}-o{args.overlap}"
    VECTORS.mkdir(parents=True, exist_ok=True)
    out = VECTORS / f"{slug}.npz"
    meta = {
        "model": args.model,
        "slug": slug,
        "dim": int(chunks.shape[1]),
        "max_chars": args.max_chars,
        "overlap": args.overlap,
        "chunks": len(ids),
        "query_prefix": query_prefix,
        "passage_prefix": passage_prefix,
        "max_seq_length": args.max_seq_length,
        "encode_device": device,
        "encode_seconds": round(seconds, 1),
        "query_cpu_ms": latency,
    }
    np.savez_compressed(
        out,
        chunk_ids=np.array(ids),
        chunks=chunks.astype(np.float32),
        queries=np.array(queries),
        query_vectors=np.asarray(encoded, dtype=np.float32),
        meta=np.array(json.dumps(meta)),
    )
    print(json.dumps(meta, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
