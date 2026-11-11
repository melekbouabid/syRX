"""Retrieval benchmark: Recall@k (k = 1, 3, 5, 10) and MRR for BM25, dense and hybrid.

Each configuration is one MLflow run in the experiment `tujya-rag` (params, metrics, the
per-question results as an artifact). Results: docs/module2/benchmark.md.

    # from the repository root, project venv (.venv) active
    python -m src.benchmark.bench_retrieval                 # Qdrant (TUJYA_QDRANT_URL)
    python -m src.benchmark.bench_retrieval --store sql     # pgvector, the store before D113
    # step 3.2: reranking and query rewriting (D117, D118), merged into the results file
    python -m src.benchmark.bench_retrieval --configs dense-names-rerank,hybrid-names-rerank
    python -m src.benchmark.bench_retrieval --configs dense-names-rewrite --fetch-k 10
    # needs TUJYA_PHARMA_DATABASE_URL (ingredient names) and MLFLOW_TRACKING_URI

Metrics, over the questions that have expected chunks (benchmark/rag_questions.jsonl):
- Recall@k: share of questions with at least one expected chunk in the top k (hit rate);
- Coverage@k: mean share of a question's expected chunks found in the top k;
- MRR@10: mean of 1 / rank of the first expected chunk (0 if none in the top 10).
Two filter settings: `none` (the raw question), and `names` (the ingredients the agent
finds in the question, as metadata filter, which is what the agent does).
A configuration is `<mode>-<filter>`, plus `-rerank` (the cross-encoder reads the first
`--fetch-k` results and reorders them; Recall@10 needs fetch_k >= 10) and/or `-rewrite` (the
LLM rewrites the question into an English query first; the filter still comes from the
question). BM25 and hybrid run in Qdrant when the index has the `Qdrant/bm25` sparse vectors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import make_engine
from app.pharma import Ingredient
from src.rag.embeddings import OllamaEmbedder
from src.rag.names import NameIndex
from src.rag.rerank import CrossEncoderReranker
from src.rag.retrieval import Mode, Retriever
from src.rag.rewrite import QueryRewriter
from src.rag.sparse import FastEmbedBM25
from src.rag.store import Filters
from src.rag.stores import open_store

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "src" / "benchmark" / "rag_questions.jsonl"
KS = (1, 3, 5, 10)
EXPERIMENT = "tujya-rag"
BASE_CONFIGS = [f"{m}-{f}" for f in ("none", "names") for m in ("bm25", "dense", "hybrid")]


def load_questions() -> list[dict[str, Any]]:
    items = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]
    return [item for item in items if item["expected_chunks"]]


def score(ranked: list[str], expected: set[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in KS:
        top = set(ranked[:k])
        result[f"recall_at_{k}"] = float(bool(top & expected))
        result[f"coverage_at_{k}"] = len(top & expected) / len(expected)
    rank = next((i for i, cid in enumerate(ranked[:10], start=1) if cid in expected), None)
    result["mrr"] = 1.0 / rank if rank else 0.0
    return result


def mean(rows: list[dict[str, float]], key: str) -> float:
    return round(sum(r[key] for r in rows) / len(rows), 4) if rows else 0.0


def parse_config(name: str) -> dict[str, Any]:
    """'hybrid-names-rerank' -> mode hybrid, filter names, rerank on, rewrite off."""
    mode, filt, *options = name.split("-")
    if mode not in ("bm25", "dense", "hybrid") or filt not in ("none", "names"):
        raise SystemExit(f"bad configuration {name!r}: <mode>-<filter>[-rerank][-rewrite]")
    if set(options) - {"rerank", "rewrite"}:
        raise SystemExit(f"bad option in {name!r}: rerank or rewrite")
    return {
        "mode": mode,
        "filter": filt,
        "rerank": "rerank" in options,
        "rewrite": "rewrite" in options,
    }


async def run_config(
    retriever: Retriever,
    names: NameIndex,
    questions: list[dict[str, Any]],
    mode: str,
    filt: str,
    *,
    rerank: bool = False,
    fetch_k: int = 20,
    queries: dict[str, str] | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], float]:
    per_question = []
    started = time.perf_counter()
    for item in questions:
        found = names.find(item["question"]) if filt == "names" else []
        filters = Filters(ingredients=tuple(found))
        query = (queries or {}).get(item["id"], item["question"])
        hits = await retriever.search(
            query, k=10, mode=cast(Mode, mode), filters=filters, rerank=rerank, fetch_k=fetch_k
        )
        ranked = [h.chunk.chunk_id for h in hits]
        scores = score(ranked, set(item["expected_chunks"]))
        per_question.append(
            {
                "id": item["id"],
                "type": item["type"],
                "language": item["language"],
                "filter_ingredients": found,
                "search_query": query,
                "ranked": ranked,
                **scores,
            }
        )
    seconds = time.perf_counter() - started
    keys = [f"recall_at_{k}" for k in KS] + [f"coverage_at_{k}" for k in KS] + ["mrr"]
    summary = {key: mean(per_question, key) for key in keys}
    for group in ("fr", "en"):
        rows = [r for r in per_question if r["language"] == group]
        summary[f"recall_at_5_{group}"] = mean(rows, "recall_at_5")
        summary[f"mrr_{group}"] = mean(rows, "mrr")
    for kind in sorted({r["type"] for r in per_question}):
        rows = [r for r in per_question if r["type"] == kind]
        summary[f"recall_at_5_{kind}"] = mean(rows, "recall_at_5")
    summary["ms_per_query"] = round(seconds * 1000 / len(questions), 1)
    return summary, per_question, seconds


def log_mlflow(name: str, params: dict[str, Any], metrics: dict[str, float], rows: Any) -> str:
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("benchmark", "retrieval")
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{name}.json"
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            mlflow.log_artifact(str(path))
        return str(run.info.run_id)


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.benchmark.bench_retrieval")
    parser.add_argument("--store", choices=["qdrant", "sql"], default=None)
    parser.add_argument(
        "--configs",
        default=",".join(BASE_CONFIGS),
        help="comma list of <mode>-<filter>[-rerank][-rewrite] (default: the six base ones)",
    )
    parser.add_argument("--fetch-k", type=int, default=None, help="candidates reranked")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "src" / "benchmark" / "results_retrieval.json"
    )
    args = parser.parse_args(argv)
    settings = Settings()
    if args.store:
        settings = settings.model_copy(update={"vector_store": args.store})
    engine = make_engine(settings.pharma_database_url)
    embedder = OllamaEmbedder(
        settings.ollama_url,
        settings.embedding_model,
        settings.embedding_dim,
        num_ctx=settings.embedding_num_ctx,
    )
    store = open_store(settings, embedder.dim, engine)
    configs = [parse_config(c.strip()) | {"name": c.strip()} for c in args.configs.split(",")]
    fetch_k = args.fetch_k or settings.rag_fetch_k
    cache = settings.rag_model_cache_dir
    retriever = Retriever(
        store,
        embedder,
        sparse=FastEmbedBM25(settings.rag_sparse_model, cache_dir=cache),
        reranker=CrossEncoderReranker(settings.rag_rerank_model, cache_dir=cache),
    )
    with Session(engine) as session:
        names = NameIndex(session.scalars(select(Ingredient.name)).all())
    questions = load_questions()
    await retriever.search("warm up", k=1, mode="dense")
    queries: dict[str, str] = {}
    if any(c["rewrite"] for c in configs):  # once per question, shared by the configs
        from src.agent.runtime import ollama_llm

        rewriter = QueryRewriter(ollama_llm(settings))
        for item in questions:
            queries[item["id"]] = await rewriter.rewrite(item["question"])
    out = args.out
    results = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    for config in configs:
        mode, filt, rerank = config["mode"], config["filter"], config["rerank"]
        name = f"retrieval-{config['name']}"
        summary, rows, seconds = await run_config(
            retriever,
            names,
            questions,
            mode,
            filt,
            rerank=rerank,
            fetch_k=fetch_k,
            queries=queries if config["rewrite"] else None,
        )
        params = {
            "mode": mode,
            "filter": filt,
            "embedding_model": embedder.model,
            "store": store.kind,
            "lexical": retriever.lexical_kind,
            "chunks": len(retriever.chunks),
            "questions": len(questions),
            "rrf_k": 60,
            "candidates": 50,
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
            "rerank": rerank,
            "rerank_model": settings.rag_rerank_model if rerank else "",
            "fetch_k": fetch_k if rerank else 0,
            "query_rewrite": config["rewrite"],
            "rewrite_llm": settings.llm_model if config["rewrite"] else "",
        }
        run_id = log_mlflow(name, params, summary, rows)
        results[name] = {**summary, "run_id": run_id}
        print(
            f"{name:34s} R@1 {summary['recall_at_1']:.3f} R@3 {summary['recall_at_3']:.3f} "
            f"R@5 {summary['recall_at_5']:.3f} R@10 {summary['recall_at_10']:.3f} "
            f"MRR {summary['mrr']:.3f} ({summary['ms_per_query']} ms/q, {seconds:.0f} s)",
            flush=True,
        )
        out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"-> {out}")


if __name__ == "__main__":
    asyncio.run(main())
