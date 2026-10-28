"""Retrieval benchmark: Recall@k (k = 1, 3, 5, 10) and MRR for BM25, dense and hybrid.

Each configuration is one MLflow run in the experiment `tujya-rag` (params, metrics, the
per-question results as an artifact). Results: docs/BENCHMARK.md.

    cd backend
    python scripts/bench_retrieval.py          # SYNERGYRX_PHARMA_DATABASE_URL, MLFLOW_TRACKING_URI

Metrics, over the questions that have expected chunks (benchmark/rag_questions.jsonl):
- Recall@k: share of questions with at least one expected chunk in the top k (hit rate);
- Coverage@k: mean share of a question's expected chunks found in the top k;
- MRR@10: mean of 1 / rank of the first expected chunk (0 if none in the top 10).
Two filter settings: `none` (the raw question), and `names` (the ingredients the agent
finds in the question, as metadata filter, which is what the agent does).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import make_engine
from app.pharma import Ingredient
from app.rag.embeddings import OllamaEmbedder
from app.rag.names import NameIndex
from app.rag.retrieval import Retriever
from app.rag.store import ChunkStore, Filters

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "benchmark" / "rag_questions.jsonl"
KS = (1, 3, 5, 10)
EXPERIMENT = "tujya-rag"


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


async def run_config(
    retriever: Retriever, names: NameIndex, questions: list[dict[str, Any]], mode: str, filt: str
) -> tuple[dict[str, float], list[dict[str, Any]], float]:
    per_question = []
    started = time.perf_counter()
    for item in questions:
        found = names.find(item["question"]) if filt == "names" else []
        filters = Filters(ingredients=tuple(found))
        hits = await retriever.search(item["question"], k=10, mode=mode, filters=filters)  # type: ignore[arg-type]
        ranked = [h.chunk.chunk_id for h in hits]
        scores = score(ranked, set(item["expected_chunks"]))
        per_question.append(
            {
                "id": item["id"],
                "type": item["type"],
                "language": item["language"],
                "filter_ingredients": found,
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


async def main() -> None:
    settings = Settings()
    engine = make_engine(settings.pharma_database_url)
    embedder = OllamaEmbedder(
        settings.ollama_url,
        settings.embedding_model,
        settings.embedding_dim,
        num_ctx=settings.embedding_num_ctx,
    )
    retriever = Retriever(ChunkStore(engine, embedder.dim), embedder)
    with Session(engine) as session:
        names = NameIndex(session.scalars(select(Ingredient.name)).all())
    questions = load_questions()
    await retriever.search("warm up", k=1, mode="dense")
    results = {}
    for filt in ("none", "names"):
        for mode in ("bm25", "dense", "hybrid"):
            name = f"retrieval-{mode}-{filt}"
            summary, rows, seconds = await run_config(retriever, names, questions, mode, filt)
            params = {
                "mode": mode,
                "filter": filt,
                "embedding_model": embedder.model,
                "chunks": len(retriever.chunks),
                "questions": len(questions),
                "rrf_k": 60,
                "candidates": 50,
                "bm25_k1": 1.2,
                "bm25_b": 0.75,
            }
            run_id = log_mlflow(name, params, summary, rows)
            results[name] = {**summary, "run_id": run_id}
            print(
                f"{name:26s} R@1 {summary['recall_at_1']:.3f} R@3 {summary['recall_at_3']:.3f} "
                f"R@5 {summary['recall_at_5']:.3f} R@10 {summary['recall_at_10']:.3f} "
                f"MRR {summary['mrr']:.3f} ({summary['ms_per_query']} ms/q, {seconds:.0f} s)"
            )
    out = ROOT / "benchmark" / "results_retrieval.json"
    out.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"-> {out}")


if __name__ == "__main__":
    asyncio.run(main())
