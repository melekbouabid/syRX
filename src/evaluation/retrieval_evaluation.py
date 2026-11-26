"""Retrieval evaluation with the metrics of the course: set metrics (Hit@k, Precision@k,
Recall@k, F1@k, R-Precision) and rank metrics (MRR, MAP, nDCG@k), from
src/evaluation/retrieval_metrics.py.

`run`: the main configurations on the served corpus v2 (collection tujya_chunks_v2, read
only: nothing is indexed or written there), the 85 benchmark questions that have qrels, top 20
kept per question, so the Recall@k / Precision@k curve covers k = 1..20. Relevant chunks: the
qrels (src/evaluation/qrels.json, chunk ids of the corpus v1) mapped to the v2 chunks by
`chunking_variants.Relevance`, as in the retrieval grid. Rewritten queries: the ones the grid
recorded (data/processed/rag/bench/queries.json), so the served configuration is replayed
exactly. `rerankers`: the reranker grid ran on the corpus v1 only; its recorded top-10
rankings (MLflow artifacts) are scored here with the same metrics, next to the v1
configuration without reranker, and never run again.

Results in src/evaluation/results/retrieval_metrics.json (+ per-question rankings in
retrieval_rankings.jsonl), one MLflow run per configuration (experiment tujya-rag-retrieval).
Run: `python -m src.evaluation.retrieval_evaluation run | rerankers | table [--no-mlflow]`.
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

from src.evaluation import retrieval_metrics as rm

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "src" / "evaluation" / "results"
OUT = RESULTS / "retrieval_metrics.json"
RANKINGS = RESULTS / "retrieval_rankings.jsonl"
GRID = RESULTS / "grid"
EXPERIMENT = "tujya-rag-retrieval"
COLLECTION = "tujya_chunks_v2"
CURVE_K = 20
TABLE_KS = (1, 3, 5, 10, 20)
# The served configuration: hybrid + query rewriting (backend/app/config.py), and the number of
# search passages the agent gives the LLM: `search_documents` default k = 5 (src/agent/tools.py),
# at most `rag_top_k` = 6 of them kept (src/agent/graph.py), 8 sources in all (rag_chain.py).
SERVED = "hybrid-rewrite"
LLM_K = 5
CONFIGS = {  # name: (mode, rewrite)
    "dense": ("dense", False),
    "bm25": ("bm25", False),
    "hybrid": ("hybrid", False),
    "hybrid-rewrite": ("hybrid", True),
}
# Reranker grid (corpus v1): the best configuration by MRR and the same one without reranker.
RERANKED = "bge-m3-hybrid-rewrite-rerank-minilm-l-12-v2-f20"
NOT_RERANKED = "bge-m3-hybrid-rewrite"


def summary(rows: list[dict[str, Any]], ks: tuple[int, ...]) -> dict[str, float]:
    """Means over the questions, at the k of the table; 4 decimals."""
    means = rm.mean_scores(r["scores"] for r in rows)
    wanted = [f"{m}_at_{k}" for k in ks for m in (*rm.SET_METRICS, "ndcg")]
    return {key: round(means[key], 4) for key in [*wanted, "r_precision", "mrr", "map"]}


def curve(rows: list[dict[str, Any]], max_k: int) -> dict[str, list[float]]:
    means = rm.mean_scores(r["scores"] for r in rows)
    return {
        metric: [round(means[f"{metric}_at_{k}"], 4) for k in range(1, max_k + 1)]
        for metric in ("recall", "precision", "hit", "f1", "ndcg")
    }


def flattening_k(recall: list[float], gain: float = 0.01) -> int:
    """Smallest k after which one more passage adds less than `gain` recall, for two steps
    in a row: where the curve flattens."""
    for k in range(1, len(recall) - 1):
        if recall[k] - recall[k - 1] < gain and recall[k + 1] - recall[k] < gain:
            return k
    return len(recall)


def scored(ranked: list[str], expected: set[str], max_k: int) -> dict[str, float]:
    return rm.score(ranked, expected, ks=range(1, max_k + 1))


# --- configurations on the served corpus v2 ------------------------------------------------


async def rank_v2(questions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Top-20 rankings of every configuration, read only on the served collection."""
    from app.config import Settings
    from src.evaluation.chunking_variants import Relevance
    from src.evaluation.retrieval_grid import SERVED_CHUNK, Runtime, load_corpus_v2, read_queries
    from src.rag.embedder import FastEmbedBM25, OllamaEmbedder
    from src.rag.retrieval import Mode, Retriever
    from src.rag.vector_store import Filters, QdrantStore

    runtime = Runtime.open()
    s: Settings = runtime.settings
    embedder = OllamaEmbedder(
        s.ollama_url, s.embedding_model, s.embedding_dim, num_ctx=s.embedding_num_ctx
    )
    store = QdrantStore.connect(s.qdrant_url, embedder.dim, COLLECTION, timeout=120)
    sparse = FastEmbedBM25(s.rag_sparse_model, cache_dir=s.rag_model_cache_dir)
    retriever = Retriever(store, embedder, sparse=sparse)
    relevance = Relevance(runtime.corpus(SERVED_CHUNK), load_corpus_v2())
    rewrites = read_queries()["rewrites"]
    await retriever.search("warm up", k=1, mode="dense")
    out: dict[str, list[dict[str, Any]]] = {}
    for name, (mode, rewrite) in CONFIGS.items():
        rows = []
        started = time.perf_counter()
        for item in questions:
            query = rewrites[item["id"]] if rewrite else item["question"]
            filters = Filters(ingredients=tuple(runtime.names.find(item["question"])))
            hits = await retriever.search(query, k=CURVE_K, mode=cast(Mode, mode), filters=filters)
            ranked = [h.chunk.chunk_id for h in hits]
            expected = relevance.expected(item)
            rows.append(
                {
                    "config": name,
                    "id": item["id"],
                    "language": item["language"],
                    "query": query,
                    "ranked": ranked,
                    "expected": sorted(expected),
                    "scores": scored(ranked, expected, CURVE_K),
                }
            )
        print(f"{name:16s} {len(rows)} questions in {time.perf_counter() - started:.1f} s")
        out[name] = rows
    return out


def params_of(name: str, corpus: str, **extra: Any) -> dict[str, Any]:
    mode, rewrite = CONFIGS.get(name, ("hybrid", True))
    return {
        "config": name,
        "corpus": corpus,
        "collection": COLLECTION if corpus == "v2" else "tujya_chunks",
        "embedding_model": "bge-m3",
        "mode": mode,
        "query_rewrite": rewrite,
        "filter": "names",
        "rrf_k": 60,
        "hybrid_candidates": 50,
        **extra,
    }


def log_run(name: str, params: dict[str, Any], record: dict[str, Any], rows: list[dict]) -> str:
    import mlflow

    from src.utils import mlflow_utils as mu

    mu.configure()
    corpus = params["corpus"]
    dataset_path = "data/processed/rag/" + ("corpus_v2" if corpus == "v2" else "corpus")
    with mu.start_tracked_run(
        EXPERIMENT,
        f"metrics-{name}" + ("" if corpus == "v2" else "-corpus-v1"),
        seed=0,
        dataset=f"rag-corpus-{corpus}+rag-questions",
        dataset_path=dataset_path,
        params={**params, "questions": len(rows)},
        tags={"benchmark": "retrieval-metrics", "plan_step": "4b"},
    ) as run:
        mlflow.log_metrics(record["metrics"])
        for metric, values in record.get("curve", {}).items():
            for k, value in enumerate(values, start=1):
                mlflow.log_metric(f"{metric}_curve", value, step=k)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{name}.json"
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            mlflow.log_artifact(str(path), "per_question")
        return str(run.info.run_id)


def read_results() -> dict[str, Any]:
    return json.loads(OUT.read_text(encoding="utf-8")) if OUT.is_file() else {}


def write_results(data: dict[str, Any]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


async def cmd_run(log: bool) -> None:
    from src.evaluation.retrieval_grid import load_questions

    questions = load_questions()  # the questions with qrels
    rankings = await rank_v2(questions)
    data = read_results()
    configs = data.setdefault("configs", {})
    for name, rows in rankings.items():
        record = {
            "params": params_of(name, "v2", top_k=CURVE_K),
            "metrics": summary(rows, TABLE_KS),
            "curve": curve(rows, CURVE_K),
            "questions": len(rows),
            "empty_expected": sum(1 for r in rows if not r["expected"]),
        }
        record["run_id"] = log_run(name, record["params"], record, rows) if log else ""
        configs[name] = record
    served = configs[SERVED]["curve"]["recall"]
    data["served"] = {
        "config": SERVED,
        "llm_k": LLM_K,
        "recall_at_llm_k": served[LLM_K - 1],
        "precision_at_llm_k": configs[SERVED]["curve"]["precision"][LLM_K - 1],
        "flattening_k": flattening_k(served),
        "flattening_rule": "first k with recall gain < 0.01 at k+1 and at k+2",
    }
    write_results(data)
    with RANKINGS.open("w", encoding="utf-8", newline="\n") as handle:
        for rows in rankings.values():
            for row in rows:
                slim = {k: v for k, v in row.items() if k != "scores"}
                handle.write(json.dumps(slim, ensure_ascii=False) + "\n")
    print(table(data))


# --- the reranker, from the rankings the grid recorded (corpus v1, top 10) ---------------------


def recorded_rows(run_id: str) -> list[dict[str, Any]]:
    """The per-question rankings a grid run logged (artifact per_question/<name>.json)."""
    import mlflow

    from src.utils import mlflow_utils as mu

    mu.configure()
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(
            mlflow.artifacts.download_artifacts(f"runs:/{run_id}/per_question", dst_path=tmp)
        )
        rows: list[dict[str, Any]] = json.loads(
            next(folder.glob("*.json")).read_text(encoding="utf-8")
        )
    return rows


def cmd_rerankers(log: bool) -> None:
    grid = {
        **json.loads((GRID / "bge-m3.json").read_text(encoding="utf-8")),
        **json.loads((GRID / "rerankers.json").read_text(encoding="utf-8")),
    }
    data = read_results()
    section = data.setdefault("corpus_v1_recorded", {})
    for name in (NOT_RERANKED, RERANKED):
        source = grid[name]
        rows = []
        for item in recorded_rows(source["run_id"]):
            expected = set(item["expected"])
            rows.append({**item, "scores": scored(item["ranked"], expected, 10)})
        params = params_of(
            name,
            "v1",
            top_k=10,
            rerank=bool(source.get("rerank")),
            rerank_model=source.get("rerank_model", ""),
            fetch_k=source.get("fetch_k", 0),
            source_run=source["run_id"],
        )
        record = {
            "params": params,
            "metrics": summary(rows, (1, 3, 5, 10)),
            "questions": len(rows),
            "source_run": source["run_id"],
            "latency_ms": source.get("latency_d96_ms") or source.get("ms_per_question"),
        }
        record["run_id"] = log_run(name, params, record, rows) if log else ""
        section[name] = record
        print(f"{name}: {record['metrics']}")
    write_results(data)


# --- table for docs/module2/results.md ---------------------------------------------------------


def table(data: dict[str, Any]) -> str:
    lines = [
        "| Configuration | Hit@5 | P@5 | R@5 | F1@5 | nDCG@5 | R@10 | R@20 | R-Prec | MRR | MAP |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, record in data.get("configs", {}).items():
        m = record["metrics"]
        lines.append(
            f"| {name}{' (served)' if name == SERVED else ''} | {m['hit_at_5']:.3f} "
            f"| {m['precision_at_5']:.3f} | {m['recall_at_5']:.3f} | {m['f1_at_5']:.3f} "
            f"| {m['ndcg_at_5']:.3f} | {m['recall_at_10']:.3f} | {m['recall_at_20']:.3f} "
            f"| {m['r_precision']:.3f} | {m['mrr']:.3f} | {m['map']:.3f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.evaluation.retrieval_evaluation")
    parser.add_argument("command", choices=["run", "rerankers", "table"])
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if args.command == "run":
        asyncio.run(cmd_run(not args.no_mlflow))
    elif args.command == "rerankers":
        cmd_rerankers(not args.no_mlflow)
    else:
        print(table(read_results()))


if __name__ == "__main__":
    main()
