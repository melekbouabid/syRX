"""Retrieval grid of step 3.3: embeddings x modes x chunking x rerankers, one MLflow run each.

The configurations are in config/rag_grid.yaml, by group (one experiment branch
`experiment/rag-<group>` per group). Results: docs/module2/benchmark_retrieval.md, MLflow
experiment `tujya-rag-retrieval`, `src/benchmark/grid/<group>.json`.

    # project venv (.venv), from the repository root; stack running (qdrant, ollama, postgres,
    # mlflow), TUJYA_PHARMA_DATABASE_URL set (ingredient names of the filter)
    python -m src.benchmark.bench_grid rewrites            # once: queries.json (LLM rewrites)
    python -m src.benchmark.bench_grid run --group bge-m3  # the group's configurations
    python -m src.benchmark.bench_grid latency --configs a,b,c   # interleaved rounds (D96)
    python -m src.benchmark.bench_grid select              # champion / challenger (D123)

Every configuration searches like the agent (src/agent/tools.py `search_documents`): the
ingredients found in the question are a metadata filter (`names`), top 10 kept. Metrics over
the questions that have expected chunks: Recall@k (k = 1, 3, 5, 10: at least one expected
chunk in the top k), MRR@10, nDCG@10 (binary relevance) and Coverage@10, by language too.
A corpus variant (`chunk` other than 1500/1) has other chunk ids: its relevant chunks are
found by src/benchmark/rechunk.py `Relevance`.

Time per question: the search (embedding of the query, Qdrant, reranker) is timed during the
run; the rewrite is the median time the LLM took in `rewrites`; for a sentence-transformers
embedding the query vector is precomputed, so its CPU encoding time measured by encode_st.py
is added. The `latency` command measures the shortlisted configurations again in interleaved
rounds after a pause (D96: this laptop slows down under sustained load).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import yaml

from src.benchmark.rechunk import Relevance, load_variant

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "config" / "rag_grid.yaml"
QUESTIONS = ROOT / "src" / "benchmark" / "rag_questions.jsonl"
BENCH = ROOT / "data" / "processed" / "rag" / "bench"
QUERIES = BENCH / "queries.json"
VECTORS = BENCH / "vectors"
RESULTS = ROOT / "src" / "benchmark" / "grid"
EXPERIMENT = "tujya-rag-retrieval"
REGISTERED = "tujya-rag-retrieval"
KS = (1, 3, 5, 10)
SERVED_CHUNK = "1500/1"


@dataclass(frozen=True)
class Config:
    embedding: str  # ollama:<model> | st:<slug>
    mode: str = "dense"
    chunk: str = SERVED_CHUNK
    rewrite: bool = False
    rerank: str = ""
    fetch_k: int = 0
    group: str = ""
    extra: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def backend(self) -> str:
        return self.embedding.split(":", 1)[0]

    @property
    def model(self) -> str:
        return self.embedding.split(":", 1)[1]

    @property
    def max_chars(self) -> int:
        return int(self.chunk.split("/")[0])

    @property
    def overlap(self) -> int:
        return int(self.chunk.split("/")[1])

    @property
    def name(self) -> str:
        parts = [self.model.lower(), self.mode]
        if self.chunk != SERVED_CHUNK:
            parts.append(f"c{self.max_chars}-o{self.overlap}")
        if self.rewrite:
            parts.append("rewrite")
        if self.rerank:
            short = self.rerank.split("/")[-1].lower().replace("ms-marco-", "")
            parts.append(f"rerank-{short}-f{self.fetch_k}")
        return "-".join(parts)

    @property
    def collection(self) -> str:
        """The served index for the served model and chunking, a bench collection otherwise."""
        if self.backend == "ollama" and self.chunk == SERVED_CHUNK:
            return ""
        suffix = "" if self.chunk == SERVED_CHUNK else f"-c{self.max_chars}-o{self.overlap}"
        return f"tujya_bench_{self.model.lower()}{suffix}".replace(".", "_")

    def params(self) -> dict[str, Any]:
        return {
            "embedding_backend": self.backend,
            "embedding_model": self.model,
            "mode": self.mode,
            "filter": "names",
            "chunk_max_chars": self.max_chars,
            "chunk_overlap_sentences": self.overlap,
            "query_rewrite": self.rewrite,
            "rerank": bool(self.rerank),
            "rerank_model": self.rerank,
            "fetch_k": self.fetch_k if self.rerank else 0,
            "top_k": 10,
            "rrf_k": 60,
            "hybrid_candidates": 50,
            "group": self.group,
        }


def load_grid(path: Path = GRID) -> tuple[dict[str, Any], dict[str, list[Config]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    groups = {
        group: [Config(**{**item, "group": group}) for item in items]
        for group, items in raw["groups"].items()
    }
    return raw["selection"], groups


def load_questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [item for item in items if item["expected_chunks"]]


# --- metrics ----------------------------------------------------------------------------


def score(ranked: list[str], expected: set[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for k in KS:
        top = set(ranked[:k])
        result[f"recall_at_{k}"] = float(bool(top & expected))
    result["coverage_at_10"] = len(set(ranked[:10]) & expected) / len(expected) if expected else 0
    rank = next((i for i, cid in enumerate(ranked[:10], start=1) if cid in expected), None)
    result["mrr"] = 1.0 / rank if rank else 0.0
    gains = sum(
        1 / math.log2(i + 1) for i, cid in enumerate(ranked[:10], start=1) if cid in expected
    )
    ideal = sum(1 / math.log2(i + 1) for i in range(1, min(len(expected), 10) + 1))
    result["ndcg_at_10"] = gains / ideal if ideal else 0.0
    return result


METRICS = [f"recall_at_{k}" for k in KS] + ["mrr", "ndcg_at_10", "coverage_at_10"]


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    def mean(items: list[dict[str, Any]], key: str) -> float:
        return round(sum(r[key] for r in items) / len(items), 4) if items else 0.0

    summary = {key: mean(rows, key) for key in METRICS}
    for language in ("fr", "en"):
        part = [r for r in rows if r["language"] == language]
        summary[f"recall_at_5_{language}"] = mean(part, "recall_at_5")
        summary[f"mrr_{language}"] = mean(part, "mrr")
    return summary


# --- embedders ----------------------------------------------------------------------------


class PrecomputedEmbedder:
    """Vectors of encode_st.py: chunks by their indexed text, queries by their text."""

    def __init__(self, path: Path, chunk_texts: dict[str, str]) -> None:
        data = np.load(path)
        self.meta: dict[str, Any] = json.loads(str(data["meta"]))
        self.model = str(self.meta["slug"])
        self.dim = int(self.meta["dim"])
        position = {str(c): i for i, c in enumerate(data["chunk_ids"])}
        chunks = data["chunks"]
        self.by_text = {text: chunks[position[cid]] for cid, text in chunk_texts.items()}
        self.queries = {
            str(q): v for q, v in zip(data["queries"], data["query_vectors"], strict=True)
        }

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            vector = self.by_text.get(text)
            if vector is None:
                vector = self.queries.get(text)
            if vector is None:
                raise KeyError(f"no precomputed vector for {text[:60]!r}: run encode_st.py")
            out.append([float(x) for x in vector])
        return out


def rows_of(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Chunk rows as src/rag/index.py `read_chunks` gives them."""
    rows: list[dict[str, Any]] = frame.to_dict("records")
    for row in rows:
        row["ingredients"] = list(
            row.get("ingredients") if row.get("ingredients") is not None else []
        )
        mentioned = row.get("mentioned_ingredients")
        row["mentioned_ingredients"] = list(mentioned if mentioned is not None else [])
        for key in ("date", "version", "license", "url", "language", "section"):
            value = row.get(key)
            row[key] = (
                ""
                if value is None or (isinstance(value, float) and math.isnan(value))
                else str(value)
            )
        row["is_retracted"] = bool(row.get("is_retracted"))
        row["chunk_index"] = int(row["chunk_index"])
        if isinstance(row.get("article_type"), float):
            row["article_type"] = None
    return rows


# --- runtime ------------------------------------------------------------------------------


@dataclass
class Runtime:
    settings: Any
    engine: Any
    names: Any
    rerankers: dict[str, Any] = field(default_factory=dict)
    corpora: dict[str, pd.DataFrame] = field(default_factory=dict)

    @classmethod
    def open(cls) -> Runtime:
        from sqlalchemy import select
        from sqlalchemy.orm import Session

        from app.config import Settings
        from app.db import make_engine
        from app.pharma import Ingredient
        from src.rag.names import NameIndex

        settings = Settings()
        engine = make_engine(settings.pharma_database_url)
        with Session(engine) as session:
            names = NameIndex(session.scalars(select(Ingredient.name)).all())
        return cls(settings, engine, names)

    def corpus(self, chunk: str) -> pd.DataFrame:
        if chunk not in self.corpora:
            max_chars, overlap = (int(x) for x in chunk.split("/"))
            self.corpora[chunk] = load_variant(max_chars, overlap)
        return self.corpora[chunk]

    def reranker(self, model: str) -> Any:
        from src.rag.rerank import CrossEncoderReranker

        if model not in self.rerankers:
            cache = self.settings.rag_model_cache_dir
            self.rerankers[model] = CrossEncoderReranker(model, cache_dir=cache)
        return self.rerankers[model]

    async def retriever(self, config: Config) -> tuple[Any, float]:
        """The retriever of a configuration (its collection indexed first if needed) and the
        query encoding time to add (sentence-transformers vectors are precomputed)."""
        from src.rag.embeddings import OllamaEmbedder
        from src.rag.index import index_chunks
        from src.rag.qdrant_store import QdrantStore
        from src.rag.retrieval import Retriever
        from src.rag.sparse import FastEmbedBM25

        s = self.settings
        sparse = FastEmbedBM25(s.rag_sparse_model, cache_dir=s.rag_model_cache_dir)
        added_ms = 0.0
        embedder: Any
        if config.backend == "ollama":
            embedder = OllamaEmbedder(
                s.ollama_url, config.model, s.embedding_dim, num_ctx=s.embedding_num_ctx
            )
        else:
            frame = self.corpus(config.chunk)
            texts = {
                str(c): f"{t}\n{x}"
                for c, t, x in zip(frame["chunk_id"], frame["title"], frame["text"], strict=True)
            }
            slug = (
                config.model
                if config.chunk == SERVED_CHUNK
                else (f"{config.model}-c{config.max_chars}-o{config.overlap}")
            )
            embedder = PrecomputedEmbedder(VECTORS / f"{slug}.npz", texts)
            added_ms = float(embedder.meta["query_cpu_ms"])
        collection = config.collection or s.qdrant_collection
        store = QdrantStore.connect(s.qdrant_url, embedder.dim, collection, timeout=120)
        if config.collection:
            done = await index_chunks(
                store, embedder, rows_of(self.corpus(config.chunk)), sparse=sparse, log=print
            )
            print(f"  index {collection}: {done}", flush=True)
        reranker = self.reranker(config.rerank) if config.rerank else None
        retriever = Retriever(store, embedder, sparse=sparse, reranker=reranker)
        await retriever.search("warm up", k=1, mode="dense") if config.backend == "ollama" else None
        return retriever, added_ms


def read_queries() -> dict[str, Any]:
    if not QUERIES.is_file():
        raise SystemExit("no rewritten queries: run `python -m src.benchmark.bench_grid rewrites`")
    data: dict[str, Any] = json.loads(QUERIES.read_text(encoding="utf-8"))
    return data


async def evaluate(
    runtime: Runtime, config: Config, questions: list[dict[str, Any]], queries: dict[str, Any]
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    from src.rag.store import Filters

    retriever, added_ms = await runtime.retriever(config)
    relevance = None
    if config.chunk != SERVED_CHUNK:
        relevance = Relevance(runtime.corpus(SERVED_CHUNK), runtime.corpus(config.chunk))
    rows, times = [], []
    for item in questions:
        found = runtime.names.find(item["question"])
        query = queries["rewrites"][item["id"]] if config.rewrite else item["question"]
        started = time.perf_counter()
        if config.rerank:
            await asyncio.sleep(0)  # the reranker runs in a worker thread, like in the API
        hits = await retriever.search(
            query,
            k=10,
            mode=cast(Any, config.mode),
            filters=Filters(ingredients=tuple(found)),
            rerank=bool(config.rerank),
            fetch_k=config.fetch_k or 10,
        )
        times.append((time.perf_counter() - started) * 1000)
        ranked = [h.chunk.chunk_id for h in hits]
        expected = relevance.expected(item) if relevance else set(item["expected_chunks"])
        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "language": item["language"],
                "query": query,
                "ranked": ranked,
                "expected": sorted(expected),
                **score(ranked, expected),
            }
        )
    summary = summarize(rows)
    rewrite_ms = float(queries["rewrite_ms_median"]) if config.rewrite else 0.0
    search_ms = statistics.median(times) + added_ms
    summary["search_ms_median"] = round(search_ms, 1)
    summary["rewrite_ms_median"] = round(rewrite_ms, 1)
    summary["ms_per_question"] = round(search_ms + rewrite_ms, 1)
    summary["chunks"] = float(len(retriever.chunks))
    summary["empty_expected"] = float(sum(1 for r in rows if not r["expected"]))
    return summary, rows


def log_run(config: Config, summary: dict[str, float], rows: list[dict[str, Any]]) -> str:
    import mlflow

    from src.utils import mlflow_utils as mu

    mu.configure()
    with mu.start_tracked_run(
        EXPERIMENT,
        config.name,
        seed=0,
        dataset="rag-corpus-v1+rag-questions",
        dataset_path="data/processed/rag/corpus",
        params={**config.params(), "questions": len(rows)},
        tags={"benchmark": "retrieval", "plan_step": "3.3", "group": config.group},
    ) as run:
        mlflow.log_metrics(summary)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{config.name}.json"
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            mlflow.log_artifact(str(path), "per_question")
        return str(run.info.run_id)


def save(group: str, name: str, record: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{group}.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    data[name] = record
    path.write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def line(name: str, s: dict[str, float]) -> str:
    return (
        f"{name:58s} R@1 {s['recall_at_1']:.3f} R@5 {s['recall_at_5']:.3f} "
        f"R@10 {s['recall_at_10']:.3f} MRR {s['mrr']:.3f} nDCG {s['ndcg_at_10']:.3f} "
        f"{s['ms_per_question']:.0f} ms"
    )


# --- commands -----------------------------------------------------------------------------


async def cmd_rewrites(runtime: Runtime, questions: list[dict[str, Any]]) -> None:
    from src.agent.runtime import ollama_llm
    from src.rag.rewrite import QueryRewriter

    rewriter = QueryRewriter(ollama_llm(runtime.settings))
    await rewriter.rewrite("warm up")
    rewrites, times = {}, []
    for item in questions:
        started = time.perf_counter()
        rewrites[item["id"]] = await rewriter.rewrite(item["question"])
        times.append((time.perf_counter() - started) * 1000)
    data = {
        "llm": runtime.settings.llm_model,
        "rewrites": rewrites,
        "rewrite_ms_median": round(statistics.median(times), 1),
        "rewrite_ms_mean": round(statistics.mean(times), 1),
        "all": sorted({*rewrites.values(), *(q["question"] for q in questions)}),
    }
    BENCH.mkdir(parents=True, exist_ok=True)
    QUERIES.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rewrites)} rewrites, median {data['rewrite_ms_median']} ms -> {QUERIES}")


async def cmd_run(runtime: Runtime, configs: list[Config], log: bool) -> None:
    questions = load_questions()
    queries = read_queries()
    for config in configs:
        summary, rows = await evaluate(runtime, config, questions, queries)
        run_id = log_run(config, summary, rows) if log else ""
        save(config.group, config.name, {**config.params(), **summary, "run_id": run_id})
        print(line(config.name, summary), flush=True)


async def cmd_latency(runtime: Runtime, configs: list[Config], selection: dict[str, Any]) -> None:
    """Interleaved rounds (D96): every round times each configuration on the same questions
    after a pause, in a rotated order; the median of the round medians is kept."""
    from src.rag.store import Filters

    questions = load_questions()[:: max(1, 85 // int(selection["latency_questions"]))]
    questions = questions[: int(selection["latency_questions"])]
    queries = read_queries()
    built = {c.name: await runtime.retriever(c) for c in configs}
    medians: dict[str, list[float]] = {c.name: [] for c in configs}
    for round_ in range(int(selection["latency_rounds"])):
        order = configs[round_ % len(configs) :] + configs[: round_ % len(configs)]
        for config in order:
            await asyncio.sleep(float(selection["latency_pause_s"]))
            retriever, added_ms = built[config.name]
            times = []
            for item in questions:
                query = queries["rewrites"][item["id"]] if config.rewrite else item["question"]
                found = runtime.names.find(item["question"])
                started = time.perf_counter()
                await retriever.search(
                    query,
                    k=10,
                    mode=cast(Any, config.mode),
                    filters=Filters(ingredients=tuple(found)),
                    rerank=bool(config.rerank),
                    fetch_k=config.fetch_k or 10,
                )
                times.append((time.perf_counter() - started) * 1000 + added_ms)
            medians[config.name].append(statistics.median(times))
    for config in configs:
        search = statistics.median(medians[config.name])
        rewrite = float(queries["rewrite_ms_median"]) if config.rewrite else 0.0
        total = round(search + rewrite, 1)
        path = RESULTS / f"{config.group}.json"
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        record = data.get(config.name, {})
        record.update({"latency_d96_ms": total, "latency_d96_rounds": medians[config.name]})
        save(config.group, config.name, record)
        if record.get("run_id"):
            import mlflow

            from src.utils import mlflow_utils as mu

            mu.configure()
            with mlflow.start_run(run_id=record["run_id"]):
                mlflow.log_metric("latency_d96_ms", total)
        print(f"{config.name:58s} {total:8.0f} ms per question (rounds {medians[config.name]})")


def all_results() -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS.glob("*.json")):
        for name, record in json.loads(path.read_text(encoding="utf-8")).items():
            found[name] = record
    return found


def choose(results: dict[str, dict[str, Any]], selection: dict[str, Any]) -> list[str]:
    """Names within the latency budget, best `metric` first (ties: faster first). The time is
    the interleaved measurement when there is one, the time of the run otherwise."""
    metric, budget = selection["metric"], float(selection["latency_budget_ms"])

    def time_of(record: dict[str, Any]) -> float:
        return float(record.get("latency_d96_ms") or record["ms_per_question"])

    within = [n for n, r in results.items() if time_of(r) <= budget and not r.get("smoke")]
    return sorted(within, key=lambda n: (-float(results[n][metric]), time_of(results[n])))


def cmd_select(selection: dict[str, Any], register: bool) -> None:
    results = all_results()
    ranked = choose(results, selection)
    best_any = max(results, key=lambda n: float(results[n][selection["metric"]]))
    print(f"best {selection['metric']} whatever the time: {best_any}")
    for status, name in zip(("champion", "challenger"), ranked, strict=False):
        r = results[name]
        took = r.get("latency_d96_ms") or r["ms_per_question"]
        print(f"{status:10s} {name}  {selection['metric']} {r[selection['metric']]}  {took} ms")
    if register and ranked:
        register_configs(results, ranked[:2], best_any, selection)


def register_configs(
    results: dict[str, dict[str, Any]], names: list[str], best_any: str, selection: dict[str, Any]
) -> None:
    """Each chosen configuration becomes a version of the registered model `tujya-rag-retrieval`
    (a pyfunc whose artifact is the configuration), with the alias champion / challenger."""
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.utils import mlflow_utils as mu

    mu.configure()
    client = MlflowClient()
    for status, name in zip(("champion", "challenger"), names, strict=False):
        record = results[name]
        config = {k: record[k] for k in config_keys() if k in record}
        with mlflow.start_run(run_id=record["run_id"]):
            info = mlflow.pyfunc.log_model(
                name="retrieval-config",
                python_model=RetrievalConfig(config),
                input_example=pd.DataFrame({"question": ["contre-indications de la metformine"]}),
                pip_requirements=["mlflow", "pandas"],
            )
        version = mu.register(
            info.model_uri,
            REGISTERED,
            alias=status,
            tags={"status": status, "config": name, "plan_step": "3.3"},
            description=(
                f"{name}: {selection['metric']} {record[selection['metric']]}, "
                f"{record.get('latency_d96_ms') or record['ms_per_question']} ms per question "
                f"on the development laptop (budget {selection['latency_budget_ms']} ms; best "
                f"whatever the time: {best_any})."
            ),
            client=client,
        )
        print(f"{REGISTERED} v{version.version} <- {status} ({name})")


def config_keys() -> list[str]:
    return list(Config("ollama:x").params())


class RetrievalConfig:  # a minimal mlflow.pyfunc.PythonModel, imported lazily
    def __new__(cls, config: dict[str, Any]) -> Any:
        import mlflow.pyfunc

        class _Model(mlflow.pyfunc.PythonModel):  # type: ignore[misc,name-defined]
            def __init__(self, settings: dict[str, Any]) -> None:
                self.settings = settings

            def predict(self, context: Any, model_input: Any, params: Any = None) -> Any:
                """The retrieval settings the agent applies, one row per question."""
                rows = len(model_input) if hasattr(model_input, "__len__") else 1
                return pd.DataFrame([self.settings] * rows)

        return _Model(config)


def pick(groups: dict[str, list[Config]], group: str, names: str) -> list[Config]:
    every = [c for items in groups.values() for c in items]
    if names:
        wanted = [n.strip() for n in names.split(",") if n.strip()]
        chosen = [c for c in every if c.name in wanted]
        missing = set(wanted) - {c.name for c in chosen}
        if missing:
            raise SystemExit(f"unknown configurations: {sorted(missing)}")
        return chosen
    if group not in groups:
        raise SystemExit(f"unknown group {group!r}: {sorted(groups)}")
    return groups[group]


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.benchmark.bench_grid")
    parser.add_argument("command", choices=["rewrites", "run", "latency", "select", "list"])
    parser.add_argument("--group", default="")
    parser.add_argument("--configs", default="", help="comma-separated configuration names")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--register", action="store_true", help="select: register the winners")
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    selection, groups = load_grid()
    if args.command == "list":
        for group, items in groups.items():
            for config in items:
                print(f"{group:22s} {config.name}")
        return
    if args.command == "select":
        cmd_select(selection, args.register)
        return
    runtime = Runtime.open()
    if args.command == "rewrites":
        await cmd_rewrites(runtime, load_questions())
    elif args.command == "run":
        await cmd_run(runtime, pick(groups, args.group, args.configs), not args.no_mlflow)
    else:
        await cmd_latency(runtime, pick(groups, args.group, args.configs), selection)


if __name__ == "__main__":
    asyncio.run(main())
