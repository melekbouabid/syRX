"""Generation benchmark, step 1: run the real agent on every benchmark question.

Writes one answers file per configuration (git-ignored: it holds the source texts, some from
PubMed abstracts): question, draft (before the safety node), answer, sources with their
text, tool calls, safety report, timings. Step 2 is ragas_evaluation.py.

A configuration is the LLM (`--llm`, an Ollama model) and the answer prompt version
(`--prompt`, src/agent/prompts.py); retrieval is the agent's served setting (step 3.3, D123),
on the corpus `--corpus` (v1: the 150 first classes, collection tujya_chunks, the corpus of
step 3.4; v2: the 380 classes, collection tujya_chunks_v2, D126).
With `--trace`, MLflow traces every question (experiment `tujya-rag-generation`): LangChain
autolog records the LangGraph nodes, the LLM calls and one span per tool call, under a root
span `question` tagged with the configuration and the question id.

    # from the repository root, project venv (.venv) active; Ollama, Qdrant, postgres running
    python -m src.evaluation.benchmark_generation --llm qwen2.5:3b --prompt v1 --trace [--resume]
    # needs TUJYA_PHARMA_DATABASE_URL (docs/demo.md)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from app.config import Settings
from app.db import make_engine
from src.agent.runtime import AgentHolder
from src.evaluation import build_qrels

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "src" / "evaluation" / "questions.jsonl"
GENERATION_DIR = ROOT / "data" / "processed" / "rag" / "bench" / "generation"
EXPERIMENT = "tujya-rag-generation"
# Qdrant collection of each corpus (src/evaluation/retrieval_grid.py COLLECTIONS, same names).
COLLECTIONS = {"v1": "tujya_chunks", "v2": "tujya_chunks_v2"}


def config_name(llm: str, prompt: str, corpus: str = "v1") -> str:
    """`qwen2.5:3b` + `v1` -> `qwen2.5-3b__v1` (a folder and MLflow-friendly name); on the
    corpus v2: `qwen2.5-3b__v1__corpus-v2`."""
    name = f"{llm.replace(':', '-').replace('/', '-')}__{prompt}"
    return name if corpus == "v1" else f"{name}__corpus-{corpus}"


def answers_path(llm: str, prompt: str, corpus: str = "v1") -> Path:
    return GENERATION_DIR / config_name(llm, prompt, corpus) / "answers.jsonl"


def record_of(item: dict[str, Any], state: dict[str, Any], seconds: float, s: Settings) -> dict:
    return {
        "id": item["id"],
        "type": item["type"],
        "language": item["language"],
        "question": item["question"],
        "reference": item["reference"],
        "expected_chunks": item["expected_chunks"],
        "draft": state.get("draft", ""),
        "answer": state.get("answer", ""),
        "sources": state.get("sources", []),
        "tool_calls": state.get("tool_calls", []),
        "safety": state.get("safety", {}),
        "timings": state.get("timings", {}),
        "seconds": round(seconds, 2),
        "model": s.llm_model,
        "prompt_version": s.llm_prompt_version,
        "search_mode": s.rag_search_mode,
        "query_rewrite": s.rag_query_rewrite,
        "rerank": s.rag_rerank,
    }


def start_tracing(settings: Settings) -> Any:
    """MLflow tracing of the agent (LangChain autolog); returns the mlflow module."""
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment(EXPERIMENT)
    mlflow.langchain.autolog(log_traces=True)
    _name_unnamed_runs()
    return mlflow


def _name_unnamed_runs() -> None:
    """LangGraph starts some runs with `serialized=None`; MLflow 3.17's tracer then fails to
    name the span, drops it, and the trace stays "not fully exported". Those spans get the
    default name instead ("chain", "tool"...)."""
    from mlflow.langchain.langchain_tracer import MlflowLangchainTracer

    original = MlflowLangchainTracer._assign_span_name
    if getattr(original, "_tujya_patched", False):
        return

    def assign(self: Any, serialized: Any, default_name: str = "unknown") -> str:
        return str(original(self, serialized or {}, default_name))

    assign._tujya_patched = True  # type: ignore[attr-defined]
    MlflowLangchainTracer._assign_span_name = assign  # type: ignore[method-assign]


async def run_question(agent: Any, item: dict[str, Any], mlflow: Any, tags: dict[str, str]):
    if mlflow is None:
        return await agent.run(item["question"])
    with mlflow.start_span(name="question", span_type="AGENT") as span:
        span.set_inputs({"question": item["question"], "id": item["id"]})
        mlflow.update_current_trace(tags={**tags, "question_id": item["id"]})
        state = await agent.run(item["question"])
        span.set_outputs({"answer": state.get("answer", ""), "safety": state.get("safety", {})})
        return state


async def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--llm", default=None, help="Ollama model (default: the served one)")
    parser.add_argument("--prompt", default=None, help="answer prompt version (v1, v2)")
    parser.add_argument("--resume", action="store_true", help="keep the answers already written")
    parser.add_argument("--trace", action="store_true", help="MLflow traces (tujya-rag-generation)")
    parser.add_argument("--limit", type=int, default=0, help="first N questions only (trial)")
    parser.add_argument("--corpus", default="v1", choices=sorted(COLLECTIONS))
    args = parser.parse_args(argv)
    overrides: dict[str, str] = {"qdrant_collection": COLLECTIONS[args.corpus]}
    if args.llm:
        overrides["llm_model"] = args.llm
    if args.prompt:
        overrides["llm_prompt_version"] = args.prompt
    settings = Settings(**overrides)
    out = answers_path(settings.llm_model, settings.llm_prompt_version, args.corpus)
    agent = AgentHolder(settings, make_engine(settings.pharma_database_url)).agent()
    items = build_qrels.load_questions(QUESTIONS)
    if args.limit:
        items = items[: args.limit]
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out.is_file():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()}
    await agent.run("Bonjour, warfarine ?")  # load the models once (not traced)
    mlflow = start_tracing(settings) if args.trace else None
    tags = {
        "benchmark": "generation",
        "config": config_name(settings.llm_model, settings.llm_prompt_version, args.corpus),
        "corpus": args.corpus,
        "llm": settings.llm_model,
        "prompt_version": settings.llm_prompt_version,
    }
    with out.open("a" if done else "w", encoding="utf-8", newline="\n") as handle:
        for number, item in enumerate(items, start=1):
            if item["id"] in done:
                continue
            started = time.perf_counter()
            state = await run_question(agent, item, mlflow, tags)
            seconds = time.perf_counter() - started
            handle.write(json.dumps(record_of(item, state, seconds, settings), ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            print(f"{number}/{len(items)} {item['id']} {seconds:.1f} s", flush=True)
    if mlflow is not None:
        mlflow.flush_trace_async_logging(terminate=True)  # export before the process exits
    print(f"-> {out}")


if __name__ == "__main__":
    asyncio.run(main())
