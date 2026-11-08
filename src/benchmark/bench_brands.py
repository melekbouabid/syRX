"""Brand questions of the benchmark (step 2.7): the real agent with and without the brand table.

    # from the repository root, project venv (.venv) active; Ollama running and
    # TUJYA_PHARMA_DATABASE_URL set (as for bench_answers.py)
    python -m src.benchmark.bench_brands               # logs to MLflow (tujya-rag)
    python -m src.benchmark.bench_brands --no-mlflow

Every `brand` question of rag_questions.jsonl ("Puis-je prendre du Panadol avec de la
warfarine ?") goes twice through the full agent (TujyaAgent.run, Ollama LLM):

- `without`: an empty brand table, i.e. the agent before step 2.7;
- `with`: config/brand_aliases.csv.

Scores of each variant (share of the questions):

- `brand_found`: every brand of the question is recognised (its source is in the sources);
- `ingredients_found`: every expected ingredient is among the agent's ingredients;
- `interaction_tool`: check_interaction was called and ran;
- `interaction_found`: it returned the DDInter interaction of the expected pair;
- `brand_cited`: the answer cites the brand's source ([n] of the brand table entry);
- `cited`: the answer cites at least one source; `disclaimer`: it ends with the disclaimer.

Results: src/benchmark/results_brands.json (no source text) and an MLflow run `bench-brands`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from app.config import Settings
from app.db import make_engine
from app.medicines import DISCLAIMER
from src.agent.runtime import AgentHolder

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "src" / "benchmark" / "rag_questions.jsonl"
OUT = ROOT / "src" / "benchmark" / "results_brands.json"
CITATION = re.compile(r"\[(\d+)\]")
METRICS = (
    "brand_found",
    "ingredients_found",
    "interaction_tool",
    "interaction_found",
    "brand_cited",
    "cited",
    "disclaimer",
)


def brand_questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [item for item in items if item["type"] == "brand"]


def pair_of(item: dict[str, Any]) -> set[str]:
    spec = json.loads((ROOT / "src" / "benchmark" / "spec.json").read_text(encoding="utf-8"))
    pairs = {s["id"]: s.get("pair") for s in spec}
    return set(pairs.get(item["id"]) or item["ingredients"][:2])


def score_state(item: dict[str, Any], state: dict[str, Any], pair: set[str]) -> dict[str, Any]:
    """The scores of one answer (pure function of the agent's final state)."""
    sources = state.get("sources", [])
    brand_sources = {s["n"] for s in sources if s.get("doc_type") == "brand"}
    found_brands = {b["brand"].lower() for b in state.get("brands", [])}
    calls = state.get("tool_calls", [])
    interaction = next(
        (r["result"] for r in state.get("tool_results", []) if r["name"] == "check_interaction"),
        None,
    )
    pairs_found = {
        frozenset((row["ingredient_a"], row["ingredient_b"]))
        for p in (interaction or {}).get("pairs", [])
        for row in p.get("interactions", [])
    }
    answer = state.get("answer", "")
    cited = {int(n) for n in CITATION.findall(answer)}
    return {
        "id": item["id"],
        "brand_found": {b.lower() for b in item["brands"]} <= found_brands,
        "ingredients_found": set(item["ingredients"]) <= set(state.get("ingredients", [])),
        "interaction_tool": any(c["name"] == "check_interaction" and c["ok"] for c in calls),
        "interaction_found": frozenset(pair) in pairs_found,
        "brand_cited": bool(brand_sources & cited),
        "cited": bool(cited),
        "disclaimer": answer.rstrip().endswith(DISCLAIMER),
        "safety_action": state.get("safety", {}).get("action"),
        "ingredients": state.get("ingredients", []),
        "tools": [c["name"] for c in calls],
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {m: round(statistics.fmean(1.0 if r[m] else 0.0 for r in rows), 4) for m in METRICS}


async def run_variant(settings: Settings, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    agent = AgentHolder(settings, make_engine(settings.pharma_database_url)).agent()
    rows = []
    for item in items:
        started = time.perf_counter()
        state = await agent.run(item["question"])
        row = score_state(item, dict(state), pair_of(item))
        row["seconds"] = round(time.perf_counter() - started, 1)
        rows.append(row)
        print(f"{item['id']}: {json.dumps({m: row[m] for m in METRICS})}", flush=True)
    return rows


async def main_async(no_mlflow: bool) -> dict[str, Any]:
    items = brand_questions()
    with_table = Settings()
    without = with_table.model_copy(update={"brand_table_file": ROOT / "config" / "none.csv"})
    results: dict[str, Any] = {"questions": len(items), "model": with_table.llm_model}
    for name, settings in (("without", without), ("with", with_table)):
        rows = await run_variant(settings, items)
        results[name] = {"metrics": summarise(rows), "per_question": rows}
    OUT.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if not no_mlflow:
        import mlflow

        from src.utils import mlflow_utils as mu

        mu.configure()
        mlflow.set_experiment("tujya-rag")
        with mlflow.start_run(run_name="bench-brands"):
            mlflow.set_tags({"stage": "benchmark", "step": "2.7", "git_commit": mu.git_commit()})
            mlflow.log_params({"questions": len(items), "llm_model": with_table.llm_model})
            for name in ("without", "with"):
                mlflow.log_metrics({f"{name}_{k}": v for k, v in results[name]["metrics"].items()})
            mlflow.log_artifact(str(OUT))
    return results


def main(argv: list[str] | None = None) -> None:
    from src.utils.console import console_utf8

    console_utf8()  # MLflow prints its run link with an emoji (cp1252 console on Windows)
    parser = argparse.ArgumentParser(prog="python -m src.benchmark.bench_brands")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    results = asyncio.run(main_async(args.no_mlflow))
    print(json.dumps({k: results[k]["metrics"] for k in ("without", "with")}, indent=2))


if __name__ == "__main__":
    main()
