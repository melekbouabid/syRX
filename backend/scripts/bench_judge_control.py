"""Control for the citation judge: does it say "yes" to a source that cannot support the
claim? Pairs cited answer sentences with the first source of another question (another
drug) and counts the judge's "yes" (its false-positive rate). Logged to MLflow.

    cd backend
    python scripts/bench_judge_control.py [--judge gemma3:4b] [--n 40]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench_generation import ANSWERS, JUDGE_PROMPT, claims
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from app.rag.safety import CITATION

OUT = Path(__file__).resolve().parents[2] / "benchmark" / "results_judge_control.json"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", default="gemma3:4b")
    parser.add_argument("--n", type=int, default=40)
    args = parser.parse_args()
    rows = [json.loads(line) for line in ANSWERS.read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if r["sources"] and r["type"] != "safety"]
    pairs = []
    for row in rows:
        for sentence in claims(row["answer"]):
            if CITATION.search(sentence):
                pairs.append((row, CITATION.sub("", sentence).strip()))
    rng = random.Random(20261028)
    rng.shuffle(pairs)
    judge = ChatOllama(
        model=args.judge,
        base_url=os.environ.get("SYNERGYRX_OLLAMA_URL", "http://localhost:11434"),
        temperature=0,
        num_ctx=2048,
        num_predict=3,
    )
    yes = 0
    used = 0
    for row, claim in pairs[: args.n]:
        others = [
            r for r in rows if not set(r["id"].split("-")[1:-1]) & set(row["id"].split("-")[1:-1])
        ]
        other = rng.choice(others)["sources"][0]
        text = f"{other['title']}\n{(other.get('text') or other.get('snippet') or '')[:1500]}"
        reply = await judge.ainvoke([HumanMessage(JUDGE_PROMPT.format(source=text, claim=claim))])
        yes += str(reply.content).strip().lower().startswith("yes")
        used += 1
    result = {"judge": args.judge, "pairs": used, "false_yes": yes, "false_yes_rate": yes / used}
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("tujya-rag")
    with mlflow.start_run(run_name="citation-judge-control"):
        mlflow.set_tag("benchmark", "judge-control")
        mlflow.log_params({"judge": args.judge, "pairs": used})
        mlflow.log_metric("false_yes_rate", result["false_yes_rate"])
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
