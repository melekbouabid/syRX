"""Generation benchmark, step 1: run the real agent on every benchmark question.

Writes data/rag/bench/answers.jsonl (git-ignored: it holds the source texts, some from
PubMed abstracts): question, draft (before the safety node), answer, sources with their
text, tool calls, safety report, timings. Step 2 is scripts/bench_generation.py.

    cd backend
    python scripts/bench_answers.py            # SYNERGYRX_PHARMA_DATABASE_URL, Ollama running
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.db import make_engine
from app.rag.runtime import AgentHolder

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "benchmark" / "rag_questions.jsonl"
OUT = ROOT / "data" / "rag" / "bench" / "answers.jsonl"


async def main() -> None:
    settings = Settings()
    agent = AgentHolder(settings, make_engine(settings.pharma_database_url)).agent()
    items = [json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # --resume keeps the answers already written (same settings) and runs the others.
    done = set()
    if "--resume" in sys.argv and OUT.is_file():
        done = {json.loads(line)["id"] for line in OUT.read_text(encoding="utf-8").splitlines()}
    await agent.run("Bonjour, warfarine ?")  # load the models once
    with OUT.open("a" if done else "w", encoding="utf-8", newline="\n") as handle:
        for number, item in enumerate(items, start=1):
            if item["id"] in done:
                continue
            started = time.perf_counter()
            state = await agent.run(item["question"])
            seconds = time.perf_counter() - started
            record = {
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
                "model": settings.llm_model,
                "search_mode": settings.rag_search_mode,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"{number}/{len(items)} {item['id']} {seconds:.1f} s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
