"""Reference-based evaluation of the recorded answers (course "Generation Evaluation").

Exact Match, token F1, ROUGE-1/2/L, smoothed BLEU (src/evaluation/generation_metrics.py) and
the semantic similarity of the answer and the reference: cosine of their bge-m3 embeddings
(Ollama, multilingual), used instead of BERTScore: half the answers are in French and the
references are English label text, which token-matching BERTScore with a CPU-sized model
handles poorly, and the project venv has no torch while bge-m3 already serves the RAG.

The answers are the recorded ones (src/evaluation/results/generation/<config>.jsonl), never
generated again; an answer is cleaned as for RAGAS (fixed texts and citation marks removed).
Only answered questions with a reference count. Results by language: the references are
English, so the lexical scores of the French answers measure the language gap more than the
content. Results in src/evaluation/results/generation_metrics.json (+ per question), one
MLflow run per configuration (experiment tujya-rag-generation).
Run: `python -m src.evaluation.generation_evaluation [--no-mlflow]`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from src.evaluation import generation_metrics as gm

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "src" / "evaluation" / "questions.jsonl"
ANSWERS = ROOT / "src" / "evaluation" / "results" / "generation"
OUT = ROOT / "src" / "evaluation" / "results" / "generation_metrics.json"
PER_QUESTION = ROOT / "src" / "evaluation" / "results" / "generation_metrics_per_question.jsonl"
EXPERIMENT = "tujya-rag-generation"
# config: (llm, prompt, corpus, role); the champion is served (backend/app/config.py).
CONFIGS = {
    "qwen2.5-3b__v1__corpus-v2": ("qwen2.5:3b", "v1", "v2", "champion (served)"),
    "qwen2.5-3b__v1": ("qwen2.5:3b", "v1", "v1", "champion on the corpus v1"),
    "qwen2.5-3b__v2": ("qwen2.5:3b", "v2", "v1", "challenger (prompt v2)"),
    "llama3.2-3b__v1": ("llama3.2:3b", "v1", "v1", "other LLM"),
}
METRICS = (
    "exact_match",
    "token_f1",
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "bleu",
    "semantic_similarity",
)


def clean_answer(answer: str) -> str:
    """The answer as RAGAS reads it: no fixed safety texts, no citation marks."""
    from src.evaluation.ragas_evaluation import CITED, claims

    return " ".join(CITED.sub("", s).strip() for s in claims(answer)).strip()


def load_references(path: Path = QUESTIONS) -> dict[str, str]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {item["id"]: item.get("reference") or "" for item in items}


def load_answers(config: str) -> list[dict[str, Any]]:
    path = ANSWERS / f"{config}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def evaluated(row: dict[str, Any], references: dict[str, str]) -> bool:
    """Answered (the RAGAS run scored it: it had sources and was not refused) with a reference."""
    return "ragas" in row and bool(references.get(row["id"])) and bool(clean_answer(row["answer"]))


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"n": len(rows), **{m: mean([r[m] for r in rows]) for m in METRICS}}
    for language in ("en", "fr"):
        part = [r for r in rows if r["language"] == language]
        result[language] = {"n": len(part), **{m: mean([r[m] for r in part]) for m in METRICS}}
    return result


async def score_config(config: str, references: dict[str, str], embedder: Any) -> list[dict]:
    rows = [r for r in load_answers(config) if evaluated(r, references)]
    answers = [clean_answer(r["answer"]) for r in rows]
    refs = [references[r["id"]] for r in rows]
    texts = answers + refs
    vectors: list[list[float]] = []
    for start in range(0, len(texts), 32):  # small batches for Ollama
        vectors.extend(await embedder.embed(texts[start : start + 32]))
    out = []
    for i, (row, answer, reference) in enumerate(zip(rows, answers, refs, strict=True)):
        similarity = gm.cosine(vectors[i], vectors[len(rows) + i])
        out.append(
            {
                "config": config,
                "id": row["id"],
                "type": row["type"],
                "language": row["language"],
                **{k: round(v, 6) for k, v in gm.scores(answer, reference).items()},
                "semantic_similarity": round(similarity, 6),
            }
        )
    return out


def log_run(config: str, summary: dict[str, Any], rows: list[dict[str, Any]], model: str) -> str:
    import mlflow

    from src.utils import mlflow_utils as mu

    llm, prompt, corpus, role = CONFIGS[config]
    mu.configure()
    with mu.start_tracked_run(
        EXPERIMENT,
        f"reference-metrics-{config}",
        seed=0,
        dataset=f"rag-corpus-{corpus}+rag-questions",
        params={
            "config": config,
            "llm": llm,
            "prompt_version": prompt,
            "corpus": corpus,
            "role": role,
            "answers": f"src/evaluation/results/generation/{config}.jsonl",
            "similarity_model": model,
            "bleu_smoothing": "exp (Chen & Cherry 2014)",
            "questions": summary["n"],
        },
        tags={"benchmark": "generation-reference-metrics", "plan_step": "4b"},
    ) as run:
        metrics = {m: summary[m] for m in METRICS}
        for language in ("en", "fr"):
            metrics.update({f"{m}_{language}": summary[language][m] for m in METRICS})
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{config}.json"
            path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            mlflow.log_artifact(str(path), "per_question")
        return str(run.info.run_id)


async def run(log: bool) -> dict[str, Any]:
    from app.config import Settings
    from src.rag.embedder import OllamaEmbedder

    s = Settings()
    embedder = OllamaEmbedder(
        s.ollama_url, s.embedding_model, s.embedding_dim, num_ctx=s.embedding_num_ctx
    )
    references = load_references()
    data: dict[str, Any] = {
        "similarity_model": embedder.model,
        "questions_with_reference": sum(1 for r in references.values() if r),
        "configs": {},
    }
    every: list[dict[str, Any]] = []
    for config, (llm, prompt, corpus, role) in CONFIGS.items():
        rows = await score_config(config, references, embedder)
        summary = summarize(rows)
        summary.update({"llm": llm, "prompt_version": prompt, "corpus": corpus, "role": role})
        summary["run_id"] = log_run(config, summary, rows, embedder.model) if log else ""
        data["configs"][config] = summary
        every.extend(rows)
        print(
            f"{config:28s} n={summary['n']} EM {summary['exact_match']:.3f} "
            f"F1 {summary['token_f1']:.3f} R-L {summary['rouge_l']:.3f} "
            f"BLEU {summary['bleu']:.3f} sim {summary['semantic_similarity']:.3f}"
        )
    OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    with PER_QUESTION.open("w", encoding="utf-8", newline="\n") as handle:
        for row in every:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.evaluation.generation_evaluation")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    asyncio.run(run(not args.no_mlflow))


if __name__ == "__main__":
    main()
