"""RAGAS metrics that read the retrieved passages, and the check of the judge against humans.

The recorded champion answers (results/generation/qwen2.5-3b__v1__corpus-v2.jsonl) keep no
passages, so the served setting (qwen2.5:3b + prompt v1, corpus v2) is generated again with its
sources: `python -m src.evaluation.benchmark_generation --corpus v2` writes
data/processed/rag/bench/generation/qwen2.5-3b__v1__corpus-v2/answers.jsonl. The recorded
results are left as they are; everything here reads that new file.

- `ragas`: faithfulness, context utilization and noise sensitivity with the local judge
  (phi4-mini, config/rag_generation.yaml), on the answered questions (`--limit N`: the first N),
  each metric timed. Results in src/evaluation/results/ragas_context.json, one MLflow run.
- `sheet`: 30 answers for human labels, half judged faithful (faithfulness = 1: every claim
  supported) and half not, in a shuffled order, with the passages the judge read (the first 4
  sources, 900 characters each, as the LLM saw them); only answers whose citations all point
  to those 4 sources. The judge's verdict is not written. JSON rows for the labelling sheet.
- `kappa`: the human labels (src/evaluation/human_labels.csv: id, label) against the judge
  (1 when faithfulness = 1); Cohen's kappa, confusion matrix, Landis & Koch; results in
  src/evaluation/results/judge_kappa.json, one MLflow run.
Run: `python -m src.evaluation.judge_check ragas | sheet --out PATH | kappa [--no-mlflow]`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

from src.evaluation import ragas_evaluation as rg
from src.evaluation.judge_agreement import cohen_kappa

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "qwen2.5-3b__v1__corpus-v2"
ANSWERS = ROOT / "data" / "processed" / "rag" / "bench" / "generation" / CONFIG / "answers.jsonl"
RECORDED = ROOT / "src" / "evaluation" / "results" / "generation" / f"{CONFIG}.jsonl"
RESULTS = ROOT / "src" / "evaluation" / "results"
RAGAS_OUT = RESULTS / "ragas_context.json"
KAPPA_OUT = RESULTS / "judge_kappa.json"
HUMAN_LABELS = ROOT / "src" / "evaluation" / "human_labels.csv"
EXPERIMENT = "tujya-rag-generation"
METRICS = ("faithfulness", "context_utilization", "noise_sensitivity")
SHEET_SIZE = 30
_CITATION = re.compile(r"\[(\d+)\]")


def load_answers(path: Path = ANSWERS) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(
            f"{path} not found: run `python -m src.evaluation.benchmark_generation "
            "--corpus v2` first"
        )
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def identical_to_recorded(rows: list[dict[str, Any]], path: Path = RECORDED) -> int:
    recorded = {
        r["id"]: r["answer"]
        for r in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }
    return sum(1 for r in rows if recorded.get(r["id"]) == r["answer"])


def shown_answer(answer: str) -> str:
    """The answer the judge reads, citation marks kept (fixed safety texts removed)."""
    return " ".join(rg.claims(answer)).strip()


def cited(answer: str) -> list[int]:
    return sorted({int(n) for n in _CITATION.findall(shown_answer(answer))})


def passage(source: dict[str, Any]) -> str:
    text = (source.get("text") or source.get("snippet") or "")[: rg.CONTEXT_CHARS]
    return f"[{source['n']}] {source['title']}\n{text}"


# --- ragas -------------------------------------------------------------------------------


def cmd_ragas(limit: int, log: bool) -> None:
    cfg = rg.load_config()
    base_url = os.environ.get("TUJYA_OLLAMA_URL", "http://127.0.0.1:11434")  # as ragas_evaluation
    rows = load_answers()
    answered = [r for r in rows if rg.answered(r)]
    subset = answered[:limit] if limit else answered
    contexts = int(cfg["ragas"]["contexts"])
    seconds: dict[str, float] = {}
    scores: dict[str, dict[str, float | None]] = {}
    for metric in METRICS:
        started = time.perf_counter()
        part = rg.run_ragas(subset, cfg, base_url, metrics=(metric,))
        seconds[metric] = round(time.perf_counter() - started, 1)
        for qid, values in part.items():
            scores.setdefault(qid, {}).update(values)
    summary = {}
    for metric in METRICS:
        values = [v[metric] for v in scores.values() if v.get(metric) is not None]
        summary[metric] = {
            "mean": round(sum(values) / len(values), 4) if values else None,
            "n": len(values),
            "seconds": seconds[metric],
        }
    data = {
        "answers": str(ANSWERS.relative_to(ROOT)).replace("\\", "/"),
        "config": CONFIG,
        "questions": len(rows),
        "answered": len(answered),
        "evaluated": len(subset),
        "identical_to_recorded": identical_to_recorded(rows),
        "judge": cfg["judge"],
        "ragas_version": rg.RAGAS_VERSION,
        "contexts": contexts,
        "metrics": summary,
        "per_question": scores,
    }
    if log:
        data["run_id"] = log_run("ragas-context", data, {m: summary[m]["mean"] for m in METRICS})
    RAGAS_OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: v for k, v in data.items() if k != "per_question"}, indent=1))


def log_run(name: str, data: dict[str, Any], metrics: dict[str, Any]) -> str:
    import mlflow

    from src.utils import mlflow_utils as mu

    mu.configure()
    params = {
        k: v for k, v in data.items() if isinstance(v, str | int | float) and k != "per_question"
    }
    with mu.start_tracked_run(
        EXPERIMENT,
        f"{name}-{CONFIG}",
        seed=0,
        dataset="rag-corpus-v2+rag-questions",
        params=params,
        tags={"benchmark": name, "plan_step": "4b"},
    ) as run:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})
        mlflow.log_dict(data, f"{name}.json")
        return str(run.info.run_id)


# --- sheet -------------------------------------------------------------------------------


def judge_verdicts() -> dict[str, int]:
    """1 when the judge found every claim supported (faithfulness = 1), else 0."""
    data = json.loads(RAGAS_OUT.read_text(encoding="utf-8"))
    return {
        qid: int(values["faithfulness"] == 1.0)
        for qid, values in data["per_question"].items()
        if values.get("faithfulness") is not None
    }


def select(rows: list[dict[str, Any]], verdicts: dict[str, int], size: int, seed: int) -> list:
    """Half judged faithful, half not (as far as they exist), shuffled; citations within the
    passages the judge read."""
    contexts = int(rg.load_config()["ragas"]["contexts"])
    usable = [
        r
        for r in rows
        if r["id"] in verdicts and cited(r["answer"]) and max(cited(r["answer"])) <= contexts
    ]
    rng = random.Random(seed)
    groups = {v: [r for r in usable if verdicts[r["id"]] == v] for v in (0, 1)}
    for group in groups.values():
        rng.shuffle(group)
    take_0 = min(len(groups[0]), size // 2)
    take_1 = min(len(groups[1]), size - take_0)
    take_0 = min(len(groups[0]), size - take_1)
    chosen = groups[0][:take_0] + groups[1][:take_1]
    rng.shuffle(chosen)  # the order must not reveal the verdict
    return chosen


def cmd_sheet(out: Path, size: int, seed: int) -> None:
    rows = load_answers()
    chosen = select(rows, judge_verdicts(), size, seed)
    contexts = int(rg.load_config()["ragas"]["contexts"])
    sheet = []
    for row in chosen:
        numbers = cited(row["answer"])
        read = row["sources"][:contexts]
        sheet.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": shown_answer(row["answer"]),
                "cited_passages": "\n\n".join(passage(s) for s in read if s["n"] in numbers),
                "other_passages_read_by_the_judge": "\n\n".join(
                    passage(s) for s in read if s["n"] not in numbers
                ),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sheet, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(sheet)} answers -> {out}")


# --- kappa -------------------------------------------------------------------------------


def read_labels(path: Path | None = None) -> dict[str, int]:
    with (path or HUMAN_LABELS).open(newline="", encoding="utf-8") as handle:
        return {row["id"]: int(row["label"]) for row in csv.DictReader(handle)}


def cmd_kappa(log: bool) -> None:
    human = read_labels()
    judge = judge_verdicts()
    missing = sorted(set(human) - set(judge))
    if missing:
        raise SystemExit(f"no judge verdict for {missing}")
    ids = sorted(human)
    result = cohen_kappa([human[i] for i in ids], [judge[i] for i in ids])
    data = {
        "config": CONFIG,
        "items": len(ids),
        "rater_a": "human (src/evaluation/human_labels.csv)",
        "rater_b": f"RAGAS faithfulness = 1 ({rg.load_config()['judge']})",
        "human_faithful": sum(human[i] for i in ids),
        "judge_faithful": sum(judge[i] for i in ids),
        **result,
        "confusion_rows": "human 1, human 0",
        "confusion_columns": "judge 1, judge 0",
    }
    if log:
        metrics = {k: data[k] for k in ("observed_agreement", "expected_agreement", "kappa")}
        data["run_id"] = log_run("judge-kappa", data, metrics)
    KAPPA_OUT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(data, indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.evaluation.judge_check")
    parser.add_argument("command", choices=["ragas", "sheet", "kappa"])
    parser.add_argument("--limit", type=int, default=0, help="ragas: the first N answers only")
    parser.add_argument("--out", type=Path, help="sheet: the JSON file to write")
    parser.add_argument("--size", type=int, default=SHEET_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    if args.command == "ragas":
        cmd_ragas(args.limit, not args.no_mlflow)
    elif args.command == "sheet":
        if args.out is None:
            parser.error("sheet needs --out")
        cmd_sheet(args.out, args.size, args.seed)
    else:
        cmd_kappa(not args.no_mlflow)


if __name__ == "__main__":
    main()
