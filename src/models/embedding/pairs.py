"""Training pairs (question, passage) for fine-tuning the embedding model (step 3.5).

    # project venv, from the repository root (or `dvc repro finetune_pairs`)
    python -m src.models.embedding.pairs

The corpus of the RAG (data/processed/rag/corpus/chunks.parquet) has no questions, only
passages. Each passage gets a question written from what it is, in English and in French
(the benchmark and the users ask in both), without an LLM, so the set is reproducible:

- a label chunk: a question about its section and ingredient ("How does {drug} work?" for
  mechanism_of_action...), one in English and one in French;
- a DDInter chunk (pair A + B): "Is there an interaction between A and B?" or its French
  form, one language per pair (alternating), the 2,350 near-identical chunks would otherwise
  outweigh the labels;
- a PubMed abstract: its article title (English), first chunk only.

Each pair also gets a hard negative: another chunk of the same ingredient (another section of
the label, another pair or article), which teaches the model to tell close passages apart.

No leakage of the benchmark (src/benchmark/rag_questions.jsonl): no pair uses as positive or
negative an expected chunk of a benchmark question, nor any chunk of a PubMed article or a
DDInter pair that a benchmark question expects, and no training question equals a benchmark
question. The label questions are also worded differently from the benchmark ones ("When
should X not be used?" instead of "What are the contraindications of X?"). `check_no_leak`
asserts it; the summary counts what was excluded.

Documents are split 90 / 10 (hash of the doc id; `finetune.seed` and `finetune.val_share` of
params.yaml): the validation pairs measure retrieval before and after training on documents
never trained on.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CHUNKS = ROOT / "data" / "processed" / "rag" / "corpus" / "chunks.parquet"
QUESTIONS = ROOT / "src" / "benchmark" / "rag_questions.jsonl"
OUT = ROOT / "data" / "processed" / "rag" / "finetune"
PARAMS = ROOT / "params.yaml"
VAL_SHARE = 0.10

# (English, French) question per label section; {en} / {fr} = the ingredient's name.
SECTION_QUESTIONS: dict[str, tuple[str, str]] = {
    "indications_and_usage": ("What is {en} used for?", "À quoi sert {fr} ?"),
    "contraindications": (
        "When should {en} not be used?",
        "Dans quels cas ne faut-il pas prendre {fr} ?",
    ),
    "boxed_warning": (
        "Does {en} carry a boxed warning, and why?",
        "Pourquoi {fr} a-t-il un avertissement encadré ?",
    ),
    "warnings_and_cautions": (
        "What serious risks should I know about {en}?",
        "Quels risques importants présente {fr} ?",
    ),
    "warnings": (
        "What serious risks should I know about {en}?",
        "Quels risques importants présente {fr} ?",
    ),
    "precautions": ("What precautions are needed with {en}?", "Quelles précautions avec {fr} ?"),
    "general_precautions": (
        "What precautions are needed with {en}?",
        "Quelles précautions avec {fr} ?",
    ),
    "adverse_reactions": (
        "What unwanted effects can {en} cause?",
        "Quels effets secondaires peut provoquer {fr} ?",
    ),
    "drug_interactions": (
        "Which medicines interact with {en}?",
        "Quels médicaments interagissent avec {fr} ?",
    ),
    "drug_and_or_laboratory_test_interactions": (
        "Can {en} change the result of a laboratory test?",
        "{fr} peut-il fausser une analyse de laboratoire ?",
    ),
    "mechanism_of_action": ("How does {en} work in the body?", "Comment agit {fr} dans le corps ?"),
    "use_in_specific_populations": (
        "Can {en} be used in pregnancy, in children or in older people?",
        "{fr} chez la femme enceinte, l'enfant ou la personne âgée ?",
    ),
    "pregnancy": (
        "Can {en} be taken during pregnancy?",
        "Peut-on prendre {fr} pendant la grossesse ?",
    ),
    "nursing_mothers": (
        "Can {en} be taken while breastfeeding?",
        "Peut-on prendre {fr} pendant l'allaitement ?",
    ),
    "pediatric_use": ("Can children take {en}?", "{fr} chez l'enfant ?"),
    "geriatric_use": ("Is {en} suitable for older patients?", "{fr} chez la personne âgée ?"),
    "information_for_patients": (
        "What should patients be told about {en}?",
        "Que faut-il dire au patient sur {fr} ?",
    ),
}
INTERACTION_QUESTIONS = (
    "Is there an interaction between {a} and {b}?",
    "Y a-t-il une interaction entre {a} et {b} ?",
)
FRENCH_EXCEPTIONS = {
    "acetaminophen": "paracétamol",
    "aspirin": "aspirine",
    "valproate": "valproate",
    "ethanol": "alcool",
}


def french_name(name: str) -> str:
    """French INN, mostly the English one plus "e" ("metformine", "amoxicilline")."""
    if name in FRENCH_EXCEPTIONS:
        return FRENCH_EXCEPTIONS[name]
    return name + "e" if re.search(r"(in|en)$", name) else name


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", text.lower())).strip()


def stable(text: str, seed: int = 0) -> int:
    return int(hashlib.sha1(f"{seed}:{text}".encode()).hexdigest()[:12], 16)


def read_chunks(path: Path = CHUNKS) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    return [r for r in rows if not r.get("is_retracted")]


def read_questions(path: Path = QUESTIONS) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def benchmark_exclusions(
    questions: list[dict[str, Any]], chunks: list[dict[str, Any]]
) -> tuple[set[str], set[str]]:
    """(chunk ids, doc ids) the training set never uses."""
    expected = {c for q in questions for c in q.get("expected_chunks") or []}
    doc_of = {c["chunk_id"]: c["doc_id"] for c in chunks}
    # A PubMed article or a DDInter pair a question expects is excluded as a whole; a label
    # only loses the expected sections (they are all the chunks of these sections).
    docs = {doc_of[c] for c in expected if c in doc_of and not doc_of[c].startswith("openfda:")}
    return expected, docs


def _pair(
    query: str, positive: dict[str, Any], negative: dict[str, Any] | None, kind: str, language: str
) -> dict[str, Any]:
    return {
        "query": query,
        "positive": passage(positive),
        "positive_id": positive["chunk_id"],
        "negative": passage(negative) if negative else "",
        "negative_id": negative["chunk_id"] if negative else "",
        "doc_id": positive["doc_id"],
        "kind": kind,
        "language": language,
    }


def passage(chunk: dict[str, Any]) -> str:
    """The text the index embeds (src/rag/index.py): title, new line, text."""
    return f"{chunk['title']}\n{chunk['text']}"


def build_pairs(
    chunks: list[dict[str, Any]], questions: list[dict[str, Any]], seed: int = 0
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    excluded_chunks, excluded_docs = benchmark_exclusions(questions, chunks)
    usable = [
        c
        for c in chunks
        if c["chunk_id"] not in excluded_chunks and c["doc_id"] not in excluded_docs
    ]
    by_ingredient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in usable:
        for name in c.get("ingredients") or []:
            by_ingredient[name].append(c)

    def negative_for(chunk: dict[str, Any]) -> dict[str, Any] | None:
        """A chunk of the same ingredient from another section / document."""
        names = chunk.get("ingredients") or []
        pool = [
            c
            for name in names
            for c in by_ingredient.get(name, [])
            if c["doc_id"] != chunk["doc_id"]
            or (c["doc_type"] == "label" and c["section"] != chunk["section"])
        ]
        if not pool:
            return None
        return pool[stable(chunk["chunk_id"], seed) % len(pool)]

    pairs: list[dict[str, Any]] = []
    for chunk in usable:
        names = chunk.get("ingredients") or []
        kind = chunk["doc_type"]
        if kind == "label" and chunk["section"] in SECTION_QUESTIONS and names:
            en_q, fr_q = SECTION_QUESTIONS[chunk["section"]]
            negative = negative_for(chunk)
            pairs.append(_pair(en_q.format(en=names[0]), chunk, negative, kind, "en"))
            pairs.append(_pair(fr_q.format(fr=french_name(names[0])), chunk, negative, kind, "fr"))
        elif kind == "interaction" and len(names) == 2:
            language = "fr" if stable(chunk["doc_id"], seed) % 2 else "en"
            template = INTERACTION_QUESTIONS[language == "fr"]
            a, b = names if language == "en" else (french_name(names[0]), french_name(names[1]))
            pairs.append(
                _pair(template.format(a=a, b=b), chunk, negative_for(chunk), kind, language)
            )
        elif kind == "abstract" and int(chunk["chunk_index"]) == 0 and len(chunk["title"]) >= 20:
            pairs.append(_pair(chunk["title"].strip(), chunk, negative_for(chunk), kind, "en"))

    benchmark_questions = {fold(q["question"]) for q in questions}
    pairs = [p for p in pairs if fold(p["query"]) not in benchmark_questions]
    # The chunks of one label section share their question: each of them answers it.
    relevant: dict[tuple[str, str], list[str]] = defaultdict(list)
    for p in pairs:
        relevant[(p["doc_id"], p["query"])].append(p["positive_id"])
    for p in pairs:
        p["relevant_ids"] = sorted(set(relevant[(p["doc_id"], p["query"])]))
    stats = {
        "chunks": len(chunks),
        "chunks_usable": len(usable),
        "excluded_benchmark_chunks": len(excluded_chunks),
        "excluded_benchmark_docs": len(excluded_docs),
        "benchmark_questions": len(questions),
    }
    return pairs, stats


def split(
    pairs: Iterable[dict[str, Any]], seed: int = 0, val_share: float = VAL_SHARE
) -> dict[str, list[dict[str, Any]]]:
    """90 / 10 by document: no document has pairs in both splits."""
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for pair in pairs:
        val = stable(pair["doc_id"], seed) % 1000 < val_share * 1000
        out["val" if val else "train"].append(pair)
    return out


def check_no_leak(
    splits: dict[str, list[dict[str, Any]]],
    questions: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> None:
    expected, docs = benchmark_exclusions(questions, chunks)
    asked = {fold(q["question"]) for q in questions}
    for name, rows in splits.items():
        for row in rows:
            ids = {row["positive_id"], row["negative_id"]} - {""}
            if ids & expected or row["doc_id"] in docs or fold(row["query"]) in asked:
                raise AssertionError(f"benchmark leak in {name}: {row['query']!r}")
    train_docs = {r["doc_id"] for r in splits["train"]}
    if train_docs & {r["doc_id"] for r in splits["val"]}:
        raise AssertionError("a document has pairs in train and val")


def write(
    splits: dict[str, list[dict[str, Any]]], stats: dict[str, Any], out: Path
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {**stats}
    for name, rows in splits.items():
        with (out / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary[name] = {
            "pairs": len(rows),
            "documents": len({r["doc_id"] for r in rows}),
            "by_kind": {
                k: sum(r["kind"] == k for r in rows) for k in ("label", "interaction", "abstract")
            },
            "by_language": {k: sum(r["language"] == k for r in rows) for k in ("en", "fr")},
            "with_negative": sum(bool(r["negative"]) for r in rows),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def read_params(path: Path = PARAMS) -> dict[str, Any]:
    import yaml

    params = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {"seed": 0, "val_share": VAL_SHARE, **(params.get("finetune") or {})}


def main(argv: list[str] | None = None) -> None:
    params = read_params()
    parser = argparse.ArgumentParser(prog="python -m src.models.embedding.pairs")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--seed", type=int, default=int(params["seed"]))
    args = parser.parse_args(argv)
    chunks, questions = read_chunks(), read_questions()
    pairs, stats = build_pairs(chunks, questions, args.seed)
    splits = split(pairs, args.seed, float(params["val_share"]))
    check_no_leak(splits, questions, chunks)
    print(json.dumps(write(splits, stats, args.out), indent=2))


if __name__ == "__main__":
    main()
