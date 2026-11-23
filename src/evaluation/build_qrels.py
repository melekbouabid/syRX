"""Build the RAG benchmark set from the corpus: src/evaluation/questions.jsonl and qrels.json.

Each item comes from a hand-written spec (`src/evaluation/question_spec.json`: question type,
ingredients, language, question text) and gets, from data/processed/rag/corpus/chunks.parquet:
- `expected_chunks`: every chunk that answers it (rules per type below), written to
  qrels.json (question id -> chunk ids, as in labs/05-Evaluation);
- `reference`: an extractive reference answer (sentences of the expected label chunks).
The candidates were then read one by one and corrected or dropped
(docs/module2/benchmark.md).

Types and their expected chunks:
- contraindications / boxed_warning / mechanism / adverse_reactions / warnings: the chunks of
  that section (warnings = warnings_and_cautions or warnings) of the ingredient's label;
- interaction (A, B): A's label chunks in interaction / warning / precaution sections that
  mention B, B's that mention A, and the DDInter chunk of the pair;
- abstract: the chunks of the given PubMed article(s) (`pmids` in the spec);
- safety: no expected chunk (dose requests, "is it safe"), scored on the answer only.

    # from the repository root, project venv (.venv) active
    python -m src.evaluation.build_qrels   # writes questions.jsonl and qrels.json here

`load_questions` joins the two files again: one item per question, with its expected chunks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from src.agent.safety import CROSS_REFERENCE, sentence_problems

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "data" / "processed" / "rag" / "corpus" / "chunks.parquet"
SPEC = ROOT / "src" / "evaluation" / "question_spec.json"
QUESTIONS = ROOT / "src" / "evaluation" / "questions.jsonl"
QRELS = ROOT / "src" / "evaluation" / "qrels.json"

SECTIONS = {
    "contraindications": ("contraindications",),
    "boxed_warning": ("boxed_warning",),
    "mechanism": ("mechanism_of_action",),
    "adverse_reactions": ("adverse_reactions",),
    "warnings": ("warnings_and_cautions", "warnings"),
}
INTERACTION_SECTIONS = (
    "drug_interactions",
    "precautions",
    "warnings_and_cautions",
    "warnings",
    "general_precautions",
    "boxed_warning",
    "contraindications",
)
_HEADING = re.compile(r"^(?:\d+(?:\.\d+)*\s+)?[A-Z][A-Z &,/()-]{3,}\s+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def usable(sentence: str) -> bool:
    """A reference sentence: no dose or dosing advice, no reporting line, not a fragment."""
    lowered = sentence.lower()
    return (
        len(sentence) > 25
        and not sentence_problems(sentence)
        and not re.search(r"\bdos(?:e|es|age|ing)\b|\d\s*mg\b|1-800|contact ", lowered)
    )


def sentences(text: str) -> list[str]:
    text = CROSS_REFERENCE.sub("", _HEADING.sub("", text.strip()))
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    found: list[str] = []
    for sentence in _SENTENCE.split(text):
        sentence = sentence.strip()
        if usable(sentence) and sentence not in found:
            found.append(sentence)
    return found


def label_chunks(chunks: list[dict[str, Any]], ingredient: str, sections: tuple[str, ...]):
    return sorted(
        (
            c
            for c in chunks
            if c["doc_type"] == "label"
            and c["ingredients"] == [ingredient]
            and c["section"] in sections
        ),
        key=lambda c: (sections.index(c["section"]), c["chunk_index"]),
    )


def build_item(spec: dict[str, Any], chunks: list[dict[str, Any]]) -> dict[str, Any]:
    kind = spec["type"]
    names = spec["ingredients"]
    expected: list[dict[str, Any]] = []
    reference = spec.get("reference", "")
    if kind in SECTIONS:
        expected = label_chunks(chunks, names[0], SECTIONS[kind])
        if not reference and expected:
            reference = " ".join(sentences(expected[0]["text"])[:2])
    elif kind in ("interaction", "brand"):
        # brand (step 2.7): a brand and a drug; `pair` is the ingredient pair DDInter lists.
        a, b = spec.get("pair", names)
        names = [a, b] if kind == "brand" else names
        for x, y in ((a, b), (b, a)):
            for chunk in label_chunks(chunks, x, INTERACTION_SECTIONS):
                if y in chunk["mentioned_ingredients"]:
                    expected.append(chunk)
        expected += [
            c
            for c in chunks
            if c["doc_type"] == "interaction" and sorted(c["ingredients"]) == sorted(names)
        ]
        if not reference:
            found = [
                s
                for c in expected
                if c["doc_type"] == "label"
                for s in sentences(c["text"])
                if re.search(rf"\b{re.escape(b if a in c['ingredients'] else a)}\b", s, re.I)
            ]
            reference = " ".join(list(dict.fromkeys(found))[:2])
    elif kind == "abstract":
        wanted = {f"pubmed:{p}" for p in spec["pmids"]}
        expected = [c for c in chunks if c["doc_id"] in wanted]
    item = {
        "id": spec["id"],
        "type": kind,
        "language": spec["language"],
        "question": spec["question"],
        "ingredients": spec["ingredients"],
        "expected_chunks": [c["chunk_id"] for c in expected],
        "expected_titles": sorted({c["title"] for c in expected}),
        "reference": reference,
    }
    if kind == "brand":
        item["brands"] = spec["brands"]
    return item


def load_questions(questions: Path = QUESTIONS, qrels: Path = QRELS) -> list[dict[str, Any]]:
    """Every benchmark question with its `expected_chunks` (from qrels.json), in file order."""
    relevant = json.loads(qrels.read_text(encoding="utf-8"))
    items = []
    for line in questions.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        merged: dict[str, Any] = {}
        for key, value in item.items():
            if key == "expected_titles":  # expected_chunks comes before it, as built
                merged["expected_chunks"] = relevant[item["id"]]
            merged[key] = value
        items.append(merged)
    return items


def main() -> None:
    chunks = pq.read_table(CHUNKS).to_pylist()
    specs = json.loads(SPEC.read_text(encoding="utf-8"))
    items = [build_item(spec, chunks) for spec in specs]
    # Brand questions (step 2.7) may have no chunk: paracetamol or ibuprofen are not
    # ingredients of the indexed pill classes; the interaction comes from check_interaction
    # (DDInter) and the brand from the brand table, so they count in the generation benchmark.
    missing = [
        i["id"] for i in items if i["type"] not in ("safety", "brand") and not i["expected_chunks"]
    ]
    if missing:
        raise SystemExit(f"no expected chunk for {missing}")
    with QUESTIONS.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            question = {k: v for k, v in item.items() if k != "expected_chunks"}
            handle.write(json.dumps(question, ensure_ascii=False) + "\n")
    qrels = {item["id"]: item["expected_chunks"] for item in items}
    QRELS.write_text(json.dumps(qrels, indent=2) + "\n", encoding="utf-8", newline="\n")
    by_type: dict[str, int] = {}
    for item in items:
        by_type[item["type"]] = by_type.get(item["type"], 0) + 1
    langs = {lang: sum(1 for i in items if i["language"] == lang) for lang in ("fr", "en")}
    print(f"{len(items)} questions -> {QUESTIONS}, {QRELS}; {by_type}; {langs}")


if __name__ == "__main__":
    main()
