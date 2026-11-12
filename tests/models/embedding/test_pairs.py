"""Training pairs of the embedding fine-tuning (src/models/embedding/pairs.py): no leakage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.models.embedding import pairs as P
from src.models.embedding.evaluate import ranks, retrieval_metrics


def chunk(
    chunk_id: str,
    doc_id: str,
    doc_type: str,
    ingredients: list[str],
    section: str,
    title: str = "",
    index: int = 0,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "doc_type": doc_type,
        "ingredients": ingredients,
        "section": section,
        "chunk_index": index,
        "title": title or f"{doc_id} {section}",
        "text": f"text of {chunk_id}",
        "is_retracted": False,
    }


CHUNKS = [
    chunk("m-ci", "openfda:met", "label", ["metformin"], "contraindications"),
    chunk("m-ar", "openfda:met", "label", ["metformin"], "adverse_reactions"),
    chunk("m-moa", "openfda:met", "label", ["metformin"], "mechanism_of_action"),
    chunk("w-ci", "openfda:war", "label", ["warfarin"], "contraindications"),
    chunk("w-di", "openfda:war", "label", ["warfarin"], "drug_interactions"),
    chunk("dd-1", "ddinter:1:2", "interaction", ["metformin", "warfarin"], "interaction"),
    chunk("dd-2", "ddinter:1:3", "interaction", ["warfarin", "aspirin"], "interaction"),
    chunk(
        "pm-1",
        "pubmed:1",
        "abstract",
        ["warfarin"],
        "abstract",
        title="Bleeding risk of warfarin with antibiotics in elderly patients",
    ),
    chunk("pm-1b", "pubmed:1", "abstract", ["warfarin"], "abstract", index=1),
    chunk(
        "pm-2",
        "pubmed:2",
        "abstract",
        ["metformin"],
        "abstract",
        title="Lactic acidosis during metformin therapy: a case series",
    ),
]
QUESTIONS = [
    {
        "question": "Quelles sont les contre-indications de la metformine ?",
        "expected_chunks": ["m-ci"],
    },
    {"question": "Can I take warfarin with aspirin?", "expected_chunks": ["dd-2", "w-di"]},
    {"question": "What does this article report?", "expected_chunks": ["pm-1b"]},
    {"question": "Can children take warfarin?", "expected_chunks": []},  # a safety question
]


def test_benchmark_chunks_and_their_documents_are_excluded() -> None:
    expected, docs = P.benchmark_exclusions(QUESTIONS, CHUNKS)
    assert expected == {"m-ci", "dd-2", "w-di", "pm-1b"}
    assert docs == {"ddinter:1:3", "pubmed:1"}  # labels lose only the expected sections
    pairs, stats = P.build_pairs(CHUNKS, QUESTIONS)
    used = {p["positive_id"] for p in pairs} | {p["negative_id"] for p in pairs}
    assert not used & {"m-ci", "dd-2", "w-di", "pm-1", "pm-1b"}
    assert stats["excluded_benchmark_chunks"] == 4
    queries = {P.fold(p["query"]) for p in pairs}
    assert P.fold("Can children take warfarin?") not in queries  # equal to a benchmark question


def test_one_question_per_language_for_labels_and_one_per_pair_for_ddinter() -> None:
    pairs, _ = P.build_pairs(CHUNKS, QUESTIONS)
    by_chunk: dict[str, list[dict[str, Any]]] = {}
    for p in pairs:
        by_chunk.setdefault(p["positive_id"], []).append(p)
    assert sorted(p["language"] for p in by_chunk["m-moa"]) == ["en", "fr"]
    assert {p["query"] for p in by_chunk["m-moa"]} == {
        "How does metformin work in the body?",
        "Comment agit metformine dans le corps ?",
    }
    assert len(by_chunk["dd-1"]) == 1
    assert by_chunk["pm-2"][0]["query"].startswith("Lactic acidosis")
    # Hard negative: same ingredient, another section or document.
    negative = by_chunk["m-ar"][0]["negative_id"]
    assert negative in {"m-moa", "dd-1", "pm-2"}


def test_split_by_document_and_leak_check() -> None:
    pairs, _ = P.build_pairs(CHUNKS, QUESTIONS)
    splits = P.split(pairs * 1)
    P.check_no_leak(splits, QUESTIONS, CHUNKS)
    leaked = dict(splits)
    leaked["train"] = [*splits["train"], {**pairs[0], "positive_id": "m-ci"}]
    with pytest.raises(AssertionError, match="benchmark leak"):
        P.check_no_leak(leaked, QUESTIONS, CHUNKS)


def test_french_names() -> None:
    assert P.french_name("metformin") == "metformine"
    assert P.french_name("ibuprofen") == "ibuprofene"
    assert P.french_name("acetaminophen") == "paracétamol"
    assert P.french_name("lisinopril") == "lisinopril"


def test_retrieval_metrics() -> None:
    scores = np.array([[0.9, 0.1, 0.5], [0.2, 0.3, 0.8], [0.4, 0.6, 0.5]])
    ids = ["a", "b", "c"]
    assert ranks(scores, ids, ["a", "b", "b"]).tolist() == [1, 2, 1]
    metrics = retrieval_metrics(scores, ids, ["a", "b", "b"])
    assert metrics["recall_at_1"] == pytest.approx(2 / 3)
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr_at_10"] == pytest.approx((1 + 0.5 + 1) / 3)
    assert metrics["ndcg_at_10"] == pytest.approx((1 + 1 / np.log2(3) + 1) / 3)
    # Several relevant chunks (one label section): the first one found counts.
    assert ranks(scores, ids, [["b", "c"], ["a", "b"], ["c"]]).tolist() == [2, 2, 2]


def test_chunks_of_a_section_share_their_question() -> None:
    two = [
        *CHUNKS,
        chunk("m-ar2", "openfda:met", "label", ["metformin"], "adverse_reactions", index=1),
    ]
    pairs, _ = P.build_pairs(two, QUESTIONS)
    english = [p for p in pairs if p["positive_id"] in ("m-ar", "m-ar2") and p["language"] == "en"]
    assert [p["relevant_ids"] for p in english] == [["m-ar", "m-ar2"], ["m-ar", "m-ar2"]]
