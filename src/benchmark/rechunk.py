"""Chunking variants of the served RAG corpus, and relevance of their chunks (step 3.3).

The served corpus (data/processed/rag/corpus/chunks.parquet, stage `rag_corpus`, frozen on the
ingredients of the 150 first classes: D119) cuts label sections and abstract paragraphs into
pieces of at most 1,500 characters that repeat one sentence (src/data/corpus/corpus.py
`split_text`). A variant cuts the same documents with another `max_chars` / `overlap`:

- label sections: the same rows of data/rag/interim/openfda_label/label_sections.parquet as the
  served chunks, through `corpus.label_chunks`;
- abstracts: the same PubMed articles, each about the same ingredients as in the served corpus,
  through `corpus.abstract_chunks`;
- DDInter rows: one short chunk per pair, never split, copied as they are.

`max_chars=1500, overlap=1` gives back the served corpus exactly (tested), so a variant
changes the chunking and nothing else.

The benchmark questions name their expected chunks by id (src/benchmark/rag_questions.jsonl),
ids of the served corpus. In a variant, a chunk is relevant to a question when it comes from
the same document and section as an expected chunk and at least half of the smaller of the two
is text they share (whole sentences: a smaller piece cut from the expected chunk, or a larger
one that holds it; not a neighbour that only repeats the overlap sentence) (`Relevance`); for
an interaction question it must also name the other ingredient (or be the DDInter row), which
is the rule that chose the expected chunks (docs/module2/benchmark.md).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.corpus import corpus

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "data" / "processed" / "rag" / "corpus" / "chunks.parquet"
LABELS = ROOT / "data" / "rag" / "interim" / "openfda_label" / "label_sections.parquet"
ARTICLES = ROOT / "data" / "rag" / "interim" / "pubmed" / "articles.parquet"
MIN_SENTENCE = 20  # shorter "sentences" (list bullets, headings) do not make two chunks overlap
MIN_SHARE = 0.5  # share of the smaller chunk's text the two must have in common
_SENTENCE = re.compile(r"(?<=[.!?;])\s+")


def variant(
    served: pd.DataFrame,
    labels: pd.DataFrame,
    articles: pd.DataFrame,
    max_chars: int,
    overlap: int,
) -> pd.DataFrame:
    """The served corpus cut again with `max_chars` / `overlap` (see the module docstring)."""
    names = {i for values in served["ingredients"] for i in values}
    label_ids = {d.split(":", 1)[1] for d in served.loc[served["doc_type"] == "label", "doc_id"]}
    sections = served.loc[served["doc_type"] == "label", ["doc_id", "section"]].drop_duplicates()
    wanted = {(d.split(":", 1)[1], s) for d, s in sections.itertuples(index=False)}
    kept = labels[labels["label_id"].isin(label_ids)]
    kept = kept[
        [(lid, s) in wanted for lid, s in zip(kept["label_id"], kept["section"], strict=True)]
    ]
    abstracts = served[served["doc_type"] == "abstract"]
    about = {
        d.split(":", 1)[1]: sorted(values)
        for d, values in zip(abstracts["doc_id"], abstracts["ingredients"], strict=True)
    }
    split = partial(corpus.split_text, max_chars=max_chars, overlap=overlap)
    original = corpus.split_text
    corpus.split_text = split  # label_chunks / abstract_chunks look it up at call time
    try:
        rows = corpus.label_chunks(kept, names)
        rows += corpus.abstract_chunks(articles[articles["pmid"].isin(list(about))], about, names)
    finally:
        corpus.split_text = original
    frame = pd.DataFrame(rows, columns=corpus.CHUNK_COLUMNS)
    ddinter = served[served["doc_type"] == "interaction"]
    return pd.concat([frame, ddinter[corpus.CHUNK_COLUMNS]], ignore_index=True)


def load_variant(max_chars: int, overlap: int) -> pd.DataFrame:
    served = pd.read_parquet(CHUNKS)
    if (max_chars, overlap) == (corpus.MAX_CHARS, corpus.OVERLAP_SENTENCES):
        return served
    return variant(served, pd.read_parquet(LABELS), pd.read_parquet(ARTICLES), max_chars, overlap)


def sentences(text: str) -> set[str]:
    return {s.strip() for s in _SENTENCE.split(text) if len(s.strip()) >= MIN_SENTENCE}


def shared(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    """Characters of the sentences two chunks share, over those of the smaller one: 1.0 when
    one holds the other, small when two neighbours repeat one overlap sentence."""
    common = sum(len(s) for s in a & b)
    smaller = min(sum(len(s) for s in a), sum(len(s) for s in b))
    return common / smaller if smaller else 0.0


@dataclass(frozen=True)
class Piece:
    chunk_id: str
    doc_id: str
    section: str
    doc_type: str
    ingredients: tuple[str, ...]
    mentioned: tuple[str, ...]
    sentences: frozenset[str]


def pieces(frame: pd.DataFrame) -> dict[str, Piece]:
    return {
        row.chunk_id: Piece(
            row.chunk_id,
            row.doc_id,
            row.section,
            row.doc_type,
            tuple(row.ingredients),
            tuple(row.mentioned_ingredients),
            frozenset(sentences(row.text)),
        )
        for row in frame.itertuples(index=False)
    }


class Relevance:
    """Which chunks of a variant answer a benchmark question (rule in the module docstring)."""

    def __init__(self, served: pd.DataFrame, chunks: pd.DataFrame) -> None:
        self.served = pieces(served)
        self.chunks = pieces(chunks)
        self.by_section: dict[tuple[str, str], list[Piece]] = {}
        for piece in self.chunks.values():
            self.by_section.setdefault((piece.doc_id, piece.section), []).append(piece)

    def expected(self, question: dict[str, Any]) -> set[str]:
        found: set[str] = set()
        for cid in question["expected_chunks"]:
            target = self.served.get(cid)
            if target is None:
                continue
            for piece in self.by_section.get((target.doc_id, target.section), []):
                same = piece.chunk_id == cid and piece.sentences == target.sentences
                close = shared(piece.sentences, target.sentences) >= MIN_SHARE
                if same or (close and self._names_other(question, piece)):
                    found.add(piece.chunk_id)
        return found

    @staticmethod
    def _names_other(question: dict[str, Any], piece: Piece) -> bool:
        if question["type"] != "interaction" or piece.doc_type == "interaction":
            return True
        others = set(question.get("ingredients") or []) - set(piece.ingredients)
        return bool(others & set(piece.mentioned))


def same_corpus(a: pd.DataFrame, b: pd.DataFrame, columns: Iterable[str] | None = None) -> bool:
    """Same chunks (ids, texts and metadata), whatever the row order."""
    cols = list(columns or ["chunk_id", "doc_id", "section", "chunk_index", "text", "title"])
    left = a[cols].sort_values("chunk_id").reset_index(drop=True)
    right = b[cols].sort_values("chunk_id").reset_index(drop=True)
    return bool(left.equals(right))
