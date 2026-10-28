"""Tujya RAG corpus: openFDA label sections, PubMed abstracts and DDInter rows -> chunks.

- labels: one chunk per section, long sections split on sentence boundaries;
- abstracts: one chunk per paragraph (structured abstracts have one per label, such as
  "RESULTS: ..."), long paragraphs split the same way;
- DDInter: one chunk per interacting pair between the class ingredients.

Every chunk carries its metadata (`CHUNK_COLUMNS`, documented in docs/RAG.md). Dosage
sections are refused here a second time, whatever the connector stored.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

MAX_CHARS = 1500
OVERLAP_SENTENCES = 1
FORBIDDEN_SECTION = re.compile(r"dosage|dose|overdos|how_supplied|package_label")

CHUNK_COLUMNS = [
    "chunk_id",
    "doc_id",
    "doc_type",  # label | abstract | interaction
    "ingredients",  # the ingredient(s) the document is about, lower-case RxNorm names
    "mentioned_ingredients",  # other class ingredients named in the chunk text
    "section",  # label section, abstract paragraph label, or "interaction"
    "chunk_index",
    "title",
    "text",
    "source_name",
    "url",
    "date",  # label effective_time, publication year, DDInter release
    "version",  # label version, PubMed record, DDInter version
    "license",
    "article_type",  # PubMed only
    "is_retracted",  # PubMed only (False elsewhere)
    "language",
    "retrieved_at",
]
SECTION_TITLES = {
    "boxed_warning": "Boxed warning",
    "indications_and_usage": "Indications and usage",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and precautions",
    "warnings": "Warnings",
    "precautions": "Precautions",
    "general_precautions": "General precautions",
    "drug_interactions": "Drug interactions",
    "drug_and_or_laboratory_test_interactions": "Drug and laboratory test interactions",
    "adverse_reactions": "Adverse reactions",
    "use_in_specific_populations": "Use in specific populations",
    "pregnancy": "Pregnancy",
    "lactation": "Lactation",
    "nursing_mothers": "Nursing mothers",
    "pediatric_use": "Pediatric use",
    "geriatric_use": "Geriatric use",
    "information_for_patients": "Information for patients",
    "mechanism_of_action": "Mechanism of action",
}
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(•\-])")
_PARAGRAPH_LABEL = re.compile(r"^([A-Z][A-Z /&,-]{2,60}):\s+")


def chunk_id(*parts: object) -> str:
    text = "|".join(str(p) for p in parts)
    return hashlib.sha1(text.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def split_text(
    text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_SENTENCES
) -> list[str]:
    """Split on sentence ends into pieces of at most ~max_chars, repeating `overlap` sentences.

    A single sentence longer than `max_chars` (tables flattened to text) is cut on spaces.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    sentences: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            sentences.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            sentences.append(sentence)
    pieces: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join([*current, sentence])) > max_chars:
            pieces.append(" ".join(current))
            current = current[-overlap:] if overlap else []
            if len(" ".join([*current, sentence])) > max_chars:
                current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def mentions(text: str, names: Iterable[str]) -> list[str]:
    lowered = text.lower()
    return sorted(n for n in names if re.search(rf"\b{re.escape(n)}\b", lowered))


def label_chunks(sections: pd.DataFrame, names: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in sections.itertuples(index=False):
        if FORBIDDEN_SECTION.search(row.section):
            raise ValueError(f"dosage section must never be indexed: {row.section}")
        product = row.brand_name or row.generic_name or row.ingredient
        heading = SECTION_TITLES.get(row.section, row.section.replace("_", " ").capitalize())
        title = f"{row.ingredient.capitalize()} label ({product}): {heading}"
        for index, piece in enumerate(split_text(row.text)):
            rows.append(
                {
                    "chunk_id": chunk_id("label", row.label_id, row.section, index),
                    "doc_id": f"openfda:{row.label_id}",
                    "doc_type": "label",
                    "ingredients": [row.ingredient],
                    "mentioned_ingredients": [
                        n for n in mentions(piece, names) if n != row.ingredient
                    ],
                    "section": row.section,
                    "chunk_index": index,
                    "title": title,
                    "text": piece,
                    "source_name": "openFDA drug label",
                    "url": row.label_url or row.source_url,
                    "date": row.effective_time,
                    "version": f"set_id {row.set_id} v{row.label_version}",
                    "license": row.license,
                    "article_type": None,
                    "is_retracted": False,
                    "language": "en",
                    "retrieved_at": row.retrieved_at,
                }
            )
    return rows


def paragraphs(abstract: str) -> list[tuple[str, str]]:
    """(label, text) per abstract paragraph; unlabelled paragraphs get "abstract"."""
    found = []
    for line in abstract.split("\n"):
        line = line.strip()
        if not line:
            continue
        match = _PARAGRAPH_LABEL.match(line)
        if match:
            found.append((match.group(1).strip().lower(), line[match.end() :].strip()))
        else:
            found.append(("abstract", line))
    return found


def abstract_chunks(
    articles: pd.DataFrame, ingredients_by_pmid: dict[str, list[str]], names: set[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in articles.itertuples(index=False):
        if not isinstance(row.abstract, str) or not row.abstract.strip():
            continue
        about = ingredients_by_pmid.get(row.pmid, [])
        index = 0
        for label, paragraph in paragraphs(row.abstract):
            for piece in split_text(paragraph):
                rows.append(
                    {
                        "chunk_id": chunk_id("pubmed", row.pmid, index),
                        "doc_id": f"pubmed:{row.pmid}",
                        "doc_type": "abstract",
                        "ingredients": about,
                        "mentioned_ingredients": [
                            n for n in mentions(piece, names) if n not in about
                        ],
                        "section": label,
                        "chunk_index": index,
                        "title": str(row.title or f"PubMed {row.pmid}"),
                        "text": piece,
                        "source_name": f"PubMed ({row.journal})" if row.journal else "PubMed",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{row.pmid}/",
                        "date": "" if pd.isna(row.year) else str(int(row.year)),
                        "version": f"PMID {row.pmid}",
                        "license": row.license,
                        "article_type": row.article_type,
                        "is_retracted": bool(row.is_retracted),
                        "language": (row.language or "eng")[:2],
                        "retrieved_at": row.retrieved_at,
                    }
                )
                index += 1
    return rows


def interaction_chunks(pairs: pd.DataFrame, retrieved_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in pairs.itertuples(index=False):
        a, b = row.ingredient_a, row.ingredient_b
        if row.severity == "Unknown":
            level = "an interaction whose severity DDInter does not grade (level Unknown)"
        else:
            level = f"an interaction with severity {row.severity}"
        text = (
            f"DDInter 2.0 lists {level} between {a} and {b}. "
            "DDInter's bulk files give only the two drugs and the level, not the mechanism."
        )
        rows.append(
            {
                "chunk_id": chunk_id("ddinter", a, b),
                "doc_id": f"ddinter:{row.ddinter_id_a}:{row.ddinter_id_b}",
                "doc_type": "interaction",
                "ingredients": [a, b],
                "mentioned_ingredients": [],
                "section": "interaction",
                "chunk_index": 0,
                "title": f"DDInter 2.0: {a} + {b} ({row.severity})",
                "text": text,
                "source_name": "DDInter 2.0",
                "url": row.source_url,
                "date": "2025",
                "version": "DDInter 2.0",
                "license": row.license,
                "article_type": None,
                "is_retracted": False,
                "language": "en",
                "retrieved_at": retrieved_at,
            }
        )
    return rows


def pmid_ingredients(raw_pubmed: Path, query_hits: pd.DataFrame) -> dict[str, list[str]]:
    """PMID -> ingredients whose query returned it (from each query's `query.json`)."""
    names: dict[str, str] = {}
    for path in raw_pubmed.glob("*/*/query.json"):
        query = json.loads(path.read_text(encoding="utf-8"))
        if query.get("kind") == "ingredient":
            names[query["query_id"]] = query["drug_names"][0]
    found: dict[str, set[str]] = {}
    for row in query_hits.itertuples(index=False):
        if row.query_id in names:
            found.setdefault(row.pmid, set()).add(names[row.query_id])
    return {pmid: sorted(values) for pmid, values in found.items()}


def build_chunks(
    label_sections: pd.DataFrame | None,
    articles: pd.DataFrame | None,
    ingredients_by_pmid: dict[str, list[str]],
    pairs: pd.DataFrame | None,
    names: set[str],
    ddinter_retrieved_at: str = "",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if label_sections is not None:
        rows += label_chunks(label_sections, names)
    if articles is not None:
        kept = articles[articles["pmid"].isin(list(ingredients_by_pmid))]
        rows += abstract_chunks(kept, ingredients_by_pmid, names)
    if pairs is not None:
        rows += interaction_chunks(pairs, ddinter_retrieved_at)
    frame = pd.DataFrame(rows, columns=CHUNK_COLUMNS)
    if frame["chunk_id"].duplicated().any():
        raise ValueError("duplicate chunk ids")
    return frame


def summary(chunks: pd.DataFrame) -> dict[str, Any]:
    return {
        "chunks": len(chunks),
        "documents": int(chunks["doc_id"].nunique()),
        "by_type": {
            str(k): {"documents": int(g["doc_id"].nunique()), "chunks": len(g)}
            for k, g in chunks.groupby("doc_type")
        },
        "ingredients_covered": len({i for v in chunks["ingredients"] for i in v}),
        "retracted_chunks": int(chunks["is_retracted"].sum()),
        "mean_chars": round(float(chunks["text"].str.len().mean()), 1) if len(chunks) else 0,
    }
