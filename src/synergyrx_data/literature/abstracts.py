"""PubMed XML parsing and the abstract store.

Flags article type, language and retraction status so downstream text mining can exclude
reviews, non-English text or retracted work on purpose rather than by accident.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from typing import Any, Protocol

import pandas as pd

from synergyrx_data.literature.sentences import split_sentences
from synergyrx_data.provenance import PROVENANCE_COLUMNS

_YEAR = re.compile(r"(1[89]|20)\d{2}")

# First match wins, so the most specific types come first.
_TYPE_RULES = (
    ("retraction_notice", ("retraction of publication",)),
    ("meta_analysis", ("meta-analysis", "systematic review")),
    ("review", ("review",)),
    ("clinical_trial", ("clinical trial", "randomized controlled trial")),
    ("case_report", ("case reports",)),
    ("commentary", ("editorial", "letter", "comment", "news")),
    ("preprint", ("preprint",)),
)


def classify_article_type(publication_types: list[str]) -> str:
    lowered = [p.lower() for p in publication_types]
    for label, needles in _TYPE_RULES:
        if any(needle in ptype for ptype in lowered for needle in needles):
            return label
    return "research"


def _text(node: ET.Element | None) -> str | None:
    if node is None:
        return None
    text = re.sub(r"\s+", " ", "".join(node.itertext())).strip()
    return text or None


def _year(article: ET.Element) -> int | None:
    for path in (
        ".//Journal/JournalIssue/PubDate/Year",
        ".//Journal/JournalIssue/PubDate/MedlineDate",
        ".//ArticleDate/Year",
        ".//PubmedData/History/PubMedPubDate/Year",
    ):
        if (value := _text(article.find(path))) and (match := _YEAR.search(value)):
            return int(match.group(0))
    return None


def _abstract(article: ET.Element) -> str | None:
    parts = []
    for node in article.findall(".//Abstract/AbstractText"):
        text = _text(node)
        if not text:
            continue
        label = node.get("Label")
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts) or None


def _article_record(article: ET.Element) -> dict[str, Any] | None:
    pmid = _text(article.find("./MedlineCitation/PMID"))
    if not pmid:
        return None
    publication_types = [
        t
        for node in article.findall(".//PublicationTypeList/PublicationType")
        if (t := _text(node))
    ]
    corrections = {
        node.get("RefType", "") for node in article.findall(".//CommentsCorrectionsList/*")
    }
    lowered = {p.lower() for p in publication_types}
    doi = _text(article.find(".//ArticleIdList/ArticleId[@IdType='doi']")) or _text(
        article.find(".//ELocationID[@EIdType='doi']")
    )
    abstract = _abstract(article)
    return {
        "pmid": pmid,
        "title": _text(article.find(".//Article/ArticleTitle")),
        "abstract": abstract,
        "journal": _text(article.find(".//Journal/ISOAbbreviation"))
        or _text(article.find(".//Journal/Title")),
        "year": _year(article),
        "doi": doi.lower() if doi else None,
        "pmcid": _text(article.find(".//ArticleIdList/ArticleId[@IdType='pmc']")),
        "language": _text(article.find(".//Article/Language")),
        "publication_types": publication_types,
        "mesh_terms": [
            t for node in article.findall(".//MeshHeading/DescriptorName") if (t := _text(node))
        ],
        "article_type": classify_article_type(publication_types),
        "is_retracted": "retracted publication" in lowered or "RetractionIn" in corrections,
        "is_retraction_notice": "retraction of publication" in lowered,
        "has_erratum": "ErratumIn" in corrections,
        "has_expression_of_concern": "ExpressionOfConcernIn" in corrections,
        "has_abstract": abstract is not None,
    }


class Readable(Protocol):
    def read(self, size: int = ..., /) -> bytes: ...


def iter_pubmed_articles(source: Readable) -> Iterator[dict[str, Any]]:
    """Stream records out of an efetch / baseline document without loading it whole.

    Book chapters (`PubmedBookArticle`) are skipped: they have no journal citation.
    """
    for _, element in ET.iterparse(source, events=("end",)):  # noqa: S314 - see import note
        if element.tag == "PubmedArticle":
            if (record := _article_record(element)) is not None:
                yield record
            element.clear()


def parse_pubmed_xml(data: bytes) -> list[dict[str, Any]]:
    """Parse an efetch response into one record per journal article."""
    return list(iter_pubmed_articles(io.BytesIO(data)))


ARTICLE_COLUMNS = [
    "pmid",
    "title",
    "abstract",
    "journal",
    "year",
    "doi",
    "pmcid",
    "language",
    "publication_types",
    "mesh_terms",
    "article_type",
    "is_retracted",
    "is_retraction_notice",
    "has_erratum",
    "has_expression_of_concern",
    "has_abstract",
]


def articles_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Records as a typed frame (empty input still yields every column)."""
    frame = pd.DataFrame(records, columns=[*ARTICLE_COLUMNS, *PROVENANCE_COLUMNS])
    for column in (
        "pmid",
        "title",
        "abstract",
        "journal",
        "doi",
        "pmcid",
        "language",
        "article_type",
        *PROVENANCE_COLUMNS,
    ):
        frame[column] = frame[column].astype("string")
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce").astype("Int64")
    for column in (
        "is_retracted",
        "is_retraction_notice",
        "has_erratum",
        "has_expression_of_concern",
        "has_abstract",
    ):
        frame[column] = frame[column].fillna(False).astype(bool)
    return frame


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per PMID: the most recently retrieved copy wins (it carries late retractions)."""
    return (
        frame.sort_values(["pmid", "retrieved_at"])
        .drop_duplicates("pmid", keep="last")
        .reset_index(drop=True)
    )


def build_abstract_store(
    articles: pd.DataFrame, open_access: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Final `abstracts.parquet`: deduplicated articles, segmented, with a PubMed URL.

    `open_access` is the Europe PMC table; it fills abstracts PubMed lacks and adds the
    open-access flags.
    """
    store = deduplicate(articles).copy()
    store["abstract_source"] = store["abstract"].notna().map({True: "pubmed", False: None})
    store["is_open_access"] = pd.Series(pd.NA, index=store.index, dtype="boolean")
    store["in_europepmc"] = pd.Series(pd.NA, index=store.index, dtype="boolean")
    if open_access is not None and not open_access.empty:
        extra = open_access.drop_duplicates("pmid").set_index("pmid")
        pmids = store["pmid"]
        store["is_open_access"] = pmids.map(extra["is_open_access"]).astype("boolean")
        store["in_europepmc"] = pmids.map(extra["in_epmc"]).astype("boolean")
        store["pmcid"] = store["pmcid"].fillna(pmids.map(extra["pmcid"]))
        fallback = pmids.map(extra["abstract"])
        fill = store["abstract"].isna() & fallback.notna()
        store.loc[fill, "abstract"] = fallback[fill]
        store.loc[fill, "abstract_source"] = "europepmc"
        store["has_abstract"] = store["abstract"].notna()
    sentences = store["abstract"].map(
        lambda text: [s.text for s in split_sentences(text)] if isinstance(text, str) else []
    )
    store["sentences"] = sentences
    store["n_sentences"] = sentences.map(len).astype("int64")
    store["pubmed_url"] = "https://pubmed.ncbi.nlm.nih.gov/" + store["pmid"].astype(str) + "/"
    ordered = [c for c in store.columns if c not in PROVENANCE_COLUMNS] + PROVENANCE_COLUMNS
    return store[ordered]
