"""The indexed documents, rebuilt from their chunks: what the Documents page lists (D114).

A document is a notice (openFDA label), an abstract (PubMed) or an interaction pair
(DDInter); the index stores its chunks, each with the document's metadata. The catalog
groups the chunks by `doc_id`, in chunk order, and filters the documents by type,
ingredient (about or mentioned), section, language, source and a text query.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from src.rag.embeddings import fold
from src.rag.store import Chunk

TYPE_ORDER = {"label": 0, "abstract": 1, "interaction": 2}


def source_group(source_name: str) -> str:
    """ "PubMed (J Test)" -> "PubMed": the journal stays in the document's own source."""
    return source_name.split(" (", 1)[0]


@dataclass(frozen=True)
class Document:
    doc_id: str
    doc_type: str
    title: str
    source_name: str
    url: str
    date: str
    version: str
    license: str
    language: str
    article_type: str | None
    is_retracted: bool
    ingredients: tuple[str, ...]
    mentioned_ingredients: tuple[str, ...]
    sections: tuple[str, ...]
    chunks: tuple[Chunk, ...]
    hashes: tuple[str, ...] = ()  # SHA-256 of each chunk (index.content_hash), same order

    @classmethod
    def from_chunks(cls, chunks: list[Chunk], hashes: dict[str, str] | None = None) -> Document:
        chunks = sorted(chunks, key=lambda c: (c.chunk_index, c.chunk_id))
        first = chunks[0]
        # Label chunks are titled "<label>: <Section>"; the other types share one title.
        title = first.title.rsplit(": ", 1)[0] if first.doc_type == "label" else first.title
        mentioned = {name for c in chunks for name in c.mentioned_ingredients}
        return cls(
            doc_id=first.doc_id,
            doc_type=first.doc_type,
            title=title,
            source_name=first.source_name,
            url=first.url,
            date=first.date,
            version=first.version,
            license=first.license,
            language=first.language,
            article_type=first.article_type,
            is_retracted=any(c.is_retracted for c in chunks),
            ingredients=tuple(first.ingredients),
            mentioned_ingredients=tuple(sorted(mentioned - set(first.ingredients))),
            sections=tuple(dict.fromkeys(c.section for c in chunks)),
            chunks=tuple(chunks),
            hashes=tuple((hashes or {}).get(c.chunk_id, "") for c in chunks),
        )

    def matches(
        self,
        *,
        doc_type: str | None = None,
        ingredient: str | None = None,
        section: str | None = None,
        language: str | None = None,
        source: str | None = None,
        query: str | None = None,
    ) -> bool:
        if doc_type and self.doc_type != doc_type:
            return False
        if language and self.language != language:
            return False
        if section and section not in self.sections:
            return False
        if source and source_group(self.source_name) != source:
            return False
        if ingredient and ingredient not in (*self.ingredients, *self.mentioned_ingredients):
            return False
        if query:
            haystack = fold(" ".join([self.title, self.doc_id, *self.ingredients]))
            if not all(word in haystack for word in fold(query).split()):
                return False
        return True


class Catalog:
    def __init__(self, chunks: Iterable[Chunk], hashes: dict[str, str] | None = None) -> None:
        groups: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            groups.setdefault(chunk.doc_id, []).append(chunk)
        documents = [Document.from_chunks(group, hashes) for group in groups.values()]
        self.documents = sorted(
            documents, key=lambda d: (TYPE_ORDER.get(d.doc_type, 9), fold(d.title), d.doc_id)
        )
        self.by_id = {d.doc_id: d for d in self.documents}
        self.chunks = sum(len(d.chunks) for d in self.documents)

    def get(self, doc_id: str) -> Document | None:
        return self.by_id.get(doc_id)

    def search(self, **criteria: str | None) -> list[Document]:
        return [d for d in self.documents if d.matches(**criteria)]

    @staticmethod
    def facets(documents: list[Document]) -> dict[str, dict[str, int]]:
        """Document counts per type, language and source, for the page's filters."""
        return {
            "doc_type": dict(Counter(d.doc_type for d in documents).most_common()),
            "language": dict(Counter(d.language for d in documents).most_common()),
            "source": dict(Counter(source_group(d.source_name) for d in documents).most_common()),
        }
