"""Step 6 — LLM: question -> search -> numbered sources -> prompt v1 -> qwen2.5:3b -> cited answer.

The agent (src/agent/graph.py) runs it as its retrieve and answer nodes; the safety check
(src/agent/safety.py) then removes doses, "safe" claims and uncited sentences.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Iterable
from typing import TYPE_CHECKING, Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from src.rag.retrieval import Hit

MAX_SOURCES = 8
SOURCE_CHARS = 900  # characters of each source given to the LLM
SNIPPET_CHARS = 700  # characters of each source shown in the app
# Cross-references of the labels ("[see Warnings (5.1)]", "(5.1)"): never shown or cited.
CROSS_REFERENCE = re.compile(r"\[\s*see [^\]]*\]|\(\s*\d+(?:\.\d+)*\s*\)", re.IGNORECASE)


def hit_source(hit: Hit) -> dict[str, Any]:
    """A retrieved chunk as a source: its text, metadata and scores."""
    chunk = hit.chunk
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "doc_type": chunk.doc_type,
        "title": chunk.title,
        "section": chunk.section,
        "url": chunk.url,
        "source_name": chunk.source_name,
        "date": chunk.date,
        "version": chunk.version,
        "license": chunk.license,
        "language": chunk.language,
        "ingredients": chunk.ingredients,
        "is_retracted": chunk.is_retracted,
        "article_type": chunk.article_type,
        "text": chunk.text,
        "snippet": chunk.text[:SNIPPET_CHARS],
        "score": round(hit.score, 5),
        # Cross-encoder score and the rank before reranking (None when not reranked).
        "rerank_score": None if hit.rerank_score is None else round(hit.rerank_score, 4),
        "retrieval_rank": hit.retrieval_rank,
    }


def number_sources(
    groups: Iterable[tuple[list[dict[str, Any]], int]], max_sources: int = MAX_SOURCES
) -> list[dict[str, Any]]:
    """The sources of the answer, numbered 1, 2, ... (key `n`): at most `limit` items of each
    group, in order, each chunk once, `max_sources` in all."""
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for items, limit in groups:
        for item in items[:limit]:
            if item["chunk_id"] not in seen and len(sources) < max_sources:
                seen.add(item["chunk_id"])
                sources.append(item)
    for number, source in enumerate(sources, start=1):
        source["n"] = number
    return sources


def source_block(source: dict[str, Any]) -> str:
    """One numbered source as the LLM reads it: number, title, origin, date and text."""
    text = CROSS_REFERENCE.sub("", source.get("text") or source.get("snippet") or "")
    return (
        f"[{source['n']}] {source['title']} ({source['source_name']}"
        f"{', ' + source['date'] if source.get('date') else ''})\n{text[:SOURCE_CHARS]}"
    )


def answer_messages(
    system_prompt: str, question: str, sources: list[dict[str, Any]], preamble: str = ""
) -> list[BaseMessage]:
    """The prompt: the answer rules, then the numbered sources and the question."""
    context = "\n\n".join(source_block(source) for source in sources)
    user = preamble + f"Sources :\n{context}\n\nQuestion : {question}"
    return [SystemMessage(system_prompt), HumanMessage(user)]


async def stream_answer(llm: BaseChatModel, messages: list[BaseMessage]) -> AsyncIterator[str]:
    """The LLM's answer, piece by piece as it is generated."""
    async for chunk in llm.astream(messages):
        text = chunk.content if isinstance(chunk.content, str) else ""
        if text:
            yield text
