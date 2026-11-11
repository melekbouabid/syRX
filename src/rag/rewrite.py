"""Optional query rewriting by the LLM before retrieval (step 3.2, D118).

The corpus is in English (openFDA labels, PubMed abstracts, DDInter); half the questions
are in French. The LLM rewrites the question as a short English search query that keeps
every drug name; retrieval then runs on that query. The question itself is unchanged for
the answer. Off by default (`TUJYA_RAG_QUERY_REWRITE`): one more LLM call per question; the
benchmark measures what it brings (docs/module2/benchmark.md).

Never trusted blindly: an empty or overlong rewrite, or an LLM error, keeps the question.
"""

from __future__ import annotations

import logging
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

log = logging.getLogger(__name__)

PROMPT = """You rewrite questions about medicines into search queries for a database of \
English drug labels, PubMed abstracts and drug interaction records.
Write one short English search query (at most 20 words) with the key terms of the question.
Keep every drug or active ingredient name. Translate other words into English.
Do not answer the question. Reply with the query only."""
MAX_CHARS = 200


class QueryRewriter:
    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm

    async def rewrite(self, question: str) -> str:
        try:
            message = await self.llm.ainvoke([SystemMessage(PROMPT), HumanMessage(question)])
        except Exception as error:  # the search still runs on the question itself
            log.warning("query rewriting failed: %s", error)
            return question
        query = clean(str(message.content))
        return query if query and len(query) <= MAX_CHARS else question


def clean(text: str) -> str:
    """First non-empty line, without a "Query:" prefix or surrounding quotes."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    query = re.sub(r"^(search )?query\s*:\s*", "", lines[0], flags=re.IGNORECASE)
    return query.strip().strip("\"'`").strip()
