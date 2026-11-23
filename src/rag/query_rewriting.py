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
from collections.abc import Iterable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.rag.chunking import fold

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
    """First non-empty line, without a "Query:" prefix or surrounding quotes. A small LLM
    sometimes glues the words ("what_are_the_contraindications_of_metformin"): BM25 would
    then see one unknown term, so underscores, and hyphens in a query without spaces,
    become spaces."""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    query = re.sub(r"^(search )?query\s*:\s*", "", lines[0], flags=re.IGNORECASE)
    query = query.replace("_", " ")
    if " " not in query.strip():
        query = query.replace("-", " ")
    return re.sub(r"\s+", " ", query).strip().strip("\"'`?").strip()


# --- Drug names of the question, in French or English ------------------------------------------
# Find drug names in a question, in French or English, and map them to ingredient names.
#
# The ingredient table uses English RxNorm / DDInter names ("warfarin", "acetaminophen").
# French INNs are mostly the English name plus an "e" ("warfarine", "metformine",
# "ibuprofène"), which the matcher tries; a few common names differ and are listed below.

FRENCH_NAMES = {
    "paracetamol": "acetaminophen",
    "acide acetylsalicylique": "aspirin",
    "acide valproique": "valproate",
    "valproate de sodium": "valproate",
    "divalproex": "valproate",
    "alcool": "ethanol",
    "millepertuis": "st. john's wort",
    "pamplemousse": "grapefruit",
    "vitamine b12": "vitamin b12",
    "vitamine b6": "vitamin b6",
    "acide folique": "folic acid",
    "hydrochlorothiazide": "hydrochlorothiazide",
    "levothyroxine sodique": "levothyroxine",
}
_WORD = re.compile(r"[a-z0-9']+")
MIN_LENGTH = 4


class NameIndex:
    def __init__(self, names: Iterable[str]) -> None:
        self.names = sorted({n.strip().lower() for n in names if n and n.strip()})
        self.index: dict[str, str] = {}
        for name in self.names:
            self.index.setdefault(fold(name), name)
        for french, english in FRENCH_NAMES.items():
            if english in self.index.values():
                self.index.setdefault(french, english)

    def lookup(self, text: str) -> str | None:
        """One name typed by a user ("Warfarine", "paracétamol") -> ingredient name."""
        words = _WORD.findall(fold(text))
        key = " ".join(words)
        for candidate in (key, " ".join(_strip_e(w) for w in words)):
            if candidate in self.index:
                return self.index[candidate]
        return None

    def find(self, question: str, max_words: int = 4) -> list[str]:
        """Every ingredient named in a question, longest names first, in order of mention."""
        words = _WORD.findall(fold(question))
        taken = [False] * len(words)
        found: list[tuple[int, str]] = []
        for size in range(max_words, 0, -1):
            for start in range(len(words) - size + 1):
                if any(taken[start : start + size]):
                    continue
                gram = words[start : start + size]
                if size == 1 and len(gram[0]) < MIN_LENGTH:
                    continue
                name = self.lookup(" ".join(gram))
                if name is not None:
                    found.append((start, name))
                    for i in range(start, start + size):
                        taken[i] = True
        ordered: list[str] = []
        for _, name in sorted(found):
            if name not in ordered:
                ordered.append(name)
        return ordered


def _strip_e(word: str) -> str:
    return word[:-1] if len(word) > MIN_LENGTH and word.endswith("e") else word
