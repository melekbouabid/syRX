"""Find drug names in a question, in French or English, and map them to ingredient names.

The ingredient table uses English RxNorm / DDInter names ("warfarin", "acetaminophen").
French INNs are mostly the English name plus an "e" ("warfarine", "metformine",
"ibuprofène"), which the matcher tries; a few common names differ and are listed below.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from app.rag.embeddings import fold

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
