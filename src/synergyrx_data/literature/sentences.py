"""Rule-based sentence segmentation for biomedical abstracts.

No model download is needed. A boundary is a `.`, `!` or `?` followed by whitespace and an
upper-case letter, digit or opening bracket, unless the text before it is a known
abbreviation ("et al.", "Fig.", "vs."), a single initial, or part of a number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_BOUNDARY = re.compile(r"[.!?][\"')\]]*\s+(?=[A-Z0-9(\[\"'])")
_ABBREVIATIONS = frozenset(
    {
        "al", "approx", "ca", "cf", "dr", "e.g", "eq", "eqs", "etc", "fig", "figs", "i.e",
        "inc", "mr", "mrs", "ms", "no", "nos", "prof", "ref", "refs", "resp", "sp", "spp",
        "st", "suppl", "vol", "vs",
    }
)  # fmt: skip
_LAST_TOKEN = re.compile(r"([A-Za-z][A-Za-z.]*)\.$")


@dataclass(frozen=True)
class Sentence:
    index: int
    start: int
    end: int
    text: str


def _is_false_boundary(before: str) -> bool:
    """`before` is the text up to and including the candidate full stop."""
    match = _LAST_TOKEN.search(before)
    if not match:
        return False
    token = match.group(1)
    if token.lower() in _ABBREVIATIONS:
        return True
    # A lone capital is an initial ("J. Smith"), not the end of a sentence.
    return len(token) == 1 and token.isupper()


def split_sentences(text: str | None) -> list[Sentence]:
    """Split text into sentences, keeping character offsets into the original string."""
    if not text:
        return []
    sentences: list[Sentence] = []
    offset = 0
    for block in text.split("\n"):  # structured abstracts: one labelled section per line
        start = 0
        for match in _BOUNDARY.finditer(block):
            stop = match.start() + 1
            closing = re.match(r"[\"')\]]*", block[stop:])
            stop += closing.end() if closing else 0
            if match.group(0)[0] == "." and _is_false_boundary(block[start : match.start() + 1]):
                continue
            _append(sentences, block, start, stop, offset)
            start = match.end()
        _append(sentences, block, start, len(block), offset)
        offset += len(block) + 1
    return sentences


def _append(sentences: list[Sentence], block: str, start: int, stop: int, offset: int) -> None:
    raw = block[start:stop]
    stripped = raw.strip()
    if not stripped:
        return
    begin = offset + start + (len(raw) - len(raw.lstrip()))
    sentences.append(Sentence(len(sentences), begin, begin + len(stripped), stripped))
