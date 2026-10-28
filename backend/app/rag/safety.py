"""Safety rules applied to every agent answer (CLAUDE.md), sentence by sentence.

1. No dose: a sentence with a quantity of medicine ("50 mg", "2 comprimés", "deux fois
   par jour") is removed. A question asking for a dose is refused before any LLM call.
2. Never "safe": a sentence calling a combination safe ("sûr", "sans danger", "sans
   risque", "safe") is removed; the standard wording is "aucune interaction connue dans
   cette base de données".
3. No invented timing advice: a timing sentence ("à 2 heures d'intervalle", "avant le
   repas") is kept only if it cites a source.
4. Citations: [n] must point to a numbered source; an answer citing nothing gets the list
   of the sources it was given; an answer without any source is replaced by "no source".
5. The disclaimer always ends the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.medicines import DISCLAIMER, NO_KNOWN_INTERACTION
from app.rag.embeddings import fold

_UNITS = (
    r"(?:mg|mcg|µg|μg|ug|g|ml|ui|iu|unites?|units?|gouttes?|drops?|comprimes?"
    r"|gelules?|tablets?|capsules?|pills?)"
    # not a lab value: "eGFR 30 mL/min", "glucose 200 mg/dL", "2 g/L"
    r"(?!\s*/\s*(?:min|dl|l\b|h\b|m2|1\.73))"
)
# Checked on folded text (lower case, no accents).
DOSE = re.compile(
    rf"\b\d+(?:[.,]\d+)?\s*{_UNITS}\b"
    r"|\b(?:\d+|une|deux|trois|quatre)\s+fois\s+par\s+(?:jour|semaine)\b"
    r"|\b(?:once|twice|three times)\s+(?:a|per)\s+(?:day|week)\b"
    r"|\bmg\s*/\s*(?:kg|j|jour|day)\b|\bposologie\b|\bdosage recommande\b",
    re.IGNORECASE,
)
# Checked on the lower-cased text with its accents: folded, "sûr" would be "sur" ("on").
SAFE = re.compile(
    r"\bsûr(?:e|s|es)?\b|\bsafe(?:ly)?\b|\bharmless\b|\bsans (?:danger|risque)\b"
    r"|\bsécuritaire\b|\ben toute sécurité\b|\bno risk\b",
    re.IGNORECASE,
)
TIMING = re.compile(
    r"\bheures? d'intervalle\b|\bespacer\b|\bespacez\b|\bavant (?:le|les) repas\b"
    r"|\bapres (?:le|les) repas\b|\ba jeun\b|\bau coucher\b|\bhours? apart\b"
    r"|\b(?:before|after|with) (?:a )?meals?\b|\bat bedtime\b|\bon an empty stomach\b",
    re.IGNORECASE,
)
DOSE_QUESTION = re.compile(
    r"\bdoses?\b|\bposologie\b|\bdosage\b|\bcombien de (?:mg|milligrammes?|comprimes?|gelules?)\b"
    r"|\bquelle quantite\b|\bhow much\b|\bhow many (?:mg|milligrams?|tablets?|pills?)\b",
    re.IGNORECASE,
)
CITATION = re.compile(r"\[(\d+)\]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?])")
CROSS_REFERENCE = re.compile(r"\[\s*see [^\]]*\]|\(\s*\d+(?:\.\d+)*\s*\)", re.IGNORECASE)
_WORDS = re.compile(r"[a-z0-9]+")
_ONLY_CITATIONS = re.compile(r"^[\s.,;]*(?:\[\d+\][\s.,;]*)+$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")

REFUSAL_DOSE = (
    "Tujya ne donne jamais de dose ni de posologie. Pour savoir quelle quantité prendre, "
    "demandez à votre pharmacien ou à votre médecin : ils tiennent compte de votre situation."
)
NO_SOURCE = (
    "Je n'ai trouvé aucune source dans la base documentaire de Tujya pour répondre à cette "
    "question, donc je ne réponds pas."
)
REMOVED_NOTE = "(Une phrase a été retirée par le contrôle de sécurité.)"


@dataclass
class SafetyReport:
    answer: str
    violations: list[str] = field(default_factory=list)
    action: str = "ok"  # ok | rewritten | refused
    cited: list[int] = field(default_factory=list)


def is_dose_question(question: str) -> bool:
    return bool(DOSE_QUESTION.search(fold(question)))


def sentence_problems(sentence: str) -> list[str]:
    """Rules 1-3 for one sentence (rule 3 only when it cites nothing)."""
    folded = fold(sentence)
    problems = []
    if DOSE.search(folded):
        problems.append("dose")
    if SAFE.search(sentence.lower()):
        problems.append("safe")
    if TIMING.search(folded) and not CITATION.search(sentence):
        problems.append("uncited_timing")
    return problems


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text) if s and s.strip()]


def _content_words(text: str) -> set[str]:
    return {w for w in _WORDS.findall(fold(text)) if len(w) > 3}


def best_source(sentence: str, sources: list[str], threshold: float = 0.6) -> int | None:
    """The 1-based source containing most of the sentence's content words, if most are there.

    Used only for a sentence the model forgot to cite: the citation then reflects text
    that is really in that source (a copied or near-copied sentence), never a guess.
    """
    words = _content_words(sentence)
    if len(words) < 4:
        return None
    scores = [len(words & _content_words(text)) / len(words) for text in sources]
    best = max(range(len(scores)), key=lambda i: scores[i], default=None)
    return best + 1 if best is not None and scores[best] >= threshold else None


def check_answer(
    text: str,
    n_sources: int,
    *,
    interaction_found: bool,
    source_texts: list[str] | None = None,
) -> SafetyReport:
    """Apply rules 1-5 to an answer that cites sources numbered 1..n_sources."""
    if n_sources == 0:
        return SafetyReport(f"{NO_SOURCE}\n\n{DISCLAIMER}", ["no_source"], "refused")
    violations: list[str] = []
    kept: list[str] = []
    removed_safe = False
    removed = 0
    for sentence in split_sentences(text.replace(DISCLAIMER, "")):
        sentence = CROSS_REFERENCE.sub("", sentence).strip()
        problems = sentence_problems(sentence)
        if problems:
            violations += problems
            removed += 1
            removed_safe = removed_safe or "safe" in problems
            continue

        numbers = [int(n) for n in CITATION.findall(sentence)]
        if any(not 1 <= n <= n_sources for n in numbers):
            violations.append("bad_citation")
        cleaned = CITATION.sub(
            lambda m: m.group(0) if 1 <= int(m.group(1)) <= n_sources else "", sentence
        )
        cleaned = _SPACE_BEFORE_PUNCT.sub(r"\1", cleaned).strip()
        if _ONLY_CITATIONS.match(cleaned):  # "[1]" alone: belongs to the sentence before
            if kept and not set(CITATION.findall(cleaned)) <= set(CITATION.findall(kept[-1])):
                kept[-1] = f"{kept[-1]} {cleaned}"
            continue
        if source_texts and not CITATION.search(cleaned):
            number = best_source(cleaned, source_texts)
            if number is not None:
                violations.append("citation_added")
                cleaned = re.sub(r"([.!?]?)$", rf" [{number}]\1", cleaned, count=1)
        kept.append(cleaned)
    if removed_safe and not interaction_found:
        kept.append(NO_KNOWN_INTERACTION)
    elif removed and kept:
        kept.append(REMOVED_NOTE)
    answer = " ".join(s for s in kept if s)
    cited = sorted({int(n) for n in CITATION.findall(answer)})
    if not cited:
        violations.append("no_citation")
        listed = ", ".join(f"[{i}]" for i in range(1, n_sources + 1))
        answer = (answer + f"\n\nSources consultées : {listed}").strip()
    if not answer.strip() or not kept:
        return SafetyReport(f"{NO_SOURCE}\n\n{DISCLAIMER}", violations, "refused")
    action = "rewritten" if set(violations) - {"citation_added"} else "ok"
    return SafetyReport(f"{answer}\n\n{DISCLAIMER}", violations, action, cited)


class SentenceFilter:
    """Streams an answer sentence by sentence, holding back sentences that break rules 1-3.

    The final safety node still checks the whole answer; this only keeps unsafe text from
    ever reaching the screen while the answer is being generated.
    """

    def __init__(self) -> None:
        self.buffer = ""

    def feed(self, text: str) -> list[str]:
        self.buffer += text
        parts = _SENTENCE.split(self.buffer)
        if len(parts) == 1:
            return []
        complete, self.buffer = parts[:-1], parts[-1]
        return [s + " " for s in complete if s.strip() and not sentence_problems(s)]

    def flush(self) -> list[str]:
        rest, self.buffer = self.buffer, ""
        return [rest] if rest.strip() and not sentence_problems(rest) else []
