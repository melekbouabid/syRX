"""Offline drug-mention detection with the synonym table.

No network and no model: sentences are tokenised and every run of up to `max_ngram` tokens
is looked up in a dictionary built from drug names and synonyms. A sentence that mentions
at least two different drugs is a candidate for relation extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import pandas as pd

from synergyrx_data.cleaning.pairs import canonical_pair
from synergyrx_data.literature.sentences import split_sentences
from synergyrx_data.provenance import PROVENANCE_COLUMNS

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_CUES = re.compile(
    r"\b(synerg\w*|antagon\w*|additiv\w*|potentiat\w*|sensiti[sz]\w*|combination\w*|"
    r"combined|co-?treatment|co-?administ\w*|enhanc\w*)\b",
    re.IGNORECASE,
)
# Short synonyms that are also ordinary words or common abbreviations in abstracts.
AMBIGUOUS_TERMS = frozenset(
    {
        "acid", "act", "aim", "all", "and", "base", "can", "car", "cat", "cell", "compound",
        "control", "dose", "drug", "fast", "gel", "gold", "her", "his", "impact", "ion",
        "iron", "lead", "led", "mab", "max", "met", "mice", "not", "pan", "set", "tam",
        "the", "top", "via", "water", "who",
    }
)  # fmt: skip
MAX_PAIRS_PER_SENTENCE = 45


@dataclass
class SynonymIndex:
    """Token-sequence dictionary: `("abt", "888") -> drug_id` and `("abt888",) -> drug_id`."""

    terms: dict[tuple[str, ...], str] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    dropped_ambiguous: int = 0
    dropped_short: int = 0

    @classmethod
    def from_drugs(
        cls, drugs: pd.DataFrame, min_length: int = 4, max_ngram: int = 6
    ) -> SynonymIndex:
        index = cls()
        claims: dict[tuple[str, ...], set[str]] = {}
        for row in drugs[["drug_id", "name", "synonyms"]].itertuples(index=False):
            index.names[row.drug_id] = row.name
            for term in [row.name, *(row.synonyms if row.synonyms is not None else [])]:
                tokens = tuple(t.lower() for t in _TOKEN.findall(str(term)))
                joined = "".join(tokens)
                # Codes that mix letters and digits ("5-FU") are specific even when short.
                mixed = any(c.isdigit() for c in joined) and any(c.isalpha() for c in joined)
                too_short = joined.isdigit() or len(joined) < (3 if mixed else min_length)
                if not tokens or len(tokens) > max_ngram or too_short:
                    index.dropped_short += 1
                    continue
                if joined in AMBIGUOUS_TERMS:
                    index.dropped_ambiguous += 1
                    continue
                claims.setdefault(tokens, set()).add(row.drug_id)
                if len(tokens) > 1:  # "ABT-888" should also match "ABT888"
                    claims.setdefault((joined,), set()).add(row.drug_id)
        for tokens, drug_ids in claims.items():
            if len(drug_ids) == 1:
                index.terms[tokens] = next(iter(drug_ids))
            else:
                index.dropped_ambiguous += 1  # the same term names two different drugs
        return index

    def find(self, text: str, max_ngram: int = 6) -> list[tuple[str, str, int, int]]:
        """Return `(drug_id, matched_text, start, end)` for each mention, longest match first."""
        tokens = [(m.group(0).lower(), m.start(), m.end()) for m in _TOKEN.finditer(text)]
        mentions = []
        position = 0
        while position < len(tokens):
            matched = False
            for size in range(min(max_ngram, len(tokens) - position), 0, -1):
                key = tuple(t[0] for t in tokens[position : position + size])
                drug_id = self.terms.get(key)
                if drug_id is not None:
                    start, end = tokens[position][1], tokens[position + size - 1][2]
                    mentions.append((drug_id, text[start:end], start, end))
                    position += size
                    matched = True
                    break
            if not matched:
                position += 1
        return mentions


def synergy_cues(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _CUES.finditer(text)})


def find_candidate_sentences(
    abstracts: pd.DataFrame,
    index: SynonymIndex,
    *,
    min_drugs: int = 2,
    max_ngram: int = 6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Sentences mentioning at least `min_drugs` different drugs, one row per sentence."""
    rows: list[dict[str, Any]] = []
    stats = {"abstracts": 0, "sentences": 0, "sentences_with_a_drug": 0}
    columns = [
        "pmid",
        "abstract",
        "year",
        "article_type",
        "language",
        "is_retracted",
        *PROVENANCE_COLUMNS,
    ]
    for record in abstracts[columns].to_dict("records"):
        if not isinstance(record["abstract"], str):
            continue
        stats["abstracts"] += 1
        for sentence in split_sentences(record["abstract"]):
            stats["sentences"] += 1
            mentions = index.find(sentence.text, max_ngram)
            drug_ids = sorted({m[0] for m in mentions})
            if drug_ids:
                stats["sentences_with_a_drug"] += 1
            if len(drug_ids) < min_drugs:
                continue
            pairs = [canonical_pair(a, b)[2] for a, b in combinations(drug_ids, 2)]
            cues = synergy_cues(sentence.text)
            rows.append(
                {
                    "pmid": record["pmid"],
                    "sentence_index": sentence.index,
                    "sentence_start": sentence.start,
                    "sentence": sentence.text,
                    "n_drugs": len(drug_ids),
                    "drug_ids": drug_ids,
                    "drug_names": [index.names.get(d, d) for d in drug_ids],
                    "mentions": sorted({m[1] for m in mentions}),
                    "pair_keys": pairs[:MAX_PAIRS_PER_SENTENCE],
                    "has_synergy_cue": bool(cues),
                    "cue_terms": cues,
                    "year": record["year"],
                    "article_type": record["article_type"],
                    "language": record["language"],
                    "is_retracted": bool(record["is_retracted"]),
                    **{c: record[c] for c in PROVENANCE_COLUMNS},
                }
            )
    frame = pd.DataFrame(rows)
    stats["candidate_sentences"] = len(frame)
    return frame, stats
