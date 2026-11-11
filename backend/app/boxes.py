"""Medicine boxes: read the printed text (OCR), then link it to active ingredients.

Step 2.7 (docs/module1/boites.md, decisions D108-D112). The pill models only know C3PI
pills, so a photo of a box (Panadol, Doliprane...) used to get a wrong pill name. A box is
recognised from its text instead:

1. OCR (RapidOCR: PP-OCR detection and recognition models in ONNX, run by ONNX Runtime
   like the pill models, no PyTorch) gives the printed lines;
2. the lines are matched with the brand table `config/brand_aliases.csv` (each row with its
   source) and with the ingredient names of the interaction table plus
   `config/inn_aliases.csv` (European boxes print the INN: "paracetamolo", "ibuprofène");
3. when neither matches, RxNorm (RxNav `approximateTerm`, public API) is asked for the
   largest printed words, only when a contact email is configured (polite client, same rule
   as the data pipeline);
4. nothing found: "boîte non reconnue", never a guess.

`decide_kind` then says whether the photo shows a pill, a box or something else
(out of distribution: the scanner refuses instead of naming a pill class).
"""

from __future__ import annotations

import csv
import json
import re
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from rapidfuzz import fuzz, process

Kind = Literal["pill", "box", "unknown"]

BRAND_MIN_SCORE = 86.0  # rapidfuzz ratio, 0-100: "voltaren" ~ "voltarene" (94)
NAME_MIN_SCORE = 88.0  # "ibuprofene" ~ "ibuprofen" (95), "cetirizina" ~ "cetirizine" (90)
MIN_FUZZY_LENGTH = 6  # shorter words must match exactly
MAX_WORDS = 3  # longest ingredient name searched, in words ("acido acetilsalicilico")
RXNAV_TERMS = 3  # largest printed words sent to RxNav when nothing else matched
_WORD = re.compile(r"[a-z]+")
# Words printed on most boxes that are never a brand (not sent to RxNav).
COMMON_WORDS = frozenset(
    (
        "compresse",
        "compressa",
        "comprime",
        "comprimes",
        "compresses",
        "capsule",
        "capsules",
        "gelules",
        "gelule",
        "tablets",
        "tablet",
        "film",
        "rivestite",
        "pellicule",
        "pellicules",
        "effervescenti",
        "effervescent",
        "effervescents",
        "bustine",
        "sachets",
        "sachet",
        "sciroppo",
        "sirop",
        "solution",
        "soluzione",
        "orale",
        "oral",
        "buvable",
        "crema",
        "creme",
        "gel",
        "pomata",
        "spray",
        "gocce",
        "gouttes",
        "adulti",
        "adultes",
        "adults",
        "enfants",
        "bambini",
        "children",
        "uso",
        "usage",
        "medicinale",
        "medicament",
        "medicine",
        "farmaco",
        "boite",
        "scatola",
        "dolore",
        "douleur",
        "febbre",
        "fievre",
        "pain",
        "fever",
        "analgesico",
        "antinfiammatorio",
        "contiene",
        "contains",
        "principio",
        "attivo",
        "actif",
        "active",
        "ogni",
        "each",
        "prodotto",
        "produit",
    )
)


class OcrUnavailable(RuntimeError):
    """The OCR engine (rapidocr) is not installed or its models cannot be loaded."""


def fold(text: str) -> str:
    """Lowercase ASCII letters and digits: accents removed, everything else a space."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
    return re.sub(r"[^a-z0-9]+", " ", ascii_text).strip()


def words(text: str) -> list[str]:
    """Letter words of a printed line; digits split glued words ("EFFERALGAN500mg")."""
    return _WORD.findall(fold(text))


# --------------------------------------------------------------------------- tables


@dataclass(frozen=True)
class Brand:
    brand: str
    ingredients: tuple[str, ...]
    printed_substances: str
    country: str
    source: str
    reference: str
    url: str

    @property
    def key(self) -> str:
        return fold(self.brand)


@dataclass(frozen=True)
class InnAlias:
    alias: str
    ingredient: str
    language: str
    source: str
    url: str


def load_brands(path: Path) -> list[Brand]:
    """config/brand_aliases.csv: brand -> ingredients (DDInter names, `;`), with the source."""
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            Brand(
                brand=row["brand"].strip(),
                ingredients=tuple(i.strip().lower() for i in row["ingredients"].split(";") if i),
                printed_substances=row["printed_substances"].strip(),
                country=row["country"].strip(),
                source=row["source"].strip(),
                reference=row["reference"].strip(),
                url=row["url"].strip(),
            )
            for row in csv.DictReader(handle)
            if row["brand"].strip()
        ]


def load_inn_aliases(path: Path) -> list[InnAlias]:
    """config/inn_aliases.csv: a printed INN spelling -> the DDInter ingredient name."""
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            InnAlias(
                alias=row["alias"].strip(),
                ingredient=row["ingredient"].strip().lower(),
                language=row["language"].strip(),
                source=row["source"].strip(),
                url=row["url"].strip(),
            )
            for row in csv.DictReader(handle)
            if row["alias"].strip()
        ]


# ---------------------------------------------------------------------------- OCR


@dataclass(frozen=True)
class OcrLine:
    text: str
    score: float  # recognition confidence, 0-1
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass(frozen=True)
class OcrResult:
    lines: list[OcrLine]
    ocr_ms: float
    engine: str

    def text_chars(self, min_score: float = 0.5) -> int:
        """Letters and digits read with at least `min_score` confidence."""
        return sum(
            len(fold(line.text).replace(" ", "")) for line in self.lines if line.score >= min_score
        )


class OcrEngine(Protocol):
    name: str

    def read(self, image: np.ndarray) -> OcrResult: ...


class RapidOcrEngine:
    """RapidOCR 3 (PP-OCRv4 mobile detection + recognition, ONNX Runtime, CPU).

    `threads` caps ONNX Runtime's intra-op threads: with one thread per core the laptop of
    the project throttles after a few images (D96), 4 threads keep ~0.5 s per box photo.
    """

    name = "rapidocr"

    def __init__(self, threads: int = 4) -> None:
        try:
            from rapidocr import RapidOCR
        except ImportError as exc:  # optional at import time: the scan still works without
            raise OcrUnavailable("rapidocr is not installed (backend/requirements.txt)") from exc
        params: dict[str, Any] = {"Global.log_level": "warning"}
        if threads:
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = threads
        self.engine = RapidOCR(params=params)

    def read(self, image: np.ndarray) -> OcrResult:
        start = time.perf_counter()
        output = self.engine(image, use_cls=False)
        elapsed = (time.perf_counter() - start) * 1000
        lines = []
        boxes = output.boxes if output.boxes is not None else []
        for text, score, points in zip(output.txts or (), output.scores or (), boxes, strict=False):
            xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
            lines.append(OcrLine(str(text), float(score), (min(xs), min(ys), max(xs), max(ys))))
        return OcrResult(lines, elapsed, self.name)


class OcrHolder:
    """Loads the OCR engine on first use (like ScannerHolder for the pill models)."""

    def __init__(self, threads: int = 4, *, enabled: bool = True) -> None:
        self.threads, self.enabled = threads, enabled
        self._engine: OcrEngine | None = None
        self._error: str | None = None
        self._lock = threading.Lock()

    def get(self) -> OcrEngine | None:
        """The engine, or None when OCR is disabled or unavailable (see `error`)."""
        if not self.enabled:
            return None
        with self._lock:
            if self._engine is None and self._error is None:
                try:
                    self._engine = RapidOcrEngine(self.threads)
                except OcrUnavailable as exc:
                    self._error = str(exc)
            return self._engine

    @property
    def error(self) -> str | None:
        return self._error if self.enabled else "OCR disabled (TUJYA_BOX_OCR_ENABLED=false)"


# ------------------------------------------------------------------------ matching


@dataclass(frozen=True)
class MatchedIngredient:
    name: str  # DDInter / RxNorm ingredient name
    method: Literal["brand", "printed_name", "rxnav"]
    matched_text: str  # what was read on the box
    score: float  # 0-1: OCR confidence x similarity of the words
    source: str
    reference: str
    url: str
    in_interaction_table: bool


@dataclass
class BoxMatch:
    brand: Brand | None = None
    brand_text: str = ""
    brand_score: float = 0.0
    ingredients: list[MatchedIngredient] = field(default_factory=list)
    rxnav_terms: list[str] = field(default_factory=list)  # words RxNav was asked about

    @property
    def recognised(self) -> bool:
        return bool(self.ingredients)

    @property
    def confidence(self) -> float:
        return max((i.score for i in self.ingredients), default=0.0)


def _grams(line: OcrLine, max_words: int) -> Iterable[tuple[str, str]]:
    """(folded n-gram, original line) for every run of 1..max_words consecutive words."""
    found = words(line.text)
    for size in range(1, max_words + 1):
        for start in range(len(found) - size + 1):
            yield " ".join(found[start : start + size]), line.text


class Choices:
    """Folded names grouped by (number of words, first two letters).

    Fuzzy matching only compares names with as many words and the same beginning: spelling
    variants keep it ("paracetamolo", "ibuprofène"), while "vitamina" is not "vitamin a"
    and "aciclovir" is acyclovir, not famciclovir.
    """

    def __init__(self, keys: Iterable[str]) -> None:
        self.keys = set(keys)
        self.groups: dict[tuple[int, str], list[str]] = {}
        for key in sorted(self.keys):
            self.groups.setdefault(self._group(key), []).append(key)

    @staticmethod
    def _group(key: str) -> tuple[int, str]:
        return len(key.split()), key[:2]

    def similar(self, candidate: str, minimum: float) -> tuple[str, float] | None:
        if len(candidate.replace(" ", "")) < MIN_FUZZY_LENGTH:
            return (candidate, 100.0) if candidate in self.keys else None
        group = self.groups.get(self._group(candidate), [])
        best = process.extractOne(candidate, group, scorer=fuzz.ratio, score_cutoff=minimum)
        return (best[0], float(best[1])) if best else None


class TextMatcher:
    """Matches OCR lines with the brand table, the INN aliases and the ingredient names."""

    def __init__(
        self,
        brands: Iterable[Brand],
        aliases: Iterable[InnAlias],
        ingredient_names: Iterable[str],
        *,
        rxnav: RxNavClient | None = None,
    ) -> None:
        self.brands = {b.key: b for b in sorted(brands, key=lambda b: -len(b.key))}
        self.interaction_names = {n.strip().lower() for n in ingredient_names if n.strip()}
        # folded printed name -> (ingredient, source, url)
        self.names: dict[str, tuple[str, str, str]] = {}
        for name in sorted(self.interaction_names):
            self.names.setdefault(fold(name), (name, "DDInter 2.0", "https://ddinter2.scbdd.com/"))
        for alias in aliases:
            self.names[fold(alias.alias)] = (alias.ingredient, alias.source, alias.url)
        self._brand_keys = Choices(self.brands)
        self._name_keys = Choices(self.names)
        self.rxnav = rxnav

    def brands_in(self, lines: Sequence[OcrLine]) -> list[tuple[Brand, str, float]]:
        """Every brand of the table found in the lines, best score first (one per brand)."""
        best: dict[str, tuple[Brand, str, float]] = {}
        for line in lines:
            for gram, text in _grams(line, 2):
                hit = self._brand_keys.similar(gram, BRAND_MIN_SCORE)
                if hit is None:  # glued words: "BRUFENANALGESICO" starts with a brand
                    hit = next(
                        ((k, 90.0) for k in self.brands if len(k) >= 5 and gram.startswith(k)),
                        None,
                    )
                if hit is None:
                    continue
                score = line.score * hit[1] / 100
                if hit[0] not in best or score > best[hit[0]][2]:
                    best[hit[0]] = (self.brands[hit[0]], text, score)
        return sorted(best.values(), key=lambda b: -b[2])

    def _brand(self, lines: Sequence[OcrLine]) -> tuple[Brand, str, float] | None:
        found = self.brands_in(lines)
        return found[0] if found else None

    def _printed_names(self, line: OcrLine) -> list[tuple[str, float]]:
        """Ingredient names printed on one line, longest first; a word is used once
        ("vitamina C" is vitamin C, not "vitamin a" + "c")."""
        found = words(line.text)
        taken = [False] * len(found)
        hits = []
        for size in range(MAX_WORDS, 0, -1):
            for start in range(len(found) - size + 1):
                if any(taken[start : start + size]):
                    continue
                hit = self._name_keys.similar(" ".join(found[start : start + size]), NAME_MIN_SCORE)
                if hit is not None:
                    hits.append(hit)
                    taken[start : start + size] = [True] * size
        return hits

    def match(self, lines: Sequence[OcrLine], *, use_rxnav: bool = True) -> BoxMatch:
        result = BoxMatch()
        found: dict[str, MatchedIngredient] = {}

        def add(item: MatchedIngredient) -> None:
            if item.name not in found or item.score > found[item.name].score:
                found[item.name] = item

        brand = self._brand(lines)
        if brand is not None:
            result.brand, result.brand_text, result.brand_score = brand
            for name in brand[0].ingredients:
                add(
                    MatchedIngredient(
                        name,
                        "brand",
                        brand[1],
                        round(brand[2], 4),
                        brand[0].source,
                        brand[0].reference,
                        brand[0].url,
                        name in self.interaction_names,
                    )
                )
        for line in lines:
            for gram, score in self._printed_names(line):
                name, source, url = self.names[gram]
                add(
                    MatchedIngredient(
                        name,
                        "printed_name",
                        line.text,
                        round(line.score * score / 100, 4),
                        source,
                        f"read on the box: {line.text!r}",
                        url,
                        name in self.interaction_names,
                    )
                )
        if not found and use_rxnav and self.rxnav is not None and self.rxnav.enabled:
            for term in rxnav_terms(lines):
                result.rxnav_terms.append(term)
                for name, url, score in self.rxnav.ingredients(term):
                    add(
                        MatchedIngredient(
                            name,
                            "rxnav",
                            term,
                            round(score, 4),
                            "RxNorm (RxNav approximateTerm)",
                            f"RxNorm match of the printed word {term!r}",
                            url,
                            name in self.interaction_names,
                        )
                    )
                if found:
                    break
        result.ingredients = sorted(found.values(), key=lambda i: (-i.score, i.name))
        return result


def rxnav_terms(lines: Sequence[OcrLine], limit: int = RXNAV_TERMS) -> list[str]:
    """The words printed largest (a brand is usually the biggest text), common words out."""
    seen: list[str] = []
    for line in sorted(lines, key=lambda line: -line.height):
        if line.score < 0.6:
            continue
        for word in words(line.text):
            if len(word) >= 5 and word not in COMMON_WORDS and word not in seen:
                seen.append(word)
        if len(seen) >= limit:
            break
    return seen[:limit]


# -------------------------------------------------------------------------- RxNav


class RxNavClient:
    """Minimal polite client of the public RxNav REST API (no key; contact email required).

    Same rule as the data pipeline (src/data/tujya_data): no request without a contact
    address in the User-Agent. Without one the fallback is off and says so.
    """

    def __init__(
        self,
        contact_email: str | None,
        *,
        base_url: str = "https://rxnav.nlm.nih.gov/REST",
        timeout: float = 5.0,
    ) -> None:
        self.contact_email = contact_email or None
        if not base_url.startswith("https://"):
            raise ValueError("the RxNav URL must be https")
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self._cached = lru_cache(maxsize=512)(self._lookup)

    @property
    def enabled(self) -> bool:
        return self.contact_email is not None

    def _get(self, path: str, params: dict[str, str]) -> tuple[dict[str, Any], str]:
        url = f"{self.base_url}/{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(  # noqa: S310  https only (checked in __init__)
            url,
            headers={
                "User-Agent": f"tujya-student-project/0.1 (mailto:{self.contact_email})",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8")), url

    def _lookup(self, term: str) -> tuple[tuple[str, str, float], ...]:
        try:
            data, url = self._get("approximateTerm.json", {"term": term, "maxEntries": "5"})
            for candidate in data.get("approximateGroup", {}).get("candidate", []) or []:
                name = fold(candidate.get("name", ""))
                similarity = max((fuzz.ratio(term, w) for w in _WORD.findall(name)), default=0.0)
                if similarity < BRAND_MIN_SCORE:
                    continue
                related, _ = self._get(f"rxcui/{candidate['rxcui']}/related.json", {"tty": "IN"})
                names = [
                    prop["name"].lower()
                    for group in related.get("relatedGroup", {}).get("conceptGroup", []) or []
                    for prop in group.get("conceptProperties", []) or []
                ]
                if names:
                    return tuple((n, url, similarity / 100) for n in sorted(set(names)))
        except (OSError, ValueError, KeyError):
            return ()
        return ()

    def ingredients(self, term: str) -> tuple[tuple[str, str, float], ...]:
        """(ingredient name, query URL, similarity 0-1) for a printed word, () if none."""
        if not self.enabled:
            return ()
        return self._cached(fold(term))


# ----------------------------------------------------------------------- the service


class BoxReader:
    """What /api/scan uses for boxes: the OCR engine and a matcher built on first use.

    The ingredient names come from the medicine database (`names` callable), the brand and
    INN tables from config/ (a missing table is an empty one: the other routes still work).
    """

    def __init__(
        self,
        ocr: OcrHolder,
        brands_file: Path,
        aliases_file: Path,
        rxnav: RxNavClient | None = None,
    ) -> None:
        self.ocr, self.rxnav = ocr, rxnav
        self.brands_file, self.aliases_file = brands_file, aliases_file
        self._matcher: TextMatcher | None = None
        self._lock = threading.Lock()

    def matcher(self, names: Callable[[], Iterable[str]]) -> TextMatcher:
        with self._lock:
            if self._matcher is None:
                brands = load_brands(self.brands_file) if self.brands_file.is_file() else []
                aliases = load_inn_aliases(self.aliases_file) if self.aliases_file.is_file() else []
                self._matcher = TextMatcher(brands, aliases, names(), rxnav=self.rxnav)
            return self._matcher

    @property
    def rxnav_status(self) -> str:
        if self.rxnav is None or not self.rxnav.enabled:
            return "disabled (no TUJYA_RXNAV_CONTACT_EMAIL)"
        return "enabled"


# -------------------------------------------------------------------------- decision


@dataclass(frozen=True)
class KindThresholds:
    pill_confidence: float = 0.5  # best pill detection needed to call the photo a pill
    box_text_chars: int = 20  # letters read (confidence >= 0.5) that make it a box


DEFAULT_THRESHOLDS = KindThresholds()


def decide_kind(
    pill_confidences: Sequence[float],
    ocr: OcrResult | None,
    match: BoxMatch | None,
    thresholds: KindThresholds = DEFAULT_THRESHOLDS,
) -> Kind:
    """Pill, box, or neither (out of distribution: the scanner refuses).

    Text that names a medicine wins over the pill detector (a box often shows pills), then
    enough printed text means a box even if nothing is recognised ("boîte non reconnue"),
    then a confident pill detection; anything else is refused.
    """
    if match is not None and (match.recognised or match.brand is not None):
        return "box"
    if ocr is not None and ocr.text_chars() >= thresholds.box_text_chars:
        return "box"
    if pill_confidences and max(pill_confidences) >= thresholds.pill_confidence:
        return "pill"
    return "unknown"
