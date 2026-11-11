"""Medicine boxes (step 2.7): text matching, kind of a photo, RxNav fallback, /api/scan.

The OCR engine is faked (lines given by the test), except in the last test, which reads a
synthetic picture of printed text with the real RapidOCR when it is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.boxes import (
    BoxMatch,
    KindThresholds,
    OcrLine,
    OcrResult,
    RxNavClient,
    TextMatcher,
    decide_kind,
    fold,
    load_brands,
    load_inn_aliases,
    rxnav_terms,
    words,
)
from app.config import PROJECT_DIR
from app.routers.scan import BOX_FOUND, BOX_NOT_RECOGNISED, NOT_A_PILL
from tests.fake_models import write_models

BRANDS = PROJECT_DIR / "config" / "brand_aliases.csv"
ALIASES = PROJECT_DIR / "config" / "inn_aliases.csv"
NAMES = {
    "acetaminophen",
    "acyclovir",
    "famciclovir",
    "ascorbic acid",
    "vitamin a",
    "ibuprofen",
    "warfarin",
    "fluconazole",
    "amoxicillin",
    "clavulanic acid",
}


def line(text: str, score: float = 0.99, height: float = 20.0) -> OcrLine:
    return OcrLine(text, score, (0.0, 0.0, 100.0, height))


@pytest.fixture(scope="module")
def matcher() -> TextMatcher:
    return TextMatcher(load_brands(BRANDS), load_inn_aliases(ALIASES), NAMES)


# ------------------------------------------------------------------- text and tables


def test_fold_and_words_split_glued_digits() -> None:
    assert fold("Paracétamol 500 mg") == "paracetamol 500 mg"
    assert words("EFFERALGAN500mg") == ["efferalgan", "mg"]


def test_every_brand_row_has_ingredients_and_a_source() -> None:
    brands = load_brands(BRANDS)
    assert len(brands) >= 30
    for brand in brands:
        assert brand.ingredients and brand.source and brand.reference, brand
        assert brand.url.startswith("https://"), brand
    panadol = next(b for b in brands if b.brand == "Panadol")
    assert panadol.ingredients == ("acetaminophen",)
    assert "medicines.org.uk" in panadol.url  # the SmPC page that names paracetamol


def test_brand_on_the_box_gives_its_ingredients_with_the_source(matcher: TextMatcher) -> None:
    match = matcher.match([line("PANADOL"), line("Advance 500 mg")])
    assert match.brand is not None and match.brand.brand == "Panadol"
    (item,) = match.ingredients
    assert (item.name, item.method) == ("acetaminophen", "brand")
    assert "emc" in item.source and item.url.startswith("https://www.medicines.org.uk/")
    assert match.recognised and 0.9 < match.confidence <= 1


def test_glued_and_misspelt_brands(matcher: TextMatcher) -> None:
    for text, expected in (("EFFERALGAN500mg", "Efferalgan"), ("Voltaren Emulgel", "Voltarene")):
        brand = matcher.match([line(text)]).brand
        assert brand is not None and brand.brand == expected, text
    two = matcher.match([line("AUGMENTIN 1 g/125 mg")])
    assert sorted(i.name for i in two.ingredients) == ["amoxicillin", "clavulanic acid"]


def test_printed_inn_in_italian_and_french(matcher: TextMatcher) -> None:
    for text in ("Paracetamolo", "paracétamol 1000 mg", "Ibuprofene 200 mg"):
        found = matcher.match([line(text)]).ingredients
        assert found and found[0].method == "printed_name", text
    assert matcher.match([line("Paracetamolo")]).ingredients[0].name == "acetaminophen"


def test_longest_name_first_and_same_beginning(matcher: TextMatcher) -> None:
    vitamin = matcher.match([line("compresse effervescenti con vitamina C")])
    assert [i.name for i in vitamin.ingredients] == ["ascorbic acid"]  # not "vitamin a"
    antiviral = matcher.match([line("crema Aciclovir 5%")])
    assert [i.name for i in antiviral.ingredients] == ["acyclovir"]  # not famciclovir


def test_unknown_text_is_not_recognised_and_nothing_is_invented(matcher: TextMatcher) -> None:
    match = matcher.match([line("Lorem ipsum dolor sit amet"), line("12 bustine")])
    assert not match.recognised and match.brand is None and match.ingredients == []
    assert matcher.match([line("mg")]).ingredients == []  # short words must match exactly


def test_low_ocr_confidence_lowers_the_match_confidence(matcher: TextMatcher) -> None:
    sure = matcher.match([line("NUROFEN", 0.99)]).confidence
    unsure = matcher.match([line("NUROFEN", 0.6)]).confidence
    assert unsure < sure and unsure == pytest.approx(0.6, abs=0.01)


# ------------------------------------------------------------------------- the kind


def result(*texts: str) -> OcrResult:
    return OcrResult([line(t) for t in texts], 1.0, "fake")


def test_decide_kind() -> None:
    recognised = BoxMatch(ingredients=[object()])  # type: ignore[list-item]
    nothing = BoxMatch()
    many_letters = result("Lorem ipsum dolor sit amet consectetur")
    assert decide_kind([0.95], result("PANADOL"), recognised) == "box"  # text wins over pills
    assert decide_kind([], many_letters, nothing) == "box"  # a box, not recognised
    assert decide_kind([0.95], result("M 30"), nothing) == "pill"  # an imprint is little text
    assert decide_kind([0.3], result(), nothing) == "unknown"  # out of distribution
    assert decide_kind([], None, None) == "unknown"
    assert decide_kind([0.45], None, None, KindThresholds(pill_confidence=0.4)) == "pill"


def test_rxnav_terms_are_the_largest_uncommon_words() -> None:
    lines = [line("compresse rivestite", height=10), line("TACHIPIRINA", height=40)]
    assert rxnav_terms(lines) == ["tachipirina"]


# ---------------------------------------------------------------------------- RxNav


def test_rxnav_is_off_without_a_contact_email() -> None:
    client = RxNavClient(None)
    assert not client.enabled and client.ingredients("panadol") == ()


def test_rxnav_requires_https() -> None:
    with pytest.raises(ValueError):
        RxNavClient("me@example.org", base_url="http://rxnav.example.org/REST")


def test_rxnav_brand_to_ingredient(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RxNavClient("me@example.org")
    answers = {
        "approximateTerm.json": {
            "approximateGroup": {"candidate": [{"rxcui": "202432", "name": "Panadol"}]}
        },
        "rxcui/202432/related.json": {
            "relatedGroup": {"conceptGroup": [{"conceptProperties": [{"name": "Acetaminophen"}]}]}
        },
    }
    calls: list[str] = []

    def fake_get(path: str, params: dict[str, str]) -> tuple[dict[str, Any], str]:
        calls.append(path)
        return answers[path], f"https://rxnav.nlm.nih.gov/REST/{path}"

    monkeypatch.setattr(client, "_get", fake_get)
    (found,) = client.ingredients("Panadol")
    assert found[0] == "acetaminophen" and found[2] == pytest.approx(1.0)
    assert client.ingredients("PANADOL") == (found,) and len(calls) == 2  # cached


def test_rxnav_is_only_asked_when_nothing_else_matched(matcher: TextMatcher) -> None:
    class Spy(RxNavClient):
        def __init__(self, contact_email: str) -> None:
            super().__init__(contact_email)
            self.asked: list[str] = []

        def ingredients(self, term: str) -> tuple[tuple[str, str, float], ...]:
            self.asked.append(term)
            return (("acetaminophen", "https://rxnav.nlm.nih.gov/REST/x", 0.95),)

    spy = Spy("me@example.org")
    with_rxnav = TextMatcher(load_brands(BRANDS), load_inn_aliases(ALIASES), NAMES, rxnav=spy)
    assert with_rxnav.match([line("PANADOL")]).ingredients[0].method == "brand"
    assert spy.asked == []
    match = with_rxnav.match([line("TACHIPIRINA", height=40)])
    assert spy.asked == ["tachipirina"] and match.ingredients[0].method == "rxnav"


# ------------------------------------------------------------------------ /api/scan


class FakeEngine:
    name = "fake"

    def __init__(self, *texts: str) -> None:
        self.texts = texts

    def read(self, image: np.ndarray) -> OcrResult:
        return result(*self.texts)


class FakeHolder:
    def __init__(self, engine: FakeEngine) -> None:
        self.engine = engine
        self.error = None

    def get(self) -> FakeEngine:
        return self.engine


def scan_with(app: FastAPI, client: TestClient, *texts: str) -> dict[str, Any]:
    app.state.boxes.ocr = FakeHolder(FakeEngine(*texts))
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    response = client.post(
        "/api/scan", files={"image": ("box.png", encoded.tobytes(), "image/png")}
    )
    assert response.status_code == 200, response.text
    body: dict[str, Any] = response.json()
    return body


@pytest.fixture
def pill_models(app: FastAPI) -> Path:
    directory: Path = app.state.settings.models_dir
    write_models(directory, [(32.0, 32.0, 12.8, 12.8, 0.95)], ["005", "007", "009"])
    return directory


@pytest.fixture
def faint_models(app: FastAPI) -> Path:
    directory: Path = app.state.settings.models_dir
    write_models(directory, [(32.0, 32.0, 12.8, 12.8, 0.30)], ["005", "007", "009"])
    return directory


def test_scan_of_a_panadol_box(
    app: FastAPI, user: TestClient, pill_models: Path, pharma: str
) -> None:
    body = scan_with(app, user, "PANADOL", "Advance", "Paracetamol 500 mg")
    assert body["kind"] == "box" and body["pills"] == []  # no pill class given to a box
    box = body["box"]
    assert box["brand"] == "Panadol" and box["brand_country"] == "GB" and box["recognised"]
    names = {i["name"]: i for i in box["ingredients"]}
    assert set(names) == {"acetaminophen"}
    assert names["acetaminophen"]["url"].startswith("https://")
    assert names["acetaminophen"]["source"] and 0 < box["confidence"] <= 1
    assert box["text"] == ["PANADOL", "Advance", "Paracetamol 500 mg"]
    assert box["rxnav"].startswith("disabled")  # no contact email in the tests
    assert body["message"] == BOX_FOUND
    assert "consultez un pharmacien ou un médecin" in body["disclaimer"]
    assert body["timings"]["ocr_ms"] is not None


def test_scan_of_an_unknown_box(
    app: FastAPI, user: TestClient, pill_models: Path, pharma: str
) -> None:
    body = scan_with(app, user, "Lorem ipsum dolor sit amet", "consectetur adipiscing elit")
    assert body["kind"] == "box" and body["pills"] == []
    assert body["box"]["recognised"] is False and body["box"]["ingredients"] == []
    assert body["message"] == BOX_NOT_RECOGNISED


def test_scan_of_a_pill_is_unchanged(
    app: FastAPI, user: TestClient, pill_models: Path, pharma: str
) -> None:
    body = scan_with(app, user, "M 30")  # an imprint: little text
    assert body["kind"] == "pill" and body["box"] is None and len(body["pills"]) == 1
    assert body["message"] is None


def test_off_topic_photo_is_refused(
    app: FastAPI, user: TestClient, faint_models: Path, pharma: str
) -> None:
    body = scan_with(app, user)  # no text, one faint "pill" (0.30 < 0.5)
    assert body["kind"] == "unknown" and body["pills"] == [] and body["box"] is None
    assert body["message"] == NOT_A_PILL


# --------------------------------------------------------------------- real RapidOCR


def test_real_rapidocr_reads_a_printed_box_name(matcher: TextMatcher) -> None:
    pytest.importorskip("rapidocr")
    from app.boxes import RapidOcrEngine

    image = np.full((240, 640, 3), 255, dtype=np.uint8)
    cv2.putText(image, "PANADOL", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 0, 0), 8)
    cv2.putText(image, "Paracetamol 500 mg", (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    ocr = RapidOcrEngine(threads=2).read(image)
    match = matcher.match(ocr.lines, use_rxnav=False)
    assert [i.name for i in match.ingredients] == ["acetaminophen"]
    assert decide_kind([], ocr, match) == "box"
