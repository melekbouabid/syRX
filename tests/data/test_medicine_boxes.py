"""Ground truth of the box photos (src/data/medicine_boxes.py, step 2.7), on a tiny fake set."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.boxes import TextMatcher
from src.data import medicine_boxes as mb

NAMES = {"acetaminophen", "ibuprofen", "ginger", "acetylsalicylic acid", "ascorbic acid"}


@pytest.fixture(scope="module")
def matcher() -> TextMatcher:
    return mb.ground_truth_matcher(NAMES)


def test_brand_is_the_first_word_unless_the_product_is_named_after_its_ingredient(
    matcher: TextMatcher,
) -> None:
    assert mb.brand_of("TACHIPIRINA*20CPR DIV 500MG", matcher) == "tachipirina"
    assert mb.brand_of("BRUFEN ANALGES*12CPR RIV 200MG", matcher) == "brufen"
    assert mb.brand_of("PARACETAMOLO*SAND 20CPR 500MG", matcher) == ""
    assert mb.brand_of("OK*10CPR", matcher) == ""


def test_active_part_skips_flavours_and_excipients() -> None:
    text = "BENAGOL gusto Ginger Principi attivi: ibuprofene 200 mg. Eccipienti: saccarosio"
    assert mb.active_part(text).strip() == "ibuprofene 200 mg"
    assert "paracetamolo" in mb.active_part("Ogni compressa contiene paracetamolo 500 mg")


def test_eval_rows(tmp_path: Path, matcher: TextMatcher) -> None:
    (tmp_path / "out").mkdir()
    records = [
        {
            "nome": "TACHIPIRINA*20CPR",
            "immagine": "t.jpg",
            "Codice EAN": "1",
            "dettagli": {
                "Principi attivi": "Principio attivo: Paracetamolo 500 mg. Eccipienti: amido"
            },
        },
        {
            "nome": "BENAGOL*16PAST",
            "immagine": "b.jpg",
            "dettagli": {
                "Principi attivi": "gusto Ginger. Principi attivi: alcool 1,2 mg; vitamina C"
            },
        },
        {"nome": "TACHIPIRINA*20CPR", "immagine": "t.jpg", "dettagli": {}},  # same photo again
        {"nome": "MISSING*1", "immagine": "absent.jpg", "dettagli": {}},  # no photo file
        {"nome": "PAGE*1", "immagine": "page.jpg", "dettagli": {}},  # an HTML page named .jpg
    ]
    for name in ("t.jpg", "b.jpg"):
        (tmp_path / "out" / name).write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b" a JPEG")
    (tmp_path / "out" / "page.jpg").write_bytes(b"<!DOCTYPE html>")
    (tmp_path / "medicines.json").write_text(json.dumps(records), encoding="utf-8")
    rows = mb.eval_rows(tmp_path, matcher)
    assert [(r["image"], r["brand"], r["ingredients"]) for r in rows] == [
        ("t.jpg", "tachipirina", "acetaminophen"),
        ("b.jpg", "benagol", "ascorbic acid"),  # the flavour "Ginger" is not an ingredient
    ]
    assert set(rows[0]) == set(mb.EVAL_COLUMNS)


def test_user_readme_explains_the_labels_file(tmp_path: Path) -> None:
    assert "labels.csv" in mb.USER_README and "dvc add" in mb.USER_README
    header = mb.USER_README.split("image,brand,ingredients")[1].splitlines()[1].strip()
    assert next(csv.reader([header]))[0].endswith(".jpg")
