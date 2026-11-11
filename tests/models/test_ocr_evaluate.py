"""Scores of the OCR comparison (src/models/ocr/evaluate.py, step 2.7), on made-up outcomes."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.boxes import OcrLine
from src.models.ocr import evaluate as ev


def outcome(
    set_: str, truth: list[str], found: list[str], kind: str, brand_read: bool | None = None
) -> ev.Outcome:
    return ev.Outcome(
        set_,
        "x.jpg",
        "b" if brand_read is not None else "",
        truth,
        found,
        [],
        brand_read,
        kind,
        0.9,
        30,
        100.0,
        "",
    )


def test_score() -> None:
    outcomes = [
        outcome("box", ["acetaminophen"], ["acetaminophen"], "box", True),
        outcome("box", ["ibuprofen", "caffeine"], ["ibuprofen", "warfarin"], "box", False),
        outcome("box", ["diclofenac"], [], "pill", True),
        outcome("box", [], [], "box"),  # no known ingredient: kind only
        outcome("pill", [], [], "pill"),
        outcome("pill", [], [], "box"),
        outcome("off_topic", [], [], "unknown"),
        outcome("off_topic", [], [], "pill"),
        outcome("off_topic", [], ["acetaminophen"], "box"),  # an invented ingredient: not refused
    ]
    m = ev.score(outcomes)
    assert m["ingredient_recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["ingredient_full_recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["ingredient_precision"] == pytest.approx(2 / 3, abs=1e-4)  # 2 true of 3 found
    assert m["wrong_ingredient_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert m["brand_read_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["box_as_box_rate"] == 0.75 and m["box_as_pill_rate"] == 0.25
    assert m["pill_as_pill_rate"] == 0.5 and m["pill_as_box_rate"] == 0.5
    assert m["off_topic_refused_rate"] == pytest.approx(1 / 3, abs=1e-4)
    assert (m["n_box"], m["n_box_known_ingredient"], m["n_pill"], m["n_off_topic"]) == (4, 3, 2, 3)


def test_brand_read() -> None:
    lines = [OcrLine("EFFERALGAN500mg", 0.9, (0, 0, 1, 1)), OcrLine("compresse", 0.9, (0, 0, 1, 1))]
    assert ev.brand_read(lines, "efferalgan")
    assert ev.brand_read([OcrLine("BRUFENANALGESICO", 0.9, (0, 0, 1, 1))], "brufen")  # glued
    assert not ev.brand_read(lines, "tachipirina")
    assert ev.brand_read(lines, "Efferalgan")  # as written in a labels.csv


def test_samples_of_the_user_set_are_empty_until_the_photos_exist(tmp_path: Path) -> None:
    assert ev.user_samples(tmp_path) == []
    (tmp_path / "labels.csv").write_text(
        "image,brand,ingredients\np.jpg,Panadol,acetaminophen\n", encoding="utf-8"
    )
    (sample,) = ev.user_samples(tmp_path)
    assert (sample.set, sample.brand, sample.truth) == ("user", "Panadol", ["acetaminophen"])


def test_samples_of_the_web_set_carry_their_set_name(tmp_path: Path) -> None:
    assert ev.web_samples(tmp_path) == []
    (tmp_path / "labels.csv").write_text(
        "image,brand,ingredients,source,author,licence\n"
        "a.jpg,Augmentin,amoxicillin;clavulanic acid,https://commons.wikimedia.org/x,Bree,CC0\n",
        encoding="utf-8",
    )
    (sample,) = ev.web_samples(tmp_path)
    assert (sample.set, sample.brand) == ("web", "Augmentin")
    assert sample.truth == ["amoxicillin", "clavulanic acid"]
    assert sample.image == tmp_path / "a.jpg"


def test_engine_configs_name_a_known_engine() -> None:
    from src.models.ocr.engines import ENGINES

    for path in sorted((ev.CONFIG_DIR).glob("*.yaml")):
        import yaml

        assert yaml.safe_load(path.read_text(encoding="utf-8"))["engine"] in ENGINES, path
