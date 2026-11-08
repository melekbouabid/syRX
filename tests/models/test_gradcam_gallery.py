"""Grad-CAM pictures of the demo gallery (src/models/gradcam_gallery.py, D104).

The geometry and the version check run everywhere; composing a picture needs PyTorch
(src/utils/gradcam.py) and is skipped without it.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from app.inference import square_crop
from src.models.gradcam_gallery import (
    Square,
    compose,
    crop_square,
    paste_heatmap,
    run,
    served_classifier,
    shrink,
)


def test_the_square_is_the_crop_the_classifier_sees() -> None:
    image = np.random.default_rng(0).integers(0, 255, (400, 500, 3), dtype=np.uint8)
    box = (200.0, 150.0, 260.0, 190.0)  # 60 x 40, far from the edges
    square = crop_square(box)
    assert square == Square(left=188, top=128, side=84)  # 60 x 1.4, centred on the box
    expected = square_crop(image, box, square.side)
    region = image[square.top : square.top + square.side, square.left : square.left + square.side]
    assert np.array_equal(region, expected)


def test_heat_maps_are_clipped_at_the_edges_and_the_larger_one_wins() -> None:
    target = np.zeros((10, 10), np.float32)
    paste_heatmap(target, np.full((4, 4), 0.5, np.float32), Square(left=-2, top=-2, side=6))
    assert target[:4, :4].min() == pytest.approx(0.5) and target[4:, :].max() == 0.0
    paste_heatmap(target, np.full((4, 4), 0.2, np.float32), Square(left=2, top=2, side=4))
    assert target[2, 2] == pytest.approx(0.5)  # kept: larger
    assert target[5, 5] == pytest.approx(0.2)
    paste_heatmap(target, np.ones((4, 4), np.float32), Square(left=20, top=20, side=4))
    assert target.max() == pytest.approx(0.5)  # fully outside: nothing changes


def test_shrink_keeps_the_proportions() -> None:
    picture = np.zeros((3000, 4000, 3), np.uint8)
    assert shrink(picture, 1280).shape == (960, 1280, 3)
    small = np.zeros((300, 400, 3), np.uint8)
    assert shrink(small, 1280) is small


def _registry(tmp_path: Path, served_run: str, trained_run: str) -> tuple[Path, Path]:
    models = tmp_path / "models"
    weights = models / "train" / "classifier"
    weights.mkdir(parents=True)
    manifest = {"classify": {"run_id": served_run, "registered_model": "c", "version": 1}}
    (models / "models.json").write_text(json.dumps(manifest), "utf-8")
    (models / "train" / "classifier.json").write_text(json.dumps({"run_id": trained_run}), "utf-8")
    return models, weights


def test_the_weights_must_be_the_served_version(tmp_path: Path) -> None:
    models, weights = _registry(tmp_path, "abc", "abc")
    assert served_classifier(models, weights)["version"] == 1
    other, other_weights = _registry(tmp_path / "other", "abc", "def")
    with pytest.raises(SystemExit, match="serves run abc"):
        served_classifier(other, other_weights)


def test_no_gallery_photo_is_an_error(tmp_path: Path) -> None:
    models, weights = _registry(tmp_path, "abc", "abc")
    args = Namespace(
        models=models,
        weights=weights,
        gallery=tmp_path / "missing.csv",
        split=tmp_path / "missing.csv",
        holdout=tmp_path / "missing.csv",
        images=tmp_path,
        out=tmp_path / "out",
    )
    with pytest.raises(SystemExit, match="no gallery photo"):
        run(args)
    assert not (tmp_path / "out").exists()


def test_the_picture_colours_only_the_squares() -> None:
    pytest.importorskip("torch")
    image = np.full((100, 200, 3), 200, np.uint8)
    heat = np.zeros((8, 8), np.float32)
    heat[4, 4] = 1.0
    picture = compose(image, [(heat, Square(left=20, top=20, side=40))])
    assert picture.shape == image.shape and picture.dtype == np.uint8
    assert (picture[90, 190] == 120).all()  # outside: the photo darkened (0.6)
    assert not (picture[40, 40] == 120).all()  # inside: blended with the heat map
