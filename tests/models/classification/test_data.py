"""The crop dataset of stage prepare_classify_v2 as the classifiers read it
(src/models/classification/data.py)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.models.classification import data as D
from tests.models.classification.helpers import make_dataset


def test_index_classes_and_holdout(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    test = D.crops(root, "test")
    assert len(test) == 12 and len(D.crops(root, "test", holdout_only=True)) == 6
    assert all(c.path.is_file() and c.relative.startswith("test/") for c in test)
    classes = D.read_classes(root)
    assert D.v1_classes(classes) == [0, 1, 2]
    assert classes[0]["v1_class_id"] == 10 and classes[5]["v1_class_id"] is None
    assert D.class_names(classes)[5] == "005"


def test_stratified_fraction_keeps_every_class(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    train = D.crops(root, "train")
    half = D.stratified_fraction(train, 0.5, seed=0)
    assert len(half) == 12
    assert sorted({c.label for c in half}) == list(range(6))
    tiny = D.stratified_fraction(train, 0.01, seed=0)
    assert len(tiny) == 6  # at least one crop per class
    assert D.stratified_fraction(train, 1.0, seed=0) == train


def test_subset_copies_index_classes_and_holdout(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    subset = D.make_subset(root, tmp_path / "subset", {"train": 5, "val": 3, "test": 6})
    assert [len(D.crops(subset, s)) for s in D.SPLITS] == [5, 3, 6]
    assert len(D.read_classes(subset)) == 6  # the class list stays complete
    listed = (subset / "test_holdout.txt").read_text(encoding="utf-8").split()
    assert sorted(listed) == sorted(c.relative for c in D.crops(subset, "test", holdout_only=True))


def test_crop_tensor_is_the_api_preprocessing(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    crop = D.crops(root, "val")[0]
    tensor = D.load_tensor(crop.path, 24)
    assert tensor.shape == (1, 3, 24, 24) and tensor.dtype == np.float32
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
    # Same formula as backend/app/inference.py: resize, BGR -> RGB, / 255.
    import cv2

    from app.inference import to_tensor

    image = cv2.resize(cv2.imread(str(crop.path)), (24, 24), interpolation=cv2.INTER_AREA)
    assert np.allclose(tensor, to_tensor(image))


def test_dataset_info_reads_the_manifest(tmp_path: Path) -> None:
    root = make_dataset(tmp_path / "data")
    (root / D.MANIFEST).write_text(json.dumps({"name": "c3pi-v2-classify", "md5": "x"}), "utf-8")
    assert D.dataset_info(root)["md5"] == "x"
    assert D.dataset_info(tmp_path)["name"] == "c3pi-v2-classify"
