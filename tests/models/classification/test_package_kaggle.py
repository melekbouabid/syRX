"""Private Kaggle dataset of the crops (src/models/classification/package_kaggle.py)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.models.classification import package_kaggle
from tests.models.classification.helpers import make_dataset


def test_package_for_kaggle(tmp_path: Path) -> None:
    data = make_dataset(tmp_path / "data")
    (data / "summary.json").write_text("{}", encoding="utf-8")
    result = package_kaggle.package(data, tmp_path / "out", owner="someone")
    folder = Path(result["folder"])
    manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["images"] == {"train": 24, "val": 12, "test": 12, "holdout": 6, "classes": 6}
    assert manifest["name"] == "c3pi-v2-classify" and manifest["label_space"] == "c3pi-v2"
    meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someone/tujya-c3pi-classify-v2"
    assert meta["licenses"] == [{"name": "ODbL-1.0"}]
    assert "ODbL" in (folder / "README.md").read_text(encoding="utf-8")
    with zipfile.ZipFile(result["zip"]) as archive:
        names = set(archive.namelist())
    assert "train/000/train_0_0.jpg" in names
    assert {"index.csv", "classes.csv", "test_holdout.txt", "MANIFEST.json"} <= names
    assert "dataset-metadata.json" not in names


def test_package_refuses_a_broken_dataset(tmp_path: Path) -> None:
    data = make_dataset(tmp_path / "data")
    next((data / "val" / "001").glob("*.jpg")).unlink()
    with pytest.raises(ValueError, match="missing"):
        package_kaggle.check_dataset(data)
    data = make_dataset(tmp_path / "data2")
    (data / "test_holdout.txt").write_text("train/000/train_0_0.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="holdout"):
        package_kaggle.check_dataset(data)
