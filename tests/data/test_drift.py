"""Reference statistics and demonstration of the data drift monitor (step 2.8):
src/data/drift_reference.py and src/data/drift_demo.py.

The last test runs the served models on real C3PI v2 photos: it needs the DVC data, the
published models and the reference of `dvc repro drift_reference`, and is skipped without them
(GitHub Actions has none of them)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from app import drift as D
from src.data import drift_demo, drift_reference


def photo(level: int = 128, size: tuple[int, int] = (480, 640)) -> np.ndarray:
    rng = np.random.default_rng(level)
    image = rng.normal(level, 40, (*size, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


def test_darken_and_blur_move_the_statistics_they_should() -> None:
    base = D.image_stats(photo())
    dark = D.image_stats(drift_demo.darken(photo()))
    blurred = D.image_stats(drift_demo.blur(photo()))
    assert dark["brightness"] == pytest.approx(base["brightness"] * 0.35, rel=0.02)
    assert blurred["sharpness"] < base["sharpness"] - 1
    assert blurred["brightness"] == pytest.approx(base["brightness"], rel=0.02)
    assert blurred["megapixels"] == base["megapixels"]


def test_photos_are_spread_over_the_split_and_mapped_to_the_raw_images(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    with index.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "path", "split"])
        for i in range(100):
            split = "train" if i % 4 else "test"
            writer.writerow([f"{i}.jpg", f"PillProjectDisc{i % 3}/images/P{i:03d}.JPG", split])
    chosen = drift_reference.photos("train", 10, index=index, images=tmp_path / "raw")
    assert len(chosen) == 10 and len(set(chosen)) == 10
    assert chosen[0] == tmp_path / "raw" / "PillProjectDisc0" / "P003.JPG"  # first sorted path
    assert all(p.parent.name.startswith("PillProjectDisc") for p in chosen)  # no images/ level
    assert len(drift_reference.photos("test", 0, index=index, images=tmp_path)) == 25


def test_reference_keeps_one_sample_per_statistic_and_the_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Found:
        def __init__(self, confidence: float) -> None:
            self.confidence, self.probabilities = confidence, [confidence / 2, 0.1]

    class Scanner:
        def scan(self, image: np.ndarray) -> object:
            found = [Found(0.9)] if image.mean() > 100 else []
            return type("Output", (), {"detections": found})()

    files = []
    for level in (60, 128, 200):
        path = tmp_path / f"{level}.png"
        import cv2

        cv2.imwrite(str(path), photo(level, (60, 80)))
        files.append(path)
    (tmp_path / "models.json").write_text(
        '{"detect": {"arch": "yolo11s", "version": 9}, "classify": {"arch": "e", "version": 8}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(drift_reference, "MODELS", tmp_path)
    params = {"reference_images": 3, "reference_split": "val", "window": 50, "psi_drift": 0.3}
    reference = drift_reference.build(files, Scanner(), params)  # type: ignore[arg-type]
    assert len(reference.values["brightness"]) == 3 and reference.values["boxes"] == [0, 1, 1]
    assert reference.values["classifier_confidence"] == [0.45, 0.45]  # photos with a pill only
    assert reference.thresholds == {"window": 50.0, "psi_drift": 0.3}
    assert reference.meta["models"]["classify"]["version"] == 8
    assert reference.meta["split"] == "val" and "val split" in reference.meta["dataset"]
    summary = drift_reference.summary(reference)
    assert summary["photos"] == 3 and summary["boxes"]["mean"] == pytest.approx(2 / 3)


REAL = (
    drift_reference.OUT / "reference.json",
    drift_reference.MODELS / "detector.onnx",
    drift_reference.MODELS / "classifier.onnx",
    drift_reference.INDEX,
)


@pytest.mark.skipif(
    not all(p.is_file() for p in REAL), reason="needs DVC data, served models and the reference"
)
def test_no_drift_on_test_photos_and_drift_on_dark_or_blurred_ones() -> None:
    from app.inference import PillScanner

    scanner = PillScanner.from_dir(drift_reference.MODELS)
    files = drift_reference.photos("test", 60)
    if not all(f.is_file() for f in files):
        pytest.skip("C3PI images not pulled (dvc pull data/raw/c3pi/images)")
    same = drift_demo.run_local("test-v2", files, lambda image: image, scanner)
    assert same["status"] in ("ok", "warning") and same["drifted"] == []
    dark = drift_demo.run_local("dark", files, drift_demo.darken, scanner)
    assert dark["status"] == "drift" and "brightness" in dark["drifted"]
    blurred = drift_demo.run_local("blurred", files, drift_demo.blur, scanner)
    assert blurred["status"] == "drift" and "sharpness" in blurred["drifted"]
