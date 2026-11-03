"""Job `quality`: image checks, perceptual hash, duplicates and rejection reasons."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("pyspark")

from src.data.spark import quality_job
from tests.data.spark.fixtures import photo, segmented


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def resave(source: Path, target: Path, quality: int = 95) -> Path:
    with Image.open(source) as image:
        image.save(target, "JPEG", quality=quality)
    return target


@pytest.fixture
def fake(tmp_path):
    """Images in images/D1/<file> and the selection rows that list them."""
    images = tmp_path / "images"
    d = images / "D1"
    files = {
        "a_ok1.JPG": ("X", "C3PI_Test", photo(d / "a_ok1.JPG", 1)),
        "a_ok2.JPG": ("X", "C3PI_Test", photo(d / "a_ok2.JPG", 2)),
        "a_ok3.JPG": ("X", "C3PI_Test", photo(d / "a_ok3.JPG", 3)),
        "b_near1.JPG": ("X", "C3PI_Test", resave(d / "a_ok1.JPG", d / "b_near1.JPG")),
        "c_corrupt.JPG": ("X", "C3PI_Test", None),
        "d_small.JPG": ("X", "C3PI_Test", photo(d / "d_small.JPG", 4, size=(30, 20))),
        "e_dark.JPG": ("X", "C3PI_Test", photo(d / "e_dark.JPG", 5, value=3)),
        "f_missing.JPG": ("X", "C3PI_Test", None),
        "z_copy2.JPG": ("X", "C3PI_Test", shutil.copy(d / "a_ok2.JPG", d / "z_copy2.JPG")),
        "g_other.JPG": ("Y", "C3PI_Test", resave(d / "a_ok3.JPG", d / "g_other.JPG")),
        "h_seg.PNG": ("Y", "MC_C3PI_REFERENCE_SEG_V1.6", segmented(d / "h_seg.PNG")),
        "i_white.JPG": ("Y", "MC_CHALLENGE_V1.0", photo(d / "i_white.JPG", 6, value=250)),
    }
    good = (d / "a_ok1.JPG").read_bytes()
    (d / "c_corrupt.JPG").write_bytes(good[: len(good) // 2])
    rows = []
    for name, (key, image_type, _) in files.items():
        file = d / name
        rows.append(
            {
                "path": f"D1/images/{name}",
                "class_key": key,
                "image_type": image_type,
                "role": "seg" if "SEG" in image_type else "consumer",
                "sha1": sha1(file) if file.exists() else "",
            }
        )
    return images, rows


def test_inspect_image_measures_and_flags(tmp_path):
    ok = quality_job.inspect_image(photo(tmp_path / "ok.JPG", 1), 16)
    assert ok["exists"] and not ok["corrupt"]
    assert (ok["width"], ok["height"], ok["mode"]) == (120, 90, "RGB")
    assert 40 < ok["brightness"] < 215 and ok["contrast"] > 0 and ok["sharpness"] > 0
    assert len(ok["dhash"]) == 64  # 256 bits in hex
    broken = tmp_path / "broken.JPG"
    broken.write_bytes((tmp_path / "ok.JPG").read_bytes()[:300])
    assert quality_job.inspect_image(broken)["corrupt"]
    assert not quality_job.inspect_image(tmp_path / "none.JPG")["exists"]
    seg = quality_job.inspect_image(segmented(tmp_path / "seg.PNG"))
    assert seg["mode"] == "RGBA" and not seg["corrupt"]


def test_dhash_is_stable_and_hamming_counts_bits(tmp_path):
    first = photo(tmp_path / "1.JPG", 1)
    again = resave(first, tmp_path / "1b.JPG")
    other = photo(tmp_path / "2.JPG", 2)
    h1, h1b, h2 = (quality_job.inspect_image(p)["dhash"] for p in (first, again, other))
    assert quality_job.hamming(h1, h1b) <= 3
    assert quality_job.hamming(h1, h2) > 40
    assert quality_job.hamming("f0", "0f") == 8


def test_banding_needs_fewer_bits_than_blocks(spark):
    table = spark.createDataFrame(
        [("p", "X", "0" * 64)], "path string, class_key string, dhash string"
    )
    with pytest.raises(ValueError):
        quality_job.near_duplicates(table, 4, 16)


def test_job_rejects_with_reasons_and_records_near_duplicates(spark, params, tmp_path, fake):
    images, rows = fake
    selection_dir = tmp_path / "selection"
    spark.createDataFrame(rows).write.parquet(str(selection_dir))
    out = tmp_path / "quality"
    summary = quality_job.run(
        spark, params, selection_dir=selection_dir, images=images, out=out, partitions=3
    )
    with (out / "rejected.csv").open(encoding="utf-8") as handle:
        reasons = {r["path"].split("/")[-1]: r["reason"] for r in csv.DictReader(handle)}
    assert reasons == {
        "c_corrupt.JPG": "corrupt",
        "d_small.JPG": "too_small",
        "e_dark.JPG": "too_dark",
        "f_missing.JPG": "missing",
        "z_copy2.JPG": "duplicate_file",
        "a_ok3.JPG": "duplicate_other_class",
        "g_other.JPG": "duplicate_other_class",
        "i_white.JPG": "too_bright",
    }
    assert summary["checked"] == 12 and summary["rejected"] == 8
    pairs = {
        (r.path_a.split("/")[-1], r.path_b.split("/")[-1])
        for r in spark.read.parquet(str(out / "near_duplicates")).collect()
    }
    # Same class, near-identical: kept, but recorded for the split.
    assert ("a_ok1.JPG", "b_near1.JPG") in pairs
    assert ("a_ok3.JPG", "g_other.JPG") in pairs
    assert summary["near_duplicate_pairs_other_class"] == 1
