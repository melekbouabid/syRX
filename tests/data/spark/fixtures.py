"""Fixtures of the Spark job tests: a local SparkSession and tiny fake C3PI files.

Loaded by tests/data/spark/conftest.py, or with `-p tests.data.spark.fixtures` where the
root conftest cannot load (the Spark image and the CI job `spark` have no web app deps):

    docker compose --profile spark exec spark-master \
        python3 -m pytest tests/data/spark --noconftest -p tests.data.spark.fixtures

The tests are skipped when pyspark or Java is missing (the Windows host has no Java).
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pytest
from PIL import Image

PARAMS: dict[str, Any] = {
    "seed": 0,
    "index": {"min_images": 3, "exclude_cameras": [], "seg_per_class": 2, "challenge": "all"},
    "quality": {
        "min_side": 40,
        "min_brightness": 10,
        "max_brightness": 245,
        "hash_size": 16,
        "near_duplicate_bits": 3,
    },
    "split": {"val": 0.2, "test": 0.2},
    "augment": {"target_per_class": 6, "size": 64},
    "rag": {"min_chars": 20, "near_duplicate": 0.3},
}


def _java_available() -> bool:
    home = os.environ.get("JAVA_HOME")
    return bool(shutil.which("java") or (home and (Path(home) / "bin").is_dir()))


@pytest.fixture(scope="session")
def spark() -> Iterator[Any]:
    pytest.importorskip("pyspark")
    if not _java_available():
        pytest.skip("Java is not installed: run these tests in the Spark container or in CI")
    from pyspark.sql import SparkSession

    session = (
        SparkSession.builder.master("local[2]")
        .appName("tujya-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def params() -> dict[str, Any]:
    return {k: dict(v) if isinstance(v, dict) else v for k, v in PARAMS.items()}


def image_xml(**fields: Any) -> str:
    """One <Image> element in the ALLXML layout (only the fields the jobs read)."""
    colours = "".join(f"<Color>{escape(c)}</Color>" for c in fields.get("colors", ["WHITE"]))
    tag = "Class" if fields["image_type"] in ("C3PI_Test", "C3PI_Reference") else "Layout"
    ndc11 = fields["ndc11"]
    return (
        "<Image>"
        f"<NDC9>{ndc11[:9]}</NDC9><NDC11>{ndc11}</NDC11>"
        f"<Part>{fields.get('part', 1)}</Part><Parts>1</Parts>"
        f"<LabeledBy>LAB</LabeledBy><GenericName>{escape(fields.get('generic', 'DRUG'))}"
        f"</GenericName><ProprietaryName>{escape(fields.get('name', 'DRUG 10MG'))}"
        "</ProprietaryName>"
        f"<File><Name>{fields['file']}</Name><Type>{fields['file'].rsplit('.', 1)[1]}</Type>"
        f"<Size>{fields.get('size', 1000)}</Size><Sha1>{fields.get('sha1', '')}</Sha1></File>"
        f"<{tag}>{fields['image_type']}</{tag}>"
        f"<Camera>{escape(fields.get('camera', ''))}</Camera>"
        f"<Background>{escape(fields.get('background', ''))}</Background>"
        f"<Illumination>{escape(fields.get('light', ''))}</Illumination>"
        f"<Imprint>A;1</Imprint>{colours}<Shape>ROUND</Shape><Size>8</Size>"
        f"<AcquisitionDate>{fields.get('date', 'Monday')}</AcquisitionDate>"
        "</Image>"
    )


def write_metadata(folder: Path, discs: dict[str, list[dict[str, Any]]]) -> Path:
    """ALLXML/<disc>.xml (ISO-8859-1, like NLM's) and directory_of_images.txt."""
    xml_dir = folder / "ALLXML"
    xml_dir.mkdir(parents=True)
    lines = []
    for disc, images in discs.items():
        body = "".join(image_xml(**image) for image in images)
        text = (
            '<?xml version="1.0" encoding="ISO-8859-1"?>\n<MedicosConsultants><ImageExport>'
            f"{body}</ImageExport></MedicosConsultants>\n"
        )
        (xml_dir / f"{disc}.xml").write_bytes(text.encode("iso-8859-1"))
        for image in images:
            path = f"{disc}/images/{image['file']}"
            lines.append(
                f"{image['ndc11']}|{image.get('part', 1)}|{path}|{image['image_type']}|"
                f"{image.get('name', 'DRUG 10MG')}"
            )
    (folder / "directory_of_images.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return folder


def photo(path: Path, seed: int, size: tuple[int, int] = (120, 90), value: int = 0) -> Path:
    """A random photo-like JPEG (each seed gives a clearly different image)."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(40, 215, (size[1] // 6, size[0] // 6, 3), dtype=np.uint8)
    image = Image.fromarray(pixels).resize(size, Image.Resampling.BILINEAR)
    if value:
        image = Image.new("RGB", size, (value, value, value))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "JPEG", quality=92)
    return path


def segmented(path: Path, size: tuple[int, int] = (120, 100)) -> Path:
    """An RGBA PNG like REFERENCE_SEG: a transparent background, a round pill, and an
    opaque banner at the bottom."""
    width, height = size
    alpha = np.zeros((height, width), dtype=np.uint8)
    yy, xx = np.mgrid[:height, :width]
    alpha[(yy - 35) ** 2 + (xx - 60) ** 2 < 20**2] = 255
    alpha[height - 15 :, :] = 255  # banner
    rgb = np.full((height, width, 3), 200, dtype=np.uint8)
    rgb[height - 15 :, :] = 30
    image = Image.fromarray(np.dstack([rgb, alpha]))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, "PNG")
    return path
