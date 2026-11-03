"""Job `augment`: offline augmentation of the rare classes, on the train split only.

Input: data/processed/c3pi/bigdata/split/images (job `split`). The sources are the TRAIN
images of each class with role consumer (photos) or seg (segmented studio PNGs); val and
test images are never read, so nothing generated can leak into them.

A class with fewer sources than `target_per_class` gets new images up to the target. The
plan (which source, which operations, with which random values) is a Spark DataFrame built
from a seeded hash of (class, index), so a rerun gives the same files; the images are then
written in parallel by the executors (`mapInPandas`). Operations:

- rotation by a random angle, the corners filled with a generated background;
- background: a segmented pill (alpha mask, NLM banner cut off) is pasted on a generated
  background (plain colour, gradient and noise); a photo is rotated onto one;
- brightness (x0.6 to x1.4) and, one time in two, a Gaussian blur (radius 0.5 to 2 px);
- the long side is resized to `size` px, JPEG quality 90.

Online augmentation (RandAugment, MixUp, CutMix) stays in the training code (step 2.3).

Outputs (data/processed/c3pi/bigdata/augmented, DVC): images/<class_id>/<name>.jpg,
manifest/ (Parquet: path, source, class, operations), summary.json.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.data.spark.quality_job import local_file
from src.data.spark.session import BIGDATA, C3PI_RAW, ROOT, write_json

IMAGES = C3PI_RAW / "images"
SPLIT = BIGDATA / "split" / "images"
OUT = BIGDATA / "augmented"
SOURCE_ROLES = ("consumer", "seg")
PLAN_SCHEMA = T.StructType(
    [
        T.StructField("class_id", T.IntegerType()),
        T.StructField("class_key", T.StringType()),
        T.StructField("index", T.IntegerType()),
        T.StructField("source", T.StringType()),
        T.StructField("role", T.StringType()),
        T.StructField("seed", T.LongType()),
    ]
)
RESULT_SCHEMA = (
    "path string, source string, class_id int, class_key string, role string, "
    "operations string, ok boolean, error string"
)


def variant_seed(seed: int, class_key: str, index: int) -> int:
    digest = hashlib.sha1(f"{seed}:{class_key}:{index}".encode(), usedforsecurity=False)
    return int(digest.hexdigest()[:12], 16)


def plan(sources: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    """Images to generate for ONE class (applyInPandas): target - len(sources) variants.

    Sources are used in turn (sorted by path), so every train image of the class serves
    before any is used twice.
    """
    sources = sources.sort_values("path").reset_index(drop=True)
    missing = max(0, target - len(sources)) if len(sources) else 0
    rows = []
    for i in range(missing):
        source = sources.iloc[i % len(sources)]
        rows.append(
            {
                "class_id": int(source["class_id"]),
                "class_key": source["class_key"],
                "index": i,
                "source": source["path"],
                "role": source["role"],
                "seed": variant_seed(seed, source["class_key"], i),
            }
        )
    return pd.DataFrame(rows, columns=[f.name for f in PLAN_SCHEMA.fields])


def background(size: tuple[int, int], rng: random.Random) -> Image.Image:
    """A plain colour with a linear gradient and a little noise."""
    width, height = size
    base = np.array([rng.randint(30, 230) for _ in range(3)], dtype=np.float32)
    shift = np.array([rng.randint(-40, 40) for _ in range(3)], dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    noise = np.random.default_rng(rng.getrandbits(32)).normal(0, 6, (height, width, 3))
    pixels = np.clip(base + shift * ramp + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def cut_pill(image: Image.Image) -> Image.Image:
    """The pill of a segmented RGBA PNG: drop the opaque NLM banner, crop to the alpha box."""
    alpha = np.asarray(image.getchannel("A")) > 0
    opaque_rows = np.flatnonzero(alpha.mean(axis=1) > 0.95)
    top_of_banner = int(opaque_rows.min()) if opaque_rows.size else alpha.shape[0]
    pill = alpha[:top_of_banner]
    rows, cols = np.flatnonzero(pill.any(axis=1)), np.flatnonzero(pill.any(axis=0))
    if rows.size == 0:
        raise ValueError("empty alpha mask")
    return image.crop((int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1))


def augment(source: Path, seed: int, size: int) -> tuple[Image.Image, str]:
    """One augmented RGB image and the list of operations applied."""
    rng = random.Random(seed)
    operations = []
    with Image.open(source) as opened:
        if opened.format == "JPEG":
            opened.draft("RGB", (size, size))
        image = opened.copy()
    angle = rng.uniform(-180, 180)
    # Work at about the output size: rotating and filling a 14-megapixel photo is slow.
    if image.mode == "RGBA":
        pill = cut_pill(image)
        pill.thumbnail((size // 2, size // 2), Image.Resampling.LANCZOS)
        pill = pill.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        side = int(max(pill.size) * rng.uniform(1.4, 2.2))
        canvas = background((side, side), rng)
        x = rng.randint(0, side - pill.width)
        y = rng.randint(0, side - pill.height)
        canvas.paste(pill, (x, y), mask=pill.getchannel("A"))
        image = canvas
        operations += [f"rotate({angle:.0f})", "background(segmented)"]
    else:
        image = image.convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        fill = background(image.size, rng)
        mask = Image.new("L", image.size, 255).rotate(angle, expand=True)
        rotated = image.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
        canvas = fill.resize(rotated.size)
        canvas.paste(rotated, (0, 0), mask=mask)
        image = canvas
        operations += [f"rotate({angle:.0f})", "background(corners)"]
    factor = rng.uniform(0.6, 1.4)
    image = ImageEnhance.Brightness(image).enhance(factor)
    operations.append(f"brightness({factor:.2f})")
    if rng.random() < 0.5:
        radius = rng.uniform(0.5, 2.0)
        image = image.filter(ImageFilter.GaussianBlur(radius))
        operations.append(f"blur({radius:.1f})")
    scale = size / max(image.size)
    image = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    return image, ";".join(operations)


def output_name(class_id: int, class_key: str, index: int) -> str:
    return f"images/{class_id:03d}/{class_key}_aug{index:03d}.jpg"


def generate(plan_df: DataFrame, images: Path, out: Path, size: int, partitions: int) -> DataFrame:
    """Write the planned images in parallel; one result row per image."""
    images_root, out_root, root = str(images), str(out), ROOT

    def batches(frames: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for frame in frames:
            rows = []
            for item in frame.itertuples(index=False):
                name = output_name(item.class_id, item.class_key, item.index)
                target = Path(out_root) / name
                shown = target.relative_to(root) if target.is_relative_to(root) else target
                row = {
                    "path": shown.as_posix(),
                    "source": item.source,
                    "class_id": item.class_id,
                    "class_key": item.class_key,
                    "role": item.role,
                    "operations": "",
                    "ok": True,
                    "error": "",
                }
                try:
                    image, operations = augment(
                        local_file(item.source, Path(images_root)), int(item.seed), size
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    image.save(target, "JPEG", quality=90)
                    row["operations"] = operations
                except Exception as error:  # reported in the manifest, the job goes on
                    row.update(ok=False, error=f"{type(error).__name__}: {error}"[:300])
                rows.append(row)
            yield pd.DataFrame(rows)

    return plan_df.repartition(partitions).mapInPandas(batches, RESULT_SCHEMA)


def run(
    spark: SparkSession,
    params: dict[str, Any],
    *,
    split_images: Path = SPLIT,
    images: Path = IMAGES,
    out: Path = OUT,
    partitions: int = 32,
) -> dict[str, Any]:
    p = params["augment"]
    target, size, seed = int(p["target_per_class"]), int(p["size"]), int(params["seed"])
    split = spark.read.parquet(str(split_images))
    sources = split.where((F.col("split") == "train") & F.col("role").isin(*SOURCE_ROLES)).select(
        "path", "class_id", "class_key", "role"
    )
    planned = sources.groupBy("class_key").applyInPandas(
        lambda frame: plan(frame, target, seed), PLAN_SCHEMA
    )
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "16")
    for old in (out / "images").glob("*/*.jpg"):  # a rerun replaces every generated image
        old.unlink()
    result = generate(planned, images, out, size, partitions)
    result.write.mode("overwrite").parquet(str(out / "manifest"))
    manifest = spark.read.parquet(str(out / "manifest"))
    before = sources.groupBy("class_id").count()
    added = manifest.where("ok").groupBy("class_id").agg(F.count("*").alias("added"))
    per_class = before.join(added, "class_id", "left").na.fill(0, ["added"])
    summary = {
        "target_per_class": target,
        "classes": per_class.count(),
        "classes_augmented": per_class.where("added > 0").count(),
        "generated": manifest.where("ok").count(),
        "failed": manifest.where("NOT ok").count(),
        "train_sources": sources.count(),
        "sources_not_train": manifest.join(
            split.where(F.col("split") != "train").select(F.col("path").alias("source")),
            "source",
        ).count(),
        "min_train_before": per_class.agg(F.min("count")).first()[0],
        "min_train_after": per_class.agg(F.min(F.col("count") + F.col("added"))).first()[0],
    }
    write_json(summary, out / "summary.json")
    return summary
