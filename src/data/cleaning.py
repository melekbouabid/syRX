"""Job `quality`: check every downloaded image of the selection and list the rejected ones.

Input: data/processed/c3pi/bigdata/index/selection (job `index`) and the images in
data/raw/c3pi/images/<disc>/<file> (DVC). Each Spark task opens its images with Pillow
(`inspect_image`, run through `mapInPandas`) and measures:

- bytes, width, height, mode; corrupt when Pillow cannot decode the whole file;
- brightness (mean grey level), contrast (standard deviation), sharpness (variance of the
  Laplacian), all on a copy whose long side is 256 px;
- a difference hash (dHash) of hash_size x hash_size bits, the perceptual hash used to find
  near-identical photos.

Reasons for rejection (one per image, first that applies): `missing`, `corrupt`,
`too_small` (short side under min_side), `too_dark` / `too_bright` (consumer and challenge
photos only: the segmented studio PNGs are mostly transparent), `duplicate_file` (same
SHA-1 as an image kept), `duplicate_other_class` (perceptually identical to an image of
another pill: the label is ambiguous; pairs of two studio shots do not count, see
`near_duplicates`). Near-identical photos of the same pill are kept but
recorded (`near_duplicates/`), so the split job puts them in the same split.

Outputs (data/processed/c3pi/bigdata/quality, DVC): metrics/ (one row per image, Parquet),
near_duplicates/ (pairs within a class), rejected.csv (path, class, type, reason),
summary.json.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.data.spark.session import BIGDATA, C3PI_RAW, write_json, write_rows

IMAGES = C3PI_RAW / "images"
SELECTION = BIGDATA / "index" / "selection"
OUT = BIGDATA / "quality"
PHOTO_TYPES = ("C3PI_Test", "MC_CHALLENGE_V1.0")  # photos whose brightness is checked
HASH_BLOCKS = 4  # the hash is compared as 4 blocks: <= 3 differing bits => one equal block
METRIC_SCHEMA = (
    "path string, exists boolean, bytes long, width int, height int, mode string, "
    "corrupt boolean, error string, brightness double, contrast double, sharpness double, "
    "dhash string"
)
METRIC_COLUMNS = [c.split()[0] for c in METRIC_SCHEMA.split(", ")]


def local_file(path: str, images: Path) -> Path:
    """Selection path `PillProjectDiscN/images/<file>` -> images/PillProjectDiscN/<file>."""
    disc, _, name = path.split("/")
    return images / disc / name


def dhash(gray: Image.Image, size: int) -> str:
    """Difference hash: is each pixel brighter than its right neighbour, size x size bits."""
    small = np.asarray(gray.resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{size * size // 4}x}"


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def _laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    lap = g[1:-1, :-2] + g[1:-1, 2:] + g[:-2, 1:-1] + g[2:, 1:-1] - 4 * g[1:-1, 1:-1]
    return float(lap.var())


def inspect_image(file: Path, hash_size: int = 16) -> dict[str, Any]:
    """Metrics of one image file (see the module docstring); never raises."""
    row: dict[str, Any] = dict.fromkeys(METRIC_COLUMNS)
    row.update(exists=file.is_file(), corrupt=False, error="")
    if not row["exists"]:
        return row
    row["bytes"] = file.stat().st_size
    try:
        with Image.open(file) as image:
            row.update(width=image.width, height=image.height, mode=image.mode)
            if image.format == "JPEG":  # decode at 1/2 to 1/8 scale: much faster, still checks
                image.draft("RGB", (max(256, image.width // 8), max(256, image.height // 8)))
            image.load()  # a truncated or damaged file fails here
            if image.mode == "RGBA":  # segmented PNG: measure the pill, not the transparency
                background = Image.new("RGB", image.size, (128, 128, 128))
                background.paste(image, mask=image.getchannel("A"))
                image = background
            gray = image.convert("L")
            scale = 256 / max(gray.size)
            small = gray.resize(
                (max(1, round(gray.width * scale)), max(1, round(gray.height * scale)))
            )
            pixels = np.asarray(small)
            row.update(
                brightness=round(float(pixels.mean()), 2),
                contrast=round(float(pixels.std()), 2),
                sharpness=round(_laplacian_variance(pixels), 2),
                dhash=dhash(gray, hash_size),
            )
    except Exception as error:  # any decoding error makes the image unusable
        row.update(corrupt=True, error=f"{type(error).__name__}: {error}"[:300])
    return row


def measure(selection: DataFrame, images: Path, hash_size: int, partitions: int) -> DataFrame:
    """One metrics row per selected image, computed in parallel by the executors."""
    root = str(images)

    def batches(frames: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for frame in frames:
            rows = []
            for path in frame["path"]:
                row = inspect_image(local_file(path, Path(root)), hash_size)
                rows.append({**row, "path": path})
            yield pd.DataFrame(rows, columns=METRIC_COLUMNS)

    return selection.select("path").repartition(partitions).mapInPandas(batches, METRIC_SCHEMA)


def _blocks(column: str, hash_size: int) -> list[F.Column]:
    """The hex hash cut into HASH_BLOCKS strings (banding for the near-duplicate join)."""
    width = hash_size * hash_size // 4 // HASH_BLOCKS
    return [F.substring(column, 1 + i * width, width) for i in range(HASH_BLOCKS)]


def near_duplicates(table: DataFrame, max_bits: int, hash_size: int) -> DataFrame:
    """Pairs of images (a < b) whose hashes differ by at most `max_bits` bits.

    Banding: with HASH_BLOCKS blocks and at most HASH_BLOCKS - 1 differing bits, two close
    hashes share at least one identical block, so an equi-join on (block index, block value)
    finds every candidate pair without comparing all pairs; the exact distance is then
    computed on the candidates only.

    Only pairs with at least one consumer photo count: studio shots (challenge, segmented,
    reference) show a small pill on a plain background, so their hashes are close across
    different pills (measured on the v2 selection: 1,468 such pairs, all studio-studio),
    while a copy of a consumer photo is still found whatever its role.
    """
    if max_bits >= HASH_BLOCKS:
        raise ValueError(f"banding finds pairs up to {HASH_BLOCKS - 1} bits apart")
    hashed = table.where(F.col("dhash").isNotNull())
    banded = hashed.select(
        "path",
        "class_key",
        "dhash",
        (F.col("role") == "consumer").alias("photo"),
        F.posexplode(F.array(*_blocks("dhash", hash_size))).alias("block", "value"),
    )
    a, b = banded.alias("a"), banded.alias("b")
    pairs = (
        a.join(b, (F.col("a.block") == F.col("b.block")) & (F.col("a.value") == F.col("b.value")))
        .where(F.col("a.path") < F.col("b.path"))
        .where(F.col("a.photo") | F.col("b.photo"))
        .select(
            F.col("a.path").alias("path_a"),
            F.col("b.path").alias("path_b"),
            F.col("a.class_key").alias("class_a"),
            F.col("b.class_key").alias("class_b"),
            F.col("a.dhash").alias("hash_a"),
            F.col("b.dhash").alias("hash_b"),
        )
        .dropDuplicates(["path_a", "path_b"])
    )
    distance = F.udf(hamming, "int")
    return pairs.withColumn("bits", distance("hash_a", "hash_b")).where(F.col("bits") <= max_bits)


def basic_reason(params: dict[str, Any]) -> F.Column:
    """Reason of an image that cannot be used at all (null when it can)."""
    photo = F.col("image_type").isin(*PHOTO_TYPES)
    return (
        F.when(~F.col("exists"), "missing")
        .when(F.col("corrupt"), "corrupt")
        .when(F.least("width", "height") < int(params["min_side"]), "too_small")
        .when(photo & (F.col("brightness") < float(params["min_brightness"])), "too_dark")
        .when(photo & (F.col("brightness") > float(params["max_brightness"])), "too_bright")
    )


def duplicate_reason(table: DataFrame, pairs: DataFrame) -> DataFrame:
    """Add `reason` for duplicates among usable images: a second copy of a file
    (`duplicate_file`), or a photo near-identical to one of another pill
    (`duplicate_other_class`)."""
    first = Window.partitionBy("sha1").orderBy("path")
    other_class = (
        pairs.where(F.col("class_a") != F.col("class_b"))
        .select(F.explode(F.array("path_a", "path_b")).alias("path"))
        .distinct()
        .withColumn("_other", F.lit(True))
    )
    reason = F.when(F.col("_rank") > 1, "duplicate_file").when(
        F.col("_other"), "duplicate_other_class"
    )
    return (
        table.withColumn("_rank", F.when(F.col("sha1") != "", F.row_number().over(first)))
        .join(other_class, "path", "left")
        .withColumn("reason", reason)
        .drop("_rank", "_other")
    )


def run(
    spark: SparkSession,
    params: dict[str, Any],
    *,
    selection_dir: Path = SELECTION,
    images: Path = IMAGES,
    out: Path = OUT,
    partitions: int = 64,
) -> dict[str, Any]:
    p = params["quality"]
    hash_size = int(p["hash_size"])
    selection = spark.read.parquet(str(selection_dir))
    spark.conf.set("spark.sql.execution.arrow.maxRecordsPerBatch", "32")
    metrics = measure(selection, images, hash_size, partitions)
    metrics.write.mode("overwrite").parquet(str(out / "metrics"))
    metrics = spark.read.parquet(str(out / "metrics"))  # measured once, reused below
    table = selection.select("path", "class_key", "image_type", "role", "sha1").join(
        metrics, "path"
    )
    # Unusable images first (a blank frame hashes like every other blank frame), then the
    # duplicates among the usable ones.
    basic = table.withColumn("reason", basic_reason(p))
    usable = basic.where(F.col("reason").isNull()).drop("reason")
    pairs = near_duplicates(usable, int(p["near_duplicate_bits"]), hash_size).cache()
    pairs.coalesce(1).write.mode("overwrite").parquet(str(out / "near_duplicates"))
    judged = (
        duplicate_reason(usable, pairs)
        .unionByName(basic.where(F.col("reason").isNotNull()))
        .cache()
    )
    judged.coalesce(4).write.mode("overwrite").parquet(str(out / "judged"))
    rejected = judged.where(F.col("reason").isNotNull()).orderBy("class_key", "path")
    columns = ["path", "class_key", "image_type", "role", "reason", "error"]
    write_rows(
        (r.asDict() for r in rejected.select(*columns).toLocalIterator()),
        out / "rejected.csv",
        columns,
    )
    reasons = {r["reason"]: r["count"] for r in rejected.groupBy("reason").count().collect()}
    by_role = {
        r["role"]: {"checked": r["checked"], "rejected": r["rejected"]}
        for r in judged.groupBy("role")
        .agg(
            F.count("*").alias("checked"),
            F.sum(F.col("reason").isNotNull().cast("int")).alias("rejected"),
        )
        .collect()
    }
    stats = judged.where(F.col("reason").isNull()).agg(
        F.avg("brightness").alias("brightness"),
        F.avg("width").alias("width"),
        F.avg("height").alias("height"),
        F.sum("bytes").alias("bytes"),
    )
    summary = {
        "checked": judged.count(),
        "rejected": sum(reasons.values()),
        "reasons": reasons,
        "by_role": by_role,
        "near_duplicate_pairs": pairs.count(),
        "near_duplicate_pairs_other_class": pairs.where("class_a != class_b").count(),
        "kept": stats.first().asDict(),
    }
    write_json(summary, out / "summary.json")
    return summary
