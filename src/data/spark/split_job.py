"""Job `split`: train / val / test of the C3PI dataset v2, stratified by class, without leakage.

Inputs: index/selection and index/classes (job `index`), quality/judged and
quality/near_duplicates (job `quality`), config/pill_classes_v1.csv (the 150 classes of v1,
only to flag them).

1. Classes: candidates that still have at least `min_images` usable consumer photos after
   the quality checks; class ids 0..K-1 by number of usable photos (ties: class key).
2. Groups. C3PI gives no identifier of the physical pill photographed: one package (NDC11)
   is one specimen, and almost every class has a single package, so keeping a whole
   specimen in one split would leave the class out of val and test. The unit kept together
   is the shooting session: the photos of one package with one camera, one background, one
   lighting and one date (front and back, repeated shots). Groups linked by a near-identical
   photo (perceptual hash, job `quality`) or by an identical file are merged (union-find),
   so a duplicate can never sit on both sides of a split.
3. Each class's groups are shuffled (seeded hash of the group, independent of Spark's
   partitioning) and fill test, then val, best fit up to each split's share of the class's
   consumer photos; the rest is train. Studio and challenge images (reference, seg,
   challenge roles) stay in train unless a near-duplicate ties them to a val/test group.
4. `check_leakage` verifies that no group, file (SHA-1) or near-duplicate pair spans train
   and val/test; the job fails otherwise.

Outputs: data/processed/c3pi/bigdata/split/images (Parquet, every kept image with its
split, class_id and group, DVC) and config/splits/c3pi_v2/ (git): classes.csv, val.csv,
test.csv (train = every other image of images/), summary.json.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from src.data.spark.session import BIGDATA, ROOT, SPLITS_V2, write_json, write_rows

INDEX = BIGDATA / "index"
QUALITY = BIGDATA / "quality"
OUT = BIGDATA / "split"
CLASSES_V1 = ROOT / "config" / "pill_classes_v1.csv"
SESSION = ["class_key", "ndc11", "camera", "background", "illumination", "acquisition_date"]
SPLITS = ("train", "val", "test")
CLASS_COLUMNS = [
    "class_id",
    "class_key",
    "ndc9",
    "part",
    "ndc11_list",
    "name",
    "generic_name",
    "labeled_by",
    "imprint",
    "color",
    "shape",
    "in_v1",
    "n_consumer",
    "n_train",
    "n_val",
    "n_test",
    "n_reference",
    "n_seg",
    "n_challenge",
]
GROUP_SCHEMA = "class_key string, group_id string, n_consumer long, split string"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # smallest id wins: deterministic


def merge_groups(images: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    """Final group of every image: its session group merged along near-duplicate pairs."""
    uf = UnionFind()
    group_of = dict(zip(images["path"], images["group_id"], strict=True))
    for group in images["group_id"]:
        uf.find(group)
    for a, b in zip(pairs["path_a"], pairs["path_b"], strict=True):
        if a in group_of and b in group_of:
            uf.union(group_of[a], group_of[b])
    for _, same in images.groupby("sha1"):  # identical files (none left after quality)
        first = same["group_id"].iloc[0]
        for other in same["group_id"].iloc[1:]:
            uf.union(first, other)
    return images["group_id"].map(uf.find)


def _order(seed: int, group_id: str) -> str:
    return hashlib.sha1(f"{seed}:{group_id}".encode(), usedforsecurity=False).hexdigest()


def assign_groups(groups: pd.DataFrame, val: float, test: float, seed: int) -> pd.DataFrame:
    """Split the groups of ONE class; used per class with Spark's applyInPandas.

    Groups are visited in a seeded pseudo-random order (independent of the row order). Test
    is filled first, then val, each with the groups that still fit in its share of the
    class's consumer photos (best fit: a large session is skipped rather than overshooting);
    a split that no group fits gets the smallest one. At least one group with consumer
    photos always stays in train; groups without consumer photos are train.
    """
    groups = groups.copy()
    groups["_order"] = [_order(seed, g) for g in groups["group_id"]]
    groups = groups.sort_values("_order").reset_index(drop=True)
    sizes = [int(n) for n in groups["n_consumer"]]
    total = sum(sizes)
    split = ["train"] * len(groups)
    free = [i for i, n in enumerate(sizes) if n > 0]  # groups with photos, not assigned yet
    for name, share in (("test", test), ("val", val)):
        want, filled = max(1, round(total * share)), 0
        for i in list(free):
            if len(free) <= 1:  # keep one group for train
                break
            if filled + sizes[i] <= want:
                split[i], filled = name, filled + sizes[i]
                free.remove(i)
        if filled == 0 and len(free) > 1:
            smallest = min(free, key=lambda i: (sizes[i], i))
            split[smallest] = name
            free.remove(smallest)
    groups["split"] = split
    return groups.drop(columns="_order")


def check_leakage(images: DataFrame, pairs: DataFrame) -> dict[str, int]:
    """Counts of leaks between train and val/test; all must be zero."""
    held_out = F.col("split") != "train"
    groups = images.groupBy("group_id").agg(F.countDistinct("split").alias("n")).where("n > 1")
    files = (
        images.where(F.col("sha1") != "")
        .groupBy("sha1")
        .agg(F.countDistinct(held_out.cast("int")).alias("n"))
        .where("n > 1")
    )
    split_of = images.select("path", "split")
    crossing = (
        pairs.join(split_of.withColumnRenamed("path", "path_a"), "path_a")
        .withColumnRenamed("split", "split_a")
        .join(split_of.withColumnRenamed("path", "path_b"), "path_b")
        .where(F.col("split_a") != F.col("split"))
    )
    return {
        "groups_in_two_splits": groups.count(),
        "files_in_train_and_heldout": files.count(),
        "near_duplicates_across_splits": crossing.count(),
    }


def _v1_keys(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["class_key"] for row in csv.DictReader(handle)}


def run(
    spark: SparkSession,
    params: dict[str, Any],
    *,
    index: Path = INDEX,
    quality: Path = QUALITY,
    out: Path = OUT,
    splits_dir: Path = SPLITS_V2,
    classes_v1: Path = CLASSES_V1,
) -> dict[str, Any]:
    seed = int(params["seed"])
    val, test = float(params["split"]["val"]), float(params["split"]["test"])
    min_images = int(params["index"]["min_images"])
    selection = spark.read.parquet(str(index / "selection")).where("in_class")
    judged = spark.read.parquet(str(quality / "judged")).select("path", "reason", "dhash")
    pairs = spark.read.parquet(str(quality / "near_duplicates")).select("path_a", "path_b")
    kept = selection.join(judged, "path").where(F.col("reason").isNull()).drop("reason")

    usable = (
        kept.where(F.col("role") == "consumer")
        .groupBy("class_key")
        .agg(F.count("*").alias("n_consumer"))
        .where(F.col("n_consumer") >= min_images)
    )
    rank = Window.orderBy(F.desc("n_consumer"), F.asc("class_key"))
    classes = usable.withColumn("class_id", F.row_number().over(rank) - 1)
    kept = kept.join(classes.select("class_key", "class_id"), "class_key")

    session = F.sha1(F.concat_ws("|", *SESSION)).substr(1, 16)
    kept = kept.withColumn(
        "group_id",
        F.when(F.col("role") == "consumer", session).otherwise(F.sha1(F.col("path")).substr(1, 16)),
    )
    local = kept.select("path", "group_id", "sha1").toPandas()
    local["group_id"] = merge_groups(local, pairs.toPandas())
    merged = spark.createDataFrame(local[["path", "group_id"]])
    kept = kept.drop("group_id").join(merged, "path")

    group_table = kept.groupBy("class_key", "group_id").agg(
        F.sum((F.col("role") == "consumer").cast("long")).alias("n_consumer")
    )
    assigned = group_table.groupBy("class_key").applyInPandas(
        lambda frame: assign_groups(frame, val, test, seed), GROUP_SCHEMA
    )
    images = kept.join(assigned.select("group_id", "split"), "group_id").cache()
    images.write.mode("overwrite").parquet(str(out / "images"))
    images = spark.read.parquet(str(out / "images"))

    leaks = check_leakage(images, spark.read.parquet(str(quality / "near_duplicates")))
    if any(leaks.values()):
        raise RuntimeError(f"leakage between splits: {leaks}")

    counts = (
        images.groupBy("class_id")
        .pivot("split", list(SPLITS))
        .agg(F.sum((F.col("role") == "consumer").cast("int")))
        .na.fill(0)
    )
    roles = (
        images.groupBy("class_id")
        .pivot("role", ["reference", "seg", "challenge"])
        .count()
        .na.fill(0)
        .withColumnsRenamed({r: f"n_{r}" for r in ("reference", "seg", "challenge")})
    )
    info = spark.read.parquet(str(index / "classes")).drop("n_consumer", "rank")
    v1 = _v1_keys(classes_v1)
    table = (
        classes.join(info, "class_key")
        .join(counts.withColumnsRenamed({s: f"n_{s}" for s in SPLITS}), "class_id")
        .join(roles, "class_id", "left")
        .na.fill(0)
        .withColumn("in_v1", F.col("class_key").isin(*v1) if v1 else F.lit(False))
        .orderBy("class_id")
    )
    class_rows = [r.asDict() for r in table.collect()]
    write_rows(class_rows, splits_dir / "classes.csv", CLASS_COLUMNS)
    for split in ("val", "test"):
        held = images.where(F.col("split") == split).orderBy("class_id", "path")
        columns = ["path", "class_id", "image_type", "sha1", "group_id"]
        write_rows(
            (r.asDict() for r in held.select(*columns).toLocalIterator()),
            splits_dir / f"{split}.csv",
            columns,
        )
    per_split = {
        f"{r['split']} {r['role']}": r["count"]
        for r in images.groupBy("split", "role").count().collect()
    }
    summary = {
        "classes": len(class_rows),
        "classes_in_v1": sum(1 for r in class_rows if r["in_v1"]),
        "candidates_dropped_after_quality": spark.read.parquet(str(index / "classes")).count()
        - len(class_rows),
        "images": images.count(),
        "groups": images.select("group_id").distinct().count(),
        "by_split_and_role": dict(sorted(per_split.items())),
        "consumer_per_class_min": min(r["n_consumer"] for r in class_rows),
        "consumer_per_class_max": max(r["n_consumer"] for r in class_rows),
        "leakage": leaks,
        "params": {"seed": seed, "val": val, "test": test, "min_images": min_images},
    }
    write_json(summary, splits_dir / "summary.json")
    write_json(summary, out / "summary.json")
    return summary
