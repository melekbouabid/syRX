"""Job `epillid`: what the ePillID benchmark shares with C3PI and with the classes of v2.

ePillID (Usuyama et al., CVPR Workshops 2020, https://github.com/usuyama/ePillID-benchmark)
is built from two NLM sources: the consumer photos and reference images of the NLM Pill
Image Recognition Challenge (taken from C3PI) and the reference images of NIH Pillbox. It
serves Tujya as the reference BENCHMARK (comparison with the literature), not as training
data: its consumer images are crops of C3PI photos, so adding them would duplicate images
we already have (and could put a copy of a test photo in train).

Input: data/raw/epillid/ePillID_data/all_labels.csv (DVC), index/images (job `index`),
config/splits/c3pi_v2/classes.csv (job `split`) and config/pill_classes_v1.csv (v1).
Each ePillID row gets its NDC9 (labeler + product code, zero-padded) and its kind:
`consumer` (fcn_mix_weight/dc_224), `challenge_reference` (fcn_mix_weight/dr_224) or
`pillbox_reference` (segmented_nih_pills_224).

Outputs (data/processed/c3pi/bigdata/epillid, DVC): images/ (ePillID rows with their
match), classes/ (v2 classes with their ePillID images), summary.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.data.spark.session import BIGDATA, ROOT, SPLITS_V2, write_json

EPILLID = ROOT / "data" / "raw" / "epillid" / "ePillID_data" / "all_labels.csv"
INDEX = BIGDATA / "index" / "images"
CLASSES_V1 = ROOT / "config" / "pill_classes_v1.csv"
OUT = BIGDATA / "epillid"


def read_epillid(spark: SparkSession, path: Path) -> DataFrame:
    frame = spark.read.option("header", True).csv(str(path))
    kind = (
        F.when(F.col("image_path").contains("/dc_"), "consumer")
        .when(F.col("image_path").contains("/dr_"), "challenge_reference")
        .otherwise("pillbox_reference")
    )
    ndc9 = F.concat(F.lpad("label_code_id", 5, "0"), F.lpad("prod_code_id", 4, "0"))
    return frame.select(
        "image_path",
        "pilltype_id",
        (F.col("is_ref") == "True").alias("is_ref"),
        (F.col("is_front") == "True").alias("is_front"),
        ndc9.alias("ndc9"),
        kind.alias("kind"),
    )


def _ndc9_of(classes: DataFrame) -> DataFrame:
    return classes.select(F.split("class_key", "_").getItem(0).alias("ndc9")).distinct()


def overlap(epillid: DataFrame, index: DataFrame, v2: DataFrame, v1: DataFrame) -> DataFrame:
    """ePillID rows flagged: product in the C3PI archive, among the v2 / v1 classes."""
    c3pi = index.select("ndc9").distinct().withColumn("in_c3pi", F.lit(True))
    in_v2 = _ndc9_of(v2).withColumn("in_v2", F.lit(True))
    in_v1 = _ndc9_of(v1).withColumn("in_v1", F.lit(True))
    out = epillid
    for table in (c3pi, in_v2, in_v1):
        out = out.join(table, "ndc9", "left")
    return out.na.fill(False, ["in_c3pi", "in_v2", "in_v1"])


def run(
    spark: SparkSession,
    params: dict[str, Any],
    *,
    epillid_csv: Path = EPILLID,
    index: Path = INDEX,
    classes_v2: Path = SPLITS_V2 / "classes.csv",
    classes_v1: Path = CLASSES_V1,
    out: Path = OUT,
) -> dict[str, Any]:
    del params  # no parameter: the job only measures the overlap
    epillid = read_epillid(spark, epillid_csv)
    v2 = spark.read.option("header", True).csv(str(classes_v2))
    v1 = spark.read.option("header", True).csv(str(classes_v1))
    rows = overlap(epillid, spark.read.parquet(str(index)), v2, v1).cache()
    rows.coalesce(1).write.mode("overwrite").parquet(str(out / "images"))
    v2_keys = v2.select("class_id", "class_key", "name", F.split("class_key", "_")[0].alias("ndc9"))
    per_class = (
        v2_keys.join(rows, "ndc9")
        .groupBy("class_id", "class_key", "name")
        .pivot("kind", ["consumer", "challenge_reference", "pillbox_reference"])
        .count()
        .na.fill(0)
        .orderBy(F.col("class_id").cast("int"))
    )
    per_class.coalesce(1).write.mode("overwrite").parquet(str(out / "classes"))
    by_kind = {}
    for r in (
        rows.groupBy("kind")
        .agg(
            F.count("*").alias("images"),
            F.countDistinct("pilltype_id").alias("pill_types"),
            F.sum(F.col("in_c3pi").cast("int")).alias("images_in_c3pi"),
            F.sum(F.col("in_v2").cast("int")).alias("images_in_v2_classes"),
            F.sum(F.col("in_v1").cast("int")).alias("images_in_v1_classes"),
        )
        .collect()
    ):
        by_kind[r["kind"]] = {k: v for k, v in r.asDict().items() if k != "kind"}
    summary = {
        "images": rows.count(),
        "pill_types": rows.select("pilltype_id").distinct().count(),
        "pill_types_in_c3pi": rows.where("in_c3pi").select("pilltype_id").distinct().count(),
        "v2_classes": v2.count(),
        "v2_classes_in_epillid": per_class.count(),
        "v2_classes_with_epillid_consumer": per_class.where("consumer > 0").count(),
        "by_kind": by_kind,
        "use": "benchmark only: ePillID images are crops of C3PI / Pillbox images",
    }
    write_json(summary, out / "summary.json")
    return summary
