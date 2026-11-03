"""Job `index`: the whole C3PI archive as one table, its statistics and the class selection.

Inputs (data/raw/c3pi/metadata, DVC):
- ALLXML/PillProjectDisc{1..110}.xml, one <Image> element per file, read with Spark's XML
  source (NDC, part, names, file name, size, SHA-1, image type, camera, background...);
- directory_of_images.txt (`NDC11|part|path|type|name`, the archive's own index; its copy
  directory_consumer_grade_images.txt has the same bytes), joined to check that both agree.

Outputs (data/processed/c3pi/bigdata/index, DVC):
- images/         Parquet partitioned by image_type, one row per image of every type
- class_stats/    one row per pill product (NDC9 + part): files and bytes per image type
- type_stats/     one row per image type and file type: files, bytes, products
- classes/        candidate classes: products with at least `min_images` consumer photos
- selection/      images of the candidates to download, with their role (consumer,
                  reference, seg, challenge); `download_list.csv` is the same list as CSV
- summary.json    counts and sizes (also read by the MLflow step)

Docs: docs/module1/big_data.md, docs/module1/dataset_c3pi.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.data.spark.session import BIGDATA, C3PI_RAW, write_json, write_rows

CONSUMER = "C3PI_Test"  # consumer-grade photos (phones, compact cameras, webcams)
SEG = "MC_C3PI_REFERENCE_SEG_V1.6"  # studio shots with an alpha mask of the pill
CHALLENGE = "MC_CHALLENGE_V1.0"  # NLM Pill Image Recognition Challenge (2016)
# Small reference JPGs of the former RxImage API, in order of preference (one per NDC11).
REFERENCE_JPG = ("MC_API_RXNAV_V1.3", "MC_API_NLMIMAGE_V1.3", "MC_SPL_SPLIMAGE_V3.0")
ROLES = ("consumer", "reference", "seg", "challenge")

METADATA = C3PI_RAW / "metadata"
OUT = BIGDATA / "index"
DIRECTORY_NAMES = ("directory_of_images.txt", "directory_consumer_grade_images.txt")

_FILE = T.StructType(
    [
        T.StructField("Name", T.StringType()),
        T.StructField("Type", T.StringType()),
        T.StructField("Size", T.StringType()),
        T.StructField("Sha1", T.StringType()),
    ]
)
# Explicit schema: NDCs keep their leading zeros, <Color> is always a list, and the <Size>
# of <File> (bytes) stays apart from the <Size> of the pill (mm).
XML_SCHEMA = T.StructType(
    [
        *(
            T.StructField(name, T.StringType())
            for name in (
                "NDC9",
                "NDC11",
                "Part",
                "Parts",
                "MedicosConsultantsID",
                "LabeledBy",
                "GenericName",
                "ProprietaryName",
            )
        ),
        T.StructField("File", _FILE),
        *(
            T.StructField(name, T.StringType())
            for name in (
                "Class",
                "Layout",
                "Camera",
                "Illumination",
                "Background",
                "Imprint",
                "Shape",
                "Size",
                "AcquisitionDate",
            )
        ),
        T.StructField("Color", T.ArrayType(T.StringType())),
    ]
)
IMAGE_COLUMNS = [
    "path",
    "disc",
    "file_name",
    "file_type",
    "file_size",
    "sha1",
    "image_type",
    "ndc11",
    "ndc9",
    "part",
    "class_key",
    "name",
    "generic_name",
    "labeled_by",
    "imprint",
    "color",
    "shape",
    "size_mm",
    "camera",
    "illumination",
    "background",
    "acquisition_date",
    "in_directory",
]


def _clean(column: str) -> F.Column:
    return F.coalesce(F.trim(F.col(column)), F.lit(""))


def read_xml(spark: SparkSession, xml_dir: Path) -> DataFrame:
    """One row per <Image> of every ALLXML file (a few files list an image twice)."""
    raw = (
        spark.read.format("xml")
        .option("rowTag", "Image")
        .option("encoding", "ISO-8859-1")  # as declared by every ALLXML file
        .schema(XML_SCHEMA)
        .load(str(xml_dir / "PillProjectDisc*.xml"))
    )
    ndc11 = _clean("NDC11")
    ndc9 = F.when(_clean("NDC9") != "", _clean("NDC9")).otherwise(F.substring(ndc11, 1, 9))
    part = F.coalesce(F.col("Part").cast("int"), F.lit(1))
    image_type = F.when(_clean("Class") != "", _clean("Class")).otherwise(_clean("Layout"))
    disc = F.regexp_extract(F.col("_metadata.file_path"), r"(PillProjectDisc\d+)\.xml$", 1)
    return raw.select(
        disc.alias("disc"),
        F.trim(F.col("File.Name")).alias("file_name"),
        F.upper(F.trim(F.col("File.Type"))).alias("file_type"),
        F.coalesce(F.col("File.Size").cast("long"), F.lit(0)).alias("file_size"),
        F.lower(_clean("File.Sha1")).alias("sha1"),
        image_type.alias("image_type"),
        ndc11.alias("ndc11"),
        ndc9.alias("ndc9"),
        part.alias("part"),
        _clean("ProprietaryName").alias("name"),
        _clean("GenericName").alias("generic_name"),
        _clean("LabeledBy").alias("labeled_by"),
        _clean("Imprint").alias("imprint"),
        F.coalesce(F.concat_ws(";", "Color"), F.lit("")).alias("color"),
        _clean("Shape").alias("shape"),
        _clean("Size").alias("size_mm"),
        _clean("Camera").alias("camera"),
        _clean("Illumination").alias("illumination"),
        _clean("Background").alias("background"),
        _clean("AcquisitionDate").alias("acquisition_date"),
    ).withColumns(
        {
            "path": F.concat_ws("/", "disc", F.lit("images"), "file_name"),
            "class_key": F.concat_ws("_", "ndc9", F.col("part").cast("string")),
        }
    )


def read_directory(spark: SparkSession, path: Path) -> DataFrame:
    """directory_of_images.txt: NDC11|part|path|type|name, no header, no quoting."""
    schema = "ndc11 string, part int, path string, image_type string, name string"
    return spark.read.option("sep", "|").option("quote", "\u0000").schema(schema).csv(str(path))


def build_images(xml: DataFrame, directory: DataFrame) -> DataFrame:
    """One row per distinct file path, flagged when the directory file also lists it."""
    listed = directory.select("path").distinct().withColumn("in_directory", F.lit(True))
    first = Window.partitionBy("path").orderBy("disc", "sha1")
    unique = xml.withColumn("_r", F.row_number().over(first)).where("_r = 1").drop("_r")
    return (
        unique.join(listed, "path", "left")
        .withColumn("in_directory", F.coalesce("in_directory", F.lit(False)))
        .select(*IMAGE_COLUMNS)
    )


def most_common(images: DataFrame, key: str, column: str) -> DataFrame:
    """The most frequent non-empty value of `column` per `key` (ties: alphabetical)."""
    counts = images.where(F.col(column) != "").groupBy(key, column).count()
    best = Window.partitionBy(key).orderBy(F.desc("count"), F.asc(column))
    return counts.withColumn("_r", F.row_number().over(best)).where("_r = 1").select(key, column)


def type_stats(images: DataFrame) -> DataFrame:
    return (
        images.groupBy("image_type", "file_type")
        .agg(
            F.count("*").alias("files"),
            F.sum("file_size").alias("bytes"),
            F.countDistinct("class_key").alias("products"),
        )
        .orderBy("image_type", "file_type")
    )


def class_stats(images: DataFrame) -> DataFrame:
    """Files and bytes of every product, per image type (one column per type)."""
    types = sorted(r[0] for r in images.select("image_type").distinct().collect())
    files = images.groupBy("class_key").pivot("image_type", types).count().na.fill(0)
    consumer = images.where((F.col("image_type") == CONSUMER) & (F.col("file_type") == "JPG"))
    extra = images.groupBy("class_key").agg(
        F.count("*").alias("files"),
        F.sum("file_size").alias("bytes"),
        F.countDistinct("ndc11").alias("ndc11_count"),
    )
    photos = consumer.groupBy("class_key").agg(
        F.count("*").alias("consumer_jpg"), F.sum("file_size").alias("consumer_bytes")
    )
    return (
        files.join(extra, "class_key")
        .join(photos, "class_key", "left")
        .na.fill(0, ["consumer_jpg", "consumer_bytes"])
    )


def select_classes(
    images: DataFrame, *, min_images: int, exclude_cameras: list[str]
) -> tuple[DataFrame, DataFrame]:
    """Candidate classes (at least `min_images` consumer JPGs) and their consumer photos.

    Ranked by number of photos, then class key: the rank is only an order, the final class
    ids come from the split job, after the quality checks.
    """
    consumer = images.where((F.col("image_type") == CONSUMER) & (F.col("file_type") == "JPG"))
    if exclude_cameras:
        consumer = consumer.where(~F.col("camera").isin(*exclude_cameras))
    counts = consumer.groupBy("class_key").agg(
        F.count("*").alias("n_consumer"),
        F.sum("file_size").alias("consumer_bytes"),
        F.countDistinct("ndc11").alias("ndc11_count"),
        F.countDistinct("camera").alias("cameras"),
    )
    candidates = counts.where(F.col("n_consumer") >= min_images)
    rank = Window.orderBy(F.desc("n_consumer"), F.asc("class_key"))
    names = [most_common(images, "class_key", c) for c in ("name", "generic_name", "labeled_by")]
    looks = [most_common(images, "class_key", c) for c in ("imprint", "color", "shape")]
    ndc_list = images.groupBy("class_key").agg(
        F.concat_ws(";", F.array_sort(F.collect_set("ndc11"))).alias("ndc11_list"),
        F.first("ndc9").alias("ndc9"),
        F.first("part").alias("part"),
    )
    classes = candidates.withColumn("rank", F.row_number().over(rank) - 1).join(
        ndc_list, "class_key", "left"
    )
    for table in [*names, *looks]:
        classes = classes.join(table, "class_key", "left")
    classes = classes.na.fill("")
    photos = consumer.join(candidates.select("class_key"), "class_key", "left_semi")
    return classes, photos


def build_selection(
    images: DataFrame,
    classes: DataFrame,
    photos: DataFrame,
    *,
    seg_per_class: int,
    challenge: str,
) -> DataFrame:
    """Every image to download for the candidate classes, with its role.

    - consumer: every consumer photo of the class (the photos of `select_classes`);
    - reference: per package (NDC11) one small RxImage JPG, first in REFERENCE_JPG order;
    - seg: up to `seg_per_class` segmented studio PNGs, a fixed pseudo-random set (by SHA-1);
    - challenge: the MC_CHALLENGE_V1.0 images, all of them or only the candidates' ones.
    """
    keys = classes.select("class_key")
    in_class = keys.withColumn("in_class", F.lit(True))
    order = F.array(*(F.lit(t) for t in REFERENCE_JPG))
    reference = (
        images.where(F.col("image_type").isin(*REFERENCE_JPG))
        .join(keys, "class_key", "left_semi")
        .withColumn("_prio", F.array_position(order, F.col("image_type")))
        .withColumn("_best", F.min("_prio").over(Window.partitionBy("class_key", "ndc11")))
        .where(F.col("_prio") == F.col("_best"))
        .drop("_prio", "_best")
    )
    seg = (
        images.where(F.col("image_type") == SEG)
        .join(keys, "class_key", "left_semi")
        .withColumn("_r", F.row_number().over(Window.partitionBy("class_key").orderBy("sha1")))
        .where(F.col("_r") <= seg_per_class)
        .drop("_r")
    )
    parts = [
        photos.withColumn("role", F.lit("consumer")),
        reference.withColumn("role", F.lit("reference")),
        seg.withColumn("role", F.lit("seg")),
    ]
    if challenge != "none":
        tests = images.where(F.col("image_type") == CHALLENGE)
        if challenge == "classes":
            tests = tests.join(keys, "class_key", "left_semi")
        parts.append(tests.withColumn("role", F.lit("challenge")))
    selection = parts[0]
    for part in parts[1:]:
        selection = selection.unionByName(part)
    return (
        selection.join(in_class, "class_key", "left")
        .withColumn("in_class", F.coalesce("in_class", F.lit(False)))
        .dropDuplicates(["path"])
    )


def _directory_file(metadata: Path) -> Path:
    for name in DIRECTORY_NAMES:
        if (metadata / name).is_file():
            return metadata / name
    raise FileNotFoundError(f"none of {DIRECTORY_NAMES} in {metadata}")


def run(
    spark: SparkSession,
    params: dict[str, Any],
    *,
    metadata: Path = METADATA,
    out: Path = OUT,
) -> dict[str, Any]:
    """Run the job; returns the summary also written to out/summary.json."""
    p = params["index"]
    xml = read_xml(spark, metadata / "ALLXML")
    directory = read_directory(spark, _directory_file(metadata))
    images = build_images(xml, directory).cache()
    images.write.mode("overwrite").partitionBy("image_type").parquet(str(out / "images"))
    types = type_stats(images)
    types.coalesce(1).write.mode("overwrite").parquet(str(out / "type_stats"))
    class_stats(images).coalesce(1).write.mode("overwrite").parquet(str(out / "class_stats"))

    classes, photos = select_classes(
        images, min_images=int(p["min_images"]), exclude_cameras=list(p["exclude_cameras"])
    )
    classes = classes.cache()
    classes.coalesce(1).write.mode("overwrite").parquet(str(out / "classes"))
    selection = build_selection(
        images,
        classes,
        photos,
        seg_per_class=int(p["seg_per_class"]),
        challenge=str(p["challenge"]),
    ).cache()
    selection.coalesce(4).write.mode("overwrite").parquet(str(out / "selection"))
    listed = selection.select("path", "sha1", "file_size", "image_type", "role").orderBy("path")
    write_rows(
        (r.asDict() for r in listed.toLocalIterator()),
        out / "download_list.csv",
        ["path", "sha1", "file_size", "image_type", "role"],
    )

    by_role = {
        r["role"]: {"files": r["files"], "bytes": r["bytes"]}
        for r in selection.groupBy("role")
        .agg(F.count("*").alias("files"), F.sum("file_size").alias("bytes"))
        .collect()
    }
    summary = {
        "images": images.count(),
        "products": images.select("class_key").distinct().count(),
        "bytes": images.agg(F.sum("file_size")).first()[0] or 0,
        "xml_rows": xml.count(),
        "directory_rows": directory.count(),
        "not_in_directory": images.where(~F.col("in_directory")).count(),
        "by_type": {
            f"{r['image_type']} {r['file_type']}": {
                "files": r["files"],
                "bytes": r["bytes"],
                "products": r["products"],
            }
            for r in types.collect()
        },
        "min_images": int(p["min_images"]),
        "candidate_classes": classes.count(),
        "selection": by_role,
        "selection_files": selection.count(),
        "selection_bytes": selection.agg(F.sum("file_size")).first()[0] or 0,
    }
    write_json(summary, out / "summary.json")
    return summary
