"""Job `rag`: the RAG corpus of module 2 at a larger scale, cleaned, deduplicated, chunked.

Inputs (DVC): the collection workspaces listed in params.yaml (`bigdata.rag.workspaces`):
data/rag (v1: 150 classes, 20 abstracts per ingredient) and, once collected, data/rag_v2
(classes of the C3PI dataset v2, 40 abstracts per ingredient,
`python -m src.data.spark.rag_sources`): openFDA label sections, PubMed articles, query hits
and the queries (raw/pubmed/*/*/query.json, which ingredient each query is about); the
ingredients of both class lists; DDInter pairs (data/processed/pharma).

1. Cleaning (Spark SQL): HTML entities and tags removed, spaces collapsed (paragraph breaks
   kept), texts shorter than `min_chars` dropped; label sections about doses are refused
   (docs/project_rules.md); retraction notices and abstracts outside English dropped.
2. Deduplication: labels by (ingredient, section, text); articles by PMID, then by
   identical normalised abstract, then near-duplicates with MinHash LSH (pyspark.ml: word
   3-shingles, Jaccard distance under `near_duplicate`), keeping the smallest PMID.
3. Chunking: the functions of the v1 corpus (src/data/corpus/corpus.py: same splitter, same
   metadata columns) run on every partition (`mapInPandas`).

Outputs: data/processed/rag/corpus_v2/chunks (Parquet partitioned by doc_type) and
summary.json. The served index still reads the v1 corpus until module 2 switches (step 3.x).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from pyspark.ml.feature import HashingTF, MinHashLSH, NGram, RegexTokenizer
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql import types as T

from src.data.corpus import corpus
from src.data.spark.session import ROOT, write_json

WORKSPACES = (ROOT / "data" / "rag", ROOT / "data" / "rag_v2")
INGREDIENT_FILES = (
    ROOT / "data" / "processed" / "pharma" / "pill_ingredients.csv",
    ROOT / "data" / "rag_v2" / "ingredients.csv",
)
PAIRS = ROOT / "data" / "processed" / "pharma" / "interactions_with_any_drug.csv"
OUT = ROOT / "data" / "processed" / "rag" / "corpus_v2"
_TYPES: dict[str, T.DataType] = {
    "ingredients": T.ArrayType(T.StringType()),
    "mentioned_ingredients": T.ArrayType(T.StringType()),
    "chunk_index": T.IntegerType(),
    "is_retracted": T.BooleanType(),
}
CHUNK_SCHEMA = T.StructType(
    [T.StructField(name, _TYPES.get(name, T.StringType())) for name in corpus.CHUNK_COLUMNS]
)
FORBIDDEN = corpus.FORBIDDEN_SECTION.pattern


def clean_text(column: str, keep_newlines: bool = False) -> F.Column:
    """HTML entities and tags out, runs of spaces collapsed (newlines kept if asked)."""
    text = F.coalesce(F.col(column), F.lit(""))
    for entity, char in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"')):
        text = F.replace(text, F.lit(entity), F.lit(char))
    text = F.regexp_replace(text, r"&#?\w+;", " ")
    text = F.regexp_replace(text, r"<[^>]{1,200}>", " ")
    if keep_newlines:
        # Java regex: \v would also match newlines, so the vertical tab is spelled \x0B.
        text = F.regexp_replace(text, r"[ \t\r\f\x0B]+", " ")
        text = F.regexp_replace(text, r" ?\n ?", "\n")
        return F.trim(F.regexp_replace(text, r"\n{2,}", "\n"))
    return F.trim(F.regexp_replace(text, r"\s+", " "))


def _existing(paths: list[Path]) -> list[str]:
    return [str(p) for p in paths if p.exists()]


def read_union(
    spark: SparkSession, relative: str, workspaces: tuple[Path, ...] = WORKSPACES
) -> DataFrame | None:
    """The same interim table of every workspace that has it, stacked."""
    paths = _existing([w / relative for w in workspaces])
    if not paths:
        return None
    frames = [spark.read.parquet(p) for p in paths]
    out = frames[0]
    for frame in frames[1:]:
        out = out.unionByName(frame, allowMissingColumns=True)
    return out


def clean_labels(labels: DataFrame, min_chars: int) -> DataFrame:
    newest = Window.partitionBy("ingredient", "section", "_hash").orderBy(F.desc("retrieved_at"))
    return (
        labels.where(~F.col("section").rlike(FORBIDDEN))
        .withColumn("text", clean_text("text"))
        .where(F.length("text") >= min_chars)
        .withColumn("_hash", F.sha1(F.lower("text")))
        .withColumn("_r", F.row_number().over(newest))
        .where("_r = 1")
        .drop("_r", "_hash")
    )


def clean_articles(articles: DataFrame, min_chars: int) -> DataFrame:
    newest = Window.partitionBy("pmid").orderBy(F.desc("retrieved_at"))
    same_text = Window.partitionBy("_hash").orderBy(F.col("pmid").cast("long"))
    language = F.lower(F.coalesce(F.col("language"), F.lit("eng")))
    return (
        articles.withColumn("_r", F.row_number().over(newest))
        .where("_r = 1")
        .where(~F.coalesce(F.col("is_retraction_notice"), F.lit(False)))
        .where(language.startswith("en"))
        .withColumn("abstract", clean_text("abstract", keep_newlines=True))
        .withColumn("title", clean_text("title"))
        .where(F.length("abstract") >= min_chars)
        .withColumn("_hash", F.sha1(F.lower(F.regexp_replace("abstract", r"\W+", ""))))
        .withColumn("_r", F.row_number().over(same_text))
        .where("_r = 1")
        .drop("_r", "_hash")
    )


def near_duplicate_pmids(articles: DataFrame, threshold: float, seed: int) -> DataFrame:
    """PMIDs to drop: the larger PMID of each pair of abstracts closer than `threshold`."""
    tokens = RegexTokenizer(inputCol="abstract", outputCol="_words", pattern=r"\W+")
    shingles = NGram(n=3, inputCol="_words", outputCol="_shingles")
    hashed = HashingTF(inputCol="_shingles", outputCol="_features", numFeatures=1 << 18)
    frame = hashed.transform(
        shingles.transform(tokens.transform(articles.select("pmid", "abstract")))
    )
    frame = frame.where(F.size("_shingles") > 0)
    model = MinHashLSH(inputCol="_features", outputCol="_hashes", numHashTables=5, seed=seed)
    fitted = model.fit(frame)
    pairs = fitted.approxSimilarityJoin(frame, frame, threshold, distCol="distance")
    a, b = F.col("datasetA.pmid"), F.col("datasetB.pmid")
    return (
        pairs.where(a.cast("long") < b.cast("long"))
        .select(b.alias("pmid"), a.alias("kept_pmid"), "distance")
        .dropDuplicates(["pmid"])
    )


def pmid_ingredients(
    spark: SparkSession, hits: DataFrame, workspaces: tuple[Path, ...] = WORKSPACES
) -> DataFrame:
    """PMID -> ingredients whose query returned it (query.json of each raw query folder)."""
    files = _existing([w / "raw" / "pubmed" for w in workspaces])
    queries = spark.read.option("multiLine", True).json([f"{f}/*/*/query.json" for f in files])
    queries = queries.where(F.col("kind") == "ingredient").select(
        "query_id", F.lower(F.col("drug_names")[0]).alias("ingredient")
    )
    return (
        hits.join(queries.distinct(), "query_id")
        .groupBy("pmid")
        .agg(F.array_sort(F.collect_set("ingredient")).alias("about"))
    )


def ingredient_names(spark: SparkSession, files: tuple[Path, ...] = INGREDIENT_FILES) -> list[str]:
    paths = _existing(list(files))
    frame = spark.read.option("header", True).csv(paths)
    return sorted(r[0] for r in frame.select(F.lower("ingredient")).distinct().collect() if r[0])


def _chunker(kind: str, names: list[str]) -> Any:
    """Per-partition chunking with the v1 functions (chunks of at most corpus.MAX_CHARS)."""

    def batches(frames: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for frame in frames:
            if kind == "label":
                rows = corpus.label_chunks(frame, set(names))
            elif kind == "abstract":
                about = dict(zip(frame["pmid"], frame["about"].map(list), strict=True))
                rows = corpus.abstract_chunks(frame, about, set(names))
            else:
                rows = corpus.interaction_chunks(frame, "")
            out = pd.DataFrame(rows, columns=corpus.CHUNK_COLUMNS)
            out["chunk_index"] = out["chunk_index"].astype("int32")
            yield out

    return batches


def run(spark: SparkSession, params: dict[str, Any], *, out: Path = OUT) -> dict[str, Any]:
    p = params["rag"]
    min_chars = int(p["min_chars"])
    # Workspaces and ingredient lists from params.yaml (paths from the repository root);
    # the module constants when the parameters do not list them.
    workspaces = tuple(ROOT / w for w in p["workspaces"]) if "workspaces" in p else WORKSPACES
    files = INGREDIENT_FILES
    if "ingredient_files" in p:
        files = tuple(ROOT / f for f in p["ingredient_files"])
    names = ingredient_names(spark, files)
    stats: dict[str, Any] = {
        "ingredients": len(names),
        "workspaces": [
            w.relative_to(ROOT).as_posix() if w.is_relative_to(ROOT) else str(w) for w in workspaces
        ],
    }
    parts: list[DataFrame] = []

    labels = read_union(spark, "interim/openfda_label/label_sections.parquet", workspaces)
    if labels is not None:
        cleaned = clean_labels(labels, min_chars).cache()
        stats["label_sections"] = {"read": labels.count(), "kept": cleaned.count()}
        parts.append(cleaned.mapInPandas(_chunker("label", names), CHUNK_SCHEMA))

    articles = read_union(spark, "interim/pubmed/articles.parquet", workspaces)
    hits = read_union(spark, "interim/pubmed/query_hits.parquet", workspaces)
    if articles is not None and hits is not None:
        about = pmid_ingredients(spark, hits, workspaces)
        cleaned = clean_articles(articles.join(about, "pmid"), min_chars).cache()
        near = near_duplicate_pmids(cleaned, float(p["near_duplicate"]), int(params["seed"]))
        near = near.cache()
        kept = cleaned.join(near.select("pmid"), "pmid", "left_anti")
        stats["articles"] = {
            "read": articles.count(),
            "distinct_pmids": articles.select("pmid").distinct().count(),
            "after_cleaning": cleaned.count(),
            "near_duplicates": near.count(),
            "kept": kept.count(),
        }
        parts.append(kept.mapInPandas(_chunker("abstract", names), CHUNK_SCHEMA))

    if PAIRS.is_file():
        pairs = spark.read.option("header", True).csv(str(PAIRS))
        both = F.lower("ingredient_a").isin(*names) & F.lower("ingredient_b").isin(*names)
        pairs = pairs.where(both).dropDuplicates(["ingredient_a", "ingredient_b"])
        stats["interaction_pairs"] = pairs.count()
        parts.append(pairs.mapInPandas(_chunker("interaction", names), CHUNK_SCHEMA))

    chunks = parts[0]
    for part in parts[1:]:
        chunks = chunks.unionByName(part)
    chunks = chunks.dropDuplicates(["chunk_id"])
    chunks.write.mode("overwrite").partitionBy("doc_type").parquet(str(out / "chunks"))
    chunks = spark.read.parquet(str(out / "chunks"))
    by_type = {
        r["doc_type"]: {"documents": r["documents"], "chunks": r["chunks"]}
        for r in chunks.groupBy("doc_type")
        .agg(F.countDistinct("doc_id").alias("documents"), F.count("*").alias("chunks"))
        .collect()
    }
    covered = chunks.select(F.explode("ingredients").alias("i")).distinct().count()
    stats.update(
        chunks=chunks.count(),
        documents=chunks.select("doc_id").distinct().count(),
        by_type=by_type,
        ingredients_covered=covered,
        max_chars=corpus.MAX_CHARS,
        mean_chars=round(chunks.agg(F.avg(F.length("text"))).first()[0] or 0.0, 1),
    )
    v1 = ROOT / "data" / "processed" / "rag" / "corpus" / "summary.json"
    if v1.is_file():
        stats["v1"] = json.loads(v1.read_text(encoding="utf-8"))
    write_json(stats, out / "summary.json")
    return stats
