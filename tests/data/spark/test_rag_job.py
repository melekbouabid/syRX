"""Job `rag`: cleaning, deduplication (exact and MinHash) and chunking of the RAG sources."""

from __future__ import annotations

import json

import pandas as pd
import pytest

pytest.importorskip("pyspark")

from pyspark.sql import functions as F

from src.data.spark import rag_job

TEXT = (
    "Warfarin is an anticoagulant whose effect is increased by many drugs that inhibit "
    "CYP2C9, which raises the risk of bleeding in older patients taking several medicines."
)


def test_clean_text_removes_markup_and_spaces(spark):
    frame = spark.createDataFrame([("<p>A &amp; B</p>   C\n\n  D",)], "t string")
    flat, lines = frame.select(
        rag_job.clean_text("t").alias("flat"), rag_job.clean_text("t", True).alias("lines")
    ).first()
    assert flat == "A & B C D"
    assert lines == "A & B C\nD"


def label(section: str, text: str, ingredient: str = "warfarin", when: str = "2026-10-01"):
    return {
        "ingredient": ingredient,
        "label_id": f"L-{ingredient}",
        "set_id": "S",
        "label_version": "3",
        "effective_time": "20240101",
        "brand_name": "Brand",
        "generic_name": ingredient,
        "label_url": "https://example.org/label",
        "section": section,
        "text": text,
        "source_url": "https://api.fda.gov",
        "retrieved_at": when,
        "license": "CC0",
    }


def article(pmid: str, abstract: str, **extra):
    row = {
        "pmid": pmid,
        "title": f"Title {pmid}",
        "abstract": abstract,
        "journal": "J",
        "year": 2020,
        "language": "eng",
        "article_type": "Journal Article",
        "is_retracted": False,
        "is_retraction_notice": False,
        "license": "NLM",
        "retrieved_at": "2026-10-01",
    }
    row.update(extra)
    return row


def test_clean_labels_refuses_doses_and_duplicates(spark):
    rows = [
        label("drug_interactions", TEXT),
        label("drug_interactions", TEXT, when="2026-11-01"),  # same text, newer copy
        label("dosage_and_administration", TEXT),
        label("warnings", "Too short."),
    ]
    kept = rag_job.clean_labels(spark.createDataFrame(pd.DataFrame(rows)), 20).collect()
    assert [(r.section, r.retrieved_at) for r in kept] == [("drug_interactions", "2026-11-01")]


def test_clean_articles_keeps_one_copy_per_abstract(spark):
    rows = [
        article("10", TEXT),
        article("10", TEXT, retrieved_at="2026-11-01"),  # same PMID fetched twice
        article("11", "  " + TEXT.upper()),  # same abstract, another PMID
        article("12", "Retracted: " + TEXT, is_retraction_notice=True),
        article("13", TEXT + " Autre.", language="fre"),
        article("14", "Short."),
    ]
    kept = rag_job.clean_articles(spark.createDataFrame(pd.DataFrame(rows)), 20).collect()
    assert [(r.pmid, r.retrieved_at) for r in kept] == [("10", "2026-11-01")]


def test_minhash_finds_near_duplicate_abstracts(spark):
    other = (
        "Metformin lowers blood glucose in type 2 diabetes; lactic acidosis is a rare but "
        "serious adverse effect reported with renal impairment and contrast agents."
    )
    rows = [
        article("20", TEXT),
        article("21", TEXT.replace("older", "elderly")),  # one word changed
        article("22", other),
    ]
    near = rag_job.near_duplicate_pmids(spark.createDataFrame(pd.DataFrame(rows)), 0.3, 0)
    assert [(r.pmid, r.kept_pmid) for r in near.collect()] == [("21", "20")]


@pytest.fixture
def workspaces(tmp_path, monkeypatch):
    v1, v2 = tmp_path / "rag", tmp_path / "rag_v2"
    for workspace, labels, articles, hits, queries in (
        (
            v1,
            [label("drug_interactions", TEXT), label("dosage_and_administration", TEXT)],
            [article("30", TEXT)],
            [("q1", "30")],
            {"q1": "warfarin"},
        ),
        (
            v2,
            [label("warnings", TEXT.replace("Warfarin", "Metformin"), "metformin")],
            [article("30", TEXT), article("31", TEXT.replace("older", "elderly"))],
            [("q2", "30"), ("q2", "31")],
            {"q2": "metformin"},
        ),
    ):
        interim = workspace / "interim"
        (interim / "openfda_label").mkdir(parents=True)
        (interim / "pubmed").mkdir()
        pd.DataFrame(labels).to_parquet(interim / "openfda_label" / "label_sections.parquet")
        pd.DataFrame(articles).to_parquet(interim / "pubmed" / "articles.parquet")
        pd.DataFrame(hits, columns=["query_id", "pmid"]).to_parquet(
            interim / "pubmed" / "query_hits.parquet"
        )
        for query_id, name in queries.items():
            folder = workspace / "raw" / "pubmed" / "current" / query_id
            folder.mkdir(parents=True)
            query = {"query_id": query_id, "kind": "ingredient", "drug_names": [name]}
            (folder / "query.json").write_text(json.dumps(query), encoding="utf-8")
    names = tmp_path / "ingredients.csv"
    names.write_text("class_id,ingredient\n0,warfarin\n1,metformin\n", encoding="utf-8")
    pairs = tmp_path / "pairs.csv"
    pairs.write_text(
        "ingredient_a,ingredient_b,severity,ddinter_id_a,ddinter_id_b,source_url,license\n"
        "metformin,warfarin,Minor,D1,D2,https://ddinter2.scbdd.com/,CC BY-NC-SA 4.0\n"
        "aspirin,warfarin,Major,D3,D2,https://ddinter2.scbdd.com/,CC BY-NC-SA 4.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(rag_job, "WORKSPACES", (v1, v2))
    monkeypatch.setattr(rag_job, "INGREDIENT_FILES", (names,))
    monkeypatch.setattr(rag_job, "PAIRS", pairs)
    return tmp_path


def test_job_builds_one_corpus_from_both_workspaces(spark, params, workspaces):
    out = workspaces / "corpus_v2"
    summary = rag_job.run(spark, params, out=out)
    chunks = spark.read.parquet(str(out / "chunks"))
    by_type = {r.doc_type: r["count"] for r in chunks.groupBy("doc_type").count().collect()}
    assert by_type == {"label": 2, "abstract": 1, "interaction": 1}
    assert chunks.where(F.col("section").rlike("dosage")).count() == 0
    abstract = chunks.where("doc_type = 'abstract'").first()
    assert abstract.doc_id == "pubmed:30"  # 31 is a near-duplicate of 30
    assert sorted(abstract.ingredients) == ["metformin", "warfarin"]  # both queries found it
    assert summary["articles"]["near_duplicates"] == 1
    assert summary["interaction_pairs"] == 1  # aspirin is not a class ingredient
    assert summary["ingredients_covered"] == 2


def test_job_reads_the_workspaces_listed_in_params(spark, params, workspaces):
    v1 = workspaces / "rag"
    params["rag"].update(
        workspaces=[str(v1)], ingredient_files=[str(workspaces / "ingredients.csv")]
    )
    summary = rag_job.run(spark, params, out=workspaces / "corpus_v1_only")
    assert summary["workspaces"] == [str(v1)]
    assert summary["label_sections"] == {"read": 2, "kept": 1}  # the dosage section is out
    assert summary["articles"]["read"] == 1
    chunks = spark.read.parquet(str(workspaces / "corpus_v1_only" / "chunks"))
    abstract = chunks.where("doc_type = 'abstract'").first()
    assert list(abstract.ingredients) == ["warfarin"]  # only the v1 query found it
