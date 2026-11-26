"""Step 1 — Extractor: source documents (openFDA labels, PubMed abstracts, DDInter) -> clean text.

The downloads run through the polite connectors of src/data/sources (cache, rate limits,
provenance); each source lands as clean interim tables (label sections, abstracts, DDInter
rows), then `corpus` cuts them into chunks (step 3). Run:
`python -m src.rag.extractor [--set key=value] <command>`, with the commands ddinter, classes,
ingredients (RxNav, step 2), interactions, labels, pubmed, corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd

from src.data.sources import pipeline
from src.data.sources.config import Settings, find_config, load_settings
from src.data.sources.http.client import PoliteClient
from src.data.sources.provenance import read_parquet, write_parquet
from src.rag import chunking
from src.rag.metadata import (
    HOST,
    ddinter_names,
    map_classes,
    norm_name,
    read_aliases,
    read_csv_rows,
    read_rows,
    served_classes,
    summarise,
    update_class_file,
    write_rows,
)

# DDInter is used under its CC BY-NC-SA 4.0 terms (non-commercial student project); the
# shared config.yaml keeps the connector opt-in.
DDINTER_OVERRIDES = [
    "sources.ddinter.enabled=true",
    "sources.ddinter.options.terms_acknowledged=true",
    "sources.ddinter.options.codes=[A, B, C, D, G, H, J, L, M, N, P, R, S, V]",
]
# The RAG sources run through the normal connectors, in their own data folder and manifest,
# so they never mix with the original project's PubMed records under data/raw/pubmed.
RAG_OVERRIDES = [
    "data_dir=data/external/rag_v1",
    "manifest_path=data/external/rag_v1/manifest.json",
]
LABEL_OVERRIDES = ["sources.openfda_label.enabled=true"]
PUBMED_OVERRIDES = [
    "sources.pubmed.options.mode=ingredients",
    "sources.pubmed.options.max_queries=100",
    "sources.pubmed.options.max_records_per_query=20",
    "sources.pubmed.options.batch_size=20",
    "sources.pubmed.options.sort=relevance",
    "sources.pubmed.options.start_date=2000/01/01",
]
# The corpus steps use the data pipeline configuration; TUJYA_CONFIG or a config.yaml in the
# working directory wins.
PIPELINE_CONFIG = Path(__file__).resolve().parents[2] / "config" / "data_pipeline" / "config.yaml"


def _paths(settings: Settings) -> dict[str, Path]:
    root = settings.project_root
    out = root / "data" / "processed" / "pharma"
    return {
        "classes": root / "config" / "pill_classes.csv",
        "classes_v2": root / "config" / "splits" / "c3pi_v2" / "classes.csv",
        "aliases": root / "config" / "ingredient_aliases.csv",
        "ingredients": out / "pill_ingredients.csv",
        "ddinter": root / "data" / "interim" / "ddinter" / "interactions.parquet",
        "between": out / "interactions_between_classes.csv",
        "any": out / "interactions_with_any_drug.csv",
        # Collection workspace of the RAG sources (raw downloads, interim tables, manifest).
        "rag": root / "data" / "external" / "rag_v1",
        "corpus": root / "data" / "processed" / "rag" / "corpus",
    }


def run_classes(settings: Settings) -> None:
    paths = _paths(settings)
    count = served_classes(paths["classes_v2"], paths["classes"])
    print(f"{count} classes written to {paths['classes']}; run `ingredients` next")


async def run_ingredients(settings: Settings) -> None:
    paths = _paths(settings)
    pills = read_csv_rows(paths["classes"])
    async with PoliteClient(settings, allowed_hosts={HOST}) as client:
        rows = await map_classes(client, pills)
    write_rows(rows, paths["ingredients"])
    summary = summarise(rows)
    update_class_file(paths["classes"], summary)
    missing = [p["class_key"] for p in pills if int(p["class_id"]) not in summary]
    by_name = sum(1 for s in summary.values() if s["ingredient_method"] == "name")
    distinct = {r.ingredient for r in rows}
    print(
        f"{len(pills)} classes: {len(summary)} mapped ({by_name} by name), "
        f"{len(missing)} without ingredient {missing}; {len(distinct)} distinct ingredients"
    )


def run_interactions(settings: Settings) -> None:
    paths = _paths(settings)
    names = {r.ingredient for r in read_rows(paths["ingredients"])}
    ddinter = pd.read_parquet(paths["ddinter"])
    aliases = read_aliases(paths["aliases"])
    known = ddinter_names(ddinter, aliases)
    unmatched = sorted(names - known)
    between = filter_interactions(ddinter, names, both=True, aliases=aliases)
    anything = filter_interactions(ddinter, names, both=False, aliases=aliases)
    between.to_csv(paths["between"], index=False)
    anything.to_csv(paths["any"], index=False)
    print(
        f"{len(names)} ingredients, {len(names) - len(unmatched)} found in DDInter; "
        f"not in DDInter: {unmatched}"
    )
    print(f"between classes: {len(between)} pairs {between['severity'].value_counts().to_dict()}")
    print(f"with any drug: {len(anything)} pairs {anything['severity'].value_counts().to_dict()}")


def run_ddinter(settings: Settings) -> None:
    (run,) = asyncio.run(pipeline.fetch(settings, ["ddinter"]))
    print(f"ddinter: {run.outcome} {run.tables} {run.message}")


def run_source(settings: Settings, source: str) -> None:
    (run,) = asyncio.run(pipeline.fetch(settings, [source]))
    print(f"{source}: {run.outcome} {run.tables} {run.stats} {run.message}")


def run_corpus(settings: Settings) -> None:
    paths = _paths(settings)
    rag = paths["rag"]
    names = {r.ingredient for r in read_rows(paths["ingredients"])}
    labels = read_parquet(rag / "interim" / "openfda_label" / "label_sections.parquet")
    articles = read_parquet(rag / "interim" / "pubmed" / "articles.parquet")
    hits = read_parquet(rag / "interim" / "pubmed" / "query_hits.parquet")
    by_pmid = chunking.pmid_ingredients(rag / "raw" / "pubmed", hits) if hits is not None else {}
    pairs = pd.read_csv(paths["between"]) if paths["between"].is_file() else None
    ddinter_date = ""
    ddinter = paths["ddinter"]
    if ddinter.is_file():
        ddinter_date = str(pd.read_parquet(ddinter, columns=["retrieved_at"])["retrieved_at"].max())
    chunks = chunking.build_chunks(labels, articles, by_pmid, pairs, names, ddinter_date)
    out = paths["corpus"]
    write_parquet(chunks, out / "chunks.parquet")
    stats = chunking.summary(chunks)
    (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.rag.extractor")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "command",
        choices=["ddinter", "classes", "ingredients", "interactions", "labels", "pubmed", "corpus"],
    )
    args = parser.parse_args(argv)
    overrides = {
        "ddinter": DDINTER_OVERRIDES,
        "labels": RAG_OVERRIDES + LABEL_OVERRIDES,
        "pubmed": RAG_OVERRIDES + PUBMED_OVERRIDES,
    }.get(args.command, [])
    settings = load_settings(find_config(None) or PIPELINE_CONFIG, [*overrides, *args.set])
    if args.command == "ddinter":
        run_ddinter(settings)
    elif args.command == "classes":
        run_classes(settings)
    elif args.command == "ingredients":
        asyncio.run(run_ingredients(settings))
    elif args.command == "labels":
        run_source(settings, "openfda_label")
    elif args.command == "pubmed":
        run_source(settings, "pubmed")
    elif args.command == "corpus":
        run_corpus(settings)
    else:
        run_interactions(settings)


# --- DDInter interactions of the class ingredients -> clean rows with a sentence each ----------
# DDInter names are matched to RxNorm names after config/ingredient_aliases.csv (step 2). Each
# pair is stored once, sorted; a pair listed twice keeps its most severe level (`conflict`).

DDINTER_URL = "https://ddinter2.scbdd.com/"
DDINTER_CITATION = (
    "Tian Y, et al. DDInter 2.0: an enhanced drug interaction resource with expanded data "
    "coverage, new interaction types, and improved user interface. Nucleic Acids Res. "
    "2025;53(D1):D1356-62. doi:10.1093/nar/gkae726"
)
SEVERITY_ORDER = {"Major": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}
COLUMNS = [
    "ingredient_a",
    "ingredient_b",
    "pair",
    "severity",
    "description",
    "conflict",
    "ddinter_id_a",
    "ddinter_id_b",
    "source",
    "source_url",
    "license",
]


def filter_interactions(
    ddinter: pd.DataFrame,
    ingredients: Iterable[str],
    *,
    both: bool = True,
    aliases: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Interactions where both drugs (or, with both=False, at least one) are in `ingredients`."""
    wanted = {norm_name(i) for i in ingredients}
    aliases = aliases or {}

    def name(value: object) -> str:
        return aliases.get(norm_name(value), norm_name(value))

    frame = pd.DataFrame(
        {
            "a": ddinter["drug_a_name"].map(name),
            "b": ddinter["drug_b_name"].map(name),
            "id_a": ddinter["drug_a_source_id"].astype("string"),
            "id_b": ddinter["drug_b_source_id"].astype("string"),
            "level": ddinter["level"].fillna("Unknown").astype("string").str.strip(),
        }
    )
    in_a, in_b = frame["a"].isin(wanted), frame["b"].isin(wanted)
    frame = frame[(in_a & in_b) if both else (in_a | in_b)]
    frame = frame[frame["a"] != frame["b"]]
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)

    swap = frame["a"] > frame["b"]
    ordered = pd.DataFrame(
        {
            "ingredient_a": frame["a"].where(~swap, frame["b"]),
            "ingredient_b": frame["b"].where(~swap, frame["a"]),
            "ddinter_id_a": frame["id_a"].where(~swap, frame["id_b"]),
            "ddinter_id_b": frame["id_b"].where(~swap, frame["id_a"]),
            "severity": frame["level"],
        }
    )
    ordered["rank"] = ordered["severity"].map(SEVERITY_ORDER).fillna(0)
    levels = ordered.groupby(["ingredient_a", "ingredient_b"])["severity"].nunique()
    best = (
        ordered.sort_values("rank", ascending=False)
        .drop_duplicates(["ingredient_a", "ingredient_b"])
        .drop(columns="rank")
        .sort_values(["ingredient_a", "ingredient_b"])
        .reset_index(drop=True)
    )
    best["conflict"] = [
        bool(levels[(a, b)] > 1)
        for a, b in zip(best["ingredient_a"], best["ingredient_b"], strict=True)
    ]
    best["pair"] = best["ingredient_a"] + " + " + best["ingredient_b"]
    best["description"] = [
        f"DDInter 2.0 lists an interaction between {a} and {b} with severity {s}."
        for a, b, s in zip(
            best["ingredient_a"], best["ingredient_b"], best["severity"], strict=True
        )
    ]
    best["source"] = "DDInter 2.0"
    best["source_url"] = DDINTER_URL
    best["license"] = "CC BY-NC-SA 4.0"
    return best[COLUMNS]


if __name__ == "__main__":
    main()
