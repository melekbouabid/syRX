"""Tujya data steps: `python -m synergyrx_data.tujya [--set key=value] <command>`.

    ddinter        DDInter 2.0, all 14 ATC files, terms acknowledged for this academic
                   project (CC BY-NC-SA 4.0); SynergyRx's own config keeps it opt-in
    ingredients    config/pill_classes.csv -> RxNav -> data/tujya/pill_ingredients.csv,
                   and the ingredient columns of config/pill_classes.csv
    interactions   DDInter (run `synergyrx fetch --source ddinter` first) ->
                   data/tujya/interactions_between_classes.csv (both drugs are class ingredients)
                   data/tujya/interactions_with_any_drug.csv (at least one is)
    labels         openFDA drug labels, one per class ingredient -> data/rag (raw, interim)
    pubmed         PubMed abstracts on interactions and adverse effects, up to 20 per
                   ingredient -> data/rag (raw, interim)
    corpus         labels + abstracts + DDInter pairs -> data/rag/corpus/chunks.parquet

Docs: docs/INTERACTIONS.md, docs/RAG.md.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path

import pandas as pd

from synergyrx_data import pipeline
from synergyrx_data.config import Settings, load_settings
from synergyrx_data.http.client import PoliteClient
from synergyrx_data.provenance import read_parquet, write_parquet
from synergyrx_data.tujya import corpus
from synergyrx_data.tujya import ingredients as ing
from synergyrx_data.tujya.interactions import ddinter_names, filter_interactions, read_aliases

# Tujya uses DDInter under its CC BY-NC-SA 4.0 terms (non-commercial student project, owner's
# decision of 2026-10-27). The shared config.yaml keeps the connector opt-in.
DDINTER_OVERRIDES = [
    "sources.ddinter.enabled=true",
    "sources.ddinter.options.terms_acknowledged=true",
    "sources.ddinter.options.codes=[A, B, C, D, G, H, J, L, M, N, P, R, S, V]",
]
# The RAG sources run through the normal connectors, in their own data folder and manifest,
# so they never mix with the SynergyRx PubMed records under data/raw/pubmed.
RAG_OVERRIDES = ["data_dir=data/rag", "manifest_path=data/rag/manifest.json"]
LABEL_OVERRIDES = ["sources.openfda_label.enabled=true"]
PUBMED_OVERRIDES = [
    "sources.pubmed.options.mode=ingredients",
    "sources.pubmed.options.max_queries=100",
    "sources.pubmed.options.max_records_per_query=20",
    "sources.pubmed.options.batch_size=20",
    "sources.pubmed.options.sort=relevance",
    "sources.pubmed.options.start_date=2000/01/01",
]
CLASS_COLUMNS = ["rxcui", "active_ingredients", "ingredient_rxcuis", "ingredient_method"]


def _paths(settings: Settings) -> dict[str, Path]:
    root = settings.project_root
    out = root / "data" / "tujya"
    return {
        "classes": root / "config" / "pill_classes.csv",
        "aliases": root / "config" / "ingredient_aliases.csv",
        "ingredients": out / "pill_ingredients.csv",
        "ddinter": root / "data" / "interim" / "ddinter" / "interactions.parquet",
        "between": out / "interactions_between_classes.csv",
        "any": out / "interactions_with_any_drug.csv",
        "rag": root / "data" / "rag",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def update_class_file(path: Path, summary: dict[int, dict[str, str]]) -> None:
    """Add or refresh the ingredient columns of config/pill_classes.csv, keeping the rest."""
    rows = _read_csv(path)
    names = [c for c in rows[0] if c not in CLASS_COLUMNS] + CLASS_COLUMNS if rows else []
    empty = dict.fromkeys(CLASS_COLUMNS, "")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **summary.get(int(row["class_id"]), empty)})


async def run_ingredients(settings: Settings) -> None:
    paths = _paths(settings)
    pills = _read_csv(paths["classes"])
    async with PoliteClient(settings, allowed_hosts={ing.HOST}) as client:
        rows = await ing.map_classes(client, pills)
    ing.write_rows(rows, paths["ingredients"])
    summary = ing.summarise(rows)
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
    names = {r.ingredient for r in ing.read_rows(paths["ingredients"])}
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
    names = {r.ingredient for r in ing.read_rows(paths["ingredients"])}
    labels = read_parquet(rag / "interim" / "openfda_label" / "label_sections.parquet")
    articles = read_parquet(rag / "interim" / "pubmed" / "articles.parquet")
    hits = read_parquet(rag / "interim" / "pubmed" / "query_hits.parquet")
    by_pmid = corpus.pmid_ingredients(rag / "raw" / "pubmed", hits) if hits is not None else {}
    pairs = pd.read_csv(paths["between"]) if paths["between"].is_file() else None
    ddinter_date = ""
    ddinter = paths["ddinter"]
    if ddinter.is_file():
        ddinter_date = str(pd.read_parquet(ddinter, columns=["retrieved_at"])["retrieved_at"].max())
    chunks = corpus.build_chunks(labels, articles, by_pmid, pairs, names, ddinter_date)
    out = rag / "corpus"
    write_parquet(chunks, out / "chunks.parquet")
    stats = corpus.summary(chunks)
    (out / "summary.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m synergyrx_data.tujya")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "command",
        choices=["ddinter", "ingredients", "interactions", "labels", "pubmed", "corpus"],
    )
    args = parser.parse_args(argv)
    overrides = {
        "ddinter": DDINTER_OVERRIDES,
        "labels": RAG_OVERRIDES + LABEL_OVERRIDES,
        "pubmed": RAG_OVERRIDES + PUBMED_OVERRIDES,
    }.get(args.command, [])
    settings = load_settings(None, [*overrides, *args.set])
    if args.command == "ddinter":
        run_ddinter(settings)
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


if __name__ == "__main__":
    main()
