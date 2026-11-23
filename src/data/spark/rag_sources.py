"""Sources of the larger RAG corpus (module 2) for the classes of the C3PI dataset v2.

    python -m src.data.spark.rag_sources ingredients   # v2 classes -> RxNav -> ingredients
    python -m src.data.spark.rag_sources labels        # openFDA, one label per ingredient
    python -m src.data.spark.rag_sources pubmed        # PubMed, up to 40 per ingredient

Run from the repository root with the project venv (.venv): network access goes through the
polite client of the data pipeline (src/data/sources: rate limits per host, retries,
cache, manifest). Everything is written to its own workspace, data/external/rag_v2 (raw,
interim, manifest.json, DVC), so the corpus of v1 (data/external/rag_v1, stage `rag_corpus`)
is left as it is.
The Spark job `rag` (src/data/spark/rag_job.py) then cleans, deduplicates and chunks both.

- ingredients: the classes of config/splits/c3pi_v2/classes.csv. Classes already in v1 keep
  their RxNav mapping (data/processed/pharma/pill_ingredients.csv); the new ones are mapped
  like v1 (NDC -> RxCUI -> ingredients, src/rag/corpus.py). Writes
  data/external/rag_v2/ingredients.csv (one row per class and ingredient) and
  data/external/rag_v2/class_ingredients.csv (`active_ingredients` per class, the input of the two
  connectors).
  It also writes data/external/rag_v2/new_ingredients.csv: the ingredients that no class of v1
  (config/pill_classes_v1.csv) has, i.e. those without any document in the v1 corpus.
- labels / pubmed: the openFDA and PubMed connectors with the new ingredients only (the
  ingredients of v1 keep their label and abstracts of data/external/rag_v1: no second version of
  a label,
  and the expected chunks of the benchmarks stay the documents they were written on); PubMed
  keeps up to 40 abstracts per ingredient (v1: 20).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from src.data.sources import pipeline
from src.data.sources.config import Settings, load_settings
from src.data.sources.http.client import PoliteClient
from src.data.sources.literature.queries import read_ingredients
from src.data.sources.logging_utils import setup_logging
from src.rag import corpus as ing

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "data_pipeline" / "config.yaml"
CLASSES_V2 = ROOT / "config" / "splits" / "c3pi_v2" / "classes.csv"
INGREDIENTS_V1 = ROOT / "data" / "processed" / "pharma" / "pill_ingredients.csv"
CLASSES_V1 = ROOT / "config" / "pill_classes_v1.csv"
WORKSPACE = "data/external/rag_v2"
INGREDIENTS = ROOT / WORKSPACE / "ingredients.csv"
CLASS_INGREDIENTS = ROOT / WORKSPACE / "class_ingredients.csv"
NEW_INGREDIENTS = ROOT / WORKSPACE / "new_ingredients.csv"
PUBMED_PER_INGREDIENT = 40
# Names of the class tables that are no medicine (the placebo tablets of a pack): never searched
# (PubMed returns pesticide studies for them); params.yaml bigdata.rag.not_drugs, same list.
NOT_DRUGS = frozenset({"inert ingredients"})
# PubMed terms for a name it does not know as written (RxNorm adds the pharmacopoeia).
PUBMED_ALIASES = {
    "omega-3 acid ethyl esters (usp)": [
        "omega-3-acid ethyl esters",
        "omega-3 fatty acid ethyl esters",
    ]
}
OVERRIDES = [
    f"data_dir={WORKSPACE}",
    f"manifest_path={WORKSPACE}/manifest.json",
]
SOURCE_OVERRIDES = {
    "openfda_label": [
        "sources.openfda_label.enabled=true",
        f"sources.openfda_label.options.ingredients_file={WORKSPACE}/new_ingredients.csv",
    ],
    "pubmed": [
        "sources.pubmed.options.mode=ingredients",
        f"sources.pubmed.options.ingredients_file={WORKSPACE}/new_ingredients.csv",
        f"sources.pubmed.options.ingredient_aliases={json.dumps(PUBMED_ALIASES)}",
        "sources.pubmed.options.max_queries=1000",
        f"sources.pubmed.options.max_records_per_query={PUBMED_PER_INGREDIENT}",
        f"sources.pubmed.options.batch_size={PUBMED_PER_INGREDIENT}",
        "sources.pubmed.options.sort=relevance",
        "sources.pubmed.options.start_date=2000/01/01",
    ],
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reuse_v1(
    classes: list[dict[str, str]], v1_rows: list[ing.IngredientRow]
) -> tuple[list[ing.IngredientRow], list[dict[str, str]]]:
    """Rows of the classes already mapped in v1 (renumbered), and the classes still to map."""
    by_key: dict[str, list[ing.IngredientRow]] = defaultdict(list)
    for row in v1_rows:
        by_key[row.class_key].append(row)
    known: list[ing.IngredientRow] = []
    missing: list[dict[str, str]] = []
    for pill in classes:
        rows = by_key.get(pill["class_key"])
        if rows:
            class_id = int(pill["class_id"])
            known.extend(replace(r, class_id=class_id) for r in rows)
        else:
            missing.append(pill)
    return known, missing


def write_class_ingredients(rows: list[ing.IngredientRow], classes: list[dict[str, str]]) -> int:
    """class_key + `active_ingredients` (";"-joined), the format the connectors read."""
    summary = ing.summarise(rows)
    with CLASS_INGREDIENTS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, ["class_id", "class_key", "active_ingredients"], lineterminator="\n"
        )
        writer.writeheader()
        for pill in classes:
            found = summary.get(int(pill["class_id"]), {})
            writer.writerow(
                {
                    "class_id": pill["class_id"],
                    "class_key": pill["class_key"],
                    "active_ingredients": found.get("active_ingredients", ""),
                }
            )
    return len(summary)


def write_new_ingredients(rows: list[ing.IngredientRow], v1_names: set[str]) -> list[str]:
    """The ingredients of the v2 classes that no v1 class has, one per row (`active_ingredients`,
    the column the connectors read); the names of `NOT_DRUGS` are left out."""
    known = {n.lower() for n in v1_names} | NOT_DRUGS
    new = sorted({r.ingredient.lower() for r in rows} - known)
    with NEW_INGREDIENTS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["active_ingredients"])
        writer.writerows([name] for name in new)
    return new


async def run_ingredients(settings: Settings) -> None:
    classes = _read_csv(CLASSES_V2)
    v1 = ing.read_rows(INGREDIENTS_V1) if INGREDIENTS_V1.is_file() else []
    known, missing = reuse_v1(classes, v1)
    async with PoliteClient(settings, allowed_hosts={ing.HOST}) as client:
        new = await ing.map_classes(client, missing)
    rows = sorted(known + new, key=lambda r: (r.class_id, r.ingredient))
    INGREDIENTS.parent.mkdir(parents=True, exist_ok=True)
    ing.write_rows(rows, INGREDIENTS)
    mapped = write_class_ingredients(rows, classes)
    names = {r.ingredient for r in rows}
    new = write_new_ingredients(rows, set(read_ingredients(CLASSES_V1)))
    print(
        f"{len(classes)} classes ({len(classes) - len(missing)} from v1, {len(missing)} mapped "
        f"with RxNav): {mapped} with ingredients, {len(names)} distinct ingredients, "
        f"{len(new)} not in v1"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.data.spark.rag_sources")
    parser.add_argument("command", choices=["ingredients", "labels", "pubmed"])
    args = parser.parse_args(argv)
    source = {"labels": "openfda_label", "pubmed": "pubmed"}.get(args.command, "")
    settings = load_settings(CONFIG, [*OVERRIDES, *SOURCE_OVERRIDES.get(source, [])])
    # Same logging as `tujya-data`: the contact address and the API key are masked everywhere.
    setup_logging(
        settings.logging.level,
        json_logs=settings.logging.json_logs,
        secrets=settings.secrets.values_to_redact(),
    )
    if args.command == "ingredients":
        asyncio.run(run_ingredients(settings))
        return
    (run,) = asyncio.run(pipeline.fetch(settings, [source]))
    print(f"{source}: {run.outcome} {run.tables} {run.stats} {run.message}")


if __name__ == "__main__":
    main()
