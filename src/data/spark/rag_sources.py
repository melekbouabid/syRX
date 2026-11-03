"""Sources of the larger RAG corpus (module 2) for the classes of the C3PI dataset v2.

    python -m src.data.spark.rag_sources ingredients   # v2 classes -> RxNav -> ingredients
    python -m src.data.spark.rag_sources labels        # openFDA, one label per ingredient
    python -m src.data.spark.rag_sources pubmed        # PubMed, up to 40 per ingredient

Run from the repository root with the project venv (.venv): network access goes through the
polite client of the data pipeline (src/data/tujya_data: rate limits per host, retries,
cache, manifest). Everything is written to its own workspace, data/rag_v2 (raw, interim,
manifest.json, DVC), so the corpus of v1 (data/rag, stage `rag_corpus`) is left as it is.
The Spark job `rag` (src/data/spark/rag_job.py) then cleans, deduplicates and chunks both.

- ingredients: the classes of config/splits/c3pi_v2/classes.csv. Classes already in v1 keep
  their RxNav mapping (data/processed/pharma/pill_ingredients.csv); the new ones are mapped
  like v1 (NDC -> RxCUI -> ingredients, src/data/corpus/ingredients.py). Writes
  data/rag_v2/ingredients.csv (one row per class and ingredient) and
  data/rag_v2/class_ingredients.csv (`active_ingredients` per class, the input of the two
  connectors).
- labels / pubmed: the openFDA and PubMed connectors with that ingredient file; PubMed keeps
  up to 40 abstracts per ingredient (v1: 20).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from src.data.corpus import ingredients as ing
from src.data.tujya_data import pipeline
from src.data.tujya_data.config import Settings, load_settings
from src.data.tujya_data.http.client import PoliteClient

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config" / "data_pipeline" / "config.yaml"
CLASSES_V2 = ROOT / "config" / "splits" / "c3pi_v2" / "classes.csv"
INGREDIENTS_V1 = ROOT / "data" / "processed" / "pharma" / "pill_ingredients.csv"
WORKSPACE = "data/rag_v2"
INGREDIENTS = ROOT / WORKSPACE / "ingredients.csv"
CLASS_INGREDIENTS = ROOT / WORKSPACE / "class_ingredients.csv"
PUBMED_PER_INGREDIENT = 40
OVERRIDES = [
    f"data_dir={WORKSPACE}",
    f"manifest_path={WORKSPACE}/manifest.json",
]
SOURCE_OVERRIDES = {
    "openfda_label": [
        "sources.openfda_label.enabled=true",
        f"sources.openfda_label.options.ingredients_file={WORKSPACE}/class_ingredients.csv",
    ],
    "pubmed": [
        "sources.pubmed.options.mode=ingredients",
        f"sources.pubmed.options.ingredients_file={WORKSPACE}/class_ingredients.csv",
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
    print(
        f"{len(classes)} classes ({len(classes) - len(missing)} from v1, {len(missing)} mapped "
        f"with RxNav): {mapped} with ingredients, {len(names)} distinct ingredients"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.data.spark.rag_sources")
    parser.add_argument("command", choices=["ingredients", "labels", "pubmed"])
    args = parser.parse_args(argv)
    source = {"labels": "openfda_label", "pubmed": "pubmed"}.get(args.command, "")
    settings = load_settings(CONFIG, [*OVERRIDES, *SOURCE_OVERRIDES.get(source, [])])
    if args.command == "ingredients":
        asyncio.run(run_ingredients(settings))
        return
    (run,) = asyncio.run(pipeline.fetch(settings, [source]))
    print(f"{source}: {run.outcome} {run.tables} {run.stats} {run.message}")


if __name__ == "__main__":
    main()
