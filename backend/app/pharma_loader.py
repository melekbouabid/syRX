"""Load the pill classes, ingredients and DDInter interactions into the pharma tables.

    python -m app.pharma_loader                       # database from SYNERGYRX_PHARMA_DATABASE_URL
    python -m app.pharma_loader --database-url postgresql+psycopg://tujya:tujya@localhost:5432/tujya

Inputs (built by the data steps, see docs/INTERACTIONS.md):
- config/pill_classes.csv                        one row per identification class
- data/tujya/pill_ingredients.csv                class -> ingredient (RxNav)
- data/tujya/interactions_with_any_drug.csv      DDInter pairs involving a class ingredient

The load is a full refresh: the five tables are dropped and created again from the models
in app/pharma.py (so a new column reaches an existing database), then filled in one
transaction. Running it twice gives the same database. Nothing else lives in these tables:
they are rebuilt from the CSV files above, which DVC versions.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from app.config import PROJECT_DIR, Settings
from app.db import make_engine
from app.pharma import Ingredient, Interaction, PharmaBase, Pill, PillIngredient, Source

SOURCES = {
    "c3pi": {
        "title": "NLM Computational Photography Project for Pill Identification (C3PI)",
        "url": "https://data.lhncbc.nlm.nih.gov/public/Pills/",
        "license": "ODbL, research and development use only",
        "citation": "U.S. National Library of Medicine, C3PI pill image collection.",
    },
    "rxnav": {
        "title": "RxNav / RxNorm (NLM)",
        "url": "https://lhncbc.nlm.nih.gov/RxNav/APIs/",
        "license": "RxNorm terms of use (UMLS licence for some sources)",
        "citation": "U.S. National Library of Medicine, RxNorm.",
    },
    "ddinter": {
        "title": "DDInter 2.0 drug-drug interactions",
        "url": "https://ddinter2.scbdd.com/",
        "license": "CC BY-NC-SA 4.0",
        "citation": (
            "Tian Y, et al. DDInter 2.0: an enhanced drug interaction resource with expanded "
            "data coverage, new interaction types, and improved user interface. Nucleic Acids "
            "Res. 2025;53(D1):D1356-62. doi:10.1093/nar/gkae726"
        ),
    },
}


@dataclass(frozen=True)
class Inputs:
    classes: Path
    ingredients: Path
    interactions: Path

    @classmethod
    def default(cls, root: Path = PROJECT_DIR) -> Inputs:
        return cls(
            classes=root / "config" / "pill_classes.csv",
            ingredients=root / "data" / "tujya" / "pill_ingredients.csv",
            interactions=root / "data" / "tujya" / "interactions_with_any_drug.csv",
        )


@dataclass(frozen=True)
class LoadReport:
    pills: int
    ingredients: int
    pill_ingredients: int
    interactions: int


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _int(value: str | None) -> int:
    return int(value) if value else 0


def _upsert_sources(db: Session) -> dict[str, Source]:
    found = {s.name: s for s in db.scalars(select(Source))}
    for name, values in SOURCES.items():
        source = found.get(name)
        if source is None:
            source = Source(name=name, **values)
            db.add(source)
            found[name] = source
        else:
            for key, value in values.items():
                setattr(source, key, value)
    db.flush()
    return found


def _ingredient_table(
    db: Session, mapped: Iterable[dict[str, str]], interactions: Iterable[dict[str, str]]
) -> dict[str, Ingredient]:
    by_name: dict[str, Ingredient] = {}
    for row in mapped:
        name = row["ingredient"].strip().lower()
        if name and name not in by_name:
            by_name[name] = Ingredient(name=name, rxcui=row.get("ingredient_rxcui") or None)
    for row in interactions:
        for side in ("a", "b"):
            name = row[f"ingredient_{side}"].strip().lower()
            ddinter_id = row.get(f"ddinter_id_{side}") or None
            ingredient = by_name.setdefault(name, Ingredient(name=name))
            if ingredient.ddinter_id is None:
                ingredient.ddinter_id = ddinter_id
    db.add_all(by_name.values())
    db.flush()
    return by_name


def load(engine: Engine, inputs: Inputs) -> LoadReport:
    # Read every input first, so a missing file leaves the current tables untouched.
    classes = _rows(inputs.classes)
    mapped = _rows(inputs.ingredients)
    interactions = _rows(inputs.interactions)
    PharmaBase.metadata.drop_all(engine)
    PharmaBase.metadata.create_all(engine)
    with Session(engine) as db, db.begin():
        sources = _upsert_sources(db)
        ingredients = _ingredient_table(db, mapped, interactions)

        pills: dict[int, Pill] = {}
        for row in classes:
            pill = Pill(
                class_id=int(row["class_id"]),
                class_key=row["class_key"],
                ndc9=row["ndc9"],
                part=_int(row.get("part")) or 1,
                ndc11_list=row.get("ndc11_list", ""),
                name=row["name"],
                generic_name=row.get("generic_name", ""),
                labeled_by=row.get("labeled_by", ""),
                imprint=row.get("imprint", ""),
                color=row.get("color", ""),
                shape=row.get("shape", ""),
                n_images=_int(row.get("n_consumer_selected"))
                + _int(row.get("n_reference_selected")),
                source=sources["c3pi"],
            )
            pills[pill.class_id] = pill
        db.add_all(pills.values())
        db.flush()

        links: set[tuple[int, str]] = set()
        for row in mapped:
            key = (int(row["class_id"]), row["ingredient"].strip().lower())
            if key in links or key[0] not in pills:
                continue
            links.add(key)
            db.add(
                PillIngredient(
                    pill=pills[key[0]], ingredient=ingredients[key[1]], source=sources["rxnav"]
                )
            )

        pairs: set[tuple[int, int]] = set()
        for row in interactions:
            a = ingredients[row["ingredient_a"].strip().lower()]
            b = ingredients[row["ingredient_b"].strip().lower()]
            if a.id == b.id:
                continue
            first, second = (a, b) if a.id < b.id else (b, a)
            if (first.id, second.id) in pairs:
                continue
            pairs.add((first.id, second.id))
            db.add(
                Interaction(
                    ingredient_a=first,
                    ingredient_b=second,
                    severity=row["severity"] or "Unknown",
                    description=row.get("description", ""),
                    source=sources["ddinter"],
                )
            )
    return LoadReport(len(pills), len(ingredients), len(links), len(pairs))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m app.pharma_loader")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args(argv)
    url = args.database_url or Settings().pharma_database_url
    engine = make_engine(url)
    try:
        report = load(engine, Inputs.default())
    finally:
        engine.dispose()
    print(f"loaded into {engine.url.render_as_string(hide_password=True)}: {report}")


if __name__ == "__main__":
    main()
