"""Step 2 — Metadata: the names that tie every document to the pill classes.

Each pill class gets its active ingredients from RxNav (NDC -> RxCUI -> base ingredients, or the
generic name as a fallback), so a label, an abstract or a DDInter row can carry the
`ingredients` it is about; DDInter names are matched to RxNorm names through
config/ingredient_aliases.csv. The other fields of each chunk (brand in the title, source,
section, language, date/year, url, version, licence) are set when it is cut, by
src/rag/chunking.py (`CHUNK_COLUMNS`), and become the Qdrant payload (src/rag/vector_store.py).
"""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.sources.http.client import PoliteClient

CLASS_COLUMNS = ["rxcui", "active_ingredients", "ingredient_rxcuis", "ingredient_method"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def update_class_file(path: Path, summary: dict[int, dict[str, str]]) -> None:
    """Add or refresh the ingredient columns of config/pill_classes.csv, keeping the rest."""
    rows = read_csv_rows(path)
    names = [c for c in rows[0] if c not in CLASS_COLUMNS] + CLASS_COLUMNS if rows else []
    empty = dict.fromkeys(CLASS_COLUMNS, "")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **summary.get(int(row["class_id"]), empty)})


def served_classes(source: Path, path: Path) -> int:
    """Write the class table of the API from the v2 class list: same ids, same columns, plus
    empty ingredient columns (filled by `ingredients`). Returns the number of classes."""
    rows = read_csv_rows(source)
    names = [c for c in rows[0] if c not in CLASS_COLUMNS] + CLASS_COLUMNS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({**dict.fromkeys(CLASS_COLUMNS, ""), **row})
    return len(rows)


# --- Active ingredients of the pill classes (RxNav) --------------------------------------------
# NDC -> RxCUI (`ndcstatus`) -> base ingredients (`historystatus`), not salts, never strengths.
# Without a resolved NDC the generic name is looked up (`approximateTerm`, `method=name`).

RXNAV = "https://rxnav.nlm.nih.gov/REST"
HOST = "rxnav.nlm.nih.gov"
SOURCE = "rxnav"


@dataclass(frozen=True)
class IngredientRow:
    """One ingredient of one class, with how it was found."""

    class_id: int
    class_key: str
    ndc11: str  # the NDC that resolved, empty for method=name
    ndc_status: str  # RxNav status of that NDC: ACTIVE, OBSOLETE, ALIEN, UNKNOWN
    product_rxcui: str
    product_name: str  # RxNorm name of the product, e.g. "glimepiride 4 MG Oral Tablet"
    ingredient_rxcui: str
    ingredient: str  # lower case RxNorm IN
    method: str  # ndc | name


async def _json(client: PoliteClient, path: str, params: dict[str, str] | None = None) -> Any:
    result = await client.fetch(f"{RXNAV}/{path}", source=SOURCE, params=params)
    return result.json()


async def ndc_product(client: PoliteClient, ndc11: str) -> tuple[str, str, str]:
    """(rxcui, status, concept name) of a package NDC; rxcui is "" when RxNorm has none."""
    status = (await _json(client, "ndcstatus.json", {"ndc": ndc11})).get("ndcStatus") or {}
    rxcui = status.get("rxcui") or ""
    if not rxcui:
        for entry in status.get("ndcHistory") or []:
            rxcui = entry.get("activeRxcui") or entry.get("originalRxcui") or ""
            if rxcui:
                break
    return rxcui, status.get("status") or "UNKNOWN", status.get("conceptName") or ""


async def product_ingredients(
    client: PoliteClient, rxcui: str
) -> tuple[str, list[tuple[str, str]]]:
    """(product name, [(ingredient rxcui, ingredient name)]) from the RxCUI's history status."""
    body = await _json(client, f"rxcui/{rxcui}/historystatus.json")
    history = body.get("rxcuiStatusHistory") or {}
    name = (history.get("attributes") or {}).get("name") or ""
    features = (history.get("definitionalFeatures") or {}).get("ingredientAndStrength") or []
    found = {(f["baseRxcui"], f["baseName"].lower()) for f in features if f.get("baseRxcui")}
    if not found:
        derived = (history.get("derivedConcepts") or {}).get("ingredientConcept") or []
        found = {
            (d["ingredientRxcui"], d["ingredientName"].lower())
            for d in derived
            if d.get("ingredientRxcui")
        }
    return name, sorted(found, key=lambda pair: pair[1])


def clean_generic_name(text: str) -> str:
    """'Warfarin Sodium Tablets, USP' -> 'Warfarin Sodium'. Only for the name fallback."""
    text = re.sub(
        r"\b(tablets?|capsules?|usp|us|oral|film[- ]coated|extended[- ]release)\b",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    return re.sub(r"[\s,]+", " ", text).strip()


async def name_ingredients(client: PoliteClient, generic: str) -> list[tuple[str, str]]:
    term = clean_generic_name(generic)
    if not term:
        return []
    body = await _json(client, "approximateTerm.json", {"term": term, "maxEntries": "1"})
    candidates = (body.get("approximateGroup") or {}).get("candidate") or []
    if not candidates:
        return []
    _, found = await product_ingredients(client, candidates[0]["rxcui"])
    return found


async def map_class(client: PoliteClient, pill: dict[str, str]) -> list[IngredientRow]:
    """All ingredient rows of one row of config/pill_classes.csv."""
    class_id, key = int(pill["class_id"]), pill["class_key"]
    rows: list[IngredientRow] = []
    for ndc11 in filter(None, pill["ndc11_list"].split(";")):
        rxcui, status, concept = await ndc_product(client, ndc11)
        if not rxcui:
            continue
        name, found = await product_ingredients(client, rxcui)
        rows.extend(
            IngredientRow(class_id, key, ndc11, status, rxcui, concept or name, ing_cui, ing, "ndc")
            for ing_cui, ing in found
        )
    if not rows:
        for ing_cui, ing in await name_ingredients(client, pill.get("generic_name", "")):
            rows.append(IngredientRow(class_id, key, "", "", "", "", ing_cui, ing, "name"))
    return rows


async def map_classes(client: PoliteClient, pills: Iterable[dict[str, str]]) -> list[IngredientRow]:
    rows: list[IngredientRow] = []
    for pill in pills:
        rows.extend(await map_class(client, pill))
    return rows


def summarise(rows: Iterable[IngredientRow]) -> dict[int, dict[str, str]]:
    """Per class: one product RxCUI, the ingredient names and RxCUIs ("; "-joined), method."""
    first: dict[int, IngredientRow] = {}
    names: dict[int, dict[str, str]] = {}
    for row in rows:
        first.setdefault(row.class_id, row)
        names.setdefault(row.class_id, {})[row.ingredient] = row.ingredient_rxcui
    return {
        class_id: {
            "rxcui": first[class_id].product_rxcui,
            "active_ingredients": "; ".join(sorted(found)),
            "ingredient_rxcuis": "; ".join(found[name] for name in sorted(found)),
            "ingredient_method": first[class_id].method,
        }
        for class_id, found in names.items()
    }


def write_rows(rows: Iterable[IngredientRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(IngredientRow)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def read_rows(path: Path) -> list[IngredientRow]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [IngredientRow(int(row.pop("class_id")), **row) for row in csv.DictReader(handle)]


# --- DDInter names -> RxNorm ingredient names (config/ingredient_aliases.csv) -----------------


def norm_name(name: object) -> str:
    return " ".join(str(name).lower().split())


def read_aliases(path: Path) -> dict[str, str]:
    """{normalised DDInter name: RxNorm ingredient} from config/ingredient_aliases.csv."""
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype="string")
    return {
        norm_name(d): norm_name(r)
        for r, d in zip(table["rxnorm_ingredient"], table["ddinter_name"], strict=True)
    }


def ddinter_names(ddinter: pd.DataFrame, aliases: Mapping[str, str] | None = None) -> set[str]:
    aliases = aliases or {}
    names = pd.concat([ddinter["drug_a_name"], ddinter["drug_b_name"]])
    return {aliases.get(norm_name(n), norm_name(n)) for n in names}
