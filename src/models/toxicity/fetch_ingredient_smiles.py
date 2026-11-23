"""Write config/ingredient_smiles.csv: one PubChem SMILES per active ingredient (empty if unknown).

Run: `python -m src.models.toxicity.fetch_ingredient_smiles`.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
INGREDIENTS = ROOT / "data" / "processed" / "pharma" / "pill_ingredients.csv"
OUT = ROOT / "config" / "ingredient_smiles.csv"
API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/SMILES/JSON"
PAGE = "https://pubchem.ncbi.nlm.nih.gov/compound/{}"
# PubChem's best match for these names is an ion or a coenzyme; look up the form in the
# pills instead (the same aliases as the openFDA label search, docs/module2/rag.md).
LOOKUP_ALIASES = {
    "valproate": "valproic acid",
    "vitamin b12": "cyanocobalamin",
    "vitamin b6": "pyridoxine",
}


def lookup(name: str) -> tuple[str, str]:
    url = API.format(urllib.parse.quote(name))
    request = urllib.request.Request(  # noqa: S310 fixed https host
        url, headers={"User-Agent": "tujya-student-project"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 fixed host
            props = json.load(response)["PropertyTable"]["Properties"][0]
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return "", ""
        raise
    return str(props["CID"]), str(props.get("SMILES", ""))


def main() -> None:
    with INGREDIENTS.open(encoding="utf-8", newline="") as f:
        names = sorted(
            {r["ingredient"].strip().lower() for r in csv.DictReader(f) if r["ingredient"]}
        )
    rows = []
    for name in names:
        cid, smiles = lookup(LOOKUP_ALIASES.get(name, name))
        rows.append(
            {
                "ingredient": name,
                "pubchem_cid": cid,
                "smiles": smiles,
                "source_url": PAGE.format(cid) if cid else "",
                "retrieved_at": time.strftime("%Y-%m-%d"),
            }
        )
        print(f"{name:30s} {cid or '-':>10s} {smiles[:60]}")
        time.sleep(0.25)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    found = sum(1 for r in rows if r["smiles"])
    print(f"{found}/{len(rows)} ingredients with a structure -> {OUT}")


if __name__ == "__main__":
    main()
