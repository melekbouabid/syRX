"""Write config/ingredient_smiles.csv: a structure (SMILES) for each active ingredient.

cd backend
python scripts/fetch_ingredient_smiles.py      # PubChem PUG-REST, 4 requests/s at most

Used by the toxicity tool to accept ingredient names (docs/TOXICITY.md). One PubChem lookup
by name per ingredient of data/tujya/pill_ingredients.csv; PubChem data is public domain.
An ingredient PubChem does not find (or a mixture such as a multivitamin) gets an empty
SMILES: the tool then says it cannot predict it.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INGREDIENTS = ROOT / "data" / "tujya" / "pill_ingredients.csv"
OUT = ROOT / "config" / "ingredient_smiles.csv"
API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/SMILES/JSON"
PAGE = "https://pubchem.ncbi.nlm.nih.gov/compound/{}"
# PubChem's best match for these names is an ion or a coenzyme; look up the form in the
# pills instead (the same aliases as the openFDA label search, docs/RAG.md).
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
