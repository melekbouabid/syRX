# Pill ingredients and interactions

How each identification class gets its active ingredient(s), and how the interactions
between those ingredients are built. Run on 2026-10-28.

```
config/pill_classes.csv (NDC11 list per class)
  -> RxNav ndcstatus      NDC11 -> product RxCUI (also obsolete NDCs, via their history)
  -> RxNav historystatus  RxCUI -> base ingredients (RxNorm IN), strengths dropped
  -> data/tujya/pill_ingredients.csv  + ingredient columns in config/pill_classes.csv
  -> DDInter 2.0 (14 ATC files, `python -m synergyrx_data.tujya ddinter`)
  -> config/ingredient_aliases.csv (3 DDInter names that differ from RxNorm)
  -> data/tujya/interactions_between_classes.csv, data/tujya/interactions_with_any_drug.csv
```

Commands (pipeline venv `.venv-dev`, from the repository root):

```bash
python -m synergyrx_data.tujya --set http.require_contact_email=false ddinter
python -m synergyrx_data.tujya --set http.require_contact_email=false ingredients
python -m synergyrx_data.tujya interactions
```

`http.require_contact_email=false` was used because no `.env` exists on this machine (same
situation as D1); with `CONTACT_EMAIL` in `.env` the flag is not needed.

## Ingredients (source: RxNav / RxNorm, NLM)

- API: https://lhncbc.nlm.nih.gov/RxNav/APIs/ (`ndcstatus`, `rxcui/{id}/historystatus`,
  `approximateTerm` as a fallback). Rate limited to 10 requests/s (NLM allows 20/s per IP).
- The base ingredient (`baseName`, e.g. "cyclobenzaprine") is kept, not the salt
  ("cyclobenzaprine hydrochloride"), because DDInter is keyed by the base name.
- Strengths are present in the RxNav answer but are **never stored** (no doses in Tujya).

Result: **149 of 150 classes mapped through their NDCs, 82 distinct ingredients**, no class
needed the name fallback. 11 classes are combination products (e.g. losartan +
hydrochlorothiazide, glyburide + metformin, folic acid + vitamins B12 and B6). Divalproex and
valproic acid both map to RxNorm "valproate".

Not mapped: class 104, "FORMULA B PLUS TAB" (vitamins and minerals, NDC 00904-7929-60). RxNav
knows the NDC only as "ALIEN" (from a non-RxNorm source) with no ingredient list. The class
stays in the vision model; the app must say that no ingredient information is available.

`data/tujya/pill_ingredients.csv` columns: class_id, class_key, ndc11, ndc_status
(ACTIVE / OBSOLETE / ALIEN / UNKNOWN), product_rxcui, product_name (RxNorm name),
ingredient_rxcui, ingredient (lower case RxNorm IN), method (`ndc` or `name`).

## Interactions (source: DDInter 2.0)

- Licence CC BY-NC-SA 4.0 (non-commercial, share-alike). Enabled for this academic project
  by the `ddinter` step, which runs the existing connector with `enabled=true`,
  `terms_acknowledged=true` and all 14 files. The shared `config.yaml` keeps DDInter opt-in
  for the SynergyRx `update` run (two existing tests pin that default). Not redistributed:
  the tables stay under `data/`, git-ignored.
- Citation: Tian Y, et al. DDInter 2.0: an enhanced drug interaction resource with expanded
  data coverage, new interaction types, and improved user interface. Nucleic Acids Res.
  2025;53(D1):D1356-62. doi:10.1093/nar/gkae726
- The connector downloaded only 8 of the 14 ATC files by default (A, B, D, H, L, P, R, V).
  The other six (C cardiovascular, G, J, M, N nervous system, S) are published at the same
  address with the same date; the `ddinter` step requests all 14 (234,981 unique rows
  instead of 160,235).
- Name matching: case-insensitive, after three aliases in `config/ingredient_aliases.csv`
  (valproate = Valproic acid, vitamin b12 = Cyanocobalamin, vitamin b6 = Pyridoxine).
  **All 82 ingredients are found in DDInter.**

| Table | Pairs | Major | Moderate | Minor | Unknown |
|---|---:|---:|---:|---:|---:|
| `interactions_between_classes.csv` (both ingredients belong to the 150 classes) | 2,350 | 125 | 826 | 72 | 1,327 |
| `interactions_with_any_drug.csv` (at least one does) | 40,508 | 4,124 | 20,352 | 1,446 | 14,586 |

Columns: ingredient_a, ingredient_b (alphabetical, so each pair appears once), pair
("a + b"), severity (DDInter level: Major, Moderate, Minor, Unknown), description, conflict
(DDInter lists the pair twice with different levels; the most severe is kept; 0 cases
here), ddinter_id_a, ddinter_id_b, source, source_url, license.

**About `description`.** DDInter's bulk files contain only the two drugs and the level; the
mechanism and management text is only on DDInter's web pages, which this project does not
scrape. The description therefore only restates what DDInter says ("DDInter 2.0 lists an
interaction between fluconazole and warfarin with severity Major."). Explanations will come
from the RAG corpus (openFDA labels), always with their citation. "Unknown" means DDInter
records an interaction whose severity it does not grade; it must never be shown as "no
interaction". A pair absent from the table is "no known interaction in this database",
never "safe".
