# Bonus: drug-combination synergy (DrugComb) — not turned into a tool

Question: from the DrugComb data already collected by the SynergyRx pipeline, can a model
predict the synergy of a combination **involving a drug it has never seen**, well enough
to justify a `predict_synergy` tool? The rule was fixed **before training** (D75): add the
tool only if the model's test RMSE is at least 10 % below the mean baseline **and** its
Pearson r is at least 0.3, on a leave-drug-out test set.

**Result: no.** The model beats the mean baseline (RMSE −7.3 %, r 0.377) but not by the
required margin, so no tool was added, and the model was not tuned further against the
test set. Cancer cell-line synergy is research data, not information about a patient's
medicines; Tujya says nothing about it in the app.

## Data (2026-10-28)

`data/parquet/` from the SynergyRx pipeline, now versioned with DVC (`data/parquet.dvc`,
6 files, 38 MB, pushed to the local remote).

| | Rows / experiments |
|---|---:|
| Rows in `synergy_pairs.parquet` | 689,081 (DrugComb 379,670, TDC 286,674, O'Neil 22,737) |
| Distinct experiments (unordered pair + cell line) | 379,670 |
| Experiments by source | DrugComb + TDC 281,760; DrugComb only 75,173; DrugComb + O'Neil 17,823; all three 4,914 |
| Agreement with DrugComb | TDC: r = 1.00 on Loewe and ZIP (286,674 shared: a copy); O'Neil: r = 0.415 on Loewe (22,737 shared; no ZIP) |
| **Kept: one source per experiment = DrugComb** | 379,670 (every experiment is in DrugComb; no replicate rows within it) |
| Both drugs with a valid structure (289 drugs have one) | **323,339 experiments, 287 drugs, 110 cell lines** |

**Target: ZIP** (DrugComb's score, combining the Loewe and Bliss references; on the kept
experiments mean −0.77, sd 5.55; Loewe is skewed, mean −9.05, sd 14.76). Licences:
DrugComb CC BY 4.0; O'Neil rows are not used.

## Model

`backend/scripts/train_synergy.py` (backend venv; 46 s on the laptop CPU):

- Drug features: sum of the two drugs' Morgan fingerprints (radius 2, 1,024 bits; order of
  the pair does not matter), with the toxicity tool's standardisation (largest fragment).
- Cell line features: 30 principal components of the standardised expression matrix
  (86 cell lines; missing for the others, which LightGBM handles) and the tissue.
- **Leave-drug-out split**: a random 15 % of the 287 drugs held out for the test set (every
  experiment containing at least one of them), another 10 % for validation (early stopping):
  178,307 train / 51,435 validation / 93,597 test (86,135 with one unseen drug, 7,462 with
  two).
- LightGBM regression (63 leaves, learning rate 0.05, feature fraction 0.5), early stopping
  at 79 rounds.

| Test set (leave-drug-out) | Mean baseline RMSE | Per-cell-line mean RMSE (r) | **LightGBM RMSE (r)** |
|---|---:|---:|---:|
| All test (93,597) | 5.597 | 5.472 (0.215) | **5.189 (0.377)** |
| One unseen drug (86,135) | 5.582 | 5.448 (0.221) | 5.165 (0.381) |
| Two unseen drugs (7,462) | 5.777 | 5.734 (0.142) | 5.455 (0.328) |
| Validation (51,435) | 5.479 | 5.286 (0.263) | 5.075 (0.376) |

RMSE gain over the mean baseline on the test set: **7.3 %** (rule: ≥ 10 %) ; Pearson 0.377
(rule: ≥ 0.3). **Rule not met → no `predict_synergy` tool.**

MLflow: experiment **`tujya-synergy`**, run `lgbm-leave-drug-out-zip`
(`77af603c48c54b06bcdb016286f7de43`): parameters, dataset counts, every metric above, tag
`clearly_better_than_mean_baseline = False`, `results.json`. No model registered.

## Why it is hard, and what would be next

- Leave-drug-out is the honest setting for a tool (a user asks about any drug), and it is
  much harder than a random split where both drugs were seen in training; published
  synergy models usually report random or leave-pair-out splits, which are not comparable.
- 323,339 experiments but only 287 drugs: for generalising to a new drug, the number of
  distinct drugs is what counts, and it is small.
- Next steps if the bonus is pursued: drug-target or monotherapy-response features instead
  of fingerprints only, per-study normalisation, a different target (Loewe, or a
  synergistic / antagonistic class), and a fresh held-out set of drugs to evaluate them,
  since this test set has now been looked at.
