"""Stratified sample of candidate sentences for manual labelling.

The sample is what a person labels to measure how good relation extraction is, so it has to
represent the candidates fairly: it is stratified by whether a synergy cue word is present,
how many drugs are mentioned and the article type, and no single abstract can dominate it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

LABELS = "synergy | antagonism | additive | no_relation | unclear"
EXPORT_COLUMNS = [
    "sample_id", "stratum", "pmid", "sentence_index", "sentence", "drug_names", "n_drugs",
    "has_synergy_cue", "cue_terms", "article_type", "year", "is_retracted", "pubmed_url",
    "label", "annotator", "comment",
]  # fmt: skip


def _stratum(row: pd.Series) -> str:
    cue = "cue" if row["has_synergy_cue"] else "nocue"
    drugs = "2drugs" if row["n_drugs"] == 2 else "3plus"
    kind = "review" if row["article_type"] in ("review", "meta_analysis") else "primary"
    return f"{cue}/{drugs}/{kind}"


def stratified_sample(
    candidates: pd.DataFrame, *, size: int = 300, max_per_pmid: int = 3, seed: int = 20260101
) -> pd.DataFrame:
    """Draw `size` sentences, proportionally per stratum with a floor for small strata.

    If fewer than `size` candidates exist, all of them are returned: the sample is never
    padded with repeats.
    """
    if candidates.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS)
    pool = candidates.copy()
    pool["stratum"] = pool.apply(_stratum, axis=1)
    # Shuffle once, then cap sentences per abstract so one long abstract cannot dominate.
    pool = pool.sample(frac=1.0, random_state=seed)
    pool = pool[pool.groupby("pmid").cumcount() < max_per_pmid]
    if len(pool) > size:
        counts = pool["stratum"].value_counts().sort_index()
        floor = min(10, size // max(len(counts), 1))
        quota = (counts / counts.sum() * size).round().astype(int).clip(lower=floor)
        quota = quota.clip(upper=counts)
        picked = pd.concat(
            [pool[pool["stratum"] == stratum].head(int(n)) for stratum, n in quota.items()]
        )
        if len(picked) > size:
            picked = picked.sample(n=size, random_state=seed)
        elif len(picked) < size:
            rest = pool.drop(picked.index).head(size - len(picked))
            picked = pd.concat([picked, rest])
        pool = picked
    pool = pool.sort_values(["stratum", "pmid", "sentence_index"]).reset_index(drop=True)
    return pd.DataFrame(
        {
            "sample_id": [f"S{i + 1:04d}" for i in range(len(pool))],
            "stratum": pool["stratum"],
            "pmid": pool["pmid"],
            "sentence_index": pool["sentence_index"],
            "sentence": pool["sentence"],
            "drug_names": pool["drug_names"].map(lambda names: "; ".join(map(str, names))),
            "n_drugs": pool["n_drugs"],
            "has_synergy_cue": pool["has_synergy_cue"],
            "cue_terms": pool["cue_terms"].map(lambda cues: "; ".join(map(str, cues))),
            "article_type": pool["article_type"],
            "year": pool["year"],
            "is_retracted": pool["is_retracted"],
            "pubmed_url": "https://pubmed.ncbi.nlm.nih.gov/" + pool["pmid"].astype(str) + "/",
            "label": "",
            "annotator": "",
            "comment": "",
        }
    )


def export_annotation_sample(
    sample: pd.DataFrame, *, csv_dir: Path, json_dir: Path, guide_dir: Path
) -> dict[str, Path]:
    """Write the sample as CSV (for spreadsheets) and JSONL (for annotation tools).

    Each file goes to the folder of its format; the labelling guide goes with the reports.
    """
    for directory in (csv_dir, json_dir, guide_dir):
        directory.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "sample_sentences.csv"
    jsonl_path = json_dir / "sample_sentences.jsonl"
    sample.to_csv(csv_path, index=False, encoding="utf-8-sig")  # BOM: opens cleanly in Excel
    records: list[dict[str, Any]] = json.loads(sample.to_json(orient="records"))
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    guide = guide_dir / "annotation_guide.md"
    guide.write_text(
        "# Annotation sample\n\n"
        "The sample is `sample_sentences.csv` (with the other CSV files) and, with the same "
        "rows, `sample_sentences.jsonl` (with the JSON files).\n\n"
        "One row per sentence that mentions at least two drugs from the synergy data.\n"
        f"Fill `label` with one of: {LABELS}. Put your initials in `annotator`.\n"
        "Label the relation the sentence itself states between the listed drugs, not what "
        "you know from elsewhere. Use `unclear` rather than guessing.\n\n"
        "`stratum` is `cue|nocue / 2drugs|3plus / primary|review`; keep it when reporting "
        "precision per stratum. Sentences come from PubMed abstracts: do not redistribute.\n",
        encoding="utf-8",
    )
    return {"csv": csv_path, "jsonl": jsonl_path, "guide": guide}
