"""Shared shape of the synergy tables produced by every drug-combination connector."""

from __future__ import annotations

import numpy as np
import pandas as pd

TEXT_COLUMNS = [
    "drug_a_name",
    "drug_b_name",
    "cell_line_name",
    "tissue",
    "study_id",
    "source_record_id",
    "drug_a_source_id",
    "drug_b_source_id",
    "drug_a_smiles",
    "drug_b_smiles",
]

# Each score keeps its own column. They measure different things and are never merged.
SCORE_COLUMNS = [
    "synergy_bliss",
    "synergy_loewe",
    "synergy_hsa",
    "synergy_zip",
    "css",
    "combo_score",
]

SYNERGY_COLUMNS = TEXT_COLUMNS + SCORE_COLUMNS


def synergy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Reorder to the common layout: missing columns become nulls, scores become floats."""
    out = pd.DataFrame(index=frame.index)
    for column in TEXT_COLUMNS:
        if column in frame.columns:
            out[column] = frame[column].astype("string").str.strip().replace("", pd.NA)
        else:
            out[column] = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in SCORE_COLUMNS:
        if column in frame.columns:
            out[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
        else:
            out[column] = np.nan
    return out.reset_index(drop=True)
