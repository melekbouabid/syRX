"""JSON exports for the frontend: drugs, cell lines, citations and a mock-prediction seed.

Nothing here is invented. The seed file holds real measured examples and the real score
distributions, so the frontend can generate plausible placeholder predictions itself; it
contains no predicted value and says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synergyrx_data.cleaning.scores import SCORE_COLUMNS
from synergyrx_data.config import Settings
from synergyrx_data.manifest import utc_now_iso


def _clean(value: Any) -> Any:
    """JSON-safe value: NaN / NA become null, numpy scalars and arrays become Python ones."""
    if value is None:
        return None
    if isinstance(value, list | tuple | np.ndarray):
        return [_clean(item) for item in value]
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def drugs_json(drugs: pd.DataFrame, max_synonyms: int = 10) -> list[dict[str, Any]]:
    return [
        {
            "id": row.drug_id,
            "name": row.name,
            "smiles": _clean(row.smiles),
            "smiles_valid": bool(row.smiles_valid),
            "inchikey": _clean(row.inchikey),
            "pubchem_cid": _clean(row.pubchem_cid),
            "chembl_id": _clean(row.chembl_id),
            "synonyms": [str(s) for s in (_clean(row.synonyms) or [])][:max_synonyms],
        }
        for row in drugs.sort_values("name", key=lambda s: s.str.lower()).itertuples(index=False)
    ]


def cell_lines_json(cell_lines: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "id": row.cell_line_id,
            "name": row.name,
            "tissue": _clean(row.tissue),
            "disease": _clean(row.disease),
            "depmap_id": _clean(row.depmap_id),
        }
        for row in cell_lines.sort_values("name", key=lambda s: s.str.lower()).itertuples(
            index=False
        )
    ]


def citations_json(abstracts: pd.DataFrame) -> list[dict[str, Any]]:
    """Citation metadata only (no abstract text, which may be under publisher copyright)."""
    return [
        {
            "pmid": row.pmid,
            "title": _clean(row.title),
            "journal": _clean(row.journal),
            "year": _clean(row.year),
            "doi": _clean(row.doi),
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{row.pmid}/",
            "is_retracted": bool(row.is_retracted),
            "is_open_access": _clean(getattr(row, "is_open_access", None)),
        }
        for row in abstracts.sort_values("pmid").itertuples(index=False)
    ]


def mock_predictions_seed(pairs: pd.DataFrame, seed: int, n_examples: int) -> dict[str, Any]:
    """Seed for placeholder predictions: a random seed, score distributions, real examples."""
    distributions: dict[str, Any] = {}
    for score in SCORE_COLUMNS:
        values = pairs[score].dropna() if score in pairs.columns else pd.Series(dtype="float64")
        if len(values):
            quantiles = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            distributions[score] = {
                "n": len(values),
                "mean": round(float(values.mean()), 4),
                "std": round(float(values.std(ddof=0)), 4),
                **{f"p{int(q * 100):02d}": round(float(v), 4) for q, v in quantiles.items()},
            }
    examples: list[dict[str, Any]] = []
    if not pairs.empty:
        picked = pairs.sample(n=min(n_examples, len(pairs)), random_state=seed).sort_values(
            ["pair_key", "cell_line_id", "source"]
        )
        for row in picked.itertuples(index=False):
            observed = {s: _clean(getattr(row, s)) for s in SCORE_COLUMNS}
            examples.append(
                {
                    "drug_a_id": row.drug_a_id,
                    "drug_b_id": row.drug_b_id,
                    "cell_line_id": row.cell_line_id,
                    "source": row.source,
                    "observed": {k: v for k, v in observed.items() if v is not None},
                }
            )
    return {
        "_note": (
            "Seed for MOCK predictions in the frontend. This file contains no model output. "
            "`score_distribution` and `examples` are measured values copied from "
            "synergy_pairs.parquet; generate placeholder predictions from `seed`."
        ),
        "generated_at": utc_now_iso(),
        "seed": seed,
        "score_distribution": distributions,
        "examples": examples,
    }


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def export_frontend(
    settings: Settings,
    drugs: pd.DataFrame,
    cell_lines: pd.DataFrame,
    pairs: pd.DataFrame,
    abstracts: pd.DataFrame | None = None,
) -> dict[str, Path]:
    directory = settings.json_root
    options = settings.export
    outputs = {
        "drugs": write_json(directory / "drugs.json", drugs_json(drugs, options.max_synonyms)),
        "cell_lines": write_json(directory / "cell_lines.json", cell_lines_json(cell_lines)),
        "mock_predictions_seed": write_json(
            directory / "mock_predictions_seed.json",
            mock_predictions_seed(pairs, options.mock_seed, options.mock_examples),
        ),
    }
    if abstracts is not None and not abstracts.empty:
        outputs["citations"] = write_json(directory / "citations.json", citations_json(abstracts))
    return outputs
