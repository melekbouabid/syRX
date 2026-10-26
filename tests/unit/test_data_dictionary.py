"""Every column the pipeline writes must be explained in docs/DATA_DICTIONARY.md."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from synergyrx_data import build
from synergyrx_data.cleaning.scores import SCORE_COLUMNS
from synergyrx_data.config import Settings
from synergyrx_data.manifest import ManifestEntry
from synergyrx_data.provenance import PROVENANCE_COLUMNS
from tests.unit.test_build_cli import seed_interim

DICTIONARY = Path(__file__).resolve().parents[2] / "docs" / "DATA_DICTIONARY.md"
GENE_COLUMN = re.compile(r"^.+ \(\d+\)$")


def documented() -> dict[str, set[str]]:
    """Section heading -> column names (the first, back-quoted cell of each table row)."""
    sections: dict[str, set[str]] = {}
    current = None
    for line in DICTIONARY.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = set()
        elif current and (match := re.match(r"^\| `([^`]+)` \|", line)):
            sections[current].add(match.group(1))
    return sections


@pytest.fixture(scope="module")
def outputs(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    from synergyrx_data.config import Secrets, load_settings

    settings = load_settings(None, project_root=tmp_path_factory.mktemp("project"), environ={})
    settings.secrets = Secrets()
    seed_interim(settings)
    for step in (build.clean, build.literature, build.report, build.export):
        step(settings)
    return settings


def columns_of(path: Path) -> set[str]:
    if path.suffix == ".parquet":
        return set(pd.read_parquet(path).columns)
    if path.suffix == ".csv":
        return set(pd.read_csv(path, nrows=0).columns)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return set(payload[0] if isinstance(payload, list) else payload)


@pytest.mark.parametrize(
    "relative",
    [
        "data/parquet/synergy_pairs.parquet",
        "data/parquet/drugs.parquet",
        "data/parquet/cell_lines.parquet",
        "data/parquet/expression.parquet",
        "data/parquet/abstracts.parquet",
        "data/parquet/candidate_sentences.parquet",
        "data/json/drugs.json",
        "data/json/cell_lines.json",
        "data/json/mock_predictions_seed.json",
        "data/json/citations.json",
        "data/csv/coverage.csv",
        "data/csv/conflicts.csv",
        "data/csv/source_agreement.csv",
        "data/csv/drug_review_queue.csv",
        "data/csv/cell_line_review_queue.csv",
        "data/csv/sample_sentences.csv",
    ],
)
def test_every_output_column_is_documented(outputs: Settings, relative: str) -> None:
    sections = documented()
    assert relative in sections, f"no section for {relative} in DATA_DICTIONARY.md"
    described = set(sections[relative])
    if relative.startswith("data/parquet/"):
        described |= set(PROVENANCE_COLUMNS)  # explained once, at the top of the document
    if "X_n" in described:  # conflicts.csv documents its per-score columns as a pattern
        described |= {
            f"{score}_{stat}" for score in SCORE_COLUMNS for stat in ("n", "min", "max", "range")
        }
    actual = {c for c in columns_of(outputs.resolve(relative)) if not GENE_COLUMN.match(c)}
    assert not actual - described, (
        f"undocumented columns in {relative}: {sorted(actual - described)}"
    )
    stale = described - actual - set(PROVENANCE_COLUMNS) - {"X_n", "X_min", "X_max", "X_range"}
    stale -= {"<SYMBOL (EntrezID)>"}
    assert not stale, f"documented but not produced in {relative}: {sorted(stale)}"


def test_provenance_and_manifest_fields_are_documented() -> None:
    text = DICTIONARY.read_text(encoding="utf-8")
    for column in PROVENANCE_COLUMNS:
        assert f"| `{column}` |" in text
    assert set(ManifestEntry.model_fields) == documented()["manifest.json"]
