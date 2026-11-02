"""Every column the pipeline writes must be explained in docs/data_dictionary.md."""

from __future__ import annotations

import re
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx

from src.data.synergyrx_data import pipeline
from src.data.synergyrx_data.config import Settings, SourceConfig
from src.data.synergyrx_data.manifest import ManifestEntry
from src.data.synergyrx_data.provenance import PROVENANCE_COLUMNS
from tests.data.pipeline.unit.test_connectors_lookup import PUBMED_XML

DICTIONARY = Path(__file__).resolve().parents[4] / "docs" / "data_pipeline" / "data_dictionary.md"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TABLES = [
    "data/interim/ddinter/interactions.parquet",
    "data/interim/pubmed/articles.parquet",
    "data/interim/pubmed/query_hits.parquet",
    "data/interim/openfda_label/label_sections.parquet",
]


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


def openfda_label() -> dict[str, object]:
    return {
        "id": "f1",
        "set_id": "set-f1",
        "version": "3",
        "effective_time": "20240101",
        "openfda": {
            "generic_name": ["FLUCONAZOLE"],
            "substance_name": ["FLUCONAZOLE"],
            "product_type": ["HUMAN PRESCRIPTION DRUG"],
            "brand_name": ["Brand"],
            "manufacturer_name": ["Maker"],
        },
        "drug_interactions": ["Interactions text."],
        "warnings": ["Warnings text."],
    }


@pytest.fixture
async def outputs(settings: Settings, mock_http: respx.MockRouter, allow_robots: None) -> Settings:
    """Every table of the three connectors, written by real runs on mocked HTTP."""
    classes = settings.resolve("config/pill_classes.csv")
    classes.parent.mkdir(parents=True, exist_ok=True)
    classes.write_text("class_id,class_key,active_ingredients\n0,x_0,fluconazole\n", "utf-8")
    settings.sources["ddinter"] = SourceConfig(
        enabled=True, options={"terms_acknowledged": True, "codes": ["A"]}
    )
    settings.sources["openfda_label"] = SourceConfig(enabled=True)
    mock_http.get(url__startswith="https://ddinter2.scbdd.com/static/").respond(
        200,
        text="DDInterID_A,Drug_A,DDInterID_B,Drug_B,Level\n"
        "DDInter1,Abacavir,DDInter424,Cobicistat,Moderate\n",
        headers={"content-type": "application/octet-stream"},
    )
    mock_http.post(f"{EUTILS}/esearch.fcgi").respond(
        json={"esearchresult": {"count": "2", "webenv": "W", "querykey": "1"}}
    )
    mock_http.get(f"{EUTILS}/efetch.fcgi").respond(
        200, content=PUBMED_XML, headers={"content-type": "text/xml"}
    )
    mock_http.get("https://api.fda.gov/drug/label.json").mock(
        return_value=httpx.Response(200, json={"meta": {}, "results": [openfda_label()]})
    )
    runs = await pipeline.fetch(settings, ["ddinter", "pubmed", "openfda_label"])
    assert [run.outcome for run in runs] == ["ok", "ok", "ok"], [run.message for run in runs]
    return settings


async def test_every_output_column_is_documented(outputs: Settings) -> None:
    sections = documented()
    for relative in TABLES:
        assert relative in sections, f"no section for {relative} in data_dictionary.md"
        described = sections[relative] | set(PROVENANCE_COLUMNS)  # explained once, at the top
        actual = set(pd.read_parquet(outputs.resolve(relative)).columns)
        assert not actual - described, (
            f"undocumented columns in {relative}: {sorted(actual - described)}"
        )
        stale = sections[relative] - actual
        assert not stale, f"documented but not produced in {relative}: {sorted(stale)}"


def test_provenance_and_manifest_fields_are_documented() -> None:
    text = DICTIONARY.read_text(encoding="utf-8")
    for column in PROVENANCE_COLUMNS:
        assert f"| `{column}` |" in text
    assert set(ManifestEntry.model_fields) == documented()["manifest.json"]
