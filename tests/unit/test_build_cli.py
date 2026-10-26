"""Offline build steps end to end (clean, literature, report, export) and the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import respx
from rich.console import Console
from typer.testing import CliRunner

from synergyrx_data import build, cli
from synergyrx_data.cli import app
from synergyrx_data.config import Settings
from synergyrx_data.connectors import REGISTRY
from synergyrx_data.connectors._synergy import synergy_frame
from synergyrx_data.literature.abstracts import articles_frame
from synergyrx_data.manifest import Manifest
from synergyrx_data.provenance import PROVENANCE_COLUMNS, has_provenance, write_parquet
from synergyrx_data.quality.drift import DriftTracker, compare, snapshot
from synergyrx_data.quality.licenses import render_licenses
from synergyrx_data.quality.reports import coverage_table, missingness_table
from tests.conftest import TEST_EMAIL

FU = "O=c1[nH]cc(F)c(=O)[nH]1"
VELIPARIB = "CC1(c2nc3c(C(N)=O)cccc3[nH]2)CCCN1"
DASATINIB = "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1"


def provenance(source: str, licence: str = "CC-BY-4.0") -> dict[str, str]:
    return {
        "source": source,
        "source_url": f"https://example.org/{source}",
        "retrieved_at": "2026-01-01T00:00:00Z",
        "source_version": "1",
        "license": licence,
    }


def seed_interim(settings: Settings) -> None:
    """Two synergy sources that overlap, DepMap models and expression, PubChem, PubMed."""
    screen_a = synergy_frame(
        pd.DataFrame(
            [
                {
                    "drug_a_name": "5-FU",
                    "drug_b_name": "ABT-888",
                    "cell_line_name": "A2058",
                    "synergy_loewe": 12.0,
                    "synergy_bliss": 6.0,
                    "drug_a_smiles": FU,
                    "drug_b_smiles": VELIPARIB,
                },
                {
                    "drug_a_name": "ABT-888",
                    "drug_b_name": "5-FU",
                    "cell_line_name": "A2058",
                    "synergy_loewe": 14.0,
                    "synergy_bliss": 8.0,
                    "drug_a_smiles": VELIPARIB,
                    "drug_b_smiles": FU,
                },
                {
                    "drug_a_name": "5-FU",
                    "drug_b_name": "DASATINIB",
                    "cell_line_name": "HT-29",
                    "synergy_loewe": -3.0,
                    "drug_a_smiles": FU,
                    "drug_b_smiles": DASATINIB,
                },
                {
                    "drug_a_name": "5-FU",
                    "drug_b_name": "BROKEN",
                    "cell_line_name": "HT-29",
                    "synergy_loewe": 1.0,
                    "drug_a_smiles": FU,
                    "drug_b_smiles": "C1CC((",
                },
            ]
        )
    ).assign(**provenance("oneil", "Unspecified"))
    screen_b = synergy_frame(
        pd.DataFrame(
            [
                {
                    "drug_a_name": "Fluorouracil",
                    "drug_b_name": "Veliparib",
                    "cell_line_name": "A-2058",
                    "synergy_loewe": -9.0,
                    "synergy_zip": 2.0,
                },
                {
                    "drug_a_name": "Fluorouracil",
                    "drug_b_name": "Mysterium",
                    "cell_line_name": "UNKNOWN-LINE",
                    "synergy_zip": 500.0,
                },
            ]
        )
    ).assign(**provenance("drugcomb"))
    write_parquet(screen_a, settings.interim_dir / "oneil" / "synergy.parquet")
    write_parquet(screen_b, settings.interim_dir / "drugcomb" / "synergy.parquet")

    models = pd.DataFrame(
        {
            "depmap_id": ["ACH-000788", "ACH-000552"],
            "cell_line_name": ["A2058", "HT-29"],
            "stripped_name": ["A2058", "HT29"],
            "ccle_name": ["A2058_SKIN", "HT29_LARGE_INTESTINE"],
            "lineage": ["Skin", "Bowel"],
            "primary_disease": ["Melanoma", "Colorectal Adenocarcinoma"],
        }
    ).assign(**provenance("depmap"))
    expression = pd.DataFrame(
        {
            "depmap_id": ["ACH-000788", "ACH-000552", "ACH-999999"],
            "TSPAN6 (7105)": [4.3, 3.1, 1.0],
            "TNMD (64102)": [0.0, 0.5, 9.0],
        }
    ).assign(**provenance("depmap"))
    write_parquet(models, settings.interim_dir / "depmap" / "models.parquet")
    write_parquet(expression, settings.interim_dir / "depmap" / "expression.parquet")

    compounds = pd.DataFrame(
        {
            "query_name": ["Fluorouracil", "Veliparib", "Mysterium"],
            "status": ["found", "found", "not_found"],
            "n_cids": [1, 1, 0],
            "pubchem_cid": pd.array([3385, 11960529, None], dtype="Int64"),
            "smiles": [FU, VELIPARIB, None],
            "connectivity_smiles": [FU, VELIPARIB, None],
            "inchikey": ["GHASVSINZRGABV-UHFFFAOYSA-N", "JNAHVYVRKWKWKQ-UHFFFAOYSA-N", None],
            "title": ["Fluorouracil", "Veliparib", None],
            "synonyms": [["5-Fluorouracil", "Adrucil"], ["ABT-888"], []],
        }
    ).assign(**provenance("pubchem", "Public domain"))
    write_parquet(compounds, settings.interim_dir / "pubchem" / "compounds.parquet")

    articles = articles_frame(
        [
            {
                "pmid": str(100 + i),
                "title": f"Study {i}",
                "abstract": "Veliparib and 5-FU were synergistic in A2058 cells. "
                "Dasatinib plus fluorouracil showed antagonism.",
                "journal": "J Tests",
                "year": 2010 + i,
                "doi": f"10.1000/t{i}",
                "language": "eng",
                "article_type": "research",
                "is_retracted": i == 0,
                "has_abstract": True,
                **provenance("pubmed", "NLM"),
            }
            for i in range(3)
        ]
    )
    write_parquet(articles, settings.interim_dir / "pubmed" / "articles.parquet")


@pytest.fixture
def built(settings: Settings) -> Settings:
    seed_interim(settings)
    build.clean(settings)
    build.literature(settings)
    build.report(settings)
    build.export(settings)
    return settings


# ------------------------------------------------------------------- clean


def test_clean_builds_the_processed_tables(built: Settings) -> None:
    processed = built.parquet_root
    drugs = pd.read_parquet(processed / "drugs.parquet").set_index("name")
    cells = pd.read_parquet(processed / "cell_lines.parquet").set_index("cell_line_id")
    pairs = pd.read_parquet(processed / "synergy_pairs.parquet")
    expression = pd.read_parquet(processed / "expression.parquet")

    # "5-FU" and "Fluorouracil" are one drug; so are "ABT-888" and "Veliparib".
    assert set(drugs.index) == {"Fluorouracil", "Veliparib", "DASATINIB", "BROKEN", "Mysterium"}
    assert drugs.loc["Fluorouracil", "pubchem_cid"] == 3385
    assert {"5-FU", "Adrucil"} <= set(drugs.loc["Fluorouracil", "synonyms"])
    assert int((~drugs["smiles_valid"]).sum()) == 2  # BROKEN and Mysterium: flagged, kept
    assert has_provenance(drugs.reset_index())
    assert "oneil" in drugs.loc["Fluorouracil", "source"]

    assert {"ACH-000788", "ACH-000552", "name:UNKNOWNLINE"} == set(cells.index)
    assert cells.loc["ACH-000788", "tissue"] == "Skin" and cells.loc["ACH-000788", "mapped"]
    assert has_provenance(cells.reset_index())

    fu, veli = sorted([drugs.loc["Fluorouracil", "drug_id"], drugs.loc["Veliparib", "drug_id"]])
    shared = pairs[(pairs["pair_key"] == f"{fu}|{veli}") & (pairs["cell_line_id"] == "ACH-000788")]
    assert len(shared) == 2  # one row per source: (A,B)+(B,A) in oneil collapsed, drugcomb separate
    oneil = shared[shared["source"] == "oneil"].iloc[0]
    drugcomb = shared[shared["source"] == "drugcomb"].iloc[0]
    assert oneil.synergy_loewe == 13.0 and oneil.n_replicates == 2
    assert drugcomb.synergy_loewe == -9.0 and pd.isna(drugcomb.synergy_bliss)
    assert bool(oneil.has_conflict) and bool(drugcomb.has_conflict) and oneil.n_sources == 2
    assert pairs["score_outlier"].sum() == 1  # the 500 is flagged, and still 500
    assert pairs["synergy_zip"].max() == 500.0
    assert all(c in pairs.columns for c in PROVENANCE_COLUMNS)
    assert pairs[["pair_key", "cell_line_id", "source", "study_id"]].duplicated().sum() == 0

    # Expression is restricted to the cell lines actually used.
    assert sorted(expression["depmap_id"]) == ["ACH-000552", "ACH-000788"]

    queue = pd.read_csv(built.csv_root / "drug_review_queue.csv")
    assert set(queue["name"]) == {"BROKEN", "Mysterium"}
    cell_queue = pd.read_csv(built.csv_root / "cell_line_review_queue.csv")
    assert list(cell_queue["cell_line_id"]) == ["name:UNKNOWNLINE"]


def test_clean_without_inputs_says_what_to_run(settings: Settings) -> None:
    with pytest.raises(build.BuildError, match="synergyrx fetch"):
        build.clean(settings)


def test_clean_is_repeatable(built: Settings) -> None:
    path = built.parquet_root / "synergy_pairs.parquet"
    before = path.read_bytes()
    result = build.clean(built)
    assert path.read_bytes() == before  # same inputs, byte-identical output
    assert not [d for d in result.drift if d.severity == "warning"]


# ------------------------------------------------------------------ report


def test_coverage_reports_distinct_drugs_and_cell_lines_per_source(built: Settings) -> None:
    """The acceptance criterion: distinct drugs and cell lines are stated per source."""
    coverage = pd.read_csv(built.csv_root / "coverage.csv").set_index("source")
    assert list(coverage.index) == ["drugcomb", "oneil", "ALL SOURCES"]
    assert coverage.loc["oneil", "distinct_drugs"] == 4
    assert coverage.loc["oneil", "distinct_cell_lines"] == 2
    assert coverage.loc["drugcomb", "distinct_drugs"] == 3
    assert coverage.loc["drugcomb", "distinct_cell_lines"] == 2
    assert coverage.loc["ALL SOURCES", "distinct_drugs"] == 5  # not 4 + 3: shared drugs count once
    assert coverage.loc["ALL SOURCES", "distinct_cell_lines"] == 3
    assert coverage.loc["ALL SOURCES", "drugs_with_valid_smiles"] == 3
    assert coverage.loc["oneil", "synergy_loewe_pct"] == 100.0
    assert coverage.loc["oneil", "synergy_zip_pct"] == 0.0

    markdown = (built.reports_root / "data_quality.md").read_text(encoding="utf-8")
    assert "**5 distinct drugs**" in markdown
    assert "unseen-drug validation protocol" in markdown
    for heading in (
        "Coverage per source",
        "Drug identity and SMILES",
        "Duplicates and conflicts",
        "Missingness",
        "Schema drift",
        "Literature",
        "Sources",
    ):
        assert f"## {heading}" in markdown
    html = (built.reports_root / "data_quality.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and "<table>" in html
    assert "<strong>5 distinct drugs</strong>" in html

    conflicts = pd.read_csv(built.csv_root / "conflicts.csv")
    assert len(conflicts) == 1 and bool(conflicts["is_conflict"].iloc[0])
    assert conflicts["synergy_loewe_range"].iloc[0] == 22.0
    agreement = pd.read_csv(built.csv_root / "source_agreement.csv")
    assert set(agreement.columns) >= {"source_a", "source_b", "score", "shared_triplets"}


def test_license_registry_lists_every_source(built: Settings) -> None:
    text = built.resolve("LICENSES.md").read_text(encoding="utf-8")
    for name, connector in REGISTRY.items():
        assert f"**{name}**" in text
        assert connector.info.license.split(" (")[0] in text
    assert "CC-BY-NC-SA-4.0" in text and "Non-commercial" in text
    assert "https://ddinter2.scbdd.com/terms/" in text
    assert text == render_licenses(built, Manifest(built))  # regenerating changes nothing


def test_missingness_table() -> None:
    frame = pd.DataFrame({"a": [1, None, None, 4], "b": [1, 2, 3, 4], "c": [None] * 4})
    table = missingness_table(frame).set_index("column")
    assert list(table.index) == ["c", "a", "b"]
    assert table.loc["a", "null_pct"] == 50.0 and table.loc["b", "nulls"] == 0
    assert coverage_table(pd.DataFrame(), frame, frame).empty


# ------------------------------------------------------------------- drift


def test_schema_drift_between_versions_is_reported(tmp_path: Path) -> None:
    v1 = pd.DataFrame({"drug_row": ["a"] * 100, "synergy_zip": [1.0] * 100, "old": [1] * 100})
    v2 = pd.DataFrame(
        {"drug_row": ["a"] * 40, "synergy_zip": ["1.0"] * 40, "study_name": ["s"] * 40}
    )
    tracker = DriftTracker(tmp_path / "schemas")
    assert [f.kind for f in tracker.check("interim/x/synergy", v1, "1.4")] == ["first_snapshot"]
    assert tracker.check("interim/x/synergy", v1, "1.4") == []  # same data: no drift
    findings = {f.kind: f for f in tracker.check("interim/x/synergy", v2, "1.5")}
    assert set(findings) == {"column_added", "column_removed", "dtype_changed", "rows_changed"}
    assert findings["column_removed"].detail == "old"
    assert "study_name" in findings["column_added"].detail
    assert findings["dtype_changed"].detail.startswith("synergy_zip: float64 -> ")
    assert "-60%" in findings["rows_changed"].detail
    assert len(tracker.history("interim/x/synergy")) == 2
    # A sample build is not compared on row counts with a full build.
    sample = snapshot(v1.head(5), "1.4", is_sample=True)
    assert compare("t", snapshot(v1, "1.4"), sample) == []


def test_wide_tables_are_fingerprinted_not_listed() -> None:
    wide = pd.DataFrame({f"gene{i}": [0.0] for i in range(600)})
    shot = snapshot(wide, "24Q4")
    assert "columns" not in shot and len(shot["columns_sha256"]) == 64
    narrower = snapshot(wide.iloc[:, :599], "25Q2")
    assert [f.kind for f in compare("expression", shot, narrower)] == ["column_removed"]


def test_drift_in_a_processed_table_reaches_the_report(built: Settings) -> None:
    pairs = pd.read_parquet(built.interim_dir / "drugcomb" / "synergy.parquet")
    write_parquet(
        pairs.assign(cell_line_name=["A-2058", "A-2058"]),
        built.interim_dir / "drugcomb" / "synergy.parquet",
    )
    result = build.clean(built)  # one cell line fewer than in the previous build
    kinds = {(d.table, d.kind) for d in result.drift}
    assert ("processed/cell_lines", "rows_changed") in kinds
    build.report(built)
    assert "rows_changed" in (built.reports_root / "data_quality.md").read_text(encoding="utf-8")


# -------------------------------------------------------- literature/export


def test_literature_outputs(built: Settings) -> None:
    abstracts = pd.read_parquet(built.parquet_root / "abstracts.parquet")
    candidates = pd.read_parquet(built.parquet_root / "candidate_sentences.parquet")
    assert len(abstracts) == 3 and (abstracts["n_sentences"] == 2).all()
    assert len(candidates) == 6 and candidates["has_synergy_cue"].all()
    assert has_provenance(abstracts) and has_provenance(candidates)
    sample = pd.read_csv(built.csv_root / "sample_sentences.csv")
    assert len(sample) == 6  # fewer than 200 exist: all of them, never padded
    assert (built.json_root / "sample_sentences.jsonl").is_file()
    result = build.literature(built)
    assert any("200-500" in warning for warning in result.warnings)


def test_frontend_exports(built: Settings) -> None:
    exports = built.json_root
    drugs = json.loads((exports / "drugs.json").read_text(encoding="utf-8"))
    cells = json.loads((exports / "cell_lines.json").read_text(encoding="utf-8"))
    seed = json.loads((exports / "mock_predictions_seed.json").read_text(encoding="utf-8"))
    citations = json.loads((exports / "citations.json").read_text(encoding="utf-8"))

    by_name = {d["name"]: d for d in drugs}
    assert set(by_name) == {"Fluorouracil", "Veliparib", "DASATINIB", "BROKEN", "Mysterium"}
    assert by_name["Fluorouracil"]["smiles"] and by_name["Fluorouracil"]["pubchem_cid"] == 3385
    assert by_name["BROKEN"]["smiles"] is None and by_name["BROKEN"]["smiles_valid"] is False
    assert set(by_name["Fluorouracil"]) == {
        "id",
        "name",
        "smiles",
        "smiles_valid",
        "inchikey",
        "pubchem_cid",
        "chembl_id",
        "synonyms",
    }
    assert {c["name"]: c["depmap_id"] for c in cells}["A2058"] == "ACH-000788"

    assert "no model output" in seed["_note"] and seed["seed"] == 42
    assert seed["score_distribution"]["synergy_loewe"]["n"] == 4
    pairs = pd.read_parquet(built.parquet_root / "synergy_pairs.parquet")
    for example in seed["examples"]:  # every example is a measured row, nothing is invented
        row = pairs[
            (pairs["drug_a_id"] == example["drug_a_id"])
            & (pairs["drug_b_id"] == example["drug_b_id"])
            & (pairs["cell_line_id"] == example["cell_line_id"])
            & (pairs["source"] == example["source"])
        ]
        assert len(row) == 1
        for score, value in example["observed"].items():
            assert row[score].iloc[0] == pytest.approx(value)

    assert set(citations[0]) == {
        "pmid",
        "title",
        "journal",
        "year",
        "doi",
        "url",
        "is_retracted",
        "is_open_access",
    }
    assert citations[0]["url"] == "https://pubmed.ncbi.nlm.nih.gov/100/"
    assert citations[0]["is_retracted"] is True
    assert "abstract" not in json.dumps(citations)  # no abstract text is redistributed


def test_sample_data_is_labelled_everywhere(built: Settings) -> None:
    manifest = Manifest(built)
    path = built.interim_dir / "oneil" / "synergy.parquet"
    manifest.record_file(path, source="oneil", layer="interim", is_sample=True)
    manifest.save()
    result = build.clean(built)
    assert any("SAMPLE" in warning for warning in result.warnings)
    build.report(built)
    assert "**SAMPLE DATA.**" in (built.reports_root / "data_quality.md").read_text("utf-8")
    assert Manifest(built).entries["data/parquet/drugs.parquet"].is_sample


# --------------------------------------------------------------------- CLI

runner = CliRunner()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary project folder with the shipped config.yaml, used as the working directory."""
    repo = Path(__file__).resolve().parents[2]
    (tmp_path / "config.yaml").write_text(
        (repo / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "console", Console(width=250))  # no line wrapping in assertions
    monkeypatch.setenv("CONTACT_EMAIL", TEST_EMAIL)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.delenv("SYNERGYRX_CONFIG", raising=False)
    return tmp_path


def tree(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def test_cli_dry_run_prints_the_plan_and_touches_nothing(project: Path) -> None:
    """The acceptance criterion: `fetch --sample 100 --dry-run` changes nothing."""
    before = tree(project)
    with respx.mock(assert_all_mocked=True) as router:  # any request would raise
        result = runner.invoke(app, ["fetch", "--sample", "100", "--dry-run"])
        assert len(router.calls) == 0
    assert result.exit_code == 0, result.output
    assert "Dry run: nothing was requested, written or cached." in result.output
    for source in ("drugcomb", "tdc", "oneil", "nci_almanac", "depmap", "pubchem", "pubmed"):
        assert source in result.output
    assert "drugbank" not in result.output  # disabled sources are not part of the plan
    assert tree(project) == before
    assert not (project / "data").exists() and not (project / "manifest.json").exists()


def test_cli_fetch_runs_a_source_and_reports_failures_with_exit_code(project: Path) -> None:
    labels = b",drug_a_name,drug_b_name,cell_line,synergy,fold\nx,5-FU,ABT-888,A2058,7.69,2\n"
    with respx.mock(assert_all_called=False) as router:
        router.get(url__regex=r"https://[^/]+/robots\.txt$").respond(404)
        base = "https://bioinf.jku.at/software/DeepSynergy/"
        router.get(base + "labels.csv").respond(200, content=labels)
        router.get(base + "smiles.csv").respond(
            200, content=f"5-FU,{FU}\nABT-888,{VELIPARIB}\n".encode()
        )
        ok = runner.invoke(app, ["--text-logs", "fetch", "--source", "oneil"])
        router.get(
            "https://zenodo.org/api/records/11102665/files/summary_table_v1.4.csv/content"
        ).respond(403)
        failed = runner.invoke(app, ["--text-logs", "fetch", "--source", "drugcomb"])
    assert ok.exit_code == 0, ok.output
    assert "synergy (1)" in ok.output
    assert failed.exit_code == 1 and "needs_attention" in failed.output
    unknown = runner.invoke(app, ["fetch", "--source", "nope", "--dry-run"])
    assert unknown.exit_code == 2 and "unknown source" in unknown.output

    status = runner.invoke(app, ["status"])
    assert status.exit_code == 0 and "needs_attention" in status.output
    validate = runner.invoke(app, ["validate", "--source", "oneil"])
    assert validate.exit_code == 0, validate.output
    assert "matches manifest" in validate.output and "0 invalid" in validate.output
    # Tampering with a raw file is caught by `validate`.
    (project / "data/raw/oneil/2017-12/labels.csv").write_bytes(b"changed")
    assert runner.invoke(app, ["validate", "--source", "oneil"]).exit_code == 1


def test_cli_overrides_and_bad_config(project: Path) -> None:
    result = runner.invoke(app, ["--set", "sources.tdc.enabled=false", "fetch", "--dry-run"])
    assert result.exit_code == 0 and "dataverse" not in result.output
    bad = runner.invoke(app, ["--set", "http.nonsense=1", "fetch", "--dry-run"])
    assert bad.exit_code == 2 and "Configuration error" in bad.output
    assert "synergyrx-data" in runner.invoke(app, ["--version"]).output


def test_cli_offline_commands_on_built_data(project: Path) -> None:
    from synergyrx_data.config import load_settings

    seed_interim(load_settings(project / "config.yaml"))
    for command in ("clean", "literature", "report", "export"):
        result = runner.invoke(app, [command])
        assert result.exit_code == 0, f"{command}: {result.output}"
    summary = runner.invoke(app, ["summary"])
    assert summary.exit_code == 0, summary.output
    assert "Distinct drugs across all sources: 5" in summary.output
    payload = json.loads(runner.invoke(app, ["summary", "--json"]).output)
    assert payload["clean"]["distinct_drugs"] == 5
    assert {row["source"] for row in payload["coverage"]} == {"drugcomb", "oneil", "ALL SOURCES"}
    assert "data/parquet/synergy_pairs.parquet" in payload["outputs"]
    sources = runner.invoke(app, ["sources"])
    assert sources.exit_code == 0 and "drugbank" in sources.output
    preflight = runner.invoke(app, ["preflight"])
    assert preflight.exit_code == 0 and "contact_email" in preflight.output


def test_cli_commands_explain_missing_inputs(project: Path) -> None:
    for command in ("clean", "literature", "report", "export"):
        result = runner.invoke(app, [command])
        assert result.exit_code == 1 and "cannot run" in result.output
    assert "No processed tables yet" in runner.invoke(app, ["summary"]).output


FORMATS = {
    "parquet": {".parquet"},
    "csv": {".csv"},
    "json": {".json", ".jsonl"},
    "reports": {".md", ".html"},
}


def test_generated_files_are_grouped_by_format_under_the_data_folder(built: Settings) -> None:
    """One folder per format, all inside `data/`: the extension tells you the folder."""
    root = built.project_root
    produced = {p.relative_to(root).parts[0] for p in root.rglob("*") if p.is_file()}
    assert produced == {"data", "manifest.json", "LICENSES.md"}  # nothing else at the root

    for folder, extensions in FORMATS.items():
        files = [p for p in (built.data_root / folder).iterdir() if p.is_file()]
        assert files, f"data/{folder} is empty"
        assert {p.suffix for p in files} <= extensions, f"wrong file type in data/{folder}"
    names = {folder: {p.name for p in (built.data_root / folder).iterdir()} for folder in FORMATS}
    assert names["parquet"] == {
        "synergy_pairs.parquet", "drugs.parquet", "cell_lines.parquet", "expression.parquet",
        "abstracts.parquet", "candidate_sentences.parquet",
    }  # fmt: skip
    assert names["csv"] == {
        "coverage.csv", "conflicts.csv", "source_agreement.csv", "drug_review_queue.csv",
        "cell_line_review_queue.csv", "sample_sentences.csv",
    }  # fmt: skip
    assert names["json"] == {
        "drugs.json", "cell_lines.json", "citations.json", "mock_predictions_seed.json",
        "sample_sentences.jsonl",
    }  # fmt: skip
    assert names["reports"] == {"data_quality.md", "data_quality.html", "annotation_guide.md"}

    # No folder of an earlier layout is created, and the manifest only knows the new paths.
    present = {p.name for p in built.data_root.iterdir() if p.is_dir()}
    assert not present & {"processed", "exports", "annotation"}
    paths = set(Manifest(built).entries)
    assert {"data/parquet/drugs.parquet", "data/csv/coverage.csv", "data/json/drugs.json",
            "data/reports/data_quality.md"} <= paths  # fmt: skip
    assert not [p for p in paths if not p.startswith("data/")]
    # The layer still says what a file is, whichever folder it is in.
    entries = Manifest(built).entries
    assert entries["data/parquet/drugs.parquet"].layer == "processed"
    assert entries["data/csv/coverage.csv"].layer == "report"
    assert entries["data/json/drugs.json"].layer == "export"


def test_format_folders_follow_the_data_dir_and_can_be_overridden(tmp_path: Path) -> None:
    from synergyrx_data.config import ConfigError, load_settings

    moved = load_settings(None, ["data_dir=elsewhere/store"], project_root=tmp_path, environ={})
    store = tmp_path / "elsewhere" / "store"
    assert (moved.parquet_root, moved.csv_root, moved.json_root, moved.reports_root) == (
        store / "parquet",
        store / "csv",
        store / "json",
        store / "reports",
    )
    custom = load_settings(None, ["csv_dir=out/tables"], project_root=tmp_path, environ={})
    assert custom.csv_root == tmp_path / "out" / "tables"
    assert custom.parquet_root == tmp_path / "data" / "parquet"  # the others stay put
    with pytest.raises(ConfigError):  # keys of the previous layout are rejected, not ignored
        load_settings(None, ["exports_dir=exports"], project_root=tmp_path, environ={})
