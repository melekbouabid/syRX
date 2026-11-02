"""The command line on a temporary repository: dry run, fetch, validate, status, summary."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import respx
from rich.console import Console
from typer.testing import CliRunner

from src.data.tujya_data import cli
from src.data.tujya_data.cli import app
from src.data.tujya_data.config import ConfigError, load_settings
from src.data.tujya_data.manifest import Manifest
from src.data.tujya_data.quality.drift import DriftTracker, compare, snapshot
from src.data.tujya_data.testing import TEST_EMAIL

runner = CliRunner()
DDINTER = "https://ddinter2.scbdd.com/static/media/download/ddinter_downloads_code_A.csv"
DDINTER_ROWS = (
    b"DDInterID_A,Drug_A,DDInterID_B,Drug_B,Level\n"
    b"DDInter1,Abacavir,DDInter424,Cobicistat,Moderate\n"
)
# DDInter opted in for these tests, one category file only, and no waiting between requests.
WITH_DDINTER = ["--set", "sources.ddinter.enabled=true",
                "--set", "sources.ddinter.options.terms_acknowledged=true",
                "--set", "sources.ddinter.options.codes=[A]"]  # fmt: skip
FAST = ["--set", "http.safety_factor=1.0", "--set", "http.default_limit.rate=1000",
        "--set", "http.ncbi.rate_without_key=1000"]  # fmt: skip


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary repository with the shipped config/data_pipeline/config.yaml.

    As in the real repository, the pipeline runs from config/data_pipeline and the project
    root is two levels up: data/ is there, the manifest stays in config/data_pipeline/.
    """
    shipped = Path(__file__).resolve().parents[4] / "config" / "data_pipeline" / "config.yaml"
    (tmp_path / "config" / "data_pipeline").mkdir(parents=True)
    (tmp_path / "config" / "data_pipeline" / "config.yaml").write_text(
        shipped.read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path / "config" / "data_pipeline")
    monkeypatch.setattr(cli, "console", Console(width=250))  # no line wrapping in assertions
    monkeypatch.setenv("CONTACT_EMAIL", TEST_EMAIL)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    monkeypatch.delenv("TUJYA_CONFIG", raising=False)
    monkeypatch.delenv("SYNERGYRX_CONFIG", raising=False)  # legacy name (D86)
    return tmp_path


def tree(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def fetch_ddinter(router: respx.MockRouter) -> None:
    router.get(url__regex=r"https://[^/]+/robots\.txt$").respond(404)
    router.get(DDINTER).respond(
        200, content=DDINTER_ROWS, headers={"content-type": "application/octet-stream"}
    )
    result = runner.invoke(app, [*FAST, *WITH_DDINTER, "--text-logs", "fetch", "-s", "ddinter"])
    assert result.exit_code == 0, result.output


def test_cli_dry_run_prints_the_plan_and_touches_nothing(project: Path) -> None:
    """The acceptance criterion: `fetch --sample 100 --dry-run` changes nothing."""
    before = tree(project)
    with respx.mock(assert_all_mocked=True) as router:  # any request would raise
        result = runner.invoke(app, ["fetch", "--sample", "100", "--dry-run"])
        assert len(router.calls) == 0
    assert result.exit_code == 0, result.output
    assert "Dry run: nothing was requested, written or cached." in result.output
    assert "pubmed" in result.output
    assert "ddinter" not in result.output  # disabled sources are not part of the plan
    assert tree(project) == before
    assert not (project / "data").exists()
    assert not (project / "config" / "data_pipeline" / "manifest.json").exists()


def test_cli_fetch_runs_a_source_and_reports_failures_with_exit_code(project: Path) -> None:
    classes = project / "config" / "pill_classes.csv"
    classes.parent.mkdir(exist_ok=True)
    classes.write_text("class_id,class_key,active_ingredients\n0,x_0,Fluorouracil\n", "utf-8")
    with respx.mock(assert_all_called=False) as router:
        router.get(url__regex=r"https://[^/]+/robots\.txt$").respond(404)
        router.get(DDINTER).respond(
            200, content=DDINTER_ROWS, headers={"content-type": "application/octet-stream"}
        )
        ok = runner.invoke(app, [*FAST, *WITH_DDINTER, "--text-logs", "fetch", "-s", "ddinter"])
        router.post("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi").respond(403)
        failed = runner.invoke(app, [*FAST, "--text-logs", "fetch", "--source", "pubmed"])
    assert ok.exit_code == 0, ok.output
    assert "interactions (1)" in ok.output
    assert failed.exit_code == 1 and "needs_attention" in failed.output
    unknown = runner.invoke(app, ["fetch", "--source", "nope", "--dry-run"])
    assert unknown.exit_code == 2 and "unknown source" in unknown.output

    status = runner.invoke(app, ["status"])
    assert status.exit_code == 0 and "needs_attention" in status.output
    validate = runner.invoke(app, ["validate", "--source", "ddinter"])
    assert validate.exit_code == 0, validate.output
    assert "matches manifest" in validate.output and "0 invalid" in validate.output
    # Tampering with a raw file is caught by `validate`.
    raw = next((project / "data" / "raw" / "ddinter").rglob("ddinter_downloads_code_A.csv"))
    raw.write_bytes(b"changed")
    assert runner.invoke(app, ["validate", "--source", "ddinter"]).exit_code == 1


def test_cli_overrides_and_bad_config(project: Path) -> None:
    result = runner.invoke(app, ["--set", "sources.pubmed.enabled=false", "fetch", "--dry-run"])
    assert result.exit_code == 0 and "eutils" not in result.output
    bad = runner.invoke(app, ["--set", "http.nonsense=1", "fetch", "--dry-run"])
    assert bad.exit_code == 2 and "Configuration error" in bad.output
    assert "tujya-data" in runner.invoke(app, ["--version"]).output


def test_cli_offline_commands_on_collected_data(project: Path) -> None:
    with respx.mock(assert_all_called=False) as router:
        fetch_ddinter(router)
    summary = runner.invoke(app, ["summary"])
    assert summary.exit_code == 0, summary.output
    assert "data/interim/ddinter/interactions.parquet" in summary.output
    payload = json.loads(runner.invoke(app, ["summary", "--json"]).output)
    assert payload["outputs"] == {"data/interim/ddinter/interactions.parquet": 1}
    assert {row["source"] for row in payload["sources"]} == {"ddinter", "pubmed", "openfda_label"}
    sources = runner.invoke(app, ["sources"])
    assert sources.exit_code == 0 and "openfda_label" in sources.output
    preflight = runner.invoke(app, ["preflight"])
    assert preflight.exit_code == 0 and "contact_email" in preflight.output


def test_cli_commands_explain_missing_inputs(project: Path) -> None:
    summary = runner.invoke(app, ["summary"])
    assert summary.exit_code == 0 and "Nothing collected yet" in summary.output


def test_sample_data_is_labelled_everywhere(project: Path) -> None:
    with respx.mock(assert_all_called=False) as router:
        fetch_ddinter(router)
    settings = load_settings(project / "config" / "data_pipeline" / "config.yaml")
    manifest = Manifest(settings)
    path = settings.interim_dir / "ddinter" / "interactions.parquet"
    manifest.record_file(path, source="ddinter", layer="interim", is_sample=True)
    manifest.save()
    assert "Sample data present for: ddinter" in runner.invoke(app, ["status"]).output
    assert "SAMPLE DATA for: ddinter" in runner.invoke(app, ["summary"]).output
    payload = json.loads(runner.invoke(app, ["summary", "--json"]).output)
    assert payload["sample_sources"] == ["ddinter"]


def test_collected_files_stay_under_the_data_folder(project: Path) -> None:
    with respx.mock(assert_all_called=False) as router:
        fetch_ddinter(router)
    produced = {p.relative_to(project).parts[0] for p in project.rglob("*") if p.is_file()}
    assert produced == {"config", "data"}  # nothing else at the root...
    # ...and only the pipeline's own records next to it (paths of the shipped config.yaml).
    records = {p.name for p in (project / "config").rglob("*") if p.is_file()}
    assert records == {"config.yaml", "manifest.json"}
    entries = Manifest(load_settings(project / "config" / "data_pipeline" / "config.yaml")).entries
    assert entries and not [p for p in entries if not p.startswith("data/")]
    assert {entry.layer for entry in entries.values()} == {"raw", "interim"}
    interactions = pd.read_parquet(project / "data/interim/ddinter/interactions.parquet")
    assert list(interactions["drug_a_name"]) == ["Abacavir"]


def test_working_folders_follow_the_data_dir(tmp_path: Path) -> None:
    moved = load_settings(None, ["data_dir=elsewhere/store"], project_root=tmp_path, environ={})
    store = tmp_path / "elsewhere" / "store"
    assert (moved.raw_dir, moved.interim_dir, moved.manual_dir, moved.state_dir) == (
        store / "raw",
        store / "interim",
        store / "manual",
        store / "state",
    )
    # Keys of earlier layouts are rejected, not ignored: the synergy outputs are gone (D85).
    for key in ("exports_dir=exports", "parquet_dir=out", "csv_dir=out"):
        with pytest.raises(ConfigError):
            load_settings(None, [key], project_root=tmp_path, environ={})


# ------------------------------------------------------------------- drift


def test_schema_drift_between_versions_is_reported(tmp_path: Path) -> None:
    v1 = pd.DataFrame({"drug_row": ["a"] * 100, "score": [1.0] * 100, "old": [1] * 100})
    v2 = pd.DataFrame({"drug_row": ["a"] * 40, "score": ["1.0"] * 40, "study_name": ["s"] * 40})
    tracker = DriftTracker(tmp_path / "schemas")
    assert [f.kind for f in tracker.check("interim/x/table", v1, "1.4")] == ["first_snapshot"]
    assert tracker.check("interim/x/table", v1, "1.4") == []  # same data: no drift
    findings = {f.kind: f for f in tracker.check("interim/x/table", v2, "1.5")}
    assert set(findings) == {"column_added", "column_removed", "dtype_changed", "rows_changed"}
    assert findings["column_removed"].detail == "old"
    assert "study_name" in findings["column_added"].detail
    assert findings["dtype_changed"].detail.startswith("score: float64 -> ")
    assert "-60%" in findings["rows_changed"].detail
    assert len(tracker.history("interim/x/table")) == 2
    # A sample build is not compared on row counts with a full build.
    sample = snapshot(v1.head(5), "1.4", is_sample=True)
    assert compare("t", snapshot(v1, "1.4"), sample) == []


def test_wide_tables_are_fingerprinted_not_listed() -> None:
    wide = pd.DataFrame({f"gene{i}": [0.0] for i in range(600)})
    shot = snapshot(wide, "24Q4")
    assert "columns" not in shot and len(shot["columns_sha256"]) == 64
    narrower = snapshot(wide.iloc[:, :599], "25Q2")
    assert [f.kind for f in compare("expression", shot, narrower)] == ["column_removed"]
