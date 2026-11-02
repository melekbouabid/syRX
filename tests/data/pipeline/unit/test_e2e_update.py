"""`synergyrx update` and `synergyrx sources --check` from end to end, on a mocked network."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pandas as pd
import pytest
import respx
from rich.console import Console
from typer.testing import CliRunner

from src.data.synergyrx_data import cli
from src.data.synergyrx_data.cli import app
from src.data.synergyrx_data.connectors import all_links
from src.data.synergyrx_data.quality.linkcheck import ENDPOINTS
from src.data.synergyrx_data.testing import TEST_EMAIL
from tests.data.pipeline.unit.test_connectors_lookup import PUBMED_XML

runner = CliRunner()
PIPELINE = Path(__file__).resolve().parents[4] / "config" / "data_pipeline"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
# DDInter opted in (as the module 2 corpus does), one category file; PubMed on the ingredients.
SOURCES = ["--set", "sources.ddinter.enabled=true",
           "--set", "sources.ddinter.options.terms_acknowledged=true",
           "--set", "sources.ddinter.options.codes=[A]"]  # fmt: skip
FAST = ["--set", "http.safety_factor=1.0", "--set", "http.default_limit.rate=1000",
        "--set", "http.ncbi.rate_without_key=1000",
        "--set", "http.hosts[www.ncbi.nlm.nih.gov].rate=1000"]  # fmt: skip


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A temporary repository laid out like the real one; the pipeline runs from
    config/data_pipeline/."""
    (tmp_path / "config" / "data_pipeline").mkdir(parents=True)
    (tmp_path / "config" / "data_pipeline" / "config.yaml").write_text(
        (PIPELINE / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    classes = tmp_path / "config" / "pill_classes.csv"
    classes.parent.mkdir(exist_ok=True)
    classes.write_text("class_id,class_key,active_ingredients\n0,x_0,Fluorouracil\n", "utf-8")
    monkeypatch.chdir(tmp_path / "config" / "data_pipeline")
    monkeypatch.setattr(cli, "console", Console(width=250))
    monkeypatch.setenv("CONTACT_EMAIL", TEST_EMAIL)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    return tmp_path


def mock_everything(router: respx.MockRouter) -> None:
    router.get(url__regex=r"https://[^/]+/robots\.txt$").respond(404)
    router.get(url__startswith="https://ddinter2.scbdd.com/static/").respond(
        200,
        text="DDInterID_A,Drug_A,DDInterID_B,Drug_B,Level\n"
        "DDInter1,Abacavir,DDInter424,Cobicistat,Moderate\n"
        "DDInter781,Fluorouracil,DDInter1951,Warfarin,Major\n",
        headers={"content-type": "application/octet-stream"},
    )
    router.post(f"{EUTILS}/esearch.fcgi").respond(
        json={"esearchresult": {"count": "2", "webenv": "W", "querykey": "1"}}
    )
    router.get(f"{EUTILS}/efetch.fcgi").respond(
        200, content=PUBMED_XML, headers={"content-type": "text/xml"}
    )


def test_update_fetches_every_enabled_source_and_a_second_run_is_free(project: Path) -> None:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        mock_everything(router)
        first = runner.invoke(app, [*FAST, *SOURCES, "update"])
        assert first.exit_code == 0, first.output
        calls_after_first = len(router.calls)
        second = runner.invoke(app, [*FAST, *SOURCES, "update"])
        assert second.exit_code == 0, second.output
        # Idempotent: the second run reads its caches and stores, and asks the network nothing.
        assert len(router.calls) == calls_after_first

    for relative in (
        "data/interim/ddinter/interactions.parquet", "data/interim/pubmed/articles.parquet",
        "data/interim/pubmed/query_hits.parquet", "data/interim/pubmed/query_log.jsonl",
        "data/state/connectors.json", "config/data_pipeline/manifest.json",
    ):  # fmt: skip
        assert (project / relative).is_file(), relative
    # Everything collected is under data/: nothing is left at the project root.
    assert {p.name for p in project.iterdir()} == {"config", "data"}

    interactions = pd.read_parquet(project / "data/interim/ddinter/interactions.parquet")
    assert set(interactions["level"]) == {"Moderate", "Major"}
    articles = pd.read_parquet(project / "data/interim/pubmed/articles.parquet").set_index("pmid")
    assert set(articles.index) == {"111", "222"} and bool(articles.loc["222", "is_retracted"])

    summary = runner.invoke(app, ["summary"])
    assert "data/interim/pubmed/articles.parquet" in summary.output
    manifest_file = project / "config" / "data_pipeline" / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    layers = {entry["layer"] for entry in manifest["files"].values()}
    assert layers == {"raw", "interim"}
    for entry in manifest["files"].values():
        assert len(entry["sha256"]) == 64 and entry["retrieved_at"]
        if entry["layer"] == "raw":
            assert entry["url"] and entry["license"] and entry["version"]


def test_update_exits_non_zero_when_a_source_needs_attention(project: Path) -> None:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        mock_everything(router)
        router.post(f"{EUTILS}/esearch.fcgi").respond(403)
        result = runner.invoke(app, [*FAST, *SOURCES, "update"])
    assert result.exit_code == 1
    assert "Needs attention" in result.output and "pubmed" in result.output
    # The other sources still ran.
    assert (project / "data/interim/ddinter/interactions.parquet").is_file()


def test_sources_check_records_every_status_and_writes_the_document(project: Path) -> None:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.get("https://ddinter2.scbdd.com/robots.txt").respond(
            200, text="User-agent: *\nDisallow: /static/\n"
        )
        router.get(url__regex=r"https://[^/]+/robots\.txt$").respond(404)
        router.get("https://ddinter2.scbdd.com/terms/").respond(410, html="<html>moved</html>")
        router.get("https://www.ncbi.nlm.nih.gov/books/NBK25501/").respond(
            200, html="<html>We are verifying you are a person, not a bot.</html>"
        )
        router.get("https://open.fda.gov/terms/").mock(side_effect=httpx.ConnectTimeout("slow"))
        blocked = router.get(url__startswith="https://ddinter2.scbdd.com/static/").respond(200)
        router.get(url__regex=r"https://.*").respond(200, html="<html>ok</html>")
        result = runner.invoke(app, [*FAST, "sources", "--check"])
    assert result.exit_code == 0, result.output
    assert not blocked.called  # robots.txt disallows it, so it was not even probed

    document = (project / "docs" / "data_pipeline" / "sources.md").read_text(encoding="utf-8")
    for _, link in all_links():
        assert f"]({link.url})" in document  # every link is listed...
    for _, _, url, _ in ENDPOINTS:
        assert f"]({url})" in document
    rows = {
        line.split("](")[1].split(")")[0]: line
        for line in document.splitlines()
        if line.startswith("| ") and "](" in line
    }
    assert "| 410 |" in rows["https://ddinter2.scbdd.com/terms/"]  # ...with the status it returned
    assert "access wall" in rows["https://www.ncbi.nlm.nih.gov/books/NBK25501/"]
    assert "no answer" in rows["https://open.fda.gov/terms/"]
    assert (
        "robots.txt of ddinter2.scbdd.com disallows this path"
        in rows["https://ddinter2.scbdd.com/static/media/download/ddinter_downloads_code_V.csv"]
    )
    assert "| 200 |" in rows["https://ddinter2.scbdd.com/"]
    assert "CC-BY-NC-SA-4.0" in rows["https://ddinter2.scbdd.com/"]  # licence and terms too
    assert "## Links that need attention" in document and "## Rate limits" in document
    status = json.loads((project / "data/state/link_status.json").read_text(encoding="utf-8"))
    assert len(status["links"]) == len(all_links()) + len(ENDPOINTS)
