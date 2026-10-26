"""`synergyrx update` and `synergyrx sources --check` from end to end, on a mocked network."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pandas as pd
import pytest
import respx
from rich.console import Console
from typer.testing import CliRunner

from synergyrx_data import cli
from synergyrx_data.cli import app
from synergyrx_data.connectors import all_links
from synergyrx_data.quality.linkcheck import ENDPOINTS
from tests.conftest import TEST_EMAIL
from tests.unit.test_connectors_lookup import PUBMED_XML

runner = CliRunner()
REPO = Path(__file__).resolve().parents[2]
FU = "O=c1[nH]cc(F)c(=O)[nH]1"
VELIPARIB = "CC1(c2nc3c(C(N)=O)cccc3[nH]2)CCCN1"
ONLY_ONEIL = [
    arg
    for source in ("drugcomb", "tdc", "nci_almanac", "depmap")
    for arg in ("--set", f"sources.{source}.enabled=false")
]
FAST = ["--set", "http.safety_factor=1.0", "--set", "http.default_limit.rate=1000",
        "--set", "http.ncbi.rate_without_key=1000",
        "--set", "http.hosts[pubchem.ncbi.nlm.nih.gov].rate=1000",
        "--set", "http.hosts[www.ebi.ac.uk].rate=1000"]  # fmt: skip


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "config.yaml").write_text(
        (REPO / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "console", Console(width=250))
    monkeypatch.setenv("CONTACT_EMAIL", TEST_EMAIL)
    monkeypatch.delenv("NCBI_API_KEY", raising=False)
    return tmp_path


def mock_everything(router: respx.MockRouter) -> None:
    router.get(url__regex=r"https://[^/]+/robots\.txt$").respond(404)
    base = "https://bioinf.jku.at/software/DeepSynergy/"
    labels = (
        ",drug_a_name,drug_b_name,cell_line,synergy,fold\n"
        "a,5-FU,ABT-888,A2058,7.69,2\nb,ABT-888,5-FU,A2058,9.69,2\nc,5-FU,ABT-888,HT29,-4.0,1\n"
    )
    router.get(base + "labels.csv").respond(200, content=labels.encode())
    router.get(base + "smiles.csv").respond(
        200, content=f"5-FU,{FU}\nABT-888,{VELIPARIB}\n".encode()
    )

    pug = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
    cids = {"5-FU": 3385, "ABT-888": 11960529}

    def name_lookup(request: httpx.Request) -> httpx.Response:
        name = parse_qs(request.content.decode())["name"][0]
        return httpx.Response(200, json={"IdentifierList": {"CID": [cids[name]]}})

    router.post(f"{pug}/name/cids/JSON").mock(side_effect=name_lookup)
    router.get(url__regex=rf"{pug}/cid/[\d,]+/property/.+").respond(
        json={
            "PropertyTable": {
                "Properties": [
                    {
                        "CID": 3385,
                        "SMILES": FU,
                        "InChIKey": "GHASVSINZRGABV-UHFFFAOYSA-N",
                        "Title": "Fluorouracil",
                    },
                    {
                        "CID": 11960529,
                        "SMILES": VELIPARIB,
                        "InChIKey": "JNAHVYVRKWKWKQ-UHFFFAOYSA-N",
                        "Title": "Veliparib",
                    },
                ]
            }
        }
    )
    router.get(url__regex=rf"{pug}/cid/[\d,]+/synonyms/JSON").respond(
        json={
            "InformationList": {
                "Information": [
                    {"CID": 3385, "Synonym": ["fluorouracil", "5-Fluorouracil", "Adrucil"]},
                    {"CID": 11960529, "Synonym": ["veliparib", "ABT-888"]},
                ]
            }
        }
    )
    router.get("https://www.ebi.ac.uk/chembl/api/data/molecule.json").respond(
        json={
            "molecules": [
                {
                    "molecule_chembl_id": "CHEMBL185",
                    "pref_name": "FLUOROURACIL",
                    "max_phase": "4.0",
                    "molecule_structures": {"standard_inchi_key": "GHASVSINZRGABV-UHFFFAOYSA-N"},
                },
            ]
        }
    )
    eutils = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    router.post(f"{eutils}/esearch.fcgi").respond(
        json={"esearchresult": {"count": "2", "webenv": "W", "querykey": "1"}}
    )
    router.get(f"{eutils}/efetch.fcgi").respond(
        200, content=PUBMED_XML, headers={"content-type": "text/xml"}
    )
    router.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search").respond(
        json={"resultList": {"result": [{"pmid": "111", "isOpenAccess": "Y", "inEPMC": "Y"}]}}
    )


def test_update_builds_everything_and_a_second_run_is_free(project: Path) -> None:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        mock_everything(router)
        first = runner.invoke(app, [*FAST, *ONLY_ONEIL, "update"])
        assert first.exit_code == 0, first.output
        calls_after_first = len(router.calls)
        second = runner.invoke(app, [*FAST, *ONLY_ONEIL, "update"])
        assert second.exit_code == 0, second.output
        # Idempotent: the second run reads its caches and stores, and asks the network nothing.
        assert len(router.calls) == calls_after_first

    for relative in (
        "data/parquet/synergy_pairs.parquet", "data/parquet/drugs.parquet",
        "data/parquet/cell_lines.parquet", "data/parquet/abstracts.parquet",
        "data/parquet/candidate_sentences.parquet", "data/json/drugs.json",
        "data/json/cell_lines.json", "data/json/mock_predictions_seed.json",
        "data/json/citations.json", "data/reports/data_quality.md",
        "data/reports/data_quality.html", "data/csv/coverage.csv",
        "data/csv/conflicts.csv", "data/csv/sample_sentences.csv",
        "LICENSES.md", "manifest.json",
    ):  # fmt: skip
        assert (project / relative).is_file(), relative
    # Everything generated is under data/: nothing is left at the project root.
    assert not [d for d in ("exports", "reports", "annotation") if (project / d).exists()]

    pairs = pd.read_parquet(project / "data/parquet/synergy_pairs.parquet")
    assert len(pairs) == 2  # (A,B) and (B,A) on A2058 became one row; HT29 is the other
    assert sorted(pairs["n_replicates"]) == [1, 2]
    drugs = pd.read_parquet(project / "data/parquet/drugs.parquet").set_index("name")
    assert set(drugs.index) == {"Fluorouracil", "Veliparib"}
    assert drugs.loc["Fluorouracil", "resolution_method"] == "source_smiles+pubchem"
    assert drugs.loc["Fluorouracil", "chembl_id"] == "CHEMBL185"
    abstracts = pd.read_parquet(project / "data/parquet/abstracts.parquet").set_index("pmid")
    assert bool(abstracts.loc["111", "is_open_access"]) and bool(
        abstracts.loc["222", "is_retracted"]
    )
    candidates = pd.read_parquet(project / "data/parquet/candidate_sentences.parquet")
    assert "synergistic" in candidates["sentence"].iloc[0]

    summary = runner.invoke(app, ["summary"])
    assert "Distinct drugs across all sources: 2" in summary.output
    manifest = json.loads((project / "manifest.json").read_text(encoding="utf-8"))
    layers = {entry["layer"] for entry in manifest["files"].values()}
    assert layers == {"raw", "interim", "processed", "report", "export"}
    for entry in manifest["files"].values():
        assert len(entry["sha256"]) == 64 and entry["retrieved_at"]
        if entry["layer"] == "raw":
            assert entry["url"] and entry["license"] and entry["version"]


def test_update_exits_non_zero_when_a_source_needs_attention(project: Path) -> None:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        mock_everything(router)
        router.post("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/cids/JSON").respond(
            403
        )
        result = runner.invoke(app, [*FAST, *ONLY_ONEIL, "update"])
    assert result.exit_code == 1
    assert "Needs attention" in result.output and "pubchem" in result.output
    # The rest of the pipeline still ran on the structures the dataset itself provides.
    drugs = pd.read_parquet(project / "data/parquet/drugs.parquet")
    assert set(drugs["resolution_method"]) == {"source_smiles"}


def test_sources_check_records_every_status_and_writes_the_document(project: Path) -> None:
    with respx.mock(assert_all_mocked=True, assert_all_called=False) as router:
        router.get("https://wiki.nci.nih.gov/robots.txt").respond(
            200, text="User-agent: *\nDisallow: /\nAllow: /spaces/*/pages/\n"
        )
        router.get(url__regex=r"https://[^/]+/robots\.txt$").respond(404)
        router.get("https://www.cancerrxgene.org/").respond(410, html="<html>moved</html>")
        router.get("https://depmap.org/portal/").respond(
            200, html="<html>We are verifying you are a person, not a bot.</html>"
        )
        router.get("https://go.drugbank.com/").mock(side_effect=httpx.ConnectTimeout("slow"))
        blocked = router.get(url__startswith="https://wiki.nci.nih.gov/download/").respond(200)
        router.get(url__regex=r"https://.*").respond(200, html="<html>ok</html>")
        result = runner.invoke(app, [*FAST, "sources", "--check"])
    assert result.exit_code == 0, result.output
    assert not blocked.called  # robots.txt disallows it, so it was not even probed

    document = (project / "docs" / "SOURCES.md").read_text(encoding="utf-8")
    for _, link in all_links():
        assert f"]({link.url})" in document  # every link is listed...
    for _, _, url, _ in ENDPOINTS:
        assert f"]({url})" in document
    rows = {
        line.split("](")[1].split(")")[0]: line
        for line in document.splitlines()
        if line.startswith("| ") and "](" in line
    }
    assert "| 410 |" in rows["https://www.cancerrxgene.org/"]  # ...with the status it returned
    assert "access wall" in rows["https://depmap.org/portal/"]
    assert "no answer" in rows["https://go.drugbank.com/"]
    assert (
        "robots.txt of wiki.nci.nih.gov disallows this path"
        in rows[
            "https://wiki.nci.nih.gov/download/attachments/338237347/ComboDrugGrowth_Nov2017.zip"
        ]
    )
    assert "| 200 |" in rows["https://zenodo.org/records/11102665"]
    assert "CC-BY-4.0" in rows["https://zenodo.org/records/11102665"]  # licence and terms too
    assert "## Links that need attention" in document and "## Rate limits" in document
    status = json.loads((project / "data/state/link_status.json").read_text(encoding="utf-8"))
    assert len(status["links"]) == len(all_links()) + len(ENDPOINTS)
