"""Guards for the source-access and compliance policy (docs/CODE_REVIEW.md)."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from synergyrx_data.connectors import REGISTRY, all_links, allowed_hosts
from synergyrx_data.quality.linkcheck import ENDPOINTS

SRC = Path(__file__).resolve().parents[2] / "src" / "synergyrx_data"
CONNECTOR_FILES = sorted((SRC / "connectors").glob("*.py"))
BRIEF_LINKS = [
    "https://drugcomb.org/",
    "https://zenodo.org/records/11102665",
    "https://arxiv.org/pdf/2102.09548",
    "https://bioinf.jku.at/software/DeepSynergy",
    "https://dtp.cancer.gov/ncialmanac",
    "https://ddinter2.scbdd.com/",
    "https://sites.broadinstitute.org/ccle/datasets",
    "https://forum.depmap.org/t/depmap-portal-datasets/3695",
    "https://depmap.org/portal/",
    "https://www.cancerrxgene.org/",
    "https://www.ncbi.nlm.nih.gov/books/NBK25501/",
    "https://dataguide.nlm.nih.gov/eutilities/how_eutilities_works.html",
    "https://dtd.nlm.nih.gov/ncbi/pubmed/doc/out/190101/index.html",
    "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/",
    "https://europepmc.org/RestfulWebService",
    "https://www.ncbi.nlm.nih.gov/research/pubtator3/",
    "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
    "https://www.ebi.ac.uk/chembl/api/data/docs",
    "https://go.drugbank.com/",
]


def calls(tree: ast.AST) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call)]


@pytest.mark.parametrize("path", CONNECTOR_FILES, ids=lambda p: p.name)
def test_no_connector_fetches_or_parses_html(path: Path) -> None:
    """API first, bulk second, HTML last: today no connector needs HTML at all."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for call in calls(tree):
        name = getattr(call.func, "attr", getattr(call.func, "id", ""))
        if name == "Link":
            continue  # links are only probed for their status by `sources --check`
        for keyword in call.keywords:
            if keyword.arg in ("kind", "expect"):
                assert "html" not in ast.unparse(keyword.value), (
                    f"{path.name} requests HTML: see docs/CODE_REVIEW.md"
                )
    assert not re.search(
        r"\b(bs4|BeautifulSoup|lxml\.html|html\.parser|selenium|playwright)\b", source
    )


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: p.name)
def test_all_network_access_goes_through_the_polite_client(path: Path) -> None:
    if path.parent.name == "http":
        return  # the client itself is the one place allowed to use httpx
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert not imported & {"requests", "urllib3", "aiohttp", "socket", "ftplib", "subprocess"}
    for call in calls(tree):
        text = ast.unparse(call.func)
        assert not text.startswith(
            ("httpx.get", "httpx.post", "httpx.Client", "httpx.AsyncClient", "urllib.request")
        )


def test_every_source_declares_licence_terms_hosts_and_citation() -> None:
    for name, connector in REGISTRY.items():
        info = connector.info
        assert info.license and info.terms_url and info.citation, name
        assert info.hosts, name
        assert info.category in {"synergy", "cell_line", "chemistry", "literature", "interaction"}
        assert info.access in {"api", "bulk", "package", "manual"}
        for link in info.links:
            assert urlsplit(link.url).hostname in info.hosts, f"{name}: {link.url}"


def test_licence_gated_and_optional_sources_are_off_by_default() -> None:
    for name in ("drugbank", "ddinter", "gdsc", "pubmed_baseline", "pubtator"):
        info = REGISTRY[name].info
        assert info.optional and not info.default_enabled, name
    assert REGISTRY["drugbank"].info.access == "manual"


def test_drugbank_connector_contains_no_network_code() -> None:
    source = (SRC / "connectors" / "drugbank_manual.py").read_text(encoding="utf-8")
    assert "self.client" not in source and "fetch_file" not in source


def test_every_link_from_the_brief_is_registered_for_the_link_check() -> None:
    registered = {link.url for _, link in all_links()}
    assert not [url for url in BRIEF_LINKS if url not in registered]
    marks = {link.url: link.origin for _, link in all_links()}
    assert marks["https://zenodo.org/records/11102665"] == "checked"
    assert marks["https://go.drugbank.com/"] == "verify"


def test_every_data_endpoint_is_on_a_declared_host() -> None:
    hosts = allowed_hosts()
    for source, _, url, _ in ENDPOINTS:
        assert source in REGISTRY
        assert urlsplit(url).hostname in hosts, url
        assert urlsplit(url).scheme == "https"
