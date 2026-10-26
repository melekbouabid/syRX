"""Optional smoke tests against the real services.

They are excluded from the default run. Start them with `pytest tests/live -m live` (or
`make live`) after putting CONTACT_EMAIL in `.env`. Each test asks for a few kilobytes: one
record, one name, the first rows of one file.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from synergyrx_data.config import Settings, load_settings
from synergyrx_data.connectors import allowed_hosts
from synergyrx_data.http.client import PoliteClient
from synergyrx_data.literature.abstracts import parse_pubmed_xml

pytestmark = pytest.mark.live
REPO = Path(__file__).resolve().parents[2]
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Real rate limits and the real identity from `.env`, but a throwaway data directory."""
    built = load_settings(REPO / "config.yaml", [f"data_dir={(tmp_path / 'data').as_posix()}"])
    if not built.secrets.contact_email and os.environ.get("SYNERGYRX_ALLOW_ANONYMOUS") != "1":
        pytest.skip("CONTACT_EMAIL is not set in .env: live requests are refused without it")
    if not built.secrets.contact_email:
        built.http.require_contact_email = False
    return built


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[PoliteClient]:
    async with PoliteClient(settings, allowed_hosts=allowed_hosts()) as live:
        yield live


def identity(settings: Settings) -> dict[str, str]:
    params = {"tool": settings.secrets.ncbi_tool}
    if settings.secrets.contact_email:
        params["email"] = settings.secrets.contact_email
    if settings.secrets.api_key:
        params["api_key"] = settings.secrets.api_key
    return params


async def test_drugcomb_first_rows(client: PoliteClient) -> None:
    url = "https://zenodo.org/api/records/11102665/files/summary_table_v1.4.csv/content"
    head = await client.head_lines(url, 4, source="drugcomb")
    lines = head.decode("utf-8").splitlines()
    assert len(lines) == 4
    assert lines[0].startswith('"block_id","drug_row","drug_col","cell_line_name"')
    assert "synergy_loewe" in lines[0]


async def test_oneil_first_rows(client: PoliteClient) -> None:
    head = await client.head_lines(
        "https://bioinf.jku.at/software/DeepSynergy/labels.csv", 3, source="oneil"
    )
    assert head.decode().splitlines()[0] == ",drug_a_name,drug_b_name,cell_line,synergy,fold"


async def test_pubchem_name_to_structure(client: PoliteClient) -> None:
    base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
    cids = await client.fetch(
        f"{base}/name/cids/JSON", source="pubchem", method="POST", data={"name": "aspirin"}
    )
    assert cids.json()["IdentifierList"]["CID"] == [2244]
    props = await client.fetch(
        f"{base}/cid/2244/property/SMILES,ConnectivitySMILES,InChIKey,Title/JSON", source="pubchem"
    )
    record = props.json()["PropertyTable"]["Properties"][0]
    assert record["InChIKey"] == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N" and record["SMILES"]


async def test_chembl_by_inchikey(client: PoliteClient) -> None:
    result = await client.fetch(
        "https://www.ebi.ac.uk/chembl/api/data/molecule.json",
        source="chembl",
        params={
            "molecule_structures__standard_inchi_key__in": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
            "only": "molecule_chembl_id,pref_name",
        },
    )
    assert result.json()["molecules"][0]["molecule_chembl_id"] == "CHEMBL25"


async def test_pubmed_history_search_and_fetch(client: PoliteClient, settings: Settings) -> None:
    search = await client.fetch(
        f"{EUTILS}/esearch.fcgi",
        source="pubmed",
        method="POST",
        data={
            "db": "pubmed",
            "term": '"Drug Synergism"[mh] AND fluorouracil[tiab] AND veliparib[tiab]',
            "usehistory": "y",
            "retmax": "0",
            "retmode": "json",
            **identity(settings),
        },
        use_cache=False,
    )
    result = search.json()["esearchresult"]
    assert int(result["count"]) >= 1 and result["webenv"]
    fetched = await client.fetch(
        f"{EUTILS}/efetch.fcgi",
        source="pubmed",
        params={
            "db": "pubmed",
            "query_key": result["querykey"],
            "WebEnv": result["webenv"],
            "retstart": "0",
            "retmax": "2",
            "retmode": "xml",
            **identity(settings),
        },
        expect=("xml",),
        use_cache=False,
    )
    articles = parse_pubmed_xml(fetched.content)
    assert 1 <= len(articles) <= 2 and articles[0]["pmid"].isdigit()
    assert "api_key" not in fetched.url and "email" not in fetched.url


async def test_europepmc_open_access_flag(client: PoliteClient) -> None:
    result = await client.fetch(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
        source="europepmc",
        params={"query": "EXT_ID:31600000 AND SRC:MED", "format": "json", "resultType": "core"},
    )
    hit = result.json()["resultList"]["result"][0]
    assert hit["pmid"] == "31600000" and hit["isOpenAccess"] in ("Y", "N")


async def test_depmap_release_listing(client: PoliteClient) -> None:
    result = await client.fetch("https://api.figshare.com/v2/articles/27993248", source="depmap")
    files = {entry["name"]: entry for entry in result.json()["files"]}
    assert "Model.csv" in files and files["Model.csv"]["computed_md5"]
    assert "OmicsExpressionProteinCodingGenesTPMLogp1.csv" in files
