"""Identifier, literature and licence-gated connectors with mocked HTTP."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pandas as pd
import pytest
import respx
from pydantic import SecretStr

from synergyrx_data import pipeline
from synergyrx_data.config import Settings, SourceConfig
from synergyrx_data.connectors._synergy import synergy_frame
from synergyrx_data.provenance import has_provenance, write_parquet
from synergyrx_data.state import MANUAL_STEP, NEEDS_ATTENTION
from tests.conftest import FAKE_API_KEY, TEST_EMAIL

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
CHEMBL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
ASPIRIN_KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


async def run(settings: Settings, source: str) -> pipeline.SourceRun:
    return (await pipeline.fetch(settings, [source]))[0]


def write_synergy(settings: Settings, names: list[tuple[str, str]]) -> None:
    frame = synergy_frame(
        pd.DataFrame(
            [
                {"drug_a_name": a, "drug_b_name": b, "cell_line_name": "A2058", "synergy_zip": 1.0}
                for a, b in names
            ]
        )
    ).assign(
        source="screen",
        source_url="u",
        retrieved_at="2026-01-01T00:00:00Z",
        source_version="1",
        license="CC0",
    )
    write_parquet(frame, settings.interim_dir / "screen" / "synergy.parquet")


# ----------------------------------------------------------------- PubChem


@pytest.fixture
def pubchem(settings: Settings, mock_http: respx.MockRouter) -> dict[str, respx.Route]:
    write_synergy(settings, [("Aspirin", "Glucose"), ("Aspirin", "Nonexistium")])
    cids = {"Aspirin": [2244], "Glucose": [5793, 107526, 64689]}

    def name_lookup(request: httpx.Request) -> httpx.Response:
        name = parse_qs(request.content.decode())["name"][0]
        if name not in cids:
            fault = {"Fault": {"Code": "PUGREST.NotFound", "Message": "No CID found"}}
            return httpx.Response(404, json=fault)
        return httpx.Response(200, json={"IdentifierList": {"CID": cids[name]}})

    properties = {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": 2244,
                    "SMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
                    "ConnectivitySMILES": "CC(=O)OC1=CC=CC=C1C(=O)O",
                    "InChIKey": ASPIRIN_KEY,
                    "Title": "Aspirin",
                },
                {
                    "CID": 5793,
                    "SMILES": "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",
                    "ConnectivitySMILES": "C(C1C(C(C(C(O1)O)O)O)O)O",
                    "InChIKey": "WQZGKKKJIJFFOK-GASJEMHNSA-N",
                    "Title": "D-Glucose",
                },
            ]
        }
    }
    synonyms = {
        "InformationList": {
            "Information": [
                {"CID": 2244, "Synonym": ["aspirin", "Acetylsalicylic acid", "50-78-2"]},
                {"CID": 5793, "Synonym": ["D-Glucose", "dextrose"]},
            ]
        }
    }
    return {
        "names": mock_http.post(f"{PUG}/name/cids/JSON").mock(side_effect=name_lookup),
        "properties": mock_http.get(url__regex=rf"{PUG}/cid/[\d,]+/property/.+").respond(
            json=properties
        ),
        "synonyms": mock_http.get(url__regex=rf"{PUG}/cid/[\d,]+/synonyms/JSON").respond(
            json=synonyms
        ),
    }


async def test_pubchem_resolves_names_and_flags_ambiguity(
    settings: Settings, pubchem: dict[str, respx.Route]
) -> None:
    result = await run(settings, "pubchem")
    assert result.outcome == "ok" and result.tables == {"compounds": 3}
    table = pd.read_parquet(settings.interim_dir / "pubchem" / "compounds.parquet").set_index(
        "query_name"
    )
    assert table.loc["Aspirin", "status"] == "found"
    assert table.loc["Aspirin", "pubchem_cid"] == 2244
    assert table.loc["Aspirin", "inchikey"] == ASPIRIN_KEY
    assert "Acetylsalicylic acid" in list(table.loc["Aspirin", "synonyms"])
    assert table.loc["Glucose", "status"] == "ambiguous" and table.loc["Glucose", "n_cids"] == 3
    assert table.loc["Glucose", "pubchem_cid"] == 5793  # PubChem's first, best-ranked hit
    assert table.loc["Nonexistium", "status"] == "not_found"
    assert pd.isna(table.loc["Nonexistium", "pubchem_cid"])
    assert has_provenance(table.reset_index())
    assert table.loc["Aspirin", "source_url"] == "https://pubchem.ncbi.nlm.nih.gov/compound/2244"
    # Properties and synonyms are requested in batches, not one call per compound.
    assert pubchem["properties"].call_count == 1 and pubchem["synonyms"].call_count == 1
    assert "2244,5793" in str(pubchem["properties"].calls.last.request.url)


async def test_pubchem_second_run_asks_nothing_and_new_names_are_incremental(
    settings: Settings, pubchem: dict[str, respx.Route], mock_http: respx.MockRouter
) -> None:
    await run(settings, "pubchem")
    calls = len(mock_http.calls)
    again = await run(settings, "pubchem")
    assert len(mock_http.calls) == calls and again.tables == {"compounds": 3}
    # A not-found name is remembered too: it is not asked again on every run.
    write_synergy(
        settings, [("Aspirin", "Glucose"), ("Aspirin", "Nonexistium"), ("Aspirin", "New")]
    )
    await run(settings, "pubchem")
    assert pubchem["names"].call_count == 4  # only "New" was looked up


async def test_pubchem_outage_keeps_progress_and_reports_needs_attention(
    settings: Settings, mock_http: respx.MockRouter
) -> None:
    write_synergy(settings, [("Aspirin", "Glucose")])
    mock_http.post(f"{PUG}/name/cids/JSON").respond(503, headers={"retry-after": "0"})
    settings.http.retry.max_attempts = 2
    settings.http.retry.backoff_base = 0.01
    result = await run(settings, "pubchem")
    assert result.outcome == NEEDS_ATTENTION and "lookups failed" in result.message


# ------------------------------------------------------------------ ChEMBL


async def test_chembl_by_inchikey_then_by_name(
    settings: Settings, pubchem: dict[str, respx.Route], mock_http: respx.MockRouter
) -> None:
    await run(settings, "pubchem")

    def molecules(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        if "molecule_structures__standard_inchi_key__in" in params:
            hit = {
                "molecule_chembl_id": "CHEMBL25",
                "pref_name": "ASPIRIN",
                "max_phase": "4.0",
                "molecule_structures": {"standard_inchi_key": ASPIRIN_KEY},
            }
            return httpx.Response(200, json={"molecules": [hit]})
        if params.get("pref_name__iexact") == "Nonexistium":
            return httpx.Response(200, json={"molecules": []})
        hit = {
            "molecule_chembl_id": "CHEMBL999",
            "pref_name": "NONEXISTIUM",
            "max_phase": None,
            "molecule_structures": None,
        }
        return httpx.Response(200, json={"molecules": [hit]})

    route = mock_http.get(CHEMBL).mock(side_effect=molecules)
    result = await run(settings, "chembl")
    assert result.outcome == "ok" and result.tables == {"molecules": 2}
    table = pd.read_parquet(settings.interim_dir / "chembl" / "molecules.parquet").set_index(
        "query"
    )
    assert table.loc[ASPIRIN_KEY, "chembl_id"] == "CHEMBL25"
    assert table.loc[ASPIRIN_KEY, "max_phase"] == 4.0
    assert table.loc["Nonexistium", "query_type"] == "name"  # found through the synonym search
    assert set(table["license"]) == {"CC-BY-SA-3.0"}
    assert route.call_count == 3  # one batch of keys, then name and synonym for the miss
    await run(settings, "chembl")
    assert route.call_count == 3  # misses are remembered as well


# ------------------------------------------------------------------ PubMed

PUBMED_XML = b"""<?xml version="1.0" ?>
<PubmedArticleSet>
<PubmedArticle>
  <MedlineCitation>
    <PMID Version="1">111</PMID>
    <Article>
      <Journal><JournalIssue><PubDate><Year>2019</Year></PubDate></JournalIssue>
        <Title>Journal of Tests</Title><ISOAbbreviation>J Tests</ISOAbbreviation></Journal>
      <ArticleTitle>Veliparib and <i>5-FU</i> act synergistically.</ArticleTitle>
      <ELocationID EIdType="doi">10.1000/TEST.111</ELocationID>
      <Abstract>
        <AbstractText Label="BACKGROUND">Fluorouracil (5-FU) is widely used.</AbstractText>
        <AbstractText Label="RESULTS">The combination of veliparib and fluorouracil was
          synergistic in A2058 cells (Fig. 2). Dasatinib alone had no effect.</AbstractText>
      </Abstract>
      <Language>eng</Language>
      <PublicationTypeList><PublicationType>Journal Article</PublicationType></PublicationTypeList>
    </Article>
    <MeshHeadingList><MeshHeading><DescriptorName>Drug Synergism</DescriptorName></MeshHeading>
    </MeshHeadingList>
  </MedlineCitation>
  <PubmedData><ArticleIdList><ArticleId IdType="pubmed">111</ArticleId>
    <ArticleId IdType="pmc">PMC111</ArticleId></ArticleIdList></PubmedData>
</PubmedArticle>
<PubmedArticle>
  <MedlineCitation>
    <PMID Version="1">222</PMID>
    <Article>
      <Journal><JournalIssue><PubDate><MedlineDate>2015 Jan-Feb</MedlineDate></PubDate>
        </JournalIssue><Title>Zhonghua Test</Title></Journal>
      <ArticleTitle>A withdrawn review.</ArticleTitle>
      <Language>chi</Language>
      <PublicationTypeList><PublicationType>Review</PublicationType>
        <PublicationType>Retracted Publication</PublicationType></PublicationTypeList>
    </Article>
    <CommentsCorrectionsList><CommentsCorrections RefType="RetractionIn"><PMID>333</PMID>
      </CommentsCorrections></CommentsCorrectionsList>
  </MedlineCitation>
  <PubmedData><ArticleIdList><ArticleId IdType="doi">10.1000/x.222</ArticleId></ArticleIdList>
  </PubmedData>
</PubmedArticle>
<PubmedBookArticle><BookDocument><PMID>999</PMID></BookDocument></PubmedBookArticle>
</PubmedArticleSet>
"""

FU_ID, VELI_ID = "GHASVSINZRGABV-UHFFFAOYSA-N", "JNAHVYVRKWKWKQ-UHFFFAOYSA-N"


def write_processed(settings: Settings) -> None:
    drugs = pd.DataFrame(
        {
            "drug_id": [FU_ID, VELI_ID, "name:dasatinib"],
            "name": ["Fluorouracil", "Veliparib", "Dasatinib"],
            "synonyms": [["5-FU", "5-fluorouracil", "51-21-8"], ["ABT-888"], []],
        }
    )
    pairs = pd.DataFrame(
        {
            "pair_key": [f"{FU_ID}|{VELI_ID}"] * 2,
            "drug_a_id": [FU_ID] * 2,
            "drug_b_id": [VELI_ID] * 2,
            "cell_line_id": ["ACH-1", "ACH-2"],
        }
    )
    write_parquet(drugs, settings.parquet_root / "drugs.parquet")
    write_parquet(pairs, settings.parquet_root / "synergy_pairs.parquet")


@pytest.fixture
def pubmed(settings: Settings, mock_http: respx.MockRouter) -> dict[str, respx.Route]:
    write_processed(settings)
    search = {
        "esearchresult": {
            "count": "2",
            "webenv": "MCID_abc",
            "querykey": "1",
            "querytranslation": "translated",
        }
    }
    return {
        "esearch": mock_http.post(f"{EUTILS}/esearch.fcgi").respond(json=search),
        "efetch": mock_http.get(f"{EUTILS}/efetch.fcgi").respond(
            200, content=PUBMED_XML, headers={"content-type": "text/xml; charset=UTF-8"}
        ),
    }


async def test_pubmed_history_search_fetch_and_flags(
    settings: Settings, pubmed: dict[str, respx.Route]
) -> None:
    result = await run(settings, "pubmed")
    assert result.outcome == "ok"
    assert result.tables == {"articles": 2, "query_hits": 2}

    search = parse_qs(pubmed["esearch"].calls.last.request.content.decode())
    assert search["usehistory"] == ["y"] and search["db"] == ["pubmed"]
    assert search["tool"] == ["synergyrx-data"] and search["email"] == [TEST_EMAIL]
    assert (search["mindate"], search["datetype"]) == (["2000/01/01"], ["edat"])
    term = search["term"][0]
    assert '"Fluorouracil"[tiab]' in term and '"5-FU"[tiab]' in term and '"ABT-888"[tiab]' in term
    assert '"Drug Synergism"[mh]' in term and "51-21-8" not in term  # CAS numbers are not searched
    fetch = dict(pubmed["efetch"].calls.last.request.url.params)
    assert (fetch["WebEnv"], fetch["query_key"], fetch["retmode"]) == ("MCID_abc", "1", "xml")

    articles = pd.read_parquet(settings.interim_dir / "pubmed" / "articles.parquet").set_index(
        "pmid"
    )
    first = articles.loc["111"]
    assert first.title == "Veliparib and 5-FU act synergistically."  # inline markup flattened
    assert first.abstract.startswith("BACKGROUND: Fluorouracil") and "\nRESULTS:" in first.abstract
    assert (first.journal, first.year, first.doi, first.pmcid) == (
        "J Tests",
        2019,
        "10.1000/test.111",
        "PMC111",
    )
    assert first.article_type == "research" and first.language == "eng"
    assert not first.is_retracted and first.has_abstract
    assert list(first.mesh_terms) == ["Drug Synergism"]
    assert first.source_url == "https://pubmed.ncbi.nlm.nih.gov/111/"
    second = articles.loc["222"]
    assert second.is_retracted and second.article_type == "review"
    assert second.language == "chi" and second.year == 2015 and not second.has_abstract
    assert "999" not in articles.index  # book chapters are skipped

    log = [
        json.loads(line)
        for line in (settings.interim_dir / "pubmed" / "query_log.jsonl").read_text().splitlines()
    ]
    assert len(log) == 1 and log[0]["count"] == 2 and log[0]["query"] == term
    assert log[0]["pair_key"] == f"{FU_ID}|{VELI_ID}"


async def test_pubmed_is_incremental_and_deduplicates_by_pmid(
    settings: Settings, pubmed: dict[str, respx.Route], mock_http: respx.MockRouter
) -> None:
    await run(settings, "pubmed")
    calls = len(mock_http.calls)
    again = await run(settings, "pubmed")
    assert len(mock_http.calls) == calls  # already searched through today
    assert again.tables == {"articles": 2, "query_hits": 2}

    # Pretend the last run was ten days ago: only the gap (plus the overlap) is asked for.
    state_path = settings.state_dir / "pubmed_queries.json"
    state = json.loads(state_path.read_text())
    ten_days_ago = datetime.now(UTC).date() - timedelta(days=10)
    for entry in state.values():
        entry["last_maxdate"] = ten_days_ago.strftime("%Y/%m/%d")
    state_path.write_text(json.dumps(state))
    result = await run(settings, "pubmed")
    search = parse_qs(pubmed["esearch"].calls.last.request.content.decode())
    assert search["mindate"] == [(ten_days_ago - timedelta(days=2)).strftime("%Y/%m/%d")]
    assert search["maxdate"] == [datetime.now(UTC).date().strftime("%Y/%m/%d")]
    # The same two PMIDs came back in the new window: still two articles, not four.
    assert result.tables["articles"] == 2
    assert result.stats["records_parsed"] == 4 and result.stats["distinct_pmids"] == 2


async def test_pubmed_api_key_is_sent_but_never_stored(
    settings: Settings,
    pubmed: dict[str, respx.Route],
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    from synergyrx_data.logging_utils import setup_logging

    settings.secrets.ncbi_api_key = SecretStr(FAKE_API_KEY)
    setup_logging("DEBUG", json_logs=True, secrets=settings.secrets.values_to_redact())
    await run(settings, "pubmed")
    assert parse_qs(pubmed["esearch"].calls.last.request.content.decode())["api_key"] == [
        FAKE_API_KEY
    ]
    assert pubmed["efetch"].calls.last.request.url.params["api_key"] == FAKE_API_KEY
    logs = capfd.readouterr().err
    assert "http.response" in logs  # DEBUG logging was really on
    assert FAKE_API_KEY not in logs and TEST_EMAIL not in logs
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert FAKE_API_KEY.encode() not in path.read_bytes(), f"API key leaked into {path}"


async def test_pubmed_without_processed_tables_explains_what_to_do(
    settings: Settings, mock_http: respx.MockRouter
) -> None:
    result = await run(settings, "pubmed")
    assert result.outcome == NEEDS_ATTENTION and "no queries" in result.message
    assert len(mock_http.calls) == 0
    plan = await pipeline.plan(settings, ["pubmed"])
    assert plan[0].action == "skip" and "clean" in plan[0].note


# ---------------------------------------------------- Europe PMC / PubTator


async def test_europepmc_adds_open_access_status(
    settings: Settings, pubmed: dict[str, respx.Route], mock_http: respx.MockRouter
) -> None:
    await run(settings, "pubmed")
    hits = {
        "resultList": {
            "result": [
                {
                    "pmid": "111",
                    "pmcid": "PMC111",
                    "doi": "10.1000/TEST.111",
                    "isOpenAccess": "Y",
                    "inEPMC": "Y",
                    "hasPDF": "Y",
                    "license": "cc by",
                    "title": "T",
                    "abstractText": "Abstract from Europe PMC.",
                    "pubYear": "2019",
                },
            ]
        }
    }
    route = mock_http.get(EPMC).respond(json=hits)
    result = await run(settings, "europepmc")
    assert result.outcome == "ok" and result.stats["mode"] == "enrich"
    query = route.calls.last.request.url.params["query"]
    assert query == "(EXT_ID:111 OR EXT_ID:222) AND SRC:MED"
    table = pd.read_parquet(settings.interim_dir / "europepmc" / "articles.parquet")
    assert list(table["pmid"]) == ["111"]  # 222 is unknown to Europe PMC: remembered, not listed
    assert bool(table["is_open_access"].iloc[0]) and table["article_license"].iloc[0] == "cc by"
    await run(settings, "europepmc")
    assert route.call_count == 1


async def test_europepmc_takes_over_when_pubmed_is_down(
    settings: Settings, mock_http: respx.MockRouter
) -> None:
    write_processed(settings)
    mock_http.post(f"{EUTILS}/esearch.fcgi").respond(403)
    assert (await run(settings, "pubmed")).outcome == NEEDS_ATTENTION
    hits = {
        "resultList": {
            "result": [
                {
                    "pmid": "555",
                    "title": "Found by fallback",
                    "abstractText": "Veliparib plus 5-FU.",
                    "pubYear": "2020",
                }
            ]
        }
    }
    route = mock_http.get(EPMC).respond(json=hits)
    result = await run(settings, "europepmc")
    assert result.stats["mode"] == "search"
    query = route.calls.last.request.url.params["query"]
    assert 'TITLE_ABS:"Fluorouracil"' in query and "SRC:MED" in query
    table = pd.read_parquet(settings.interim_dir / "europepmc" / "articles.parquet")
    assert list(table["pmid"]) == ["555"] and table["found_by_query"].notna().all()


async def test_pubtator_annotations(
    settings: Settings, pubmed: dict[str, respx.Route], mock_http: respx.MockRouter
) -> None:
    await run(settings, "pubmed")
    payload = {
        "PubTator3": [
            {
                "id": "111",
                "passages": [
                    {
                        "infons": {"type": "abstract"},
                        "annotations": [
                            {
                                "text": "veliparib",
                                "infons": {"type": "Chemical", "identifier": "MESH:C521013"},
                                "locations": [{"offset": 40, "length": 9}],
                            },
                            {"text": "", "infons": {"type": "Chemical"}, "locations": []},
                        ],
                    }
                ],
            }
        ]
    }
    route = mock_http.get(url__startswith="https://www.ncbi.nlm.nih.gov/research/pubtator3-api/")
    route.respond(json=payload)
    result = await run(settings, "pubtator")
    assert result.tables == {"annotations": 1}
    row = pd.read_parquet(settings.interim_dir / "pubtator" / "annotations.parquet").iloc[0]
    assert (row.pmid, row.entity_type, row.text, row.identifier, row.start, row.length) == (
        "111",
        "Chemical",
        "veliparib",
        "MESH:C521013",
        40,
        9,
    )
    assert route.calls.last.request.url.params["pmids"] == "111,222"


# ----------------------------------------------------- baseline (optional)


async def test_pubmed_baseline_downloads_only_listed_files_and_checks_md5(
    settings: Settings, mock_http: respx.MockRouter, allow_robots: None
) -> None:
    assert (await pipeline.plan(settings, ["pubmed_baseline"]))[0].action == "skip"
    archive = gzip.compress(PUBMED_XML)
    base = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed26n0007.xml.gz"
    mock_http.get(base + ".md5").respond(
        200,
        text=f"MD5(pubmed26n0007.xml.gz)= {hashlib.md5(archive).hexdigest()}\n",
        headers={"content-type": "text/plain"},
    )
    data = mock_http.get(base).respond(
        200, content=archive, headers={"content-type": "application/x-gzip"}
    )
    settings.sources["pubmed_baseline"] = SourceConfig(enabled=True, options={"files": [7]})
    result = await run(settings, "pubmed_baseline")
    assert result.outcome == "ok" and result.tables == {"articles": 1}  # only 111 has an abstract
    assert (await run(settings, "pubmed_baseline")).outcome == "unchanged"
    assert data.call_count == 1


# --------------------------------------------------------- licence-gated


async def test_ddinter_refuses_until_terms_are_acknowledged(
    settings: Settings, mock_http: respx.MockRouter, allow_robots: None
) -> None:
    files = mock_http.get(url__startswith="https://ddinter2.scbdd.com/static/").respond(
        200,
        text="DDInterID_A,Drug_A,DDInterID_B,Drug_B,Level\n"
        "DDInter1,Abacavir,DDInter424,Cobicistat,Moderate\n",
        headers={"content-type": "application/octet-stream"},
    )
    result = await run(settings, "ddinter")
    assert result.outcome == MANUAL_STEP and "CC BY-NC-SA" in result.message
    assert not files.called

    settings.sources["ddinter"] = SourceConfig(
        enabled=True, options={"terms_acknowledged": True, "codes": ["A", "B"]}
    )
    result = await run(settings, "ddinter")
    assert result.outcome == "ok" and files.call_count == 2
    table = pd.read_parquet(settings.interim_dir / "ddinter" / "interactions.parquet")
    assert len(table) == 1  # the same pair is listed in both category files
    assert set(table["license"]) == {"CC-BY-NC-SA-4.0"}


async def test_drugbank_never_touches_the_network(
    settings: Settings, mock_http: respx.MockRouter
) -> None:
    result = await run(settings, "drugbank")
    assert result.outcome == MANUAL_STEP
    assert "never downloaded automatically" in result.message
    assert "data/manual/drugbank/drugbank vocabulary.csv" in result.message

    manual = settings.manual_dir / "drugbank"
    manual.mkdir(parents=True)
    (manual / "drugbank vocabulary.csv").write_text(
        "DrugBank ID,Accession Numbers,Common name,CAS,UNII,Synonyms,Standard InChI Key\n"
        f"DB00945,APRD00264,Acetylsalicylic acid,50-78-2,R16CO5Y76E,Aspirin | ASA,{ASPIRIN_KEY}\n",
        encoding="utf-8",
    )
    result = await run(settings, "drugbank")
    assert result.outcome == "ok" and result.tables == {"vocabulary": 1}
    row = pd.read_parquet(settings.interim_dir / "drugbank" / "vocabulary.parquet").iloc[0]
    assert row.drugbank_id == "DB00945" and list(row.synonyms) == ["Aspirin", "ASA"]
    assert len(mock_http.calls) == 0  # not a single request, in either case


async def test_pubmed_baseline_respects_the_real_robots_rule_and_reads_manual_files(
    settings: Settings, mock_http: respx.MockRouter
) -> None:
    """ftp.ncbi.nlm.nih.gov really serves `Disallow: /`: nothing is fetched from it."""
    mock_http.get("https://ftp.ncbi.nlm.nih.gov/robots.txt").respond(
        200, text="User-agent: *\nDisallow: /\n"
    )
    files = mock_http.get(url__startswith="https://ftp.ncbi.nlm.nih.gov/pubmed/").respond(200)
    settings.sources["pubmed_baseline"] = SourceConfig(enabled=True, options={"files": [7]})
    result = await run(settings, "pubmed_baseline")
    assert result.outcome == MANUAL_STEP and not files.called
    assert "data/manual/pubmed_baseline/pubmed26n0007.xml.gz" in result.message

    manual = settings.manual_dir / "pubmed_baseline"
    manual.mkdir(parents=True)
    (manual / "pubmed26n0007.xml.gz").write_bytes(gzip.compress(PUBMED_XML))
    result = await run(settings, "pubmed_baseline")
    assert result.outcome == "ok" and result.tables == {"articles": 1}
    assert not files.called


async def test_a_sample_run_does_not_advance_the_real_pubmed_state(
    settings: Settings, pubmed: dict[str, respx.Route]
) -> None:
    """Otherwise the first full run would start from today and miss everything older."""
    settings.sample = 1
    sampled = await run(settings, "pubmed")
    assert sampled.stats["queries_truncated"] == 1  # PubMed has 2 records, the sample took 1
    assert pubmed["efetch"].calls.last.request.url.params["retmax"] == "1"
    assert (settings.state_dir / "pubmed_queries.sample.json").is_file()
    assert not (settings.state_dir / "pubmed_queries.json").exists()

    settings.sample = None
    await run(settings, "pubmed")
    search = parse_qs(pubmed["esearch"].calls.last.request.content.decode())
    assert search["mindate"] == ["2000/01/01"]  # the full run still starts at the beginning
    assert pubmed["efetch"].calls.last.request.url.params["retmax"] == "2"
    log = (settings.interim_dir / "pubmed" / "query_log.jsonl").read_text().splitlines()
    assert [json.loads(line)["truncated"] for line in log] == [True, False]
