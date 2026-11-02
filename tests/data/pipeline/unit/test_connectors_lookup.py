"""PubMed and DDInter connectors with mocked HTTP (openFDA labels: module 2 corpus tests)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs

import pandas as pd
import pytest
import respx
from pydantic import SecretStr

from src.data.tujya_data import pipeline
from src.data.tujya_data.config import Settings, SourceConfig
from src.data.tujya_data.state import MANUAL_STEP, NEEDS_ATTENTION
from src.data.tujya_data.testing import FAKE_API_KEY, TEST_EMAIL

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


async def run(settings: Settings, source: str) -> pipeline.SourceRun:
    return (await pipeline.fetch(settings, [source]))[0]


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


def write_classes(settings: Settings) -> None:
    """One pill class with one active ingredient, so one PubMed query, plus its synonyms."""
    path = settings.project_root / "config" / "pill_classes.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("class_id,class_key,active_ingredients\n0,x_0,Fluorouracil\n", "utf-8")
    settings.sources["pubmed"] = SourceConfig(
        enabled=True,
        options={"ingredient_aliases": {"fluorouracil": ["5-FU", "5-fluorouracil", "51-21-8"]}},
    )


@pytest.fixture
def pubmed(settings: Settings, mock_http: respx.MockRouter) -> dict[str, respx.Route]:
    write_classes(settings)
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
    assert search["tool"] == ["tujya-data"] and search["email"] == [TEST_EMAIL]
    assert (search["mindate"], search["datetype"]) == (["2000/01/01"], ["edat"])
    term = search["term"][0]
    assert '"fluorouracil"[tiab]' in term and '"5-FU"[tiab]' in term
    assert '"5-fluorouracil"[tiab]' in term
    assert '"Drug Interactions"[mh]' in term
    assert "51-21-8" not in term  # CAS numbers are not searched
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
    assert log[0]["kind"] == "ingredient" and log[0]["pair_key"] is None


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
    from src.data.tujya_data.logging_utils import setup_logging

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


async def test_pubmed_without_ingredients_explains_what_to_do(
    settings: Settings, mock_http: respx.MockRouter
) -> None:
    result = await run(settings, "pubmed")
    assert result.outcome == NEEDS_ATTENTION and "no queries" in result.message
    assert len(mock_http.calls) == 0
    plan = await pipeline.plan(settings, ["pubmed"])
    assert plan[0].action == "skip" and "pill_classes.csv" in plan[0].note


async def test_pubmed_rejects_the_removed_synergy_query_modes(
    settings: Settings, mock_http: respx.MockRouter
) -> None:
    """`pairs` and `drugs` read the original synergy tables, which no longer exist (D85)."""
    settings.sources["pubmed"] = SourceConfig(enabled=True, options={"mode": "pairs"})
    result = await run(settings, "pubmed")
    assert result.outcome == NEEDS_ATTENTION and "unknown PubMed query mode" in result.message
    assert len(mock_http.calls) == 0


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
