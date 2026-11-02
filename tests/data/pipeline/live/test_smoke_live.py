"""Optional smoke tests against the real services.

They are excluded from the default run. Start them with `pytest tests/live -m live` after
putting CONTACT_EMAIL in `.env`. Each test asks for a few kilobytes.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from src.data.synergyrx_data.config import Settings, load_settings
from src.data.synergyrx_data.connectors import allowed_hosts
from src.data.synergyrx_data.http.client import PoliteClient
from src.data.synergyrx_data.literature.abstracts import parse_pubmed_xml

pytestmark = pytest.mark.live
PIPELINE = Path(__file__).resolve().parents[4] / "config" / "data_pipeline"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Real rate limits and the real identity from `.env`, but a throwaway data directory."""
    built = load_settings(PIPELINE / "config.yaml", [f"data_dir={(tmp_path / 'data').as_posix()}"])
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


async def test_pubmed_history_search_and_fetch(client: PoliteClient, settings: Settings) -> None:
    search = await client.fetch(
        f"{EUTILS}/esearch.fcgi",
        source="pubmed",
        method="POST",
        data={
            "db": "pubmed",
            "term": '"Drug Interactions"[mh] AND fluconazole[tiab] AND warfarin[tiab]',
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
