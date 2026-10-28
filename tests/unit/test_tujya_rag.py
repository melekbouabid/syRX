"""Tujya RAG corpus: openFDA label connector, PubMed ingredient queries and chunking."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pandas as pd
import pytest
import respx

from synergyrx_data import pipeline
from synergyrx_data.config import Settings, SourceConfig
from synergyrx_data.connectors.openfda import API, choose_label, label_sections
from synergyrx_data.literature.queries import build_topic_query, read_ingredients
from synergyrx_data.tujya import corpus

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def write_classes(settings: Settings, ingredients: list[str]) -> None:
    path = settings.resolve("config/pill_classes.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{i},x_{i},{value}" for i, value in enumerate(ingredients)]
    path.write_text("class_id,class_key,active_ingredients\n" + "\n".join(rows) + "\n", "utf-8")


def label(
    label_id: str,
    substances: list[str],
    *,
    product_type: str = "HUMAN PRESCRIPTION DRUG",
    effective_time: str = "20240101",
    sections: tuple[str, ...] = ("drug_interactions", "warnings"),
) -> dict[str, object]:
    return {
        "id": label_id,
        "set_id": f"set-{label_id}",
        "version": "3",
        "effective_time": effective_time,
        "openfda": {
            "generic_name": [" AND ".join(substances)],
            "substance_name": substances,
            "product_type": [product_type],
            "brand_name": ["Brand"],
            "manufacturer_name": ["Maker"],
        },
        "dosage_and_administration": ["Take 100 mg once daily."],
        "overdosage": ["Doses above 2 g caused seizures."],
        **{section: [f"{section} text for {label_id}."] for section in sections},
    }


def test_choose_label_prefers_single_ingredient_prescription_then_sections_then_date() -> None:
    combo = label("combo", ["FLUCONAZOLE", "OTHER"], effective_time="20260101")
    otc = label("otc", ["FLUCONAZOLE"], product_type="HUMAN OTC DRUG", effective_time="20260101")
    thin = label("thin", ["FLUCONAZOLE"], effective_time="20260101", sections=("warnings",))
    full_old = label("full_old", ["FLUCONAZOLE"], effective_time="20200101")
    full_new = label("full_new", ["FLUCONAZOLE"], effective_time="20230101")
    unrelated = label("unrelated", ["ASPIRIN"])
    terms = ["fluconazole"]
    assert choose_label([combo, otc, thin, full_old, full_new, unrelated], terms) == full_new
    assert choose_label([combo, otc], terms) == otc
    assert choose_label([unrelated], terms) is None
    # Salts match the base ingredient: "CYCLOBENZAPRINE HYDROCHLORIDE" is about cyclobenzaprine.
    salt = label("salt", ["CYCLOBENZAPRINE HYDROCHLORIDE"])
    assert choose_label([salt], ["cyclobenzaprine"]) == salt


def test_label_sections_never_include_dosage() -> None:
    sections = dict(label_sections(label("a", ["X"])))
    assert set(sections) == {"warnings", "drug_interactions"}
    assert all("mg" not in text for text in sections.values())


@pytest.fixture
def openfda(settings: Settings, mock_http: respx.MockRouter) -> respx.Route:
    write_classes(settings, ["fluconazole", "glyburide; metformin", "unknownium"])
    settings.sources["openfda_label"] = SourceConfig(enabled=True)
    labels = {
        "fluconazole": [label("f1", ["FLUCONAZOLE"]), label("f0", ["FLUCONAZOLE", "OTHER"])],
        "glyburide": [label("g1", ["GLYBURIDE"])],
        "metformin": [label("m1", ["METFORMIN HYDROCHLORIDE"])],
    }

    def search(request: httpx.Request) -> httpx.Response:
        query = request.url.params["search"]
        for name, found in labels.items():
            if f'generic_name:"{name}"' in query:
                return httpx.Response(200, json={"meta": {}, "results": found})
        return httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})

    return mock_http.get(API).mock(side_effect=search)


async def test_openfda_connector_keeps_one_label_per_ingredient(
    settings: Settings, openfda: respx.Route
) -> None:
    (run,) = await pipeline.fetch(settings, ["openfda_label"])
    assert run.outcome == "needs_attention"  # unknownium has no label
    assert "unknownium" in run.message
    assert run.tables == {"label_sections": 6}

    first = openfda.calls[0].request.url.params
    assert first["sort"] == "effective_time:desc" and first["limit"] == "25"
    assert 'openfda.product_type:"HUMAN PRESCRIPTION DRUG"' in first["search"]
    # unknownium: prescription search, then any product type, both 404
    assert len(openfda.calls) == 5

    sections = pd.read_parquet(settings.interim_dir / "openfda_label" / "label_sections.parquet")
    assert set(sections["ingredient"]) == {"fluconazole", "glyburide", "metformin"}
    assert set(sections["label_id"]) == {"f1", "g1", "m1"}
    assert not sections["section"].str.contains("dosage|overdos").any()
    assert not sections["text"].str.contains("mg").any()
    row = sections.iloc[0]
    assert row.label_url == "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=set-f1"
    assert row.license.startswith("CC0") and row.source_version == "20240101"

    raw = json.loads(
        (settings.raw_dir / "openfda_label" / "current" / "labels" / "fluconazole.json").read_text(
            encoding="utf-8"
        )
    )
    assert [c["id"] for c in raw["candidates"]] == ["f1", "f0"]
    assert "dosage_and_administration" not in raw["label"] and "overdosage" not in raw["label"]

    calls = len(openfda.calls)
    (again,) = await pipeline.fetch(settings, ["openfda_label"])
    assert len(openfda.calls) == calls + 2  # stored labels are reused; only the miss retries
    assert again.tables == {"label_sections": 6}


def test_ingredient_queries_are_about_interactions_and_adverse_effects(tmp_path: Path) -> None:
    classes = tmp_path / "classes.csv"
    classes.write_text(
        "class_id,active_ingredients\n0,Glyburide; metformin\n1,metformin\n2,\n", "utf-8"
    )
    assert read_ingredients(classes) == ["glyburide", "metformin"]
    assert read_ingredients(tmp_path / "missing.csv") == []
    query = build_topic_query("valproate", ["valproic acid"])
    assert query.kind == "ingredient" and query.drug_names == ("valproate",)
    assert '"valproic acid"[tiab]' in query.pubmed and '"Drug Interactions"[mh]' in query.pubmed
    assert "adverse event*" in query.pubmed and "hasabstract[text]" in query.pubmed
    assert "Neoplasms" not in query.pubmed and "synerg" not in query.pubmed
    assert query.query_id != build_topic_query("metformin", []).query_id


async def test_pubmed_ingredient_mode(settings: Settings, mock_http: respx.MockRouter) -> None:
    write_classes(settings, ["fluconazole", "metformin"])
    settings.sources["pubmed"] = SourceConfig(
        enabled=True,
        options={"mode": "ingredients", "max_records_per_query": 1, "sort": "relevance"},
    )
    search = {"esearchresult": {"count": "5", "webenv": "W", "querykey": "1"}}
    esearch = mock_http.post(f"{EUTILS}/esearch.fcgi").respond(json=search)
    xml = (
        b"<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>42</PMID><Article>"
        b"<ArticleTitle>T</ArticleTitle><Abstract><AbstractText>A.</AbstractText></Abstract>"
        b"</Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"
    )
    mock_http.get(f"{EUTILS}/efetch.fcgi").respond(
        200, content=xml, headers={"content-type": "text/xml"}
    )
    (run,) = await pipeline.fetch(settings, ["pubmed"])
    assert run.outcome == "ok" and run.stats["queries"] == 2
    sent = parse_qs(esearch.calls[0].request.content.decode())
    assert sent["sort"] == ["relevance"] and '"fluconazole"[tiab]' in sent["term"][0]
    hits = pd.read_parquet(settings.interim_dir / "pubmed" / "query_hits.parquet")
    by_pmid = corpus.pmid_ingredients(settings.raw_dir / "pubmed", hits)
    assert by_pmid == {"42": ["fluconazole", "metformin"]}


# ------------------------------------------------------------------ chunking


def test_split_text_respects_the_limit_and_overlaps() -> None:
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    pieces = corpus.split_text(text, max_chars=300)
    assert len(pieces) > 5 and all(len(p) <= 300 for p in pieces)
    assert pieces[1].startswith(pieces[0].split(". ")[-1].rstrip("."))  # one sentence repeated
    assert corpus.split_text("short") == ["short"] and corpus.split_text("  ") == []
    assert all(len(p) <= 100 for p in corpus.split_text("x" * 250, max_chars=100))


def _sections() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ingredient": "fluconazole",
                "label_id": "f1",
                "set_id": "s1",
                "label_version": "3",
                "effective_time": "20240101",
                "brand_name": "Diflucan",
                "generic_name": "FLUCONAZOLE",
                "section": "drug_interactions",
                "text": "Fluconazole increases warfarin exposure. Monitor prothrombin time.",
                "label_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=s1",
                "source_url": API,
                "license": "CC0",
                "retrieved_at": "2026-10-28T00:00:00Z",
            }
        ]
    )


def _articles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pmid": "42",
                "title": "Metformin and fluconazole",
                "abstract": "BACKGROUND: Metformin is common.\nRESULTS: Fluconazole raised levels.",
                "journal": "J Test",
                "year": 2020,
                "language": "eng",
                "article_type": "research",
                "is_retracted": True,
                "license": "NLM",
                "retrieved_at": "2026-10-28T00:00:00Z",
            },
            {"pmid": "43", "title": "Not ours", "abstract": "x", "journal": None, "year": None},
        ]
    )


def _pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ingredient_a": "fluconazole",
                "ingredient_b": "warfarin",
                "severity": "Major",
                "ddinter_id_a": "D1",
                "ddinter_id_b": "D2",
                "source_url": "https://ddinter2.scbdd.com/",
                "license": "CC BY-NC-SA 4.0",
            },
            {
                "ingredient_a": "fluconazole",
                "ingredient_b": "metformin",
                "severity": "Unknown",
                "ddinter_id_a": "D1",
                "ddinter_id_b": "D3",
                "source_url": "https://ddinter2.scbdd.com/",
                "license": "CC BY-NC-SA 4.0",
            },
        ]
    )


def test_build_chunks_metadata() -> None:
    names = {"fluconazole", "metformin", "warfarin"}
    chunks = corpus.build_chunks(
        _sections(), _articles(), {"42": ["metformin"]}, _pairs(), names, "2026-10-28"
    )
    assert list(chunks.columns) == corpus.CHUNK_COLUMNS
    assert chunks["doc_type"].value_counts().to_dict() == {
        "abstract": 2,
        "interaction": 2,
        "label": 1,
    }
    lab = chunks[chunks["doc_type"] == "label"].iloc[0]
    assert lab.ingredients == ["fluconazole"] and lab.mentioned_ingredients == ["warfarin"]
    assert lab.title == "Fluconazole label (Diflucan): Drug interactions"
    assert lab.date == "20240101" and lab.url.endswith("setid=s1")
    abstracts = chunks[chunks["doc_type"] == "abstract"]
    assert list(abstracts["section"]) == ["background", "results"]
    assert abstracts["is_retracted"].all() and set(abstracts["doc_id"]) == {"pubmed:42"}
    assert abstracts.iloc[1].mentioned_ingredients == ["fluconazole"]
    assert abstracts.iloc[0].url == "https://pubmed.ncbi.nlm.nih.gov/42/"
    unknown = chunks[chunks["title"].str.contains("Unknown")].iloc[0]
    assert "does not grade" in unknown.text and "no interaction" not in unknown.text
    assert chunks["chunk_id"].is_unique
    summary = corpus.summary(chunks)
    assert summary["chunks"] == 5 and summary["retracted_chunks"] == 2
    # Same input, same ids: indexing can be idempotent.
    again = corpus.build_chunks(
        _sections(), _articles(), {"42": ["metformin"]}, _pairs(), names, "2026-10-28"
    )
    assert list(again["chunk_id"]) == list(chunks["chunk_id"])


def test_dosage_sections_are_refused_by_the_chunker() -> None:
    bad = _sections().assign(section="dosage_and_administration")
    with pytest.raises(ValueError, match="dosage"):
        corpus.build_chunks(bad, None, {}, None, set())
