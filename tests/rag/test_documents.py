"""The indexed documents: catalog (src/rag/documents.py) and GET /api/documents (D114).

The API tests run the app on a Qdrant in-memory store holding the fake corpus, and ask a
chat question through it, so the agent's path on Qdrant is covered too.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agent.runtime import AgentHolder
from src.rag.documents import Catalog, Document, source_group
from src.rag.embeddings import FakeEmbedder
from src.rag.index import index_chunks
from src.rag.qdrant_store import QdrantStore
from src.rag.store import Chunk
from tujya_testing.fixtures import USER, sign_in
from tujya_testing.rag_fakes import CORPUS, DIM, FakeChatModel


def chunks() -> list[Chunk]:
    return [Chunk(**row) for row in CORPUS]


def test_catalog_groups_chunks_into_documents() -> None:
    parts = chunks()
    # Two chunks of the fluconazole label, given out of order.
    second = replace(parts[0], chunk_id="flu-di-2", chunk_index=1, section="warnings",
                     title="Fluconazole label (Diflucan): Warnings",
                     mentioned_ingredients=["amiodarone"])  # fmt: skip
    catalog = Catalog([second, *parts], {"flu-di": "a" * 64})
    assert len(catalog.documents) == len(CORPUS) and catalog.chunks == len(CORPUS) + 1
    label = catalog.get("doc:flu-di")
    assert label is not None
    assert label.title == "Fluconazole label (Diflucan)"  # the section suffix is dropped
    assert [c.chunk_id for c in label.chunks] == ["flu-di", "flu-di-2"]
    assert label.sections == ("drug_interactions", "warnings")
    assert label.mentioned_ingredients == ("amiodarone", "warfarin")
    assert label.hashes == ("a" * 64, "")
    # Labels first, then abstracts, then interactions.
    assert [d.doc_type for d in catalog.documents] == [
        "label", "label", "label", "abstract", "abstract", "interaction"
    ]  # fmt: skip
    assert catalog.get("doc:none") is None


def test_catalog_filters_and_facets() -> None:
    catalog = Catalog(chunks())

    def ids(found: list[Document]) -> set[str]:
        return {d.doc_id.removeprefix("doc:") for d in found}

    assert ids(catalog.search(ingredient="warfarin")) == {
        "flu-di", "war-ci", "pm-retracted", "pm-ok", "ddi-fw"
    }  # fmt: skip
    assert ids(catalog.search(doc_type="label", ingredient="warfarin")) == {"flu-di", "war-ci"}
    assert ids(catalog.search(section="contraindications")) == {"war-ci"}
    assert ids(catalog.search(source="PubMed")) == {"pm-retracted", "pm-ok"}
    assert ids(catalog.search(query="COUMADIN warf")) == {"war-ci"}  # words, any case
    assert ids(catalog.search(query="coumadin losartan")) == set()  # every word must match
    assert catalog.search(language="fr") == []
    facets = Catalog.facets(catalog.documents)
    assert facets["doc_type"] == {"label": 3, "abstract": 2, "interaction": 1}
    assert facets["source"] == {"openFDA drug label": 3, "PubMed": 2, "DDInter 2.0": 1}
    assert source_group("PubMed (J Test)") == "PubMed"


@pytest.fixture
def qdrant() -> QdrantStore:
    store = QdrantStore.connect(":memory:", DIM)
    asyncio.run(index_chunks(store, FakeEmbedder(DIM), [dict(r) for r in CORPUS]))
    return store


@pytest.fixture
def llm() -> FakeChatModel:
    return FakeChatModel(prompts=[])


@pytest.fixture
def client(
    app: FastAPI, pharma: str, qdrant: QdrantStore, llm: FakeChatModel,
    browser: Callable[[], TestClient],
) -> TestClient:  # fmt: skip
    app.state.agents = AgentHolder(
        app.state.settings, app.state.pharma_engine, llm=llm, embedder=FakeEmbedder(DIM),
        store=qdrant,
    )  # fmt: skip
    return sign_in(browser(), USER)


def test_documents_need_sign_in(app: FastAPI, anonymous: TestClient) -> None:
    assert anonymous.get("/api/documents").status_code == 401


def test_list_documents_with_filters(client: TestClient) -> None:
    body = client.get("/api/documents").json()
    assert body["total"] == len(CORPUS) and len(body["items"]) == len(CORPUS)
    index = body["index"]
    assert (index["store"], index["collection"], index["points"]) == (
        "qdrant", "tujya_chunks", len(CORPUS)
    )  # fmt: skip
    assert index["embedding_model"] == "fake-hash" and index["dim"] == DIM
    assert index["documents"] == len(CORPUS) and index["indexed_at"]
    assert index["dashboard_url"] == "http://127.0.0.1:6333/dashboard"
    first = body["items"][0]
    assert {"doc_id", "doc_type", "source_name", "license", "language", "date"} <= set(first)
    assert first["chunk_count"] == 1

    page = client.get("/api/documents", params={"ingredient": "Warfarin ", "doc_type": "label"})
    assert {d["doc_id"] for d in page.json()["items"]} == {"doc:flu-di", "doc:war-ci"}
    assert page.json()["facets"]["doc_type"] == {"label": 2}
    paged = client.get("/api/documents", params={"limit": 2, "offset": 4}).json()
    assert paged["total"] == len(CORPUS) and len(paged["items"]) == 2
    assert client.get("/api/documents", params={"limit": 0}).status_code == 422


def test_one_document_with_its_chunks(client: TestClient) -> None:
    body = client.get("/api/documents/doc:flu-di").json()
    assert body["title"] == "Fluconazole label (Diflucan)"
    assert body["mentioned_ingredients"] == ["warfarin"]
    (chunk,) = body["chunks"]
    assert chunk["chunk_id"] == "flu-di" and "prothrombin" in chunk["text"]
    assert len(chunk["content_hash"]) == 64
    assert client.get("/api/documents/doc:nothing").status_code == 404


def test_dose_sentences_are_hidden(client: TestClient, qdrant: QdrantStore) -> None:
    row = {**CORPUS[2], "text": "Dizziness was common. Patients took 50 mg twice daily."}
    asyncio.run(index_chunks(qdrant, FakeEmbedder(DIM), [row]))
    text = client.get("/api/documents/doc:los-aw").json()["chunks"][0]["text"]
    assert "Dizziness was common." in text and "50 mg" not in text


def test_empty_index_answers_503(app: FastAPI, browser: Callable[[], TestClient]) -> None:
    app.state.agents = AgentHolder(
        app.state.settings, app.state.pharma_engine, embedder=FakeEmbedder(DIM),
        store=QdrantStore.connect(":memory:", DIM),
    )  # fmt: skip
    response = sign_in(browser(), USER).get("/api/documents")
    assert response.status_code == 503 and "src.rag.index" in response.json()["detail"]


def test_unreachable_store_answers_503(app: FastAPI, browser: Callable[[], TestClient]) -> None:
    down = QdrantStore.connect("http://127.0.0.1:9", DIM)  # nothing listens on port 9
    app.state.agents = AgentHolder(
        app.state.settings, app.state.pharma_engine, embedder=FakeEmbedder(DIM), store=down
    )
    client = sign_in(browser(), USER)
    response = client.get("/api/documents")
    assert response.status_code == 503 and "qdrant" in response.json()["detail"]
    chat = client.post("/api/chat", json={"question": "warfarine ?"})
    assert chat.status_code == 503


def test_chat_answers_from_qdrant(client: TestClient, llm: FakeChatModel) -> None:
    llm.tool_calls = [{"name": "search_documents", "args": {"query": "fluconazole warfarin"}}]
    llm.answer = "Le fluconazole augmente le temps de prothrombine sous warfarine [2]."
    response = client.post(
        "/api/chat", json={"question": "La warfarine et le fluconazole interagissent-ils ?"}
    )
    assert response.status_code == 200, response.text
    titles = [c["title"] for c in response.json()["citations"]]
    assert any("Fluconazole label" in t for t in titles)
    assert not any("retracted" in t.lower() for t in titles)
