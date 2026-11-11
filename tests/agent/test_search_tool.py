"""The search_documents tool with the step 3.2 options: reranking, query rewriting, sources
with their metadata (fake reranker and LLM, SQL store of the test corpus)."""

from __future__ import annotations

import asyncio

from fastapi import FastAPI

from src.agent.tools import Toolbox
from src.rag.embeddings import FakeEmbedder
from src.rag.retrieval import RetrieverHolder
from src.rag.rewrite import QueryRewriter
from tujya_testing.rag_fakes import DIM, FakeChatModel, build_index


class WordReranker:
    name = "fake-reranker"

    def __init__(self) -> None:
        self.queries: list[str] = []

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.queries.append(query)
        words = set(query.lower().split())
        return [float(sum(w in t.lower() for w in words)) for t in texts]


def toolbox(app: FastAPI, pharma: str, *, rerank: bool, rewriter: QueryRewriter | None) -> Toolbox:
    store = asyncio.run(build_index(pharma))
    settings = app.state.settings.model_copy(update={"rag_rerank": rerank, "rag_fetch_k": 6})
    holder = RetrieverHolder(store, FakeEmbedder(DIM), reranker=WordReranker())
    return Toolbox(settings, app.state.pharma_engine, holder, rewriter=rewriter)


def test_search_reranks_when_the_settings_ask(app: FastAPI, pharma: str) -> None:
    tools = toolbox(app, pharma, rerank=True, rewriter=None)
    result = asyncio.run(tools.search_documents("prothrombin time", k=2, mode="dense"))
    assert result["reranked"] is True and result["search_query"] == "prothrombin time"
    assert len(result["results"]) == 2
    first = result["results"][0]
    assert first["rerank_score"] is not None and first["retrieval_rank"] >= 1
    # Sources carry their metadata for the chat (step 3.2).
    for key in ("doc_id", "doc_type", "section", "source_name", "date", "version", "license"):
        assert key in first
    reranker = tools.retrievers.reranker
    assert isinstance(reranker, WordReranker) and reranker.queries == ["prothrombin time"]


def test_search_without_reranking(app: FastAPI, pharma: str) -> None:
    tools = toolbox(app, pharma, rerank=False, rewriter=None)
    result = asyncio.run(tools.search_documents("prothrombin time", k=2, mode="dense"))
    assert result["reranked"] is False
    assert all(r["rerank_score"] is None for r in result["results"])


def test_search_runs_on_the_rewritten_query(app: FastAPI, pharma: str) -> None:
    llm = FakeChatModel(answer="warfarin fluconazole bleeding")
    tools = toolbox(app, pharma, rerank=True, rewriter=QueryRewriter(llm))
    result = asyncio.run(tools.search_documents("La warfarine et le fluconazole ?", k=3))
    assert result["query"] == "La warfarine et le fluconazole ?"
    assert result["search_query"] == "warfarin fluconazole bleeding"
    reranker = tools.retrievers.reranker
    assert isinstance(reranker, WordReranker)
    assert reranker.queries == ["warfarin fluconazole bleeding"]  # reranked on it too


def test_agent_holder_wires_the_rewriter_only_when_asked(app: FastAPI, pharma: str) -> None:
    from src.agent.runtime import AgentHolder

    llm = FakeChatModel()
    store = asyncio.run(build_index(pharma))
    on = app.state.settings.model_copy(update={"rag_query_rewrite": True})
    holder = AgentHolder(
        on, app.state.pharma_engine, llm=llm, embedder=FakeEmbedder(DIM), store=store
    )
    rewriter = holder.toolbox().rewriter
    assert rewriter is not None and rewriter.llm is llm
    off = AgentHolder(
        app.state.settings,
        app.state.pharma_engine,
        llm=llm,
        embedder=FakeEmbedder(DIM),
        store=store,
    )
    assert off.toolbox().rewriter is None
