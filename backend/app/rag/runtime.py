"""Builds the agent's pieces once per app: chunk store, retriever, LLM, toolbox, agent.

Production: Ollama (`SYNERGYRX_OLLAMA_URL`) serves the LLM (`SYNERGYRX_LLM_MODEL`) and the
embeddings (`SYNERGYRX_EMBEDDING_MODEL`). Tests replace `llm` and `embedder` with fakes.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from sqlalchemy import Engine

from app.config import Settings
from app.rag.agent import TujyaAgent
from app.rag.embeddings import Embedder, OllamaEmbedder
from app.rag.retrieval import RetrieverHolder
from app.rag.store import ChunkStore
from app.rag.tools import Toolbox
from app.scanner import ScannerHolder


def ollama_llm(settings: Settings) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    options: dict[str, Any] = {}
    if settings.llm_model.startswith("qwen3"):
        options["reasoning"] = False  # no hidden "thinking" tokens: faster, same answer
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_url,
        temperature=settings.llm_temperature,
        num_ctx=settings.llm_num_ctx,
        keep_alive="30m",
        client_kwargs={"timeout": settings.llm_timeout_seconds},
        **options,
    )


class AgentHolder:
    def __init__(
        self,
        settings: Settings,
        pharma_engine: Engine,
        scanner: ScannerHolder | None = None,
        *,
        llm: BaseChatModel | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings, self.pharma_engine, self.scanner = settings, pharma_engine, scanner
        self.llm = llm
        self.embedder = embedder
        self._toolbox: Toolbox | None = None
        self._agent: TujyaAgent | None = None

    @property
    def model_name(self) -> str:
        return str(getattr(self.llm, "model", None) or self.settings.llm_model)

    def toolbox(self) -> Toolbox:
        if self._toolbox is None:
            embedder = self.embedder or OllamaEmbedder(
                self.settings.ollama_url,
                self.settings.embedding_model,
                self.settings.embedding_dim,
            )
            store = ChunkStore(self.pharma_engine, embedder.dim)
            self._toolbox = Toolbox(
                self.settings, self.pharma_engine, RetrieverHolder(store, embedder), self.scanner
            )
        return self._toolbox

    def agent(self) -> TujyaAgent:
        if self._agent is None:
            self.llm = self.llm or ollama_llm(self.settings)
            self._agent = TujyaAgent(self.llm, self.toolbox(), top_k=self.settings.rag_top_k)
        return self._agent
