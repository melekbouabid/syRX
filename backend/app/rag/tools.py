"""The agent's tools, written once and exposed twice: as LangChain tools (LangGraph agent)
and as MCP tools (app/mcp_server.py).

Every tool returns plain JSON-serialisable data with the sources it used, never a dose
(the corpus has no dosage section) and never the word "safe".

Extension point: a new tool is one async method on `Toolbox` plus one entry in
`TOOL_SPECS`. Planned (docs/ROADMAP.md, Bonus), not implemented:
    predict_toxicity(smiles)       single-molecule toxicity, model trained on Tox21 / ClinTox
    predict_synergy(drug_a, drug_b) cancer synergy from the DrugComb data of the pipeline
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.config import Settings
from app.medicines import DISCLAIMER, PharmaUnavailable, search_pills
from app.pharma import Ingredient
from app.rag.names import NameIndex
from app.rag.retrieval import Hit, RetrieverHolder
from app.rag.store import Filters
from app.scanner import ScannerHolder

DOC_TYPES = ("label", "abstract", "interaction")
SNIPPET_CHARS = 700
INFO_SECTIONS = (
    "boxed_warning",
    "indications_and_usage",
    "contraindications",
    "warnings_and_cautions",
    "warnings",
    "drug_interactions",
    "mechanism_of_action",
)


class ToolError(RuntimeError):
    """A tool could not run (missing index, models or tables); the message says why."""


def hit_source(hit: Hit) -> dict[str, Any]:
    chunk = hit.chunk
    return {
        "chunk_id": chunk.chunk_id,
        "doc_type": chunk.doc_type,
        "title": chunk.title,
        "section": chunk.section,
        "url": chunk.url,
        "source_name": chunk.source_name,
        "date": chunk.date,
        "license": chunk.license,
        "ingredients": chunk.ingredients,
        "is_retracted": chunk.is_retracted,
        "article_type": chunk.article_type,
        "text": chunk.text,
        "snippet": chunk.text[:SNIPPET_CHARS],
        "score": round(hit.score, 5),
    }


@dataclass
class Toolbox:
    settings: Settings
    pharma_engine: Engine
    retrievers: RetrieverHolder
    scanner: ScannerHolder | None = None
    _names: NameIndex | None = None

    # ----------------------------------------------------------- helpers

    def names(self) -> NameIndex:
        if self._names is None:
            with Session(self.pharma_engine) as session:
                self._names = NameIndex(session.scalars(select(Ingredient.name)).all())
        return self._names

    def normalise(self, values: list[str]) -> list[str]:
        index = self.names()
        found = []
        for value in values:
            name = index.lookup(value) or value.strip().lower()
            if name and name not in found:
                found.append(name)
        return found

    def retriever(self) -> Any:
        retriever = self.retrievers.get()
        if retriever is None:
            raise ToolError("the document index is empty: run `python -m app.rag.index`")
        return retriever

    # ------------------------------------------------------------- tools

    async def identify_pills(
        self, image_path: str | None = None, image_base64: str | None = None
    ) -> dict[str, Any]:
        """Detect and identify pills on a photo (the /api/scan models)."""
        from app.routers.scan import _run  # the route's own pipeline

        if self.scanner is None:
            raise ToolError("no scanner configured")
        if image_base64:
            data = base64.b64decode(image_base64)
        elif image_path:
            path = Path(image_path)
            if not await run_in_threadpool(path.is_file):
                raise ToolError(f"no image at {image_path}")
            data = await run_in_threadpool(path.read_bytes)
        else:
            raise ToolError("give image_path or image_base64")
        scanner = await run_in_threadpool(self.scanner.get)

        def run() -> dict[str, Any]:
            with Session(self.pharma_engine) as session:
                result = _run(scanner, data, self.settings, session)
            return result.model_dump(mode="json")

        return await run_in_threadpool(run)

    async def check_interaction(
        self, drugs: list[str] | None = None, pills: list[int] | None = None
    ) -> dict[str, Any]:
        """DDInter interactions between every pair of drugs and pill classes."""
        from app.routers.medicines import interactions_report

        names = self.normalise(drugs or [])
        pill_ids = list(pills or [])
        if len(names) + len(pill_ids) < 2:
            raise ToolError("give at least two drugs or pills")

        def run() -> dict[str, Any]:
            with Session(self.pharma_engine) as session:
                return interactions_report(session, pill_ids, names).model_dump(mode="json")

        try:
            return await run_in_threadpool(run)
        except PharmaUnavailable as exc:
            raise ToolError("the medicine tables are not loaded") from exc

    async def get_drug_info(self, drug: str, sections: list[str] | None = None) -> dict[str, Any]:
        """Pill classes containing the ingredient and excerpts of its openFDA label."""
        name = self.normalise([drug])[0]

        def pills() -> list[dict[str, Any]]:
            with Session(self.pharma_engine) as session:
                found = search_pills(session, name, 20)
            return [
                {"class_id": p.class_id, "name": p.name, "active_ingredients": p.ingredients}
                for p in found
                if name in p.ingredients
            ]

        try:
            classes = await run_in_threadpool(pills)
        except PharmaUnavailable:
            classes = []
        wanted = tuple(sections or INFO_SECTIONS)
        retriever = self.retriever()
        excerpts = []
        for section in wanted:
            chunks = [
                c
                for c in retriever.chunks
                if c.doc_type == "label" and name in c.ingredients and c.section == section
            ]
            if chunks:
                first = min(chunks, key=lambda c: c.chunk_index)
                excerpts.append(hit_source(Hit(first, 1.0)))
        return {
            "ingredient": name,
            "known": name in self.names().names,
            "pill_classes": classes,
            "label_sections": excerpts,
            "disclaimer": DISCLAIMER,
        }

    async def search_documents(
        self,
        query: str,
        ingredients: list[str] | None = None,
        doc_types: list[str] | None = None,
        sections: list[str] | None = None,
        k: int = 5,
        exclude_retracted: bool = True,
        mode: str | None = None,
    ) -> dict[str, Any]:
        """Search labels, PubMed abstracts and DDInter (mode from settings: dense by default,
        bm25 or hybrid on request)."""
        filters = Filters(
            ingredients=tuple(self.normalise(ingredients or [])),
            doc_types=tuple(t for t in (doc_types or []) if t in DOC_TYPES),
            sections=tuple(sections or []),
            exclude_retracted=exclude_retracted,
        )
        retriever = self.retriever()
        k = max(1, min(int(k), 20))
        mode = mode or self.settings.rag_search_mode
        hits = await retriever.search(query, k=k, mode=mode, filters=filters)  # type: ignore[arg-type]
        return {
            "query": query,
            "filters": {
                "ingredients": list(filters.ingredients),
                "doc_types": list(filters.doc_types),
                "sections": list(filters.sections),
                "exclude_retracted": exclude_retracted,
            },
            "results": [hit_source(hit) for hit in hits],
        }


# ------------------------------------------------------------------ schemas


class IdentifyPillsArgs(BaseModel):
    image_path: str | None = Field(None, description="Path of a photo on the server")
    image_base64: str | None = Field(None, description="The photo, base64-encoded")


class CheckInteractionArgs(BaseModel):
    drugs: list[str] = Field(
        default_factory=list,
        description="Active ingredient names, e.g. ['warfarin', 'fluconazole']",
    )
    pills: list[int] = Field(default_factory=list, description="Pill class_ids from a scan")


class DrugInfoArgs(BaseModel):
    drug: str = Field(description="Active ingredient name, e.g. 'metformin'")
    sections: list[str] | None = Field(
        None, description="Label sections, e.g. ['contraindications', 'drug_interactions']"
    )


class SearchArgs(BaseModel):
    query: str = Field(description="What to look for, preferably in English")
    ingredients: list[str] = Field(default_factory=list, description="Only about these drugs")
    doc_types: list[str] = Field(
        default_factory=list, description="label, abstract or interaction (empty: all)"
    )
    sections: list[str] = Field(default_factory=list, description="Label sections to keep")
    k: int = Field(5, ge=1, le=20, description="Number of results")


TOOL_SPECS: dict[str, tuple[str, type[BaseModel], str]] = {
    "identify_pills": (
        "identify_pills",
        IdentifyPillsArgs,
        "Detect the pills on a photo and identify each one (top-5 with probabilities). "
        "Identification is a suggestion, never a certainty.",
    ),
    "check_interaction": (
        "check_interaction",
        CheckInteractionArgs,
        "Check DDInter for interactions between two or more drugs or pill classes: "
        "severity, description, source. Never use the word safe.",
    ),
    "get_drug_info": (
        "get_drug_info",
        DrugInfoArgs,
        "Pill classes containing an active ingredient and excerpts of its openFDA label "
        "(no dosage section).",
    ),
    "search_documents": (
        "search_documents",
        SearchArgs,
        "Search openFDA labels, PubMed abstracts and DDInter rows (bge-m3 embeddings, BM25 "
        "or hybrid), with filters on ingredient, document type and section.",
    ),
}
# The agent's model is never given identify_pills: a chat question carries no image (the
# app sends the confirmed pills instead). MCP clients get all four.
CHAT_TOOLS = ("check_interaction", "get_drug_info", "search_documents")


def langchain_tools(toolbox: Toolbox, names: tuple[str, ...] = CHAT_TOOLS) -> list[BaseTool]:
    tools: list[BaseTool] = []
    for name in names:
        tool_name, schema, description = TOOL_SPECS[name]
        tools.append(
            StructuredTool.from_function(
                coroutine=getattr(toolbox, name),
                name=tool_name,
                description=description,
                args_schema=schema,
            )
        )
    return tools
