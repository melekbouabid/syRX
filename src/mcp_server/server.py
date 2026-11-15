"""Tujya MCP server: the agent's six tools for any MCP client (official Python MCP SDK).

    # project venv (.venv) active, from any folder
    python -m src.mcp_server            # stdio (Claude Desktop, IDEs)
    python -m src.mcp_server --transport streamable-http --port 8765
                                                           # http://127.0.0.1:8765/mcp

Same tools, same code and the same toolbox as the LangGraph agent (agent/tools.py, built by
agent/runtime.py): identify_pills (the served detector and 380-class classifier),
check_interaction, get_drug_info, search_documents (the served retrieval setting of step 3.3:
bge-m3 dense, ingredient filter, query rewriting by the LLM; D123), lookup_brand (medicine
brands, step 2.7), predict_toxicity (bonus). Settings come from the TUJYA_* environment
(database, Ollama URL for the query embeddings and the rewriting LLM, models directory); see
docs/module2/mcp.md for client configuration.
"""

from __future__ import annotations

import argparse
from typing import Any

from langchain_core.language_models import BaseChatModel
from mcp.server.mcpserver import MCPServer
from sqlalchemy import distinct, func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import Settings
from app.db import make_engine
from app.inference import ScannerHolder
from app.medicines import DISCLAIMER
from app.pharma import Pill, PillIngredient
from src.agent.runtime import AgentHolder
from src.agent.tools import TOOL_SPECS, Toolbox, ToolError
from src.rag.embeddings import Embedder

INSTRUCTIONS = (
    "Medicine information for the {classes} Tujya pill classes and their {ingredients} active "
    "ingredients: pill identification from a photo, DDInter interactions, openFDA label "
    "excerpts, and search of labels, PubMed abstracts and DDInter rows ({indexed} active "
    "ingredients have indexed documents; the others get pill data and DDInter interactions "
    "only). Cite the returned sources. Never give doses; never call a combination safe (say "
    "'aucune interaction connue dans cette base de données'). predict_toxicity is a model "
    "prediction on in vitro tests, not a clinical evaluation: report each molecule "
    "separately, never conclude about a mixture. "
    f"{DISCLAIMER}"
)


def catalogue(toolbox: Toolbox) -> dict[str, int | str]:
    """Pill classes and active ingredients of the pharma database, ingredients with documents."""
    try:
        with Session(toolbox.pharma_engine) as session:
            classes: int | str = session.scalar(select(func.count()).select_from(Pill)) or 0
            ingredients: int | str = (
                session.scalar(select(func.count(distinct(PillIngredient.ingredient_id)))) or 0
            )
    except SQLAlchemyError:  # no database yet: the server still starts
        classes = ingredients = "the"
    try:
        indexed: int | str = len(toolbox.indexed_ingredients())
    except ToolError:  # no index yet
        indexed = "no"
    return {"classes": classes, "ingredients": ingredients, "indexed": indexed}


def build_server(toolbox: Toolbox) -> MCPServer:
    instructions = INSTRUCTIONS.format(**catalogue(toolbox))
    server: MCPServer = MCPServer("tujya", instructions=instructions)

    @server.tool(name="identify_pills", description=TOOL_SPECS["identify_pills"][2])
    async def identify_pills(
        image_path: str | None = None, image_base64: str | None = None
    ) -> dict[str, Any]:
        return await toolbox.identify_pills(image_path=image_path, image_base64=image_base64)

    @server.tool(name="check_interaction", description=TOOL_SPECS["check_interaction"][2])
    async def check_interaction(
        drugs: list[str] | None = None, pills: list[int] | None = None
    ) -> dict[str, Any]:
        return await toolbox.check_interaction(drugs=drugs, pills=pills)

    @server.tool(name="get_drug_info", description=TOOL_SPECS["get_drug_info"][2])
    async def get_drug_info(drug: str, sections: list[str] | None = None) -> dict[str, Any]:
        return await toolbox.get_drug_info(drug=drug, sections=sections)

    @server.tool(name="search_documents", description=TOOL_SPECS["search_documents"][2])
    async def search_documents(
        query: str,
        ingredients: list[str] | None = None,
        doc_types: list[str] | None = None,
        sections: list[str] | None = None,
        k: int = 5,
        exclude_retracted: bool = True,
    ) -> dict[str, Any]:
        return await toolbox.search_documents(
            query,
            ingredients=ingredients,
            doc_types=doc_types,
            sections=sections,
            k=k,
            exclude_retracted=exclude_retracted,
        )

    @server.tool(name="lookup_brand", description=TOOL_SPECS["lookup_brand"][2])
    async def lookup_brand(name: str) -> dict[str, Any]:
        return await toolbox.lookup_brand(name=name)

    @server.tool(name="predict_toxicity", description=TOOL_SPECS["predict_toxicity"][2])
    async def predict_toxicity(molecules: list[str]) -> dict[str, Any]:
        return await toolbox.predict_toxicity(molecules=molecules)

    return server


def default_toolbox(
    settings: Settings, embedder: Embedder | None = None, llm: BaseChatModel | None = None
) -> Toolbox:
    """The agent's own toolbox (AgentHolder): same store, retriever, query rewriter, scanner
    and toxicity model, so an MCP client gets exactly what the chat uses."""
    engine = make_engine(settings.pharma_database_url)
    scanner = ScannerHolder(
        settings.models_dir,
        providers=settings.onnx_providers,
        threads=settings.onnx_threads,
        conf=settings.scan_detection_confidence,
    )
    holder = AgentHolder(settings, engine, scanner, llm=llm, embedder=embedder)
    return holder.toolbox()


# Strictly local (docs/infra.md): the HTTP transport only ever listens on the loopback.
LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.mcp_server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1", choices=LOOPBACK_HOSTS)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    server = build_server(default_toolbox(Settings()))
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
