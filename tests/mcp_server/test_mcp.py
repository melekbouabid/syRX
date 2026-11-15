"""The MCP server lists the six tools and runs them (in-process MCP client, fake index)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from app.config import Settings
from src.mcp_server.server import build_server, default_toolbox, main
from src.rag.embeddings import FakeEmbedder
from tujya_testing.pharma_data import load_pharma
from tujya_testing.rag_fakes import DIM, FakeChatModel, build_index


def _settings(tmp_path: Path, **extra: Any) -> Settings:
    url = f"sqlite:///{(tmp_path / 'pharma.db').as_posix()}"
    load_pharma(url, tmp_path / "inputs")
    asyncio.run(build_index(url))
    return Settings(
        _env_file=None,
        pharma_database_url=url,
        models_dir=tmp_path / "none",
        vector_store="sql",  # build_index wrote the fake corpus into this database
        **{"rag_query_rewrite": False, **extra},  # no LLM in these tests unless given
    )


def _server(tmp_path: Path) -> Any:
    return build_server(default_toolbox(_settings(tmp_path), FakeEmbedder(DIM)))


def _payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if structured:
        return dict(structured.get("result", structured))
    return dict(json.loads(result.content[0].text))


def test_tools_are_listed_with_their_schemas(tmp_path: Path) -> None:
    server = _server(tmp_path)

    async def run() -> Any:
        async with Client(server) as client:
            return await client.list_tools()

    tools = {tool.name: tool for tool in asyncio.run(run()).tools}
    assert set(tools) == {
        "identify_pills",
        "check_interaction",
        "get_drug_info",
        "search_documents",
        "lookup_brand",
        "predict_toxicity",
    }
    search = tools["search_documents"]
    schema = getattr(search, "input_schema", None) or search.inputSchema
    assert schema["required"] == ["query"] and "ingredients" in schema["properties"]
    assert "Never use the word safe" in (tools["check_interaction"].description or "")


def test_call_check_interaction_and_search(tmp_path: Path) -> None:
    server = _server(tmp_path)

    async def run() -> tuple[Any, Any, Any]:
        async with Client(server) as client:
            interaction = await client.call_tool(
                "check_interaction", {"drugs": ["Warfarine", "fluconazole"]}
            )
            search = await client.call_tool(
                "search_documents", {"query": "warfarin pregnancy", "doc_types": ["label"]}
            )
            missing = await client.call_tool("identify_pills", {"image_path": "nope.jpg"})
            return interaction, search, missing

    interaction, search, missing = asyncio.run(run())
    report = _payload(interaction)
    pair = report["pairs"][0]
    assert pair["highest_severity"] == "Major"
    assert {pair["item_a"]["label"], pair["item_b"]["label"]} == {"warfarin", "fluconazole"}
    assert report["disclaimer"].startswith("Information uniquement")
    found = _payload(search)["results"]
    assert (
        found and found[0]["chunk_id"] == "war-ci" and all(r["doc_type"] == "label" for r in found)
    )
    assert missing.is_error if hasattr(missing, "is_error") else missing.isError


def test_call_lookup_brand(tmp_path: Path) -> None:
    server = _server(tmp_path)

    async def run() -> tuple[Any, Any]:
        async with Client(server) as client:
            known = await client.call_tool("lookup_brand", {"name": "Doliprane"})
            unknown = await client.call_tool("lookup_brand", {"name": "Zorglubine"})
            return known, unknown

    known, unknown = (_payload(r) for r in asyncio.run(run()))
    assert known["found"] and known["brand"] == "Doliprane" and known["country"] == "FR"
    (item,) = known["ingredients"]
    assert item["name"] == "acetaminophen" and item["method"] == "brand"
    assert item["url"].startswith("https://base-donnees-publique.medicaments.gouv.fr/")
    assert not unknown["found"] and unknown["ingredients"] == [] and unknown["message"]
    assert known["disclaimer"].startswith("Information uniquement")


def test_http_transport_refuses_a_non_loopback_host() -> None:
    # The project runs only on this PC: never listen on 0.0.0.0 or a public address.
    for host in ("0.0.0.0", "192.168.1.10"):  # noqa: S104
        with pytest.raises(SystemExit):
            main(["--transport", "streamable-http", "--host", host])


def test_the_server_uses_the_agents_toolbox_and_retrieval_setting(tmp_path: Path) -> None:
    """Same retrieval as the chat (D123): with query rewriting on, search_documents searches
    the LLM's English query, exactly like the agent's tool."""
    llm = FakeChatModel(prompts=[], answer="warfarin contraindications")
    toolbox = default_toolbox(
        _settings(tmp_path, rag_query_rewrite=True), FakeEmbedder(DIM), llm=llm
    )
    assert toolbox.rewriter is not None and toolbox.rewriter.llm is llm
    assert toolbox.scanner is not None and toolbox.toxicity is not None

    async def run() -> Any:
        async with Client(build_server(toolbox)) as client:
            return await client.call_tool("search_documents", {"query": "warfarine grossesse"})

    found = _payload(asyncio.run(run()))
    assert found["query"] == "warfarine grossesse"
    assert found["search_query"] == "warfarin contraindications"


def test_instructions_give_the_real_catalogue(tmp_path: Path) -> None:
    server = _server(tmp_path)

    async def run() -> Any:
        async with Client(server) as client:
            return client.instructions

    text = asyncio.run(run()) or ""
    # 4 classes and 3 active ingredients in the fake tables; fluconazole, warfarin and
    # losartan have documents in the fake corpus.
    assert "4 Tujya pill classes and their 3 active ingredients" in text
    assert "3 active ingredients have indexed documents" in text
    assert "150" not in text and "Information uniquement" in text


def test_an_ingredient_without_documents_is_said_so(tmp_path: Path) -> None:
    """A class of v2 whose ingredient has no indexed document (D119): pill data, no label
    excerpt, and an explicit note instead of a silent empty result."""
    server = _server(tmp_path)

    async def run() -> tuple[Any, Any, Any]:
        async with Client(server) as client:
            info = await client.call_tool("get_drug_info", {"drug": "hydrochlorothiazide"})
            covered = await client.call_tool("get_drug_info", {"drug": "warfarin"})
            search = await client.call_tool(
                "search_documents", {"query": "side effects", "ingredients": ["losartan", "hctz"]}
            )
            return info, covered, search

    info, covered, search = (_payload(r) for r in asyncio.run(run()))
    assert [c["class_id"] for c in info["pill_classes"]] == [5]
    assert info["label_sections"] == [] and info["not_indexed"] == ["hydrochlorothiazide"]
    assert "No document is indexed yet for hydrochlorothiazide" in info["coverage_note"]
    assert "check_interaction" in info["coverage_note"]
    assert covered["documents_indexed"] == ["warfarin"] and "not_indexed" not in covered
    assert covered["label_sections"]
    assert search["documents_indexed"] == ["losartan"] and search["not_indexed"] == ["hctz"]
