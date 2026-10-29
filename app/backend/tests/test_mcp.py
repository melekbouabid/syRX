"""The MCP server lists the four tools and runs them (in-process MCP client, fake index)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from mcp import Client

from app.config import Settings
from app.mcp_server import build_server, default_toolbox, main
from app.rag.embeddings import FakeEmbedder
from tests.pharma_data import load_pharma
from tests.rag_fakes import DIM, build_index


def _server(tmp_path: Path) -> Any:
    url = f"sqlite:///{(tmp_path / 'pharma.db').as_posix()}"
    load_pharma(url, tmp_path / "inputs")
    asyncio.run(build_index(url))
    settings = Settings(_env_file=None, pharma_database_url=url, models_dir=tmp_path / "none")
    return build_server(default_toolbox(settings, FakeEmbedder(DIM)))


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


def test_http_transport_refuses_a_non_loopback_host() -> None:
    # The project runs only on this PC: never listen on 0.0.0.0 or a public address.
    for host in ("0.0.0.0", "192.168.1.10"):  # noqa: S104
        with pytest.raises(SystemExit):
            main(["--transport", "streamable-http", "--host", host])
