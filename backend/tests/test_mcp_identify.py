"""identify_pills through the MCP server (step 3.6): the same pipeline as /api/scan, with the
tiny fake ONNX models of fake_models.py; then the identified pills go to check_interaction."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from mcp import Client

from src.agent.mcp_server import build_server, default_toolbox
from src.rag.init_qdrant import FakeEmbedder
from tests.fake_models import write_models
from tests.test_scan import BOXES, NAMES, photo


def _payload(result: Any) -> dict[str, Any]:
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    if structured:
        return dict(structured.get("result", structured))
    return dict(json.loads(result.content[0].text))


def test_identify_then_check_the_identified_pills(
    app: FastAPI, pharma: str, tmp_path: Path
) -> None:
    settings = app.state.settings.model_copy(update={"rag_query_rewrite": False})
    write_models(settings.models_dir, BOXES, NAMES)
    image = tmp_path / "pills.png"
    image.write_bytes(photo())
    server = build_server(default_toolbox(settings, FakeEmbedder(16)))

    async def run() -> tuple[Any, Any]:
        async with Client(server) as client:
            scan = _payload(await client.call_tool("identify_pills", {"image_path": str(image)}))
            pills = [p["candidates"][0]["class_id"] for p in scan["pills"]]
            check = await client.call_tool("check_interaction", {"pills": pills})
            return scan, _payload(check)

    scan, check = asyncio.run(run())
    tops = sorted(p["candidates"][0]["class_id"] for p in scan["pills"])
    assert tops == [5, 7]  # HYZAAR (losartan + hydrochlorothiazide) and warfarin
    assert all(len(p["candidates"]) == 5 for p in scan["pills"])
    assert "consultez un pharmacien ou un médecin" in scan["disclaimer"]
    severities = {pair["highest_severity"] for pair in check["pairs"]}
    assert "Moderate" in severities  # losartan + warfarin in the fake DDInter table
