"""Start the MCP server as a subprocess (stdio), list its tools and call them on real data.

python -m src.agent.mcp_smoke
(project venv; uses the TUJYA_* environment of this shell; the stack running: postgres,
qdrant, ollama, and the served models in models/)

1. identify_pills on a test-split photo of the demo gallery (never seen in training, D105),
   preferably of one of the 230 classes added in v2 (class id 150 or more);
2. check_interaction between the identified pill and warfarin;
3. get_drug_info on its first active ingredient (says when no document is indexed for it);
4. search_documents with the served retrieval setting (query rewriting on: `search_query`).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from app.config import Settings
from app.gallery import load_gallery

ROOT = Path(__file__).resolve().parents[2]  # repository root


def gallery_photo() -> tuple[Path, int]:
    """A photo of the demo gallery (the API's own rule: v2 test split, never trained on,
    D105), a class of v2 first."""
    settings = Settings()
    photos = load_gallery(
        settings.demo_gallery_file,
        settings.demo_split_file,
        settings.demo_holdout_file,
        settings.demo_images_dir,
        settings.demo_gradcam_dir,
    )
    if not photos:
        raise SystemExit("no gallery photo on disk: dvc pull data/raw/c3pi/images")
    photo = sorted(photos, key=lambda p: p.class_id < 150)[0]
    return photo.file, photo.class_id


def payload(result: Any) -> dict[str, Any]:
    text = result.content[0].text if result.content else "{}"
    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        raise SystemExit(f"tool error: {text}")
    return dict(json.loads(text))


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.agent.mcp_server"],
        cwd=str(ROOT),
        env=dict(os.environ),
    )
    async with Client(params) as client:
        tools = await client.list_tools()
        print("tools:", [tool.name for tool in tools.tools])
        print("instructions:", (client.instructions or "")[:160], "...")
        image, expected = gallery_photo()
        scan = payload(await client.call_tool("identify_pills", {"image_path": str(image)}))
        print(f"identify_pills {image.name} (true class {expected}) ->")
        for pill in scan["pills"]:
            top = pill["candidates"][0]
            print(
                f"  class {top['class_id']} {top['name']} {top['active_ingredients']} "
                f"p={top['probability']:.3f} confirm={pill['needs_confirmation']}"
            )
        if not scan["pills"]:
            print("  no pill found:", scan.get("message"))
            return
        top = scan["pills"][0]["candidates"][0]
        check = payload(
            await client.call_tool(
                "check_interaction", {"pills": [top["class_id"]], "drugs": ["warfarin"]}
            )
        )
        print("check_interaction ->", check["message"])
        for pair in check["pairs"]:
            print(
                f"  {pair['item_a']['label']} + {pair['item_b']['label']}: "
                f"{pair['highest_severity']}"
            )
        ingredient = (top["active_ingredients"] or ["warfarin"])[0]
        info = payload(await client.call_tool("get_drug_info", {"drug": ingredient}))
        print(
            f"get_drug_info {ingredient} -> {len(info['pill_classes'])} pill classes, "
            f"{len(info['label_sections'])} label sections; "
            f"{info.get('coverage_note', 'documents indexed')}"
        )
        search = payload(
            await client.call_tool(
                "search_documents",
                {"query": "Quels sont les effets indésirables de la metformine ?", "k": 3},
            )
        )
        print(
            f"search_documents -> search_query {search['search_query']!r}:",
            [r["title"] for r in search["results"]],
        )


if __name__ == "__main__":
    asyncio.run(main())
