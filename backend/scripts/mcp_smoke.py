"""Start the MCP server as a subprocess (stdio), list its tools and call two of them.

cd backend
python scripts/mcp_smoke.py            # uses the SYNERGYRX_* environment of this shell
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

BACKEND = Path(__file__).resolve().parent.parent


async def main() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp_server"],
        cwd=str(BACKEND),
        env=dict(os.environ),
    )
    async with Client(params) as client:
        tools = await client.list_tools()
        print("tools:", [tool.name for tool in tools.tools])
        for name, args in (
            ("check_interaction", {"drugs": ["warfarine", "fluconazole"]}),
            ("search_documents", {"query": "metformin lactic acidosis", "k": 3}),
        ):
            result = await client.call_tool(name, args)
            text = result.content[0].text if result.content else "{}"
            data = json.loads(text)
            if name == "check_interaction":
                print(name, "->", data["message"], data["pairs"][0]["highest_severity"])
            else:
                print(name, "->", [r["title"] for r in data["results"]])


if __name__ == "__main__":
    asyncio.run(main())
