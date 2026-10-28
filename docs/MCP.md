# Tujya MCP server

`backend/app/mcp_server.py` exposes the agent's four tools through the Model Context
Protocol, with the official Python SDK (`mcp` 2.3, `MCPServer`, formerly FastMCP). It is the
same code as the LangGraph agent's tools (`backend/app/rag/tools.py`), so an MCP client gets
exactly what the agent gets.

| Tool | Arguments | What it returns |
|---|---|---|
| `identify_pills` | `image_path` or `image_base64` | detected pills, top-5 classes with probabilities, `needs_confirmation` |
| `check_interaction` | `drugs` (names, French or English), `pills` (class_ids) | DDInter pairs with severity, description, source; never "safe" |
| `get_drug_info` | `drug`, `sections` | pill classes with the ingredient, openFDA label excerpts (no dosage section) |
| `search_documents` | `query`, `ingredients`, `doc_types`, `sections`, `k`, `exclude_retracted` | hybrid BM25 + bge-m3 hits with title, section, URL, source, licence, text |

The server's `instructions` repeat the safety rules (no doses, never "safe", cite sources,
disclaimer); a client's own model is responsible for following them in its answers.

## Run it locally

Needs what the tools read: the medicine tables and the RAG index in PostgreSQL
(`docker compose up -d postgres ollama`, then `pharma-init` / `rag-init` or the commands in
docs/INFRA.md and docs/RAG.md), Ollama for the query embeddings, and `models/` for
`identify_pills`. From `backend/` with the backend venv:

```bash
# PowerShell: $env:SYNERGYRX_PHARMA_DATABASE_URL = "postgresql+psycopg://tujya:tujya@localhost:5432/tujya"
export SYNERGYRX_PHARMA_DATABASE_URL=postgresql+psycopg://tujya:tujya@localhost:5432/tujya
python -m app.mcp_server                                     # stdio transport
python -m app.mcp_server --transport streamable-http --port 8765   # http://127.0.0.1:8765/mcp
python scripts/mcp_smoke.py                                  # starts it over stdio, lists, calls
```

Smoke test on 2026-10-28 (`scripts/mcp_smoke.py`):

```
tools: ['identify_pills', 'check_interaction', 'get_drug_info', 'search_documents']
check_interaction -> 1 paire(s) sur 1 avec une interaction connue dans cette base de données. Major
search_documents -> ['Metformin label (METFORMIN HYDROCHLORIDE): Warnings and precautions',
                     'Metformin label (METFORMIN HYDROCHLORIDE): Boxed warning', ...]
```

## Connect an MCP client

**Claude Desktop** (`claude_desktop_config.json`) or any client using the same format
(Cursor, VS Code `mcp.json` with `servers` instead of `mcpServers`):

```json
{
  "mcpServers": {
    "tujya": {
      "command": "C:/path/to/Proj/backend/.venv/Scripts/python.exe",
      "args": ["-m", "app.mcp_server"],
      "cwd": "C:/path/to/Proj/backend",
      "env": {
        "SYNERGYRX_PHARMA_DATABASE_URL": "postgresql+psycopg://tujya:tujya@localhost:5432/tujya",
        "SYNERGYRX_OLLAMA_URL": "http://localhost:11434"
      }
    }
  }
}
```

**Claude Code**:

```bash
claude mcp add tujya --env SYNERGYRX_PHARMA_DATABASE_URL=postgresql+psycopg://tujya:tujya@localhost:5432/tujya \
  -- C:/path/to/Proj/backend/.venv/Scripts/python.exe -m app.mcp_server
```

(run it from `backend/`, or the client must start the server with `cwd` = `backend/`).

**HTTP clients** (MCP Inspector, remote agents): start the streamable HTTP transport and
connect to `http://127.0.0.1:8765/mcp`. It listens on localhost only by default; there is no
authentication, so do not expose it.

## Tests

`backend/tests/test_mcp.py` (in-process MCP client, SQLite tables, fake embeddings): the
four tools are listed with their input schemas; `check_interaction` (French name) returns
DDInter Major for warfarin + fluconazole; `search_documents` with a `doc_types` filter
returns labels only; `identify_pills` on a missing file returns an MCP error result.
