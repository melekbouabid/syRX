"""Ask the Tujya agent one question from the command line (real index and Ollama).

# from the repository root, backend venv (app/backend/.venv) active
python -m module2_ai_engineering.agent.ask "La warfarine interagit-elle avec le fluconazole ?"
python -m module2_ai_engineering.agent.ask "Effets indésirables ?" --pills 9 120 --database-url postgresql+psycopg://...
"""

from __future__ import annotations

import argparse
import asyncio
import json

from app.config import Settings
from app.db import make_engine
from module2_ai_engineering.agent.runtime import AgentHolder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--pills", type=int, nargs="*", default=[])
    parser.add_argument("--database-url")
    parser.add_argument("--json", action="store_true", help="print the whole final state")
    args = parser.parse_args()
    settings = Settings()
    if args.database_url:
        settings.pharma_database_url = args.database_url
    holder = AgentHolder(settings, make_engine(settings.pharma_database_url))
    state = asyncio.run(holder.agent().run(args.question, args.pills))
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=1, default=str))
        return
    for call in state.get("tool_calls", []):
        print(f"tool {call['name']} ({call['chosen_by']}) {call['args']} -> {call['summary']}")
    for source in state.get("sources", []):
        print(f"[{source['n']}] {source['title']} {source['url']}")
    print("\n" + state.get("answer", ""))
    print(f"\nsafety: {state.get('safety')}\ntimings: {state.get('timings')}")


if __name__ == "__main__":
    main()
