"""Register the agent's prompts in the MLflow Prompt Registry (a changed text is a new version).

Run: `python -m src.evaluation.register_prompts` (MLflow running).
"""

from __future__ import annotations

import os
from typing import Any

from src.agent.prompts import ANSWER_PROMPTS, REGISTRY_NAMES, TOOLS_PROMPT

USER_TEMPLATE = {
    "fr": "{{context}}Sources :\n{{sources}}\n\nQuestion : {{question}}",
    "en": "{{context}}Sources :\n{{sources}}\n\nQuestion : {{question}}",
}
TOOLS_USER = "Question: {{question}}\nActive ingredients mentioned or confirmed: {{ingredients}}"


def answer_template(version: str, language: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ANSWER_PROMPTS[version][language]},
        {"role": "user", "content": USER_TEMPLATE[language]},
    ]


def tools_template() -> list[dict[str, str]]:
    return [{"role": "system", "content": TOOLS_PROMPT}, {"role": "user", "content": TOOLS_USER}]


def wanted() -> list[tuple[str, int, list[dict[str, str]], str]]:
    """(registry name, version number, template, commit message), in registration order."""
    items = [(REGISTRY_NAMES["tools"], 1, tools_template(), "tool planner of the agent")]
    for version in sorted(ANSWER_PROMPTS, key=lambda v: int(v.removeprefix("v"))):
        for language in ("fr", "en"):
            message = {
                "v1": "first answer prompt (docs/module2/README.md)",
                "v2": "direct answer first, one source fact per sentence",
            }.get(version, f"answer prompt {version}")
            items.append(
                (
                    REGISTRY_NAMES[language],
                    int(version.removeprefix("v")),
                    answer_template(version, language),
                    message,
                )
            )
    return items


def registered_template(name: str, number: int) -> Any:
    import mlflow

    try:
        return mlflow.genai.load_prompt(f"prompts:/{name}/{number}").template
    except Exception:  # not registered yet
        return None


def main() -> None:
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    for name, number, template, message in wanted():
        existing = registered_template(name, number)
        if existing is not None:
            if existing != template:
                raise SystemExit(f"{name} version {number} differs from src/agent/prompts.py")
            print(f"{name} v{number}: already registered")
            continue
        if number > 1 and registered_template(name, number - 1) is None:
            raise SystemExit(f"{name}: version {number - 1} must be registered first")
        prompt = mlflow.genai.register_prompt(
            name=name,
            template=template,
            commit_message=message,
            tags={"project": "tujya", "plan_step": "3.4", "source": "src/agent/prompts.py"},
        )
        if prompt.version != number:
            raise SystemExit(f"{name}: registry gave version {prompt.version}, expected {number}")
        print(f"{name} v{prompt.version}: registered")


if __name__ == "__main__":
    main()
