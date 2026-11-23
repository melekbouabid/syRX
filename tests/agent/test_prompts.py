"""Versioned prompts of the agent (step 3.4): every version keeps the safety rules, the setting
chooses the version, an unknown version is refused."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI

from src.agent.graph import TujyaAgent
from src.agent.prompts import ANSWER_PROMPTS, DEFAULT_PROMPT_VERSION, answer_prompt
from src.agent.runtime import AgentHolder
from src.rag.init_qdrant import FakeEmbedder
from tests.support.rag_fakes import DIM, FakeChatModel, build_index

RULES = {
    "fr": [
        "JAMAIS de dose",
        "sûre",
        "aucune interaction connue dans cette base de données",
        "conseil d'horaire",
        "marque",
        "entre crochets",
        "UNIQUEMENT|N'ajoute rien",
    ],
    "en": [
        "NEVER give a dose",
        '"safe"',
        "no known interaction in this database",
        "timing",
        "brand",
        "in brackets",
        "ONLY|Add nothing",
    ],
}


@pytest.mark.parametrize("version", sorted(ANSWER_PROMPTS))
@pytest.mark.parametrize("language", ["fr", "en"])
def test_every_prompt_version_keeps_the_safety_rules(version: str, language: str) -> None:
    text = answer_prompt(version, language)
    for rule in RULES[language]:
        assert any(option in text for option in rule.split("|")), (version, language, rule)


def test_versions_differ_and_v1_is_the_default() -> None:
    assert DEFAULT_PROMPT_VERSION == "v1"
    assert ANSWER_PROMPTS["v1"]["fr"] != ANSWER_PROMPTS["v2"]["fr"]
    assert ANSWER_PROMPTS["v1"]["en"] != ANSWER_PROMPTS["v2"]["en"]


def test_unknown_version_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown prompt version"):
        answer_prompt("v9", "fr")
    with pytest.raises(ValueError):
        TujyaAgent(FakeChatModel(), toolbox=None, prompt_version="v9")  # type: ignore[arg-type]


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_the_setting_chooses_the_answer_prompt(app: FastAPI, pharma: str, version: str) -> None:
    asyncio.run(build_index(pharma))
    settings = app.state.settings.model_copy(update={"llm_prompt_version": version})
    llm = FakeChatModel(prompts=[])
    agent = AgentHolder(settings, app.state.pharma_engine, llm=llm, embedder=FakeEmbedder(DIM))
    state = asyncio.run(agent.agent().run("What does warfarin interact with?"))
    assert state["answer"]
    system = str(llm.prompts[-1][0].content)
    assert system == answer_prompt(version, "en")
