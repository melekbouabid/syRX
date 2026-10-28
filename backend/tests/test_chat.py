"""POST /api/chat and the LangGraph agent, with a scripted LLM and fake embeddings."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.medicines import DISCLAIMER
from app.rag.embeddings import FakeEmbedder
from app.rag.runtime import AgentHolder
from tests.conftest import USER, sign_in
from tests.rag_fakes import DIM, FakeChatModel, build_index


@pytest.fixture
def llm() -> FakeChatModel:
    return FakeChatModel(prompts=[])


@pytest.fixture
def chat_app(app: FastAPI, pharma: str, llm: FakeChatModel) -> FastAPI:
    asyncio.run(build_index(pharma))
    app.state.agents = AgentHolder(
        app.state.settings, app.state.pharma_engine, llm=llm, embedder=FakeEmbedder(DIM)
    )
    return app


@pytest.fixture
def client(chat_app: FastAPI, browser: Callable[[], TestClient]) -> TestClient:
    return sign_in(browser(), USER)


def ask(client: TestClient, question: str, **extra: object) -> dict:
    response = client.post("/api/chat", json={"question": question, **extra})
    assert response.status_code == 200, response.text
    return response.json()


def test_chat_needs_sign_in(chat_app: FastAPI, anonymous: TestClient) -> None:
    assert anonymous.post("/api/chat", json={"question": "warfarine ?"}).status_code == 401


def test_interaction_question_calls_tools_and_cites(client: TestClient, llm: FakeChatModel) -> None:
    llm.tool_calls = [
        {"name": "search_documents", "args": {"query": "fluconazole warfarin interaction"}},
        {"name": "not_a_tool", "args": {}},  # ignored
    ]
    llm.answer = (
        "Le fluconazole augmente le temps de prothrombine chez les patients sous warfarine [2]. "
        "DDInter classe cette interaction comme majeure [1]."
    )
    body = ask(client, "La warfarine et le fluconazole interagissent-ils ?")
    assert body["language"] == "fr"
    assert body["ingredients"] == ["warfarin", "fluconazole"]
    names = [(c["name"], c["chosen_by"]) for c in body["tool_calls"]]
    assert ("search_documents", "llm") in names and ("check_interaction", "rule") in names
    assert all(c["ok"] for c in body["tool_calls"])
    first = body["citations"][0]
    assert first["n"] == 1 and first["doc_type"] == "interaction" and "Major" in first["title"]
    assert {c["n"] for c in body["citations"] if c["cited"]} == {1, 2}
    assert all(c["url"] and c["title"] for c in body["citations"])
    assert "retracted" not in json.dumps(body["citations"]).lower()
    assert body["answer"].endswith(DISCLAIMER) and body["safety"]["action"] == "ok"
    assert body["model"] == "fake-llm" and body["timings"]["total_ms"] > 0
    # The answer prompt holds the numbered sources and the rules.
    system, user = llm.prompts[-1]
    assert "JAMAIS de dose" in str(system.content) and "[1] DDInter 2.0" in str(user.content)


def test_unsafe_answer_is_rewritten(client: TestClient, llm: FakeChatModel) -> None:
    llm.answer = (
        "Le losartan peut donner des vertiges [1]. Prenez 50 mg le matin [1]. "
        "Cette association est sans danger. Voir aussi [12]."
    )
    body = ask(client, "Effets indésirables du losartan ?")
    answer = body["answer"]
    assert "50 mg" not in answer and "sans danger" not in answer and "[12]" not in answer
    assert set(body["safety"]["violations"]) >= {"dose", "safe", "bad_citation"}
    assert body["safety"]["action"] == "rewritten" and answer.endswith(DISCLAIMER)


def test_dose_question_is_refused_without_calling_the_llm(
    client: TestClient, llm: FakeChatModel
) -> None:
    body = ask(client, "Quelle dose de warfarine dois-je prendre ?")
    assert body["safety"]["action"] == "refused" and "dose_question" in body["safety"]["violations"]
    assert "jamais de dose" in body["answer"] and body["answer"].endswith(DISCLAIMER)
    assert body["tool_calls"] == [] and body["citations"] == [] and llm.prompts == []


def test_planner_failure_falls_back_to_rules(client: TestClient, llm: FakeChatModel) -> None:
    llm.fail_planner = True
    llm.answer = "Le losartan peut donner des vertiges [1]."
    body = ask(client, "What are the adverse reactions of losartan?", pills=[5])
    assert body["language"] == "en"
    assert {c["name"] for c in body["tool_calls"]} == {"check_interaction", "search_documents"}
    assert {"losartan", "hydrochlorothiazide"} <= set(body["ingredients"])  # from pill 5
    assert body["citations"] and body["answer"].endswith(DISCLAIMER)


def test_stream_sends_tools_tokens_then_final(client: TestClient, llm: FakeChatModel) -> None:
    llm.answer = "Le fluconazole est sûr. Il augmente le temps de prothrombine [1]."
    response = client.post(
        "/api/chat", json={"question": "fluconazole et warfarine ?", "stream": True}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = []
    for block in response.text.strip().split("\n\n"):
        name = block.split("\n")[0].removeprefix("event: ")
        data = json.loads(block.split("\n")[1].removeprefix("data: "))
        events.append((name, data))
    kinds = [name for name, _ in events]
    assert kinds[0] == "tool" and kinds[-1] == "final" and "token" in kinds
    streamed = "".join(d["text"] for name, d in events if name == "token")
    assert "sûr" not in streamed and "prothrombine" in streamed  # held back while streaming
    final = events[-1][1]
    assert "sûr" not in final["answer"] and final["answer"].endswith(DISCLAIMER)


def test_no_index_answers_503(app: FastAPI, pharma: str, browser: Callable[[], TestClient]) -> None:
    app.state.agents = AgentHolder(
        app.state.settings, app.state.pharma_engine, llm=FakeChatModel(), embedder=FakeEmbedder(DIM)
    )
    response = sign_in(browser(), USER).post("/api/chat", json={"question": "warfarine ?"})
    assert response.status_code == 503 and "app.rag.index" in response.json()["detail"]


def test_chat_route_in_openapi(chat_app: FastAPI, anonymous: TestClient) -> None:
    paths = anonymous.get("/openapi.json").json()["paths"]
    assert "/api/chat" in paths and "post" in paths["/api/chat"]
