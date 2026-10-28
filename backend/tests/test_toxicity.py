"""Bonus toxicity tool: predictions with a fake model, the route, the agent and the MCP tool."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp import Client

from app.config import Settings
from app.mcp_server import build_server, default_toolbox
from app.medicines import DISCLAIMER
from app.rag.embeddings import FakeEmbedder
from app.rag.runtime import AgentHolder
from app.toxicity import (
    FP_BITS,
    MIXTURE_NOTE,
    TASKS,
    WORDING,
    ToxHolder,
    ToxicityUnavailable,
    ToxModel,
    fingerprint,
    predict_molecules,
    scaffold,
    standardise,
    tanimoto_max,
)
from tests.conftest import USER, sign_in
from tests.rag_fakes import DIM, FakeChatModel, build_index

WARFARIN = "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O"
FLUCONAZOLE = "C1=CC(=C(C=C1F)F)C(CN2C=NC=N2)(CN3C=NC=N3)O"
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
FORBIDDEN = ("safe", "sûr", "sans danger", "non toxique", "not toxic", "harmless")


def fake_model() -> ToxModel:
    """Each task returns a fixed probability; the training set holds warfarin only."""
    predictors = {
        task: (lambda x, p=0.05 + 0.05 * i: np.full(len(x), p)) for i, task in enumerate(TASKS)
    }
    train = np.packbits(fingerprint(standardise(WARFARIN)).astype(bool)).reshape(1, -1)
    return ToxModel(
        predictors=predictors,
        metrics=dict.fromkeys(TASKS, 0.7),
        version="test",
        train_fps=train,
    )


def fake_holder() -> ToxHolder:
    return ToxHolder(
        Path("unused"),
        Path("unused.csv"),
        model=fake_model(),
        smiles={"warfarin": WARFARIN, "fluconazole": FLUCONAZOLE},
    )


def test_featurisation_is_deterministic_and_strips_salts() -> None:
    mol = standardise(f"{WARFARIN}.[Na+].[Cl-]")
    assert mol is not None and mol.GetNumAtoms() == standardise(WARFARIN).GetNumAtoms()
    fp = fingerprint(mol)
    assert fp.shape == (FP_BITS,) and fp.dtype == np.uint8 and fp.sum() > 10
    assert np.array_equal(fp, fingerprint(standardise(WARFARIN)))
    assert scaffold(standardise(ASPIRIN)) == "c1ccccc1"
    assert standardise("not a smiles") is None and standardise("") is None
    packed = np.packbits(fp.astype(bool)).reshape(1, -1)
    assert tanimoto_max(fp, packed) == pytest.approx(1.0)


def test_each_molecule_is_predicted_separately_with_the_wording() -> None:
    result = predict_molecules(fake_holder(), ["warfarin", ASPIRIN])
    assert result["wording"] == WORDING
    assert result["wording"] == (
        "prédiction d'un modèle sur des tests in vitro, pas une évaluation clinique"
    )
    assert result["mixture_note"] == MIXTURE_NOTE and result["disclaimer"] == DISCLAIMER
    first, second = result["molecules"]
    assert first["resolved_as"] == "ingredient" and first["ingredient"] == "warfarin"
    assert second["resolved_as"] == "smiles" and second["ingredient"] is None
    for molecule in result["molecules"]:
        tasks = [p["task"] for p in molecule["predictions"]]
        assert tasks == list(TASKS)  # all 12 Tox21 assays, per molecule
        assert all(
            0 <= p["probability"] <= 1 and p["test_roc_auc"] == 0.7 for p in molecule["predictions"]
        )
    assert first["max_similarity_to_training"] == 1.0 and first["in_domain"]
    assert second["max_similarity_to_training"] < 1.0
    text = json.dumps(result, ensure_ascii=False).lower()
    assert not any(word in text for word in FORBIDDEN)
    assert "mixture" not in json.dumps(list(result["molecules"]))  # nothing about the pair


def test_names_resolver_invalid_input_and_at_most_two() -> None:
    result = predict_molecules(
        fake_holder(),
        ["Warfarine", "nope((", "fluconazole"],
        resolve_name=lambda value: "warfarin" if value.lower() == "warfarine" else None,
    )
    assert [m["ingredient"] for m in result["molecules"]] == ["warfarin"]
    assert result["errors"][0]["query"] == "nope(("  # third molecule ignored: two at most


def test_missing_model_says_how_to_publish(tmp_path: Path) -> None:
    holder = ToxHolder(tmp_path / "none", tmp_path / "none.csv")
    with pytest.raises(ToxicityUnavailable, match=r"train_tox21\.py publish"):
        predict_molecules(holder, ["CCO"])


def test_real_lightgbm_bundle_loads(tmp_path: Path) -> None:
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(80, FP_BITS)).astype(np.float32)
    y = (x[:, :5].sum(axis=1) > 2).astype(int)
    for task in TASKS:
        params = {"objective": "binary", "verbose": -1, "min_data_in_leaf": 5}
        lgb.train(params, lgb.Dataset(x, y), num_boost_round=3).save_model(
            str(tmp_path / f"{task}.txt")
        )
    meta = {"tasks": list(TASKS), "test_roc_auc": dict.fromkeys(TASKS, 0.75), "version": "9"}
    (tmp_path / "model.json").write_text(json.dumps(meta), encoding="utf-8")
    model = ToxModel.from_dir(tmp_path)
    out = model.predict(standardise(ASPIRIN))
    assert len(out["predictions"]) == 12 and model.version == "9"
    assert all(0 <= p["probability"] <= 1 for p in out["predictions"])


# ------------------------------------------------------------------ route


@pytest.fixture
def tox_app(app: FastAPI) -> FastAPI:
    app.state.toxicity = fake_holder()
    return app


def test_route_needs_sign_in(tox_app: FastAPI, anonymous: TestClient) -> None:
    assert anonymous.post("/api/toxicity", json={"molecules": ["warfarin"]}).status_code == 401


def test_route_predicts_one_or_two_molecules(tox_app: FastAPI, user: TestClient) -> None:
    response = user.post("/api/toxicity", json={"molecules": ["warfarin", "fluconazole"]})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["wording"] == WORDING and len(body["molecules"]) == 2
    assert body["model"]["version"] == "test" and body["disclaimer"] == DISCLAIMER
    too_many = {"molecules": ["warfarin", "fluconazole", ASPIRIN]}
    assert user.post("/api/toxicity", json=too_many).status_code == 422
    assert user.post("/api/toxicity", json={"molecules": []}).status_code == 422


def test_route_without_model_answers_503(app: FastAPI, user: TestClient, tmp_path: Path) -> None:
    app.state.toxicity = ToxHolder(tmp_path / "none", tmp_path / "none.csv")
    response = user.post("/api/toxicity", json={"molecules": ["CCO"]})
    assert response.status_code == 503 and "publish" in response.json()["detail"]


# ------------------------------------------------------------------ agent and MCP


def test_agent_adds_the_tool_cites_it_and_drops_non_toxic_claims(
    app: FastAPI, pharma: str, browser: Callable[[], TestClient]
) -> None:
    asyncio.run(build_index(pharma))
    llm = FakeChatModel(prompts=[])
    llm.answer = (
        "La warfarine n'est pas toxique [1]. "
        "Le modèle prédit une probabilité d'activité de 0,60 dans le test SR-p53 [1]."
    )
    app.state.agents = AgentHolder(
        app.state.settings,
        app.state.pharma_engine,
        llm=llm,
        embedder=FakeEmbedder(DIM),
        toxicity=fake_holder(),
    )
    client = sign_in(browser(), USER)
    response = client.post("/api/chat", json={"question": "La warfarine est-elle toxique ?"})
    assert response.status_code == 200, response.text
    body = response.json()
    calls = {(c["name"], c["chosen_by"]) for c in body["tool_calls"]}
    assert ("predict_toxicity", "rule") in calls
    first = body["citations"][0]
    assert first["doc_type"] == "prediction" and "in vitro" in first["title"]
    answer = body["answer"]
    assert "pas toxique" not in answer and "SR-p53" in answer
    # Both LLM sentences are about the prediction: replaced by the fixed summary.
    assert body["safety"]["violations"].count("toxicity_sentence_replaced") == 2
    assert "0,60 dans le test" not in answer
    # The fixed summary, written from the tool output: fake probabilities >= 0.5 for 3 tasks.
    assert f"Modèle Tox21 pour warfarin ({WORDING}) : probabilité d'activité" in answer
    assert "dans 3 des 12 tests : SR-HSE (0,50), SR-MMP (0,55), SR-p53 (0,60) [1]." in answer
    assert answer.endswith(DISCLAIMER)


def test_summary_alone_when_every_llm_sentence_is_removed(
    app: FastAPI, pharma: str, browser: Callable[[], TestClient]
) -> None:
    asyncio.run(build_index(pharma))
    llm = FakeChatModel(prompts=[], answer="La warfarine n'est pas identifiée comme toxique [1].")
    app.state.agents = AgentHolder(
        app.state.settings,
        app.state.pharma_engine,
        llm=llm,
        embedder=FakeEmbedder(DIM),
        toxicity=fake_holder(),
    )
    client = sign_in(browser(), USER)
    body = client.post("/api/chat", json={"question": "Is warfarin toxic?"}).json()
    assert "identifiée comme toxique" not in body["answer"]
    assert body["answer"].startswith("Tox21 model for warfarin (a model prediction on in vitro")
    assert body["safety"]["action"] == "rewritten" and 1 in body["safety"]["cited"]
    _, user_prompt = llm.prompts[-1]
    assert WORDING in str(user_prompt.content) and MIXTURE_NOTE in str(user_prompt.content)


def test_mcp_exposes_predict_toxicity(tmp_path: Path) -> None:
    from tests.pharma_data import load_pharma

    url = f"sqlite:///{(tmp_path / 'pharma.db').as_posix()}"
    load_pharma(url, tmp_path / "inputs")
    asyncio.run(build_index(url))
    settings = Settings(_env_file=None, pharma_database_url=url, models_dir=tmp_path / "none")
    toolbox = default_toolbox(settings, FakeEmbedder(DIM))
    toolbox.toxicity = fake_holder()
    server = build_server(toolbox)

    async def run() -> Any:
        async with Client(server) as client:
            return await client.call_tool("predict_toxicity", {"molecules": ["warfarine"]})

    result = asyncio.run(run())
    structured = getattr(result, "structured_content", None) or getattr(
        result, "structuredContent", None
    )
    data = (
        dict(structured.get("result", structured))
        if structured
        else json.loads(result.content[0].text)
    )
    assert data["wording"] == WORDING
    assert data["molecules"][0]["ingredient"] == "warfarin"  # French name resolved
