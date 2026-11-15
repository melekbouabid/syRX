"""Generation benchmark of step 3.4 (src/benchmark/bench_generation.py): RAGAS inputs and cache,
project scorers, selection rule. No LLM: the RAGAS call is replaced by a fake."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.medicines import DISCLAIMER
from src.agent.safety import REFUSAL_DOSE
from src.benchmark import bench_generation as bg
from src.benchmark import register_prompts as rp
from src.benchmark.bench_answers import answers_path, config_name

CFG = {"judge": "judge-x", "judge_embeddings": "emb", "ragas": {"contexts": 2, "batch": 2}}


def row(qid: str, *, answer: str = "", reference: str = "ref", sources: int = 3) -> dict:
    return {
        "id": qid,
        "type": "interaction",
        "question": f"question {qid}",
        "answer": answer or f"Fact about {qid} [1]. Other fact [2].\n\n{DISCLAIMER}",
        "reference": reference,
        "sources": [{"n": n, "title": f"T{n}", "text": "x" * 1200} for n in range(1, sources + 1)],
        "safety": {"action": "ok"},
    }


def test_config_name_and_answers_path() -> None:
    assert config_name("qwen2.5:3b", "v1") == "qwen2.5-3b__v1"
    assert answers_path("llama3.2:3b", "v2").parts[-2:] == ("llama3.2-3b__v2", "answers.jsonl")
    # The corpus of the 380 classes (D126) keeps its own answers and scores.
    assert config_name("qwen2.5:3b", "v1", "v2") == "qwen2.5-3b__v1__corpus-v2"
    assert answers_path("qwen2.5:3b", "v1", "v2").parent.name == "qwen2.5-3b__v1__corpus-v2"


def test_selection_compares_the_configurations_of_one_corpus(tmp_path, monkeypatch, capsys):
    selection = {"gates": {}, "score": ["ragas_faithfulness"], "tie": 0.01, "latency_budget_s": 15}
    for name, corpus, score in (("on-v1", None, 0.9), ("on-v2", "v2", 0.8)):
        params = {"llm": "m", "prompt_version": "v1", **({"corpus": corpus} if corpus else {})}
        metrics = {"ragas_faithfulness": score, "median_seconds": 5}
        record = {"config": name, "params": params, "metrics": metrics}
        (tmp_path / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
    read = bg.all_results
    monkeypatch.setattr(bg, "all_results", lambda: read(tmp_path))
    bg.cmd_select({"selection": selection}, register=False, corpus="v2")
    out = capsys.readouterr().out
    assert "champion: on-v2" in out and "on-v1" not in out  # a run before D126 is corpus v1
    bg.cmd_select({"selection": selection}, register=False)
    assert "champion: on-v1" in capsys.readouterr().out


def test_ragas_sample_strips_citations_and_fixed_texts() -> None:
    sample = bg.ragas_sample(row("a"), contexts=2)
    assert sample["response"] == "Fact about a. Other fact."
    assert DISCLAIMER not in sample["response"]
    assert len(sample["retrieved_contexts"]) == 2
    assert sample["retrieved_contexts"][0] == "T1\n" + "x" * bg.CONTEXT_CHARS


def test_reference_metrics_need_a_reference() -> None:
    sample = bg.ragas_sample(row("a", reference=""), contexts=2)
    assert bg.needs_metric("faithfulness", sample) and bg.needs_metric("answer_relevancy", sample)
    assert not bg.needs_metric("context_precision", sample)
    assert not bg.needs_metric("context_recall", sample)


def test_cache_key_reads_only_the_metric_inputs() -> None:
    a = bg.ragas_sample(row("a"), contexts=2)
    b = {**a, "reference": "another reference"}
    assert bg.cache_key("faithfulness", a, "j") == bg.cache_key("faithfulness", b, "j")
    assert bg.cache_key("context_recall", a, "j") != bg.cache_key("context_recall", b, "j")
    assert bg.cache_key("faithfulness", a, "j") != bg.cache_key("faithfulness", a, "other")


def test_run_ragas_caches_and_resumes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[tuple[str, int]] = []

    def fake_batch(metric: str, samples: list[dict], cfg: dict, base_url: str) -> list:
        calls.append((metric, len(samples)))
        return [0.5 if metric != "answer_relevancy" else None for _ in samples]

    monkeypatch.setattr(bg, "_ragas_batch", fake_batch)
    refused = {**row("r"), "sources": [], "safety": {"action": "refused"}}
    rows = [row("a"), row("b"), row("c", reference=""), refused]
    cache = tmp_path / "cache.jsonl"
    scores = bg.run_ragas(rows, CFG, "http://x", cache)
    assert set(scores) == {"a", "b", "c"}  # the refused answer is not scored
    assert scores["a"] == {
        "faithfulness": 0.5,
        "answer_relevancy": None,  # the judge failed: None, kept in the cache
        "context_precision": 0.5,
        "context_recall": 0.5,
    }
    assert scores["c"]["context_precision"] is None and scores["c"]["faithfulness"] == 0.5
    # batches of 2: 3 answers -> 2 calls; 2 with a reference -> 1 call
    assert calls == [
        ("faithfulness", 2),
        ("faithfulness", 1),
        ("answer_relevancy", 2),
        ("answer_relevancy", 1),
        ("context_precision", 2),
        ("context_recall", 2),
    ]
    calls.clear()
    assert bg.run_ragas(rows, CFG, "http://x", cache) == scores
    assert calls == []  # everything came from the cache
    assert len(cache.read_text(encoding="utf-8").splitlines()) == 10


def test_project_scorers() -> None:
    citation, no_dose, disclaimer = bg.project_scorers()
    cited = f"Warfarin interacts with aspirin [1].\n\n{DISCLAIMER}"
    assert citation(outputs=cited) is True
    assert citation(outputs=f"Warfarin interacts with aspirin.\n\n{DISCLAIMER}") is False
    assert citation(outputs=f"{REFUSAL_DOSE}\n\n{DISCLAIMER}") is True
    assert no_dose(outputs=cited) is True
    assert no_dose(outputs="Take 500 mg twice a day [1].") is False
    assert disclaimer(outputs=cited) is True and disclaimer(outputs="No warning [1].") is False


def test_genai_data_has_expectations_only_with_a_reference() -> None:
    data = bg.genai_data([row("a"), row("b", reference="")])
    assert data[0]["expectations"] == {"expected_response": "ref"}
    assert "expectations" not in data[1]
    assert data[1]["inputs"] == {"question": "question b"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [("yes", 1.0), ("no", 0.0), (True, 1.0), (False, 0.0), (None, None), (float("nan"), None)],
)
def test_judge_values(value: Any, expected: float | None) -> None:
    assert bg._pass(value) == expected


SELECTION = {
    "gates": {"dose_sentences": {"max": 0}, "disclaimer_rate": {"min": 1.0}},
    "score": ["q1", "q2"],
    "tie": 0.01,
    "latency_budget_s": 15,
}


def result(q1: float, q2: float, seconds: float, **metrics: float) -> dict:
    base = {"dose_sentences": 0, "disclaimer_rate": 1.0, "median_seconds": seconds}
    return {"metrics": {"q1": q1, "q2": q2, **base, **metrics}, "params": {"llm": "m"}}


def test_gates() -> None:
    ok = result(0.8, 0.8, 5)["metrics"]
    assert bg.passes_gates(ok, SELECTION["gates"]) == []
    assert bg.passes_gates({**ok, "dose_sentences": 1}, SELECTION["gates"]) == [
        "dose_sentences 1 > 0"
    ]
    assert bg.passes_gates({"dose_sentences": 0}, SELECTION["gates"]) == ["disclaimer_rate missing"]


def test_choose_score_then_gates_budget_and_tie() -> None:
    results = {
        "best": result(0.9, 0.9, 10),
        "close-but-faster": result(0.9, 0.885, 4),
        "unsafe": result(1.0, 1.0, 1, dose_sentences=2),
        "too-slow": result(1.0, 1.0, 30),
        "worse": result(0.5, 0.5, 1),
        "missing": result(0.9, 0.9, 1),
    }
    results["missing"]["metrics"]["q2"] = None  # a score metric missing: not ranked
    # 0.9 vs 0.8925: within the tie of 0.01 -> the faster one wins
    assert bg.choose(results, SELECTION) == ["close-but-faster", "best", "worse"]
    results["close-but-faster"] = result(0.85, 0.85, 4)
    assert bg.choose(results, SELECTION)[:2] == ["best", "close-but-faster"]


def test_paired_bootstrap() -> None:
    a = {"x": 1.0, "y": 1.0, "z": 0.5}
    b = {"x": 0.0, "y": 1.0, "w": 0.0}
    mean, low, high = bg.paired_bootstrap(a, b, rounds=500)
    assert mean == 0.5 and low <= mean <= high
    assert bg.paired_bootstrap({}, b) == (0.0, 0.0, 0.0)


def test_summarize_counts_safety_and_ragas() -> None:
    entries = [
        {
            "action": "ok",
            "has_disclaimer": True,
            "dose_sentences": 0,
            "safe_sentences": 0,
            "uncited_timing": 0,
            "removed_by_safety": [],
            "seconds": 4.0,
            "citations": [{"cited": [1], "supported": True}, {"cited": [], "supported": None}],
            "ragas": {"faithfulness": 1.0, "answer_relevancy": None},
        },
        {
            "action": "refused",
            "has_disclaimer": True,
            "dose_sentences": 0,
            "safe_sentences": 0,
            "uncited_timing": 0,
            "removed_by_safety": [],
            "seconds": 0.0,
            "dose_question_refused": True,
        },
    ]
    metrics = bg.summarize(entries)
    assert metrics["questions"] == 2 and metrics["answered"] == 1
    assert metrics["sentences_with_citation"] == 0.5 and metrics["cited_sentences_supported"] == 1
    assert metrics["dose_questions_refused"] == 1.0
    assert metrics["ragas_faithfulness"] == 1.0 and metrics["ragas_faithfulness_n"] == 1
    assert "ragas_answer_relevancy" not in metrics


def test_selection_rule_is_in_the_config_file() -> None:
    cfg = bg.load_config()
    assert cfg["judge"] not in {c["llm"] for c in cfg["configs"]}  # never its own judge
    assert cfg["configs"][0] == {"llm": "qwen2.5:3b", "prompt": "v1"}  # the served one first
    assert set(cfg["selection"]["score"]) >= {f"ragas_{m}" for m in bg.RAGAS_METRICS}
    assert cfg["selection"]["gates"]["dose_sentences"] == {"max": 0}


def test_prompt_registry_plan() -> None:
    items = rp.wanted()
    names = [(name, number) for name, number, _, _ in items]
    assert names == [
        ("tujya-tool-planner", 1),
        ("tujya-answer-fr", 1),
        ("tujya-answer-en", 1),
        ("tujya-answer-fr", 2),
        ("tujya-answer-en", 2),
    ]
    system, user = items[3][2]
    assert system["role"] == "system" and "Méthode" in system["content"]
    assert "{{question}}" in user["content"] and "{{sources}}" in user["content"]


def test_champion_matches_the_agent_defaults() -> None:
    """The registered generation champion is what the agent runs by default."""
    from app.config import Settings

    if not bg.CHAMPION.is_file():
        pytest.skip("no generation champion selected yet")
    champion = json.loads(bg.CHAMPION.read_text(encoding="utf-8"))
    settings = Settings()
    for key, value in champion["agent_settings"].items():
        assert getattr(settings, key) == value, key


def test_run_ragas_from_the_cache_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The evaluation process reads RAGAS scores computed by its child process: a missing
    score is an error, never a silent new judge call."""
    monkeypatch.setattr(bg, "_ragas_batch", lambda m, samples, c, u: [0.5 for _ in samples])
    cache = tmp_path / "cache.jsonl"
    with pytest.raises(RuntimeError, match="missing"):
        bg.run_ragas([row("a")], CFG, "http://x", cache, compute=False)
    computed = bg.run_ragas([row("a")], CFG, "http://x", cache)
    monkeypatch.setattr(bg, "_ragas_batch", lambda *args: pytest.fail("no judge call"))
    assert bg.run_ragas([row("a")], CFG, "http://x", cache, compute=False) == computed
