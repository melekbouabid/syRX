"""Selection of the answers for human labels: balanced verdicts, shuffled, verdict not shown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.evaluation import judge_check as jc


def row(qid: str, answer: str, sources: int = 4) -> dict[str, Any]:
    return {
        "id": qid,
        "question": f"question {qid}",
        "answer": answer,
        "sources": [
            {"n": n, "title": f"title {n}", "text": f"text {n} " * 300}
            for n in range(1, sources + 1)
        ],
    }


def test_cited_numbers_and_shown_answer() -> None:
    answer = "Warfarin interacts with aspirin [1]. Bleeding risk rises [3][1]."
    assert jc.cited(answer) == [1, 3]
    assert "[3]" in jc.shown_answer(answer)


def test_select_balances_verdicts_and_keeps_citations_the_judge_read() -> None:
    rows = [row(f"q{i}", f"Claim {i} [1].") for i in range(40)] + [row("far", "Claim [6].", 8)]
    verdicts = {r["id"]: int(i % 4 == 0) for i, r in enumerate(rows)}  # 10 faithful, 30 not
    verdicts["far"] = 0
    chosen = jc.select(rows, verdicts, 30, seed=0)
    assert len(chosen) == 30
    assert "far" not in {r["id"] for r in chosen}  # cites a source the judge did not read
    assert sum(verdicts[r["id"]] for r in chosen) == 10  # every faithful one, then unfaithful
    again = jc.select(rows, verdicts, 30, seed=0)
    assert [r["id"] for r in again] == [r["id"] for r in chosen]  # seeded
    order = [verdicts[r["id"]] for r in chosen]
    assert order != sorted(order) and order != sorted(order, reverse=True)  # shuffled


def test_sheet_never_contains_the_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [row(f"q{i}", f"Claim {i} [2].") for i in range(6)]
    monkeypatch.setattr(jc, "load_answers", lambda path=None: rows)
    monkeypatch.setattr(jc, "judge_verdicts", lambda: {r["id"]: i % 2 for i, r in enumerate(rows)})
    out = tmp_path / "sheet.json"
    jc.cmd_sheet(out, 4, seed=1)
    sheet = json.loads(out.read_text(encoding="utf-8"))
    assert len(sheet) == 4
    assert set(sheet[0]) == {
        "id",
        "question",
        "answer",
        "cited_passages",
        "other_passages_read_by_the_judge",
    }
    assert sheet[0]["cited_passages"].startswith("[2] title 2")
    assert len(sheet[0]["cited_passages"]) <= len("[2] title 2\n") + 900
    assert "faith" not in json.dumps(sheet).lower()


def test_kappa_reads_the_labels_against_the_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels = tmp_path / "human_labels.csv"
    labels.write_text("id,label\na,1\nb,0\nc,1\nd,0\n", encoding="utf-8")
    monkeypatch.setattr(jc, "HUMAN_LABELS", labels)
    monkeypatch.setattr(jc, "KAPPA_OUT", tmp_path / "kappa.json")
    monkeypatch.setattr(jc, "judge_verdicts", lambda: {"a": 1, "b": 0, "c": 0, "d": 0})
    jc.cmd_kappa(log=False)
    data = json.loads((tmp_path / "kappa.json").read_text(encoding="utf-8"))
    assert data["confusion"] == [[1, 1], [0, 2]]
    assert data["observed_agreement"] == 0.75
    assert data["kappa"] == pytest.approx(0.5)
