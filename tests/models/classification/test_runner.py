"""The classifier runner (src/models/classification/runner.py) with the fake model family:
the Optuna search, retraining, test scores, ONNX export and export folder are real."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

onnx = pytest.importorskip("onnx")

from src.models.classification import runner  # noqa: E402
from tests.models.classification.conftest import FAKE, HELPERS  # noqa: E402
from tests.models.classification.helpers import make_dataset  # noqa: E402


@pytest.fixture(autouse=True)
def fake_family(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(runner.BACKENDS, "torch", HELPERS)


def test_runner_writes_a_complete_export(exported: SimpleNamespace) -> None:
    out, summary = exported.base / "full", exported.full
    assert json.loads((out / "run.json").read_text(encoding="utf-8")) == json.loads(
        json.dumps(summary, default=str)
    )
    assert summary["schema"] == runner.SCHEMA and summary["task"] == "classify"
    assert summary["arch"] == "fake" and summary["smoke"] is False
    assert [t["number"] for t in summary["trials"]] == [0, 1, 2]
    best = max(summary["trials"], key=lambda t: t["value"])
    assert summary["search"]["best_trial"] == best["number"]
    assert summary["final"]["params"]["head_epochs"] == 2  # the `final` override
    assert summary["final"]["params"]["lr"] == best["params"]["lr"]
    metrics = summary["final"]["metrics"]
    for key in (
        "val_top1",
        "val_top5",
        "val_f1_macro",
        "test_v2_top1",
        "test_v2_top5",
        "test_v2_f1_macro",
        "test_v2_all_top1",
        "test_v2_v1c_top1",
        "parity_top1_agreement",
    ):
        assert 0.0 <= metrics[key] <= 1.0, key
    assert metrics["test_v2_images"] == 6  # the holdout: one test crop per class
    assert metrics["test_v2_all_images"] == 12
    assert metrics["test_v2_v1c_images"] == 3  # holdout crops of the three v1 classes
    assert summary["dataset"]["classes"] == 6 and summary["dataset"]["label_space"] == "c3pi-v2"
    assert summary["final"]["latency"]["kaggle_onnx_cpu_latency_ms"] > 0
    for name in (
        "weights/best.pt",
        "model/classifier.onnx",
        "classes.csv",
        "trials.csv",
        "config.yaml",
        "requirements.txt",
        "confusion/confusion_test_v2.csv",
        "confusion/per_class_test_v2.csv",
        "confusion/top_confusions_test_v2.csv",
    ):
        assert (out / name).is_file(), name
    assert len(summary["artifacts"]["gradcam_pictures"]) == 6  # at most 8, 6 holdout crops
    assert not (out / "work").exists()


def test_the_class_list_travels_with_the_onnx_model(exported: SimpleNamespace) -> None:
    import ast

    model = onnx.load(str(exported.base / "full" / "model" / "classifier.onnx"))
    meta = {p.key: p.value for p in model.metadata_props}
    names = ast.literal_eval(meta["names"])  # what backend/app/inference.py reads
    assert names == {i: f"{i:03d}" for i in range(6)}
    assert json.loads(meta["class_keys"])["0"] == "000000000_1"
    assert meta["label_space"] == "c3pi-v2" and meta["num_classes"] == "6"
    assert meta["task"] == "classify" and meta["imgsz"].startswith("[")


def test_smoke_run_uses_a_subset(exported: SimpleNamespace) -> None:
    smoke = exported.smoke
    assert smoke["smoke"] is True and len(smoke["trials"]) == 2
    assert smoke["dataset"]["images"] == {"train": 6, "val": 4, "test": 4}
    assert smoke["final"]["params"]["imgsz"] == 32
    assert not (exported.base / "smoke" / "subset").exists()


def test_runner_reports_why_every_trial_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.models.classification import helpers

    def broken(*args: object, **kwargs: object) -> None:
        raise KeyError("no such setting")

    monkeypatch.setattr(helpers, "train", broken)
    data = make_dataset(tmp_path / "data")
    with pytest.raises(RuntimeError, match="no such setting"):
        runner.run_experiment(FAKE, data, tmp_path / "out", smoke=True, log=lambda _: None)
