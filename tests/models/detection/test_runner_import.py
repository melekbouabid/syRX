"""The experiment runner and the MLflow import, end to end, on a temporary SQLite store.

The model family is the fake one of helpers.py (no PyTorch): the runner's Optuna search,
retraining, test scores, ONNX export and export folder are real, and so is the import
(parent run, one child run per trial, Registry, promote_if_better).
"""

from __future__ import annotations

import json
import shutil
import uuid
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

mlflow = pytest.importorskip("mlflow")
pytest.importorskip("onnx")

from mlflow import MlflowClient  # noqa: E402

from src.models.detection import import_kaggle, runner  # noqa: E402
from src.utils import mlflow_utils as mu  # noqa: E402
from tests.models.detection.helpers import make_dataset, tiny_detector  # noqa: E402

FAKE = Path(__file__).parent / "configs" / "detector_fake.yaml"


@pytest.fixture(autouse=True)
def fake_family(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(runner.BACKENDS, "yolo", "tests.models.detection.helpers")


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    base = tmp_path_factory.mktemp("mlflow")
    return SimpleNamespace(
        uri="sqlite:///" + (base / "mlflow.db").as_posix(),
        artifacts=(base / "artifacts").as_uri(),
    )


@pytest.fixture
def tracking(store: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store.uri)
    monkeypatch.chdir(tmp_path)
    mu.configure()
    mlflow.set_registry_uri(store.uri)
    experiment = f"test-{uuid.uuid4().hex[:8]}"
    mlflow.create_experiment(experiment, artifact_location=f"{store.artifacts}/{experiment}")
    model = f"detector-{uuid.uuid4().hex[:8]}"
    params = tmp_path / "params.yaml"
    params.write_text(
        f"registry:\n  detector:\n    name: {model}\n    metric: test_mAP50_95\n"
        "    min_gain: 0.005\ncompare:\n  detector:\n    metric: test_v2_mAP50_95\n",
        encoding="utf-8",
    )
    yield SimpleNamespace(experiment=experiment, model=model, params=params, client=MlflowClient())
    while mlflow.active_run():
        mlflow.end_run()


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """One full run and one smoke run of the fake family (shared: the runner is the slow part)."""
    base = tmp_path_factory.mktemp("runner")
    data = make_dataset(base / "data")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(runner.BACKENDS, "yolo", "tests.models.detection.helpers")
        full = runner.run_experiment(FAKE, data, base / "full", log=lambda _: None)
        smoke = runner.run_experiment(FAKE, data, base / "smoke", smoke=True, log=lambda _: None)
    return SimpleNamespace(base=base, data=data, full=full, smoke=smoke)


def test_runner_writes_a_complete_export(exported: SimpleNamespace) -> None:
    out, summary = exported.base / "full", exported.full
    assert json.loads((out / "run.json").read_text(encoding="utf-8")) == json.loads(
        json.dumps(summary, default=str)
    )
    assert summary["schema"] == runner.SCHEMA
    assert summary["arch"] == "fake"
    assert summary["smoke"] is False
    assert [t["number"] for t in summary["trials"]] == [0, 1, 2]
    assert all(t["state"] == "COMPLETE" for t in summary["trials"])
    best = max(summary["trials"], key=lambda t: t["value"])
    assert summary["search"]["best_trial"] == best["number"]
    assert summary["final"]["params"] == best["params"]
    assert summary["final"]["params"]["workers"] == 0  # fixed values reach the training
    metrics = summary["final"]["metrics"]
    for key in (
        "val_mAP50_95",
        "test_v2_mAP50_95",
        "test_v2_mAP50",
        "test_v2_precision",
        "test_v2_recall",
        "test_v2_all_mAP50_95",
    ):
        assert 0.0 <= metrics[key] <= 1.0
    assert metrics["test_v2_images"] == 2  # the holdout
    assert metrics["test_v2_all_images"] == 4  # the whole test split
    assert summary["final"]["latency"]["kaggle_onnx_cpu_latency_ms"] > 0
    assert summary["dataset"]["images"] == {"train": 6, "val": 3, "test": 4}
    assert summary["dataset"]["holdout"] == 2
    assert len(summary["final"]["history"]) == 3
    for name in (
        "weights/best.pt",
        "model/detector.onnx",
        "trials.csv",
        "config.yaml",
        "requirements.txt",
    ):
        assert (out / name).is_file(), name
    assert not (out / "work").exists()


def test_runner_resumes_a_finished_search(exported: SimpleNamespace) -> None:
    again = runner.run_experiment(FAKE, exported.data, exported.base / "full", log=lambda _: None)
    assert len(again["trials"]) == 3  # the Optuna study (optuna.db) was complete
    assert again["search"]["best_trial"] == exported.full["search"]["best_trial"]


def test_smoke_run_uses_a_subset(exported: SimpleNamespace) -> None:
    smoke = exported.smoke
    assert smoke["smoke"] is True
    assert len(smoke["trials"]) == 2
    assert smoke["dataset"]["images"] == {"train": 4, "val": 2, "test": 3}
    assert smoke["final"]["params"]["imgsz"] == 64
    assert not (exported.base / "smoke" / "subset").exists()


def child_runs(tracking: SimpleNamespace, parent_id: str) -> list:
    experiment = tracking.client.get_experiment_by_name(tracking.experiment)
    return tracking.client.search_runs(
        [experiment.experiment_id], f"tags.mlflow.parentRunId = '{parent_id}'"
    )


def test_smoke_import_never_reaches_the_registry(
    exported: SimpleNamespace, tracking: SimpleNamespace, tmp_path: Path
) -> None:
    result = import_kaggle.import_export(
        exported.base / "smoke",
        experiment=tracking.experiment,
        data_dir=exported.data,
        params_path=tracking.params,
        summary_dir=tmp_path,
    )
    assert result["smoke"] is True
    assert "version" not in result
    parent = tracking.client.get_run(result["run_id"])
    assert parent.info.run_name == "smoke-detector-fake"
    assert parent.data.tags["smoke"] == "true"
    assert parent.data.tags["stage"] == "compare"
    assert parent.data.metrics["onnx_cpu_latency_ms"] > 0
    children = child_runs(tracking, result["run_id"])
    assert len(children) == 2
    assert {c.data.tags["smoke"] for c in children} == {"true"}
    assert tracking.client.search_model_versions(f"name='{tracking.model}'") == []
    assert (tmp_path / "smoke-detector_fake.json").is_file()


def test_import_creates_parent_and_trial_runs_and_promotes(
    exported: SimpleNamespace, tracking: SimpleNamespace, tmp_path: Path
) -> None:
    export = exported.base / "full"
    archive = tmp_path / "detector_fake_export.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in export.rglob("*"):
            if path.is_file():
                handle.write(path, f"detector-fake/{path.relative_to(export).as_posix()}")
    folder = import_kaggle.load_export(archive, tmp_path / "unzipped")
    result = import_kaggle.import_export(
        folder,
        experiment=tracking.experiment,
        data_dir=exported.data,
        params_path=tracking.params,
        summary_dir=None,
    )
    parent = tracking.client.get_run(result["run_id"])
    tags, params, metrics = parent.data.tags, parent.data.params, parent.data.metrics
    assert parent.info.run_name == "detector-fake"
    assert tags["smoke"] == "false"
    assert tags["arch"] == "fake"
    assert tags["dataset"] == "c3pi-v2-detect"
    assert tags["git_commit"] == exported.full["code"]["git_commit"]
    assert params["search_n_trials"] == "3"
    assert params["lr0"] == str(exported.full["final"]["params"]["lr0"])
    assert metrics["test_v2_mAP50_95"] == pytest.approx(
        exported.full["final"]["metrics"]["test_v2_mAP50_95"]
    )
    history = tracking.client.get_metric_history(result["run_id"], "loss")
    assert len(history) == 3
    children = child_runs(tracking, result["run_id"])
    assert sorted(c.data.tags["trial_number"] for c in children) == ["0", "1", "2"]
    assert all("val_mAP50_95" in c.data.metrics for c in children)
    assert {c.data.tags["stage"] for c in children} == {"hpo"}
    artifacts = {a.path for a in tracking.client.list_artifacts(result["run_id"])}
    assert {"export", "onnx", "weights", "reproducibility"} <= artifacts
    # No champion yet: the first version becomes champion.
    assert result["version"] == 1
    assert result["alias"] == mu.CHAMPION
    # Same model again: equal score, not promoted (strictly better is required).
    again = import_kaggle.import_export(
        folder,
        experiment=tracking.experiment,
        data_dir=exported.data,
        params_path=tracking.params,
        summary_dir=None,
    )
    assert again["promoted"] is False
    assert again["alias"] == mu.CHALLENGER


def test_import_refuses_a_champion_not_scored_on_the_same_split(
    exported: SimpleNamespace, tracking: SimpleNamespace
) -> None:
    mlflow.set_experiment(tracking.experiment)
    with mlflow.start_run():
        mlflow.log_metric("test_mAP50_95", 0.86)  # the v1 split only
        info = mu.log_model(tiny_detector(), name="detector")
    mu.register(info.model_uri, tracking.model, alias=mu.CHAMPION)
    with pytest.raises(import_kaggle.ImportErrorKaggle, match="evaluate_champion"):
        import_kaggle.import_export(
            exported.base / "full",
            experiment=tracking.experiment,
            data_dir=exported.data,
            params_path=tracking.params,
            summary_dir=None,
        )


def test_import_promotes_over_a_weaker_reevaluated_champion(
    exported: SimpleNamespace, tracking: SimpleNamespace
) -> None:
    mlflow.set_experiment(tracking.experiment)
    with mlflow.start_run():
        mlflow.log_metrics({"test_mAP50_95": 0.86, "test_v2_mAP50_95": 0.0})
        info = mu.log_model(tiny_detector(), name="detector")
    mu.register(info.model_uri, tracking.model, alias=mu.CHAMPION)
    result = import_kaggle.import_export(
        exported.base / "full",
        experiment=tracking.experiment,
        data_dir=exported.data,
        params_path=tracking.params,
        summary_dir=None,
        measure_latency=False,
    )
    assert result["promoted"] is True
    assert (
        str(tracking.client.get_model_version_by_alias(tracking.model, "challenger").version) == "1"
    )


def test_bad_exports_are_refused(tmp_path: Path, exported: SimpleNamespace) -> None:
    with pytest.raises(import_kaggle.ImportErrorKaggle, match=r"run.json"):
        import_kaggle.load_export(tmp_path, tmp_path / "work")
    broken = tmp_path / "broken"
    shutil.copytree(exported.base / "smoke", broken)
    run = json.loads((broken / "run.json").read_text(encoding="utf-8"))
    run["schema"] = 99
    (broken / "run.json").write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(import_kaggle.ImportErrorKaggle, match="schema"):
        import_kaggle.read_run(broken)


def test_runner_reports_why_every_trial_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.models.detection import helpers

    def broken(*args: object, **kwargs: object) -> None:
        raise KeyError("no such setting")

    monkeypatch.setattr(helpers, "train", broken)
    data = make_dataset(tmp_path / "data")
    with pytest.raises(RuntimeError, match="no such setting"):
        runner.run_experiment(FAKE, data, tmp_path / "out", smoke=True, log=lambda _: None)
