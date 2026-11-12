"""Fixtures of the classifier tests: a temporary MLflow store (SQLite) and the exports of
one full run and one smoke run of the fake model family (tests/models/classification/helpers.py,
no PyTorch)."""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest
from mlflow import MlflowClient

from src.models.classification import runner
from src.utils import mlflow_utils as mu
from tests.models.classification.helpers import make_dataset

FAKE = Path(__file__).parent / "configs" / "classifier_fake.yaml"
HELPERS = "tests.models.classification.helpers"


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
    model = f"classifier-{uuid.uuid4().hex[:8]}"
    params = tmp_path / "params.yaml"
    params.write_text(
        f"registry:\n  classifier:\n    name: {model}\n    metric: test_top1\n"
        "    min_gain: 0.005\ncompare:\n  classifier:\n    metric: test_v2_top1\n"
        "    guard_metric: test_v2_v1c_top1\n    guard_max_drop: 0.0\n",
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
        patch.setitem(runner.BACKENDS, "torch", HELPERS)
        full = runner.run_experiment(FAKE, data, base / "full", log=lambda _: None)
        smoke = runner.run_experiment(FAKE, data, base / "smoke", smoke=True, log=lambda _: None)
    return SimpleNamespace(base=base, data=data, full=full, smoke=smoke)
