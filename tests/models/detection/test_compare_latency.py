"""Interleaved latency of the compared detectors (src/models/detection/compare_latency.py)
and the MLflow query of the comparison table (src/models/detection/compare.py `collect`).

Tiny ONNX detectors registered in a temporary SQLite MLflow store; the timing itself is the
real one of src/models/optimize.py (fresh process per round) on a synthetic dataset.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

mlflow = pytest.importorskip("mlflow")
onnx = pytest.importorskip("onnx")

from mlflow import MlflowClient  # noqa: E402

from src.models import optimize  # noqa: E402
from src.models.detection import compare, compare_latency  # noqa: E402
from src.utils import mlflow_utils as mu  # noqa: E402
from tests.models.detection.helpers import make_dataset, tiny_detector  # noqa: E402

SETTINGS = {
    "latency_images": 2,
    "latency_runs": 2,
    "latency_warmup": 1,
    "latency_rounds": 2,
    "latency_cooldown_s": 0,
}


@pytest.fixture
def tracking(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    base = tmp_path_factory.mktemp("mlflow-latency")
    uri = "sqlite:///" + (base / "mlflow.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(base)  # nothing may land in the repository (./mlruns)
    mu.configure()
    mlflow.set_registry_uri(uri)
    experiment = f"test-{uuid.uuid4().hex[:8]}"
    mlflow.create_experiment(experiment, artifact_location=(base / "art").as_uri())
    yield SimpleNamespace(
        experiment=experiment, model=f"detector-{uuid.uuid4().hex[:8]}", client=MlflowClient()
    )
    while mlflow.active_run():
        mlflow.end_run()


def _version(tracking: SimpleNamespace, arch: str, size: int, tags: dict[str, str]) -> str:
    mlflow.set_experiment(tracking.experiment)
    with mlflow.start_run(run_name=f"detector-{arch}") as run:
        mlflow.log_metric("onnx_cpu_latency_ms", 999.0)  # an older single timing
        info = mu.log_model(tiny_detector(size), name="detector")
    version = mu.register(info.model_uri, tracking.model, tags={"arch": arch, **tags})
    assert version.run_id == run.info.run_id
    return str(version.version)


def test_onnx_parameters_counts_the_float_weights(tmp_path: Path) -> None:
    path = tmp_path / "tiny.onnx"
    onnx.save(tiny_detector(32, boxes=4), str(path))
    assert compare_latency.onnx_parameters(path) == 1 * 5 * 4 + 1  # raw boxes + scalar zero


def test_every_fp32_detector_is_timed_together_and_logged(
    tracking: SimpleNamespace, tmp_path: Path
) -> None:
    data = make_dataset(tmp_path / "detect", holdout=2)
    baseline = _version(tracking, "base", 32, {})
    candidate = _version(tracking, "cand", 64, {})
    variant = _version(tracking, "cand", 64, {"precision": "fp16", "variant_of": candidate})
    old = _version(tracking, "old", 32, {"status": "superseded"})
    baseline_run = tracking.client.get_model_version(tracking.model, baseline).run_id
    mlflow.set_experiment(tracking.experiment)
    with mlflow.start_run(
        run_name="evaluate-detector-v1-c3pi-v2",
        tags={"stage": "compare-eval", "evaluated_run_id": baseline_run},
    ) as evaluation:
        mlflow.log_metric("test_v2_mAP50_95", 0.5)

    summary = compare_latency.measure_all(
        name=tracking.model,
        experiment=tracking.experiment,
        data=data,
        settings=SETTINGS,
        client=tracking.client,
    )

    assert set(summary["models"]) == {f"v{baseline}", f"v{candidate}"}  # no variant, no old
    assert f"v{variant}" not in summary["models"] and f"v{old}" not in summary["models"]
    assert summary["order"] == [f"v{baseline}", f"v{candidate}"]  # smallest file first
    assert "2 rounds x 2 passes" in summary["protocol"]
    row = summary["models"][f"v{baseline}"]
    assert row["targets"] == [baseline_run, evaluation.info.run_id]
    assert len(row["round_means_ms"]) == 2
    for run_id in row["targets"]:
        run = tracking.client.get_run(run_id)
        latency = run.data.metrics["onnx_cpu_latency_ms"]
        assert latency == pytest.approx(row["latency"]["onnx_cpu_latency_ms"])
        assert latency != 999.0  # the interleaved figure is now the latest value
        assert run.data.metrics["latency_rounds"] == 2.0
        assert run.data.metrics["model_size_mb"] > 0
        assert run.data.metrics["onnx_parameters"] == 21.0
        assert run.data.tags["latency_protocol"].startswith("interleaved rounds")
        history = tracking.client.get_metric_history(run_id, "onnx_cpu_latency_round_ms")
        assert sorted(m.step for m in history) == [0, 1]


def test_dry_run_logs_nothing(
    tracking: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = make_dataset(tmp_path / "detect", holdout=2)
    version = _version(tracking, "base", 32, {})
    calls: list[list[str]] = []

    def fake(task: str, paths: dict[str, Path], data: Path, settings: dict, rounds_out: dict):
        calls.append(list(paths))
        rounds_out.update({key: [{"onnx_cpu_latency_ms": 5.0}] for key in paths})
        return {key: {"onnx_cpu_latency_ms": 5.0, "onnx_cpu_latency_p95_ms": 6.0} for key in paths}

    monkeypatch.setattr(optimize, "measure_latency", fake)
    summary = compare_latency.measure_all(
        name=tracking.model,
        experiment=tracking.experiment,
        data=data,
        settings=SETTINGS,
        log=False,
        client=tracking.client,
    )
    assert calls == [[f"v{version}"]]
    assert summary["models"][f"v{version}"]["round_means_ms"] == [5.0]
    run_id = tracking.client.get_model_version(tracking.model, version).run_id
    assert tracking.client.get_run(run_id).data.metrics["onnx_cpu_latency_ms"] == 999.0


def test_collect_reads_candidates_and_baseline_without_smoke_runs(
    tracking: SimpleNamespace,
) -> None:
    # The real server refuses `tags.stage IN (...)`: the SQLite store runs the same query.
    baseline = _version(tracking, "yolo11n", 32, {"status": "archived"})
    champion = _version(tracking, "yolo11s", 32, {"status": "champion"})
    _version(tracking, "yolo11s", 32, {"precision": "fp16", "status": "variant"})
    baseline_run = tracking.client.get_model_version(tracking.model, baseline).run_id
    champion_run = tracking.client.get_model_version(tracking.model, champion).run_id
    tracking.client.set_tag(champion_run, "stage", "compare")
    tracking.client.set_tag(champion_run, "smoke", "false")
    tracking.client.set_tag(champion_run, "arch", "yolo11s")
    tracking.client.log_metric(champion_run, "test_v2_mAP50_95", 0.8885)
    tracking.client.log_metric(champion_run, "model_size_mb", 37.9)
    # Final validation score at step 0, then a per-epoch history under the same key.
    tracking.client.log_metric(champion_run, "val_mAP50_95", 0.8063, step=0)
    for epoch, value in ((1, 0.70), (2, 0.8042)):
        tracking.client.log_metric(champion_run, "val_mAP50_95", value, step=epoch)
    mlflow.set_experiment(tracking.experiment)
    with mlflow.start_run(
        tags={
            "stage": "compare-eval",
            "smoke": "false",
            "arch": "yolo11n",
            "evaluated_run_id": baseline_run,
        }
    ):
        mlflow.log_metric("test_v2_mAP50_95", 0.7888)
    with mlflow.start_run(tags={"stage": "compare", "smoke": "true", "arch": "smoke"}):
        mlflow.log_metric("test_v2_mAP50_95", 0.1)
    with mlflow.start_run(tags={"stage": "optimize", "smoke": "false", "arch": "other"}):
        mlflow.log_metric("test_v2_mAP50_95", 0.9)

    rows = compare.collect(tracking.experiment, tracking.model, tracking.client)

    assert [r.arch for r in rows] == ["yolo11s", "yolo11n (baseline v1)"]
    assert (rows[0].status, rows[0].version, rows[0].size_mb) == ("champion", champion, 37.9)
    assert rows[0].val_map == 0.8063  # the retained checkpoint, not the last epoch
    assert (rows[1].status, rows[1].version) == ("archived", baseline)


def test_classifiers_are_timed_with_their_task(
    tracking: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = make_dataset(tmp_path / "detect", holdout=2)
    _version(tracking, "cls", 32, {})
    tasks: list[str] = []

    def fake(task: str, paths: dict[str, Path], data: Path, settings: dict, rounds_out: dict):
        tasks.append(task)
        rounds_out.update({key: [{"onnx_cpu_latency_ms": 3.0}] for key in paths})
        return {key: {"onnx_cpu_latency_ms": 3.0, "onnx_cpu_latency_p95_ms": 4.0} for key in paths}

    monkeypatch.setattr(optimize, "measure_latency", fake)
    compare_latency.measure_all(
        task="classify",
        name=tracking.model,
        experiment=tracking.experiment,
        data=data,
        settings=SETTINGS,
        log=False,
        client=tracking.client,
    )
    assert tasks == ["classify"]  # the crops of the classifier, not the detector's photos
