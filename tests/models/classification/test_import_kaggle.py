"""Import of a classifier export into MLflow (src/models/classification/import_kaggle.py) on
a temporary SQLite store: parent run, one child run per trial, Registry, guard of D100 and
promote_if_better."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

mlflow = pytest.importorskip("mlflow")
onnx = pytest.importorskip("onnx")

from src.models.classification import import_kaggle  # noqa: E402
from src.utils import mlflow_utils as mu  # noqa: E402
from tests.models.classification.helpers import tiny_classifier  # noqa: E402


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
    assert result["smoke"] is True and "version" not in result
    parent = tracking.client.get_run(result["run_id"])
    assert parent.info.run_name == "smoke-classifier-fake"
    assert parent.data.tags["smoke"] == "true" and parent.data.tags["task"] == "classify"
    assert parent.data.metrics["onnx_cpu_latency_ms"] > 0
    assert {c.data.tags["smoke"] for c in child_runs(tracking, result["run_id"])} == {"true"}
    assert tracking.client.search_model_versions(f"name='{tracking.model}'") == []
    assert (tmp_path / "smoke-classifier_fake.json").is_file()


def _register_v1(tracking: SimpleNamespace, metrics: dict[str, float]) -> str:
    """A 150-class champion of v1 (here 3 outputs), re-evaluated on the v2 split."""
    mlflow.set_experiment(tracking.experiment)
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        info = mu.log_model(tiny_classifier(classes=3), name="classifier")
    version = mu.register(
        info.model_uri,
        tracking.model,
        alias=mu.CHAMPION,
        tags={"baseline": "true", "dataset": "c3pi-v1", "task": "classify"},
    )
    return str(version.version)


def _zip(export: Path, archive: Path) -> Path:
    with zipfile.ZipFile(archive, "w") as handle:
        for path in export.rglob("*"):
            if path.is_file():
                handle.write(path, f"classifier-fake/{path.relative_to(export).as_posix()}")
    return archive


def test_import_creates_runs_registers_and_promotes_over_the_v1_champion(
    exported: SimpleNamespace, tracking: SimpleNamespace, tmp_path: Path
) -> None:
    v1 = _register_v1(tracking, {"test_top1": 0.93, "test_v2_top1": 0.3, "test_v2_v1c_top1": 0.5})
    folder = import_kaggle.load_export(
        _zip(exported.base / "full", tmp_path / "classifier_fake_export.zip"), tmp_path / "unzip"
    )
    result = import_kaggle.import_export(
        folder,
        experiment=tracking.experiment,
        data_dir=exported.data,
        params_path=tracking.params,
        summary_dir=None,
        measure_latency=False,
    )
    parent = tracking.client.get_run(result["run_id"])
    tags, params, metrics = parent.data.tags, parent.data.params, parent.data.metrics
    assert parent.info.run_name == "classifier-fake"
    assert tags["stage"] == "compare" and tags["plan_step"] == "2.3"
    assert tags["label_space"] == "c3pi-v2" and tags["git_commit"]
    assert params["num_classes"] == "6" and params["search_n_trials"] == "3"
    assert metrics["test_v2_top1"] == pytest.approx(
        exported.full["final"]["metrics"]["test_v2_top1"]
    )
    children = child_runs(tracking, result["run_id"])
    assert sorted(c.data.tags["trial_number"] for c in children) == ["0", "1", "2"]
    assert all("val_top1" in c.data.metrics for c in children)
    artifacts = {a.path for a in tracking.client.list_artifacts(result["run_id"])}
    assert {"export", "onnx", "weights", "confusion", "gradcam", "reproducibility"} <= artifacts
    assert result["eligible"] is True and result["guard_reference"] == v1
    assert result["promoted"] is True and result["alias"] == mu.CHAMPION
    assert mu.alias_version(tracking.model, mu.CHALLENGER) == v1
    version = tracking.client.get_model_version(tracking.model, str(result["version"]))
    assert version.tags["label_space"] == "c3pi-v2" and version.tags["num_classes"] == "6"


def test_guard_archives_a_model_worse_on_the_v1_classes(
    exported: SimpleNamespace, tracking: SimpleNamespace
) -> None:
    # The reference is better on the like-for-like metric than any export can be.
    _register_v1(tracking, {"test_v2_top1": 0.0, "test_v2_v1c_top1": 1.01})
    result = import_kaggle.import_export(
        exported.base / "full",
        experiment=tracking.experiment,
        data_dir=exported.data,
        params_path=tracking.params,
        summary_dir=None,
        measure_latency=False,
    )
    assert result["eligible"] is False and result["promoted"] is False
    assert result["alias"] == mu.ARCHIVED
    assert "not eligible" in result["promotion"]
    assert mu.alias_version(tracking.model, mu.CHALLENGER) is None


def test_import_refuses_a_champion_not_scored_on_the_same_split(
    exported: SimpleNamespace, tracking: SimpleNamespace
) -> None:
    _register_v1(tracking, {"test_top1": 0.93})  # the v1 split only
    with pytest.raises(import_kaggle.ImportErrorKaggle, match="evaluate_champion"):
        import_kaggle.import_export(
            exported.base / "full",
            experiment=tracking.experiment,
            data_dir=exported.data,
            params_path=tracking.params,
            summary_dir=None,
        )


def test_import_refuses_a_detector_export(exported: SimpleNamespace, tmp_path: Path) -> None:
    folder = tmp_path / "detector"
    folder.mkdir()
    run = json.loads((exported.base / "smoke" / "run.json").read_text(encoding="utf-8"))
    run["task"] = "detect"
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    with pytest.raises(import_kaggle.ImportErrorKaggle, match="not a classifier"):
        import_kaggle.import_export(folder, summary_dir=None, register=False)
