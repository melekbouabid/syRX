"""MLflow helpers (src/utils/mlflow_utils.py) on a temporary local tracking store.

The store is a SQLite database in a temporary folder (the backend MLflow recommends; the
file store is in maintenance mode), never the project's server. Models are tiny ONNX graphs.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

mlflow = pytest.importorskip("mlflow")
onnx = pytest.importorskip("onnx")

from mlflow import MlflowClient  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

from src.utils import mlflow_utils as mu  # noqa: E402

LOCK = """schema: '2.0'
stages:
  prepare:
    cmd: python -m src.data.prepare
    outs:
    - path: data/processed/c3pi/datasets
      hash: md5
      md5: 5cb2cd2c3526eb63d114a5dbcdb4f032.dir
"""


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(  # noqa: S603
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", *args],  # noqa: S607
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A git repository with a dvc.lock and a params.yaml, all committed."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    root = tmp_path / "repo"
    root.mkdir()
    (root / "dvc.lock").write_text(LOCK, encoding="utf-8")
    (root / "params.yaml").write_text("train:\n  seed: 0\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "chore: initial")
    return root


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    base = tmp_path_factory.mktemp("mlflow")
    return SimpleNamespace(
        uri="sqlite:///" + (base / "mlflow.db").as_posix(),
        artifacts=(base / "artifacts").as_uri(),
    )


@pytest.fixture
def tracking(store: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """MLflow pointed at the temporary store, a fresh experiment and model name per test."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store.uri)
    monkeypatch.chdir(tmp_path)  # nothing may land in the repository (./mlruns)
    assert mu.configure() == store.uri
    mlflow.set_registry_uri(store.uri)
    experiment = f"test-{uuid.uuid4().hex[:8]}"
    mlflow.create_experiment(experiment, artifact_location=f"{store.artifacts}/{experiment}")
    yield SimpleNamespace(
        experiment=experiment, model=f"pill-{uuid.uuid4().hex[:8]}", client=MlflowClient()
    )
    while mlflow.active_run():
        mlflow.end_run()


def tiny_onnx() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def new_version(tracking: SimpleNamespace, metrics: dict[str, float]) -> str:
    """A run with `metrics`, its ONNX model logged and registered: the new version number."""
    mlflow.set_experiment(tracking.experiment)
    with mlflow.start_run():
        mlflow.log_metrics(metrics)
        info = mu.log_model(tiny_onnx(), name="model")
    return str(mu.register(info.model_uri, tracking.model).version)


# --- provenance ---------------------------------------------------------------------------


def test_provenance_tags_identify_commit_and_data_version(repo: Path) -> None:
    tags = mu.provenance_tags("c3pi-v1", 7, root=repo)
    assert tags["git_commit"] == _git(repo, "rev-parse", "HEAD")
    assert tags["dvc_rev"] == _git(repo, "hash-object", "dvc.lock")
    assert tags["dataset"] == "c3pi-v1"
    assert tags["seed"] == "7"
    assert "git_dirty" not in tags

    (repo / "params.yaml").write_text("train:\n  seed: 1\n", encoding="utf-8")
    assert mu.provenance_tags("c3pi-v1", 7, root=repo)["git_dirty"] == "true"


def test_dvc_rev_outside_git_is_the_git_blob_id(tmp_path: Path) -> None:
    assert mu.dvc_rev(tmp_path) == "none"
    content = LOCK.encode()
    (tmp_path / "dvc.lock").write_bytes(content)
    blob = hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()
    rev = mu.dvc_rev(tmp_path)
    # Inside a git work tree `git hash-object` gives the same id; outside, the fallback does.
    assert rev == blob


def test_dataset_md5_reads_dvc_lock(repo: Path) -> None:
    md5 = "5cb2cd2c3526eb63d114a5dbcdb4f032.dir"
    assert mu.dataset_md5("data/processed/c3pi/datasets", repo) == md5
    assert mu.dataset_md5("data/processed/c3pi/datasets/", repo) == md5
    assert mu.dataset_md5("data/elsewhere", repo) is None


def test_frozen_requirements_pins_the_environment() -> None:
    lines = mu.frozen_requirements().splitlines()
    assert f"pytest=={pytest.__version__}" in lines
    assert any(line.lower().startswith("mlflow") and "==" in line for line in lines)
    names = [line.split("==")[0].lower().replace("_", "-") for line in lines]
    assert names == sorted(names)
    assert len(names) == len(set(names))


# --- tracked runs and models --------------------------------------------------------------


def test_start_tracked_run_logs_tags_params_and_artefacts(
    tracking: SimpleNamespace, repo: Path
) -> None:
    with mu.start_tracked_run(
        tracking.experiment,
        "baseline",
        seed=3,
        dataset="c3pi-v1",
        dataset_path="data/processed/c3pi/datasets",
        params={"epochs": 40},
        tags={"task": "classify"},
        root=repo,
    ) as run:
        run_id = run.info.run_id
    data = tracking.client.get_run(run_id).data
    assert data.tags["git_commit"] == _git(repo, "rev-parse", "HEAD")
    assert data.tags["dvc_rev"] == _git(repo, "hash-object", "dvc.lock")
    assert data.tags["dataset"] == "c3pi-v1"
    assert data.tags["dataset_md5"] == "5cb2cd2c3526eb63d114a5dbcdb4f032.dir"
    assert data.tags["seed"] == "3"
    assert data.tags["task"] == "classify"
    assert data.params == {"seed": "3", "epochs": "40"}
    files = {a.path for a in tracking.client.list_artifacts(run_id, "reproducibility")}
    assert files == {
        "reproducibility/requirements.txt",
        "reproducibility/dvc.lock",
        "reproducibility/params.yaml",
    }
    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="reproducibility/requirements.txt"
    )
    assert f"pytest=={pytest.__version__}" in Path(local).read_text(encoding="utf-8")


def test_log_model_and_register_with_alias(tracking: SimpleNamespace) -> None:
    mlflow.set_experiment(tracking.experiment)
    with mlflow.start_run() as run:
        info = mu.log_model(tiny_onnx(), name="detector", metrics={"test_mAP50_95": 0.86})
    logged = mlflow.get_logged_model(info.model_id)
    assert logged.name == "detector"
    assert {m.key: m.value for m in logged.metrics} == {"test_mAP50_95": 0.86}
    assert tracking.client.get_run(run.info.run_id).data.metrics["test_mAP50_95"] == 0.86

    version = mu.register(
        info.model_uri,
        tracking.model,
        alias=mu.CHAMPION,
        tags={"task": "detect"},
        description="baseline",
    )
    assert str(version.version) == "1"
    assert version.run_id == run.info.run_id
    assert version.tags["task"] == "detect"
    assert version.description == "baseline"
    assert mu.alias_version(tracking.model, mu.CHAMPION) == "1"
    assert mu.alias_version(tracking.model, mu.CHALLENGER) is None
    assert mu.version_metric(tracking.model, 1, "test_mAP50_95") == 0.86


# --- promote_if_better --------------------------------------------------------------------


def _status(tracking: SimpleNamespace, version: str) -> str | None:
    return tracking.client.get_model_version(tracking.model, version).tags.get(mu.STATUS_TAG)


def test_first_version_becomes_champion(tracking: SimpleNamespace) -> None:
    v1 = new_version(tracking, {"top1": 0.90})
    result = mu.promote_if_better(tracking.model, v1, "top1", min_gain=0.01)
    assert result.promoted
    assert result.status == mu.CHAMPION
    assert result.champion_before is None
    assert mu.alias_version(tracking.model, mu.CHAMPION) == v1
    assert _status(tracking, v1) == mu.CHAMPION
    assert (
        "first champion" in tracking.client.get_model_version(tracking.model, v1).tags["promotion"]
    )


def test_better_version_is_promoted_and_old_champion_becomes_challenger(
    tracking: SimpleNamespace,
) -> None:
    v1 = new_version(tracking, {"top1": 0.90})
    mu.promote_if_better(tracking.model, v1, "top1")
    v2 = new_version(tracking, {"top1": 0.95})
    result = mu.promote_if_better(tracking.model, v2, "top1", min_gain=0.01)
    assert result.promoted
    assert (result.champion_before, result.champion_value) == (v1, 0.90)
    assert result.value == 0.95
    assert mu.alias_version(tracking.model, mu.CHAMPION) == v2
    assert mu.alias_version(tracking.model, mu.CHALLENGER) == v1
    assert _status(tracking, v2) == mu.CHAMPION
    assert _status(tracking, v1) == mu.CHALLENGER
    tags = tracking.client.get_model_version(tracking.model, v2).tags
    assert tags["promoted_from"] == v1
    assert tags["promotion"].startswith("promoted")

    # A third, even better version: v2 becomes challenger, v1 loses every alias.
    v3 = new_version(tracking, {"top1": 0.97})
    assert mu.promote_if_better(tracking.model, v3, "top1", min_gain=0.01).promoted
    assert mu.alias_version(tracking.model, mu.CHAMPION) == v3
    assert mu.alias_version(tracking.model, mu.CHALLENGER) == v2
    assert _status(tracking, v1) == mu.ARCHIVED


def test_worse_version_is_not_promoted(tracking: SimpleNamespace) -> None:
    v1 = new_version(tracking, {"top1": 0.90})
    mu.promote_if_better(tracking.model, v1, "top1")
    v2 = new_version(tracking, {"top1": 0.80})
    result = mu.promote_if_better(tracking.model, v2, "top1", min_gain=0.0)
    assert not result.promoted
    assert result.status == mu.CHALLENGER  # no challenger yet: the runner-up
    assert mu.alias_version(tracking.model, mu.CHAMPION) == v1
    assert mu.alias_version(tracking.model, mu.CHALLENGER) == v2
    assert (
        tracking.client.get_model_version(tracking.model, v2)
        .tags["promotion"]
        .startswith("not promoted")
    )

    # Worse than both the champion and the challenger: archived, aliases unchanged.
    v3 = new_version(tracking, {"top1": 0.70})
    result = mu.promote_if_better(tracking.model, v3, "top1")
    assert not result.promoted
    assert result.status == mu.ARCHIVED
    assert mu.alias_version(tracking.model, mu.CHAMPION) == v1
    assert mu.alias_version(tracking.model, mu.CHALLENGER) == v2
    assert _status(tracking, v3) == mu.ARCHIVED


def test_gain_below_min_gain_is_not_promoted(tracking: SimpleNamespace) -> None:
    v1 = new_version(tracking, {"top1": 0.900})
    mu.promote_if_better(tracking.model, v1, "top1")
    v2 = new_version(tracking, {"top1": 0.903})  # better, but by less than min_gain
    result = mu.promote_if_better(tracking.model, v2, "top1", min_gain=0.005)
    assert not result.promoted
    assert mu.alias_version(tracking.model, mu.CHAMPION) == v1
    # Equal score: not strictly better, never promoted even with min_gain = 0.
    v3 = new_version(tracking, {"top1": 0.900})
    assert not mu.promote_if_better(tracking.model, v3, "top1", min_gain=0.0).promoted
    assert mu.alias_version(tracking.model, mu.CHAMPION) == v1


def test_lower_is_better_metric(tracking: SimpleNamespace) -> None:
    v1 = new_version(tracking, {"latency_ms": 30.0})
    mu.promote_if_better(tracking.model, v1, "latency_ms", greater_is_better=False)
    v2 = new_version(tracking, {"latency_ms": 20.0})
    result = mu.promote_if_better(
        tracking.model, v2, "latency_ms", min_gain=5.0, greater_is_better=False
    )
    assert result.promoted
    assert mu.alias_version(tracking.model, mu.CHAMPION) == v2


def test_champion_promoted_again_is_a_no_op(tracking: SimpleNamespace) -> None:
    v1 = new_version(tracking, {"top1": 0.9})
    mu.promote_if_better(tracking.model, v1, "top1")
    result = mu.promote_if_better(tracking.model, v1, "top1", min_gain=0.5)
    assert result.promoted
    assert result.reason == "already champion"
    assert mu.alias_version(tracking.model, mu.CHAMPION) == v1


def test_missing_metric_raises(tracking: SimpleNamespace) -> None:
    v1 = new_version(tracking, {"top1": 0.9})
    with pytest.raises(ValueError, match="no metric 'top5'"):
        mu.promote_if_better(tracking.model, v1, "top5")
    assert mu.alias_version(tracking.model, mu.CHAMPION) is None


# Scores of step 2.2 on the comparison split (test_v2_mAP50_95, docs/decisions.md D98).
STEP_2_2 = {"yolov8s": 0.8830, "yolov10n": 0.8860, "yolo11s": 0.8885, "fasterrcnn": 0.8218}


def _step_2_2_registry(tracking: SimpleNamespace) -> dict[str, str]:
    """Baseline (challenger) and YOLOv8n (champion in place) as before the four imports."""
    baseline = new_version(tracking, {"map": 0.7888})
    mu.promote_if_better(tracking.model, baseline, "map")
    incumbent = new_version(tracking, {"map": 0.8774})
    mu.promote_if_better(tracking.model, incumbent, "map", min_gain=0.005)
    return {"yolo11n": baseline, "yolov8n": incumbent}


def _import(tracking: SimpleNamespace, order: list[str]) -> tuple[str, str]:
    versions = _step_2_2_registry(tracking)
    for arch in order:
        versions[arch] = new_version(tracking, {"map": STEP_2_2[arch]})
        mu.promote_if_better(tracking.model, versions[arch], "map", min_gain=0.005)
    by_version = {v: arch for arch, v in versions.items()}
    return (
        by_version[mu.alias_version(tracking.model, mu.CHAMPION) or ""],
        by_version[mu.alias_version(tracking.model, mu.CHALLENGER) or ""],
    )


def test_best_first_imports_give_the_rule_of_d98(tracking: SimpleNamespace) -> None:
    # Best first (import_kaggle.promotion_order): the champion is the best candidate that
    # beats the champion in place by min_gain, the challenger the second best model.
    best_first = sorted(STEP_2_2, key=lambda arch: -STEP_2_2[arch])
    assert _import(tracking, best_first) == ("yolo11s", "yolov10n")


def test_sequential_promotion_depends_on_the_import_order(tracking: SimpleNamespace) -> None:
    # Why the imports are sorted: YOLOv10n imported before YOLO11s is promoted, and YOLO11s
    # (+0.0025 over it) then misses min_gain.
    assert _import(tracking, ["yolov10n", "yolo11s", "yolov8s", "fasterrcnn"]) == (
        "yolov10n",
        "yolo11s",
    )
