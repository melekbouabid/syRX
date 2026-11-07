"""Served variant of the champion (src/models/serving.py) and its publication (publish.py).

MLflow runs on a temporary SQLite store (never the project's server), as in
tests/utils/test_mlflow_utils.py; the models are tiny ONNX graphs tagged like the FP16 / INT8
variants of src/models/optimize.py.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

mlflow = pytest.importorskip("mlflow")
onnx = pytest.importorskip("onnx")

from mlflow import MlflowClient  # noqa: E402
from onnx import TensorProto, helper  # noqa: E402

from src.models import publish, serving  # noqa: E402
from src.utils import mlflow_utils as mu  # noqa: E402


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    base = tmp_path_factory.mktemp("mlflow-serving")
    return SimpleNamespace(
        uri="sqlite:///" + (base / "mlflow.db").as_posix(),
        artifacts=(base / "artifacts").as_uri(),
    )


@pytest.fixture
def tracking(store: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", store.uri)
    monkeypatch.chdir(tmp_path)  # nothing may land in the repository (./mlruns)
    mu.configure()
    mlflow.set_registry_uri(store.uri)
    experiment = f"test-{uuid.uuid4().hex[:8]}"
    mlflow.create_experiment(experiment, artifact_location=f"{store.artifacts}/{experiment}")
    mlflow.set_experiment(experiment)
    yield SimpleNamespace(model=f"pill-{uuid.uuid4().hex[:8]}", client=MlflowClient())
    while mlflow.active_run():
        mlflow.end_run()


def tiny_onnx(op: str = "Identity") -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph([helper.make_node(op, ["x"], ["y"])], "g", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.metadata_props.add(key="op", value=op)
    return model


def version(tracking: SimpleNamespace, op: str = "Identity", **tags: str) -> str:
    with mlflow.start_run():
        info = mu.log_model(tiny_onnx(op), name="model")
    return str(mu.register(info.model_uri, tracking.model, tags=tags or None).version)


def champion(tracking: SimpleNamespace, v: str) -> None:
    tracking.client.set_registered_model_alias(tracking.model, serving.CHAMPION, v)


def variant(tracking: SimpleNamespace, of: str, precision: str = "int8-static") -> str:
    return version(tracking, "Relu", precision=precision, variant_of=of)


def test_without_served_alias_the_champion_is_served(tracking: SimpleNamespace) -> None:
    v1 = version(tracking)
    champion(tracking, v1)
    resolved = serving.served_version(tracking.model, tracking.client)
    assert (resolved.version, resolved.precision, resolved.variant_of) == (v1, "fp32", None)


def test_served_variant_of_the_champion(tracking: SimpleNamespace) -> None:
    v1 = version(tracking)
    champion(tracking, v1)
    v2 = variant(tracking, v1)
    serving.set_served(tracking.model, v2, v1, "int8-static", tracking.client)
    resolved = serving.served_version(tracking.model, tracking.client)
    assert (resolved.version, resolved.precision, resolved.variant_of) == (v2, "int8-static", v1)
    tags = tracking.client.get_model_version(tracking.model, v1).tags
    assert tags[serving.SERVED_PRECISION_TAG] == "int8-static"
    # The champion alias never moves to the quantized copy (promote_if_better compares FP32).
    assert mu.alias_version(tracking.model, serving.CHAMPION, tracking.client) == v1


def test_served_equal_to_champion_is_fp32(tracking: SimpleNamespace) -> None:
    v1 = version(tracking)
    champion(tracking, v1)
    serving.set_served(tracking.model, v1, v1, "fp32", tracking.client)
    assert serving.served_version(tracking.model, tracking.client).version == v1


def test_variant_of_a_former_champion_is_not_served(tracking: SimpleNamespace) -> None:
    v1 = version(tracking)
    champion(tracking, v1)
    v2 = variant(tracking, v1)
    serving.set_served(tracking.model, v2, v1, "int8-static", tracking.client)
    v3 = version(tracking)  # a new FP32 champion, not optimised yet
    champion(tracking, v3)
    resolved = serving.served_version(tracking.model, tracking.client)
    assert (resolved.version, resolved.precision) == (v3, "fp32")
    assert "optimize" in resolved.note


def test_no_champion_raises(tracking: SimpleNamespace) -> None:
    version(tracking)
    with pytest.raises(LookupError):
        serving.served_version(tracking.model, tracking.client)


def test_publish_copies_the_served_variant(
    tracking: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    v1 = version(tracking)
    champion(tracking, v1)
    v2 = variant(tracking, v1)
    serving.set_served(tracking.model, v2, v1, "int8-static", tracking.client)
    monkeypatch.setitem(publish.REGISTERED, "detect", tracking.model)
    monkeypatch.setitem(publish.REGISTERED, "classify", tracking.model)

    out = tmp_path / "models"
    manifest = publish.publish(out, use_runs=False, versions={"detect": None, "classify": None})
    saved = json.loads((out / "models.json").read_text(encoding="utf-8"))
    assert saved == manifest
    for task, filename in publish.FILES.items():
        entry = manifest[task]
        assert (entry["version"], entry["precision"], entry["variant_of"]) == (
            int(v2),
            "int8-static",
            int(v1),
        )
        model = onnx.load(str(out / filename))
        assert model.graph.node[0].op_type == "Relu"
        assert entry["sha256"] == publish.sha256(out / filename)

    exact = publish.publish(
        tmp_path / "exact", use_runs=False, versions={"detect": None, "classify": None}, exact=True
    )
    assert exact["detect"]["version"] == int(v1)
    assert exact["detect"]["precision"] == "fp32"
    pinned = publish.publish(
        tmp_path / "pinned", use_runs=False, versions={"detect": int(v1), "classify": int(v2)}
    )
    assert (pinned["detect"]["version"], pinned["classify"]["version"]) == (int(v1), int(v2))


def test_manifest_carries_architecture_and_metrics(
    tracking: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with mlflow.start_run():
        mlflow.log_metrics({"val_mAP50_95": 0.85, "test_v2_mAP50_95": 0.87, "train_box_loss": 1.0})
        mlflow.log_metric("val_box_loss", 0.5)
        info = mu.log_model(tiny_onnx("Identity"), name="model")
    v1 = str(mu.register(info.model_uri, tracking.model, tags={"arch": "yolov8n"}).version)
    champion(tracking, v1)
    with mlflow.start_run():
        mlflow.log_metrics({"test_v2_mAP50_95": 0.86, "onnx_cpu_latency_ms": 20.0, "time": 3.0})
        info = mu.log_model(tiny_onnx("Relu"), name="model")
    tags = {"arch": "yolov8n", "precision": "fp16", "variant_of": v1}
    v2 = str(mu.register(info.model_uri, tracking.model, tags=tags).version)
    serving.set_served(tracking.model, v2, v1, "fp16", tracking.client)
    monkeypatch.setitem(publish.REGISTERED, "detect", tracking.model)
    monkeypatch.setitem(publish.REGISTERED, "classify", tracking.model)

    manifest = publish.publish(
        tmp_path / "models", use_runs=False, versions={"detect": None, "classify": None}
    )

    entry = manifest["detect"]
    assert entry["arch"] == "yolov8n"
    # The variant's own scores, completed by the validation scores of its champion.
    assert entry["metrics"] == {
        "onnx_cpu_latency_ms": 20.0,
        "test_v2_mAP50_95": 0.86,
        "val_mAP50_95": 0.85,
    }
