"""The optimisation chain of src/models/optimize.py on a tiny classifier and a temporary MLflow.

The champion is the tiny convolutional classifier of test_quantize.py, the dataset a few 8x8
crops per class: the whole chain (variants, scores, latency in a fresh process, one run per
variant, Registry versions, alias `served`, publication) runs in seconds, without a GPU and
without the project's MLflow server.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

mlflow = pytest.importorskip("mlflow")
onnx = pytest.importorskip("onnx")

from mlflow import MlflowClient  # noqa: E402

from src.models import optimize, publish, serving  # noqa: E402
from src.models import quantize as q  # noqa: E402
from src.utils import mlflow_utils as mu  # noqa: E402
from tests.models.detection.helpers import make_dataset, tiny_detector  # noqa: E402
from tests.models.test_quantize import tiny_classifier  # noqa: E402

SETTINGS = {
    "seed": 0,
    "max_drop": 1.0,  # the tiny model is random: any variant is "accurate enough"
    "min_speedup": -10.0,  # and "fast enough": the fastest one wins
    "latency_images": 2,
    "latency_runs": 3,
    "latency_warmup": 1,
    "classifier": {"calibration_images": 4, "calibration_method": "minmax", "exclude_head": False},
    "detector": {"calibration_images": 2, "calibration_method": "minmax", "exclude_head": True},
}


def make_crops(root: Path, per_class: int = 3) -> Path:
    rng = np.random.default_rng(0)
    for split in ("val", "test"):
        for name in ("001", "002", "003", "004"):
            folder = root / split / name
            folder.mkdir(parents=True)
            for i in range(per_class):
                image = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
                cv2.imwrite(str(folder / f"{i}.png"), image)
    return root


@pytest.fixture
def crops(tmp_path: Path) -> Path:
    return make_crops(tmp_path / "classify")


@pytest.fixture
def classifier(tmp_path: Path) -> Path:
    path = tmp_path / "classifier.onnx"
    onnx.save(tiny_classifier(), str(path))
    return path


def test_spread_picks_evenly() -> None:
    assert optimize.spread(list(range(10)), 5) == [0, 2, 4, 6, 8]
    assert optimize.spread([1, 2], 5) == [1, 2]
    assert optimize.spread([1, 2, 3], 0) == [1, 2, 3]


def test_classifier_data_and_scores(crops: Path, classifier: Path) -> None:
    assert len(optimize.crop_files(crops, "test")) == 12
    assert optimize.class_names(classifier) == ["001", "002", "003", "004"]
    tensors = optimize.calibration_tensors("classify", classifier, 5, crops)
    assert len(tensors) == 5 and tensors[0].shape == (1, 3, 8, 8)
    assert tensors[0].dtype == np.float32 and float(tensors[0].max()) <= 1.0
    scores = optimize.score_classifier(classifier, crops)
    assert scores["test_images"] == 12
    assert 0.0 <= scores["test_top1"] <= scores["test_top5"] == 1.0  # 4 classes < top-5


def test_detector_scores_use_the_comparison_split(tmp_path: Path) -> None:
    data = make_dataset(tmp_path / "detect", holdout=2)
    model = tmp_path / "detector.onnx"
    onnx.save(tiny_detector(32), str(model))
    scores = optimize.score_detector(model, data)
    assert scores["test_v2_images"] == 2  # test_holdout.txt only, not the whole test split
    assert scores["test_v2_mAP50_95"] == 0.0  # the tiny detector finds nothing
    tensors = optimize.calibration_tensors("detect", model, 2, data)
    assert len(tensors) == 2 and tensors[0].shape == (1, 3, 32, 32)
    assert len(optimize.latency_files("detect", data, 5)) == 2


def test_build_variants_makes_the_four_precisions(
    crops: Path, classifier: Path, tmp_path: Path
) -> None:
    variants = optimize.build_variants("classify", classifier, tmp_path / "work", SETTINGS, crops)
    assert list(variants) == list(q.PRECISIONS)
    assert variants[q.FP32][0] == classifier
    assert variants[q.INT8_STATIC][1]["calibration_images"] == 4
    assert variants[q.INT8_STATIC][1]["calibration_split"] == "val"
    for path, _ in variants.values():
        assert optimize.class_names(path) == ["001", "002", "003", "004"]


@pytest.fixture
def tracking(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    base = tmp_path_factory.mktemp("mlflow-optimize")
    uri = "sqlite:///" + (base / "mlflow.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(base)  # nothing may land in the repository (./mlruns)
    mu.configure()
    mlflow.set_registry_uri(uri)
    mlflow.create_experiment(optimize.EXPERIMENT, artifact_location=(base / "art").as_uri())
    name = f"pill-{uuid.uuid4().hex[:8]}"
    monkeypatch.setitem(serving.REGISTERED, "classify", name)
    yield SimpleNamespace(model=name, client=MlflowClient())
    while mlflow.active_run():
        mlflow.end_run()


def test_optimise_registers_variants_and_sets_served(
    tracking: SimpleNamespace, crops: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow.set_experiment(optimize.EXPERIMENT)
    with mlflow.start_run():
        info = mu.log_model(tiny_classifier(), name="classifier")
    champion = mu.register(info.model_uri, tracking.model, alias=serving.CHAMPION).version

    report = optimize.optimise("classify", settings=SETTINGS, data=crops, client=tracking.client)

    assert report["champion_version"] == str(champion)
    assert set(report["variants"]) == set(q.PRECISIONS)
    assert report["variants"][q.FP32]["version"] is None  # FP32 is the champion itself
    for precision in (q.FP16, q.INT8_STATIC, q.INT8_DYNAMIC):
        v = report["variants"][precision]["version"]
        tags = tracking.client.get_model_version(tracking.model, v).tags
        assert tags[serving.PRECISION_TAG] == precision
        assert tags[serving.VARIANT_OF_TAG] == str(champion)
        run = tracking.client.get_run(report["variants"][precision]["run_id"])
        assert run.data.tags["stage"] == "optimize"
        assert "git_commit" in run.data.tags and "dvc_rev" in run.data.tags
        assert run.data.params["precision"] == precision
        assert {
            "test_top1",
            "onnx_cpu_latency_ms",
            "onnx_cpu_latency_p95_ms",
            "model_size_mb",
            "accuracy_drop",
        } <= set(run.data.metrics)
    # The champion alias stays on the FP32 model; `served` names the chosen precision.
    assert mu.alias_version(tracking.model, serving.CHAMPION, tracking.client) == str(champion)
    resolved = serving.served_version(tracking.model, tracking.client)
    assert resolved.precision == report["chosen"]
    assert resolved.version == (report["variants"][report["chosen"]]["version"] or str(champion))

    table = optimize.markdown_table({"classify": report})
    assert table.count("\n") == 2 + len(q.PRECISIONS)
    assert "(servie)" in table

    monkeypatch.setitem(publish.REGISTERED, "detect", tracking.model)
    out = tmp_path / "models"
    manifest = publish.publish(out, use_runs=False, versions={"detect": None, "classify": None})
    assert manifest["classify"]["precision"] == report["chosen"]
    assert json.loads((out / "models.json").read_text(encoding="utf-8")) == manifest


def test_latency_rounds_are_interleaved_and_kept_as_medians(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    timings = {"a": iter([10.0, 30.0, 20.0]), "b": iter([5.0, 7.0, 6.0])}

    def fake_latency(path: Path, files: object, *, runs: int, warmup: int) -> dict[str, float]:
        calls.append(path.name)
        value = next(timings[path.name])
        return {"onnx_cpu_latency_ms": value, "onnx_cpu_latency_p95_ms": 2 * value}

    monkeypatch.setattr(optimize, "isolated_latency", fake_latency)
    monkeypatch.setattr(optimize, "latency_files", lambda task, data, n: [])
    settings = {"latency_rounds": 3, "latency_cooldown_s": 0, "latency_runs": 2}
    out = optimize.measure_latency(
        "classify", {"fp32": tmp_path / "a", "fp16": tmp_path / "b"}, tmp_path, settings
    )
    assert calls == ["a", "b", "a", "b", "a", "b"]  # one pass over every variant per round
    assert out["fp32"]["onnx_cpu_latency_ms"] == 20.0
    assert out["fp32"]["onnx_cpu_latency_p95_ms"] == 40.0
    assert out["fp16"]["onnx_cpu_latency_ms"] == 6.0
    assert out["fp16"]["latency_rounds"] == 3.0


def test_a_new_optimisation_supersedes_the_older_variants(
    tracking: SimpleNamespace, crops: Path
) -> None:
    mlflow.set_experiment(optimize.EXPERIMENT)
    with mlflow.start_run():
        info = mu.log_model(tiny_classifier(), name="classifier")
    champion = mu.register(info.model_uri, tracking.model, alias=serving.CHAMPION).version

    first = optimize.optimise("classify", settings=SETTINGS, data=crops, client=tracking.client)
    second = optimize.optimise("classify", settings=SETTINGS, data=crops, client=tracking.client)

    def status(report: dict) -> set[str]:
        return {
            tracking.client.get_model_version(tracking.model, v["version"]).tags["status"]
            for v in report["variants"].values()
            if v["version"]
        }

    assert status(first) == {"superseded"}
    assert status(second) == {"variant"}
    assert mu.alias_version(tracking.model, serving.CHAMPION, tracking.client) == str(champion)


def test_latency_budget_reaches_every_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[float] = []

    def fake_latency(path: Path, files: object, **kwargs: float) -> dict[str, float]:
        seen.append(kwargs["budget_s"])
        return {"onnx_cpu_latency_ms": 1.0}

    monkeypatch.setattr(optimize, "isolated_latency", fake_latency)
    monkeypatch.setattr(optimize, "latency_files", lambda task, data, n: [])
    settings = {"latency_rounds": 2, "latency_cooldown_s": 0, "latency_budget_s": 1.5}
    optimize.measure_latency(
        "detect", {"a": tmp_path / "a", "b": tmp_path / "b"}, tmp_path, settings
    )
    assert seen == [1.5] * 4


def test_a_v2_champion_is_scored_on_the_comparison_split_of_step_2_3(
    tracking: SimpleNamespace, tmp_path: Path
) -> None:
    from tests.models.classification import helpers as h

    data = h.make_dataset(tmp_path / "classify_v2")
    v2 = h.tiny_classifier(metadata={"label_space": "c3pi-v2"})
    scores = optimize.score_classifier_v2(_saved(v2, tmp_path / "v2.onnx"), data)
    assert scores["test_v2_images"] == h.CLASSES  # test_holdout.txt: one crop per class
    assert scores["test_v2_all_images"] == h.CLASSES * h.PER_SPLIT["test"]
    assert scores["test_v2_v1c_images"] == len(h.V1_IDS)

    mlflow.set_experiment(optimize.EXPERIMENT)
    with mlflow.start_run():
        info = mu.log_model(v2, name="classifier")
    mu.register(
        info.model_uri, tracking.model, alias=serving.CHAMPION, tags={"label_space": "c3pi-v2"}
    )
    champion = tracking.client.get_model_version_by_alias(tracking.model, serving.CHAMPION)
    assert optimize.task_spec("classify", champion) is optimize.CLASSIFY_V2
    assert optimize.task_spec("classify") is optimize.SPECS["classify"]  # v1 model of step 2.1

    report = optimize.optimise("classify", settings=SETTINGS, data=data, client=tracking.client)
    assert report["metric"] == "test_v2_top1"
    fp32 = report["variants"][q.FP32]["metrics"]
    assert fp32["test_v2_top1"] == pytest.approx(scores["test_v2_top1"])
    run = tracking.client.get_run(report["variants"][q.FP16]["run_id"])
    assert run.data.params["test_split"] == optimize.CLASSIFY_V2.test_split


def _saved(model: onnx.ModelProto, path: Path) -> Path:
    onnx.save(model, str(path))
    return path
