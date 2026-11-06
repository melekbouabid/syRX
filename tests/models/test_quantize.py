"""FP16 / INT8 variants of an ONNX model and the choice of the served one (src/models/quantize.py).

The model is a tiny convolutional classifier built with onnx.helper (input 1x3x8x8, output
1x4 probabilities, Ultralytics-like metadata): conversions run in a fraction of a second.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from src.models import quantize as q

SIZE = 8


def tiny_classifier(seed: int = 0) -> onnx.ModelProto:
    rng = np.random.default_rng(seed)
    weight = rng.normal(0, 0.5, (4, 3, 3, 3)).astype(np.float32)
    bias = rng.normal(0, 0.1, (4,)).astype(np.float32)
    nodes = [
        helper.make_node(
            "Conv", ["images", "w", "b"], ["conv"], pads=[1, 1, 1, 1], name="/model.0/conv/Conv"
        ),
        helper.make_node("Relu", ["conv"], ["act"], name="/model.0/act/Relu"),
        helper.make_node(
            "GlobalAveragePool", ["act"], ["pool"], name="/model.1/pool/GlobalAveragePool"
        ),
        helper.make_node("Flatten", ["pool"], ["flat"], name="/model.1/Flatten"),
        helper.make_node("Softmax", ["flat"], ["output0"], axis=1, name="/model.1/Softmax"),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 4])],
        [numpy_helper.from_array(weight, "w"), numpy_helper.from_array(bias, "b")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    for key, value in {
        "names": "{0: '001', 1: '002', 2: '003', 3: '004'}",
        "imgsz": "[8, 8]",
    }.items():
        model.metadata_props.add(key=key, value=value)
    return model


@pytest.fixture
def fp32(tmp_path: Path) -> Path:
    path = tmp_path / "fp32.onnx"
    onnx.save(tiny_classifier(), str(path))
    return path


def tensors(n: int, seed: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.random((1, 3, SIZE, SIZE), dtype=np.float32) for _ in range(n)]


def run(path: Path, x: np.ndarray) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out: np.ndarray = session.run(None, {"images": x})[0]
    return out


def metadata(path: Path) -> dict[str, str]:
    return {p.key: p.value for p in onnx.load(str(path)).metadata_props}


def op_types(path: Path) -> set[str]:
    return {node.op_type for node in onnx.load(str(path)).graph.node}


def test_fp16_keeps_float32_inputs_outputs_and_metadata(fp32: Path, tmp_path: Path) -> None:
    fp16 = q.to_fp16(fp32, tmp_path / "out" / "fp16.onnx")
    model = onnx.load(str(fp16))
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert any(i.data_type == TensorProto.FLOAT16 for i in model.graph.initializer)
    assert metadata(fp16) == metadata(fp32)
    x = tensors(1)[0]
    out = run(fp16, x)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, run(fp32, x), atol=1e-2)


def test_int8_static_is_calibrated_qdq_close_to_fp32(fp32: Path, tmp_path: Path) -> None:
    int8 = q.to_int8_static(fp32, tmp_path / "int8.onnx", tensors(16))
    assert {"QuantizeLinear", "DequantizeLinear"} <= op_types(int8)
    assert metadata(int8)["names"] == metadata(fp32)["names"]
    for x in tensors(4, seed=7):
        np.testing.assert_allclose(run(int8, x), run(fp32, x), atol=0.05)
    assert q.size_mb(int8) > 0


def test_int8_static_needs_calibration_images(fp32: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="calibration"):
        q.to_int8_static(fp32, tmp_path / "int8.onnx", [])


def test_int8_static_can_leave_nodes_in_float(fp32: Path, tmp_path: Path) -> None:
    everything = q.head_nodes(fp32, "/model.")
    int8 = q.to_int8_static(fp32, tmp_path / "int8.onnx", tensors(8), exclude_nodes=everything)
    assert "QuantizeLinear" not in op_types(int8)


def test_int8_dynamic_runs_without_calibration(fp32: Path, tmp_path: Path) -> None:
    int8 = q.to_int8_dynamic(fp32, tmp_path / "dyn.onnx")
    assert metadata(fp32).items() <= metadata(int8).items()  # plus ONNX Runtime markers
    x = tensors(1)[0]
    np.testing.assert_allclose(run(int8, x), run(fp32, x), atol=0.05)


def test_calibration_reader_rewinds() -> None:
    reader = q.CalibrationReader("images", tensors(3))
    first = [reader.get_next() for _ in range(4)]
    assert first[-1] is None and len(reader) == 3
    reader.rewind()
    again, initial = reader.get_next(), first[0]
    assert again is not None and initial is not None
    assert np.array_equal(again["images"], initial["images"])


def test_output_module_prefix_and_head_nodes(fp32: Path) -> None:
    assert q.output_module_prefix(fp32) == "/model.1/"
    assert q.head_nodes(fp32, "/model.1/") == [
        "/model.1/pool/GlobalAveragePool",
        "/model.1/Flatten",
        "/model.1/Softmax",
    ]
    assert q.head_nodes(fp32, "") == []


def r(precision: str, accuracy: float, latency: float, size: float = 10.0) -> q.VariantResult:
    return q.VariantResult(precision, accuracy, latency, latency * 1.2, size)


def test_choose_the_fastest_variant_within_the_accuracy_budget() -> None:
    results = [
        r(q.FP32, 0.90, 20.0),
        r(q.FP16, 0.90, 25.0),  # slower on CPU
        r(q.INT8_STATIC, 0.85, 8.0),  # too inaccurate
        r(q.INT8_DYNAMIC, 0.895, 12.0),
    ]
    choice = q.choose_variant(results, max_drop=0.01, min_speedup=0.05)
    assert choice.precision == q.INT8_DYNAMIC
    assert "int8-dynamic" in choice.reason and "40% faster" in choice.reason


def test_fp32_is_kept_without_an_admissible_variant() -> None:
    results = [r(q.FP32, 0.90, 20.0), r(q.FP16, 0.90, 19.5), r(q.INT8_STATIC, 0.70, 5.0)]
    assert q.choose_variant(results, max_drop=0.01, min_speedup=0.05).precision == q.FP32


def test_equal_latency_prefers_the_smaller_file() -> None:
    results = [
        r(q.FP32, 0.9, 20.0, 12.0),
        r(q.INT8_STATIC, 0.9, 10.0, 3.0),
        r(q.INT8_DYNAMIC, 0.9, 10.0, 3.5),
    ]
    assert q.choose_variant(results, max_drop=0.01, min_speedup=0.05).precision == q.INT8_STATIC


def test_choose_needs_the_fp32_reference() -> None:
    with pytest.raises(ValueError, match="FP32"):
        q.choose_variant([r(q.FP16, 0.9, 10.0)], max_drop=0.01, min_speedup=0.05)
