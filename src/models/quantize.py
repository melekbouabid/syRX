"""ONNX weight optimisation: FP16 and INT8 variants of an exported model (step 2.4).

Pure functions on ONNX files, without MLflow, used by src/models/optimize.py:

- `to_fp16`: weights and activations in float16 (`onnxruntime.transformers.float16`, the
  converter of onnxconverter-common shipped with ONNX Runtime). Inputs and outputs stay
  float32 (`keep_io_types`), so the API feeds the model exactly as before.
- `to_int8_static`: post-training static quantization in the QDQ format (QuantizeLinear /
  DequantizeLinear pairs around each quantized operator), int8 weights per channel and int8
  activations whose ranges are calibrated on images of the validation split
  (`CalibrationReader`), never on the test split.
- `to_int8_dynamic`: int8 weights only, activation ranges computed at run time (no
  calibration); the fallback when the static quantization loses too much precision.
- `copy_metadata`: the Ultralytics metadata (`names`, `imgsz`, ...) of the original model,
  which the API reads, copied onto every variant.
- `choose_variant`: the rule that picks the served variant (docs/module1/optimisation_onnx.md).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

FP32, FP16, INT8_STATIC, INT8_DYNAMIC = "fp32", "fp16", "int8-static", "int8-dynamic"
PRECISIONS = (FP32, FP16, INT8_STATIC, INT8_DYNAMIC)
CALIBRATION_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}


class CalibrationReader(CalibrationDataReader):
    """Feeds the calibration tensors (1 x 3 x S x S float32) to the static quantizer."""

    def __init__(self, input_name: str, tensors: Iterable[np.ndarray]) -> None:
        self.input_name = input_name
        self._tensors = list(tensors)
        self._iter: Iterator[np.ndarray] = iter(self._tensors)

    def get_next(self) -> dict[str, np.ndarray] | None:
        tensor = next(self._iter, None)
        return None if tensor is None else {self.input_name: tensor}

    def rewind(self) -> None:
        self._iter = iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)


def input_name(model: Path) -> str:
    return str(onnx.load(str(model), load_external_data=False).graph.input[0].name)


def copy_metadata(source: Path, target: Path) -> None:
    """Give `target` the metadata_props of `source` (class names and image size of YOLO)."""
    original = onnx.load(str(source))
    model = onnx.load(str(target))
    kept = {p.key: p.value for p in original.metadata_props}
    kept.update({p.key: p.value for p in model.metadata_props if p.key not in kept})
    del model.metadata_props[:]
    for key, value in kept.items():
        model.metadata_props.add(key=key, value=value)
    onnx.save_model(model, str(target))


def to_fp16(source: Path, target: Path) -> Path:
    """FP16 weights and activations, float32 inputs and outputs."""
    from onnxruntime.transformers.float16 import convert_float_to_float16

    model = convert_float_to_float16(onnx.load(str(source)), keep_io_types=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(target))
    copy_metadata(source, target)
    return target


def _preprocess(source: Path, work: Path) -> Path:
    """Shape inference and graph cleanup recommended before quantizing (ONNX Runtime)."""
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prepared = work / "prepared.onnx"
    try:
        quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)
    except Exception:  # optional step: quantize the original graph instead
        return source
    return prepared


def to_int8_static(
    source: Path,
    target: Path,
    tensors: Sequence[np.ndarray],
    *,
    method: str = "minmax",
    per_channel: bool = True,
    exclude_nodes: Sequence[str] = (),
) -> Path:
    """QDQ int8 model calibrated on `tensors` (validation images, preprocessed as served)."""
    if not tensors:
        raise ValueError("static quantization needs calibration images")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        prepared = _preprocess(source, Path(tmp))
        quantize_static(
            str(prepared),
            str(target),
            CalibrationReader(input_name(source), tensors),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            calibrate_method=CALIBRATION_METHODS[method],
            nodes_to_exclude=list(exclude_nodes),
            extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
        )
    copy_metadata(source, target)
    return target


def to_int8_dynamic(source: Path, target: Path) -> Path:
    """Int8 weights, activations quantized at run time (no calibration)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        prepared = _preprocess(source, Path(tmp))
        quantize_dynamic(str(prepared), str(target), weight_type=QuantType.QUInt8)
    copy_metadata(source, target)
    return target


def size_mb(path: Path) -> float:
    return path.stat().st_size / 1e6


def head_nodes(model: Path, prefix: str) -> list[str]:
    """Names of the nodes whose name starts with `prefix` (e.g. the YOLO box decoding)."""
    if not prefix:
        return []
    graph = onnx.load(str(model), load_external_data=False).graph
    return [node.name for node in graph.node if node.name.startswith(prefix)]


def output_module_prefix(model: Path) -> str:
    """Module of the node that produces the first output: "/model.22/" for YOLOv8n's head.

    Ultralytics names the exported nodes after the PyTorch modules ("/model.22/Concat_3"),
    so this prefix selects the whole detection head, whatever the architecture.
    """
    graph = onnx.load(str(model), load_external_data=False).graph
    output = graph.output[0].name
    producer = next((n.name for n in graph.node if output in n.output), "")
    if "/" not in producer.strip("/"):
        return ""  # not an Ultralytics module path: nothing to select
    return producer.rsplit("/", 1)[0] + "/"


@dataclass(frozen=True)
class VariantResult:
    """Measured figures of one variant (accuracy = the Registry metric of the model)."""

    precision: str
    accuracy: float
    latency_ms: float
    latency_p95_ms: float
    size_mb: float


@dataclass(frozen=True)
class Choice:
    precision: str
    reason: str


def choose_variant(
    results: Sequence[VariantResult], *, max_drop: float, min_speedup: float
) -> Choice:
    """The served variant: the fastest one that loses at most `max_drop` of accuracy.

    FP32 is the reference. A variant is admissible when its accuracy is at most `max_drop`
    below FP32 (absolute, e.g. 0.01 = one point of mAP or top-1) and its mean latency at least
    `min_speedup` lower (relative, e.g. 0.05 = 5 %: smaller gains are within the measurement
    noise). Among the admissible variants the lowest mean latency wins (then the smaller
    file); without any, FP32 is kept.
    """
    by_precision = {r.precision: r for r in results}
    if FP32 not in by_precision:
        raise ValueError("the FP32 reference is missing")
    ref = by_precision[FP32]
    admissible = [
        r
        for r in results
        if r.precision != FP32
        and ref.accuracy - r.accuracy <= max_drop
        and r.latency_ms <= ref.latency_ms * (1 - min_speedup)
    ]
    if not admissible:
        return Choice(
            FP32,
            f"no variant within {max_drop} of FP32 accuracy and {min_speedup:.0%} faster",
        )
    best = min(admissible, key=lambda r: (r.latency_ms, r.size_mb))
    return Choice(
        best.precision,
        f"{best.precision}: accuracy {best.accuracy:.4f} vs {ref.accuracy:.4f} "
        f"(drop {ref.accuracy - best.accuracy:+.4f} <= {max_drop}), latency "
        f"{best.latency_ms:.1f} ms vs {ref.latency_ms:.1f} ms "
        f"({1 - best.latency_ms / ref.latency_ms:.0%} faster), size {best.size_mb:.1f} MB "
        f"vs {ref.size_mb:.1f} MB",
    )
