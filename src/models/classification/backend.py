"""What a classifier backend returns, without importing PyTorch.

A backend (src/models/classification/torch_backend.py; a fake one in the tests) is a module
with five functions, called by src/models/classification/runner.py:

- `train(hp, data_root, work, *, model, pretrained, epochs, name, seed, fraction=1.0,
  patience=100, max_minutes=None, log=print) -> TrainResult`;
- `predict(weights, files, imgsz) -> np.ndarray` (class probabilities, crops x classes);
- `export_onnx(weights, imgsz, target, metadata) -> Path`;
- `parameter_count(weights) -> int`;
- `gradcam_pictures(weights, items, imgsz, out_dir, names) -> list[Path]`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainResult:
    weights: Path  # best weights (validation top-1)
    run_dir: Path
    history: list[dict[str, float]] = field(default_factory=list)
