"""What a classifier backend returns, without importing PyTorch.

A backend module provides train, predict, export_onnx, parameter_count and gradcam_pictures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainResult:
    weights: Path  # best weights (validation top-1)
    run_dir: Path
    history: list[dict[str, float]] = field(default_factory=list)
