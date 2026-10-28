"""Tujya vision: C3PI subset, automatic pill boxes, training, evaluation, ONNX export."""

import os

# Ultralytics installs missing packages on the fly by default; on 2026-10-28 that silently
# added onnxruntime-gpu to the venv during an evaluation. Dependencies come from
# requirements.txt only.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
