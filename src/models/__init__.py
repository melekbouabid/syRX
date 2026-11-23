"""Models: training, evaluation, ONNX export and publication of the vision models."""

import os

# Ultralytics installs missing packages on the fly by default; dependencies come from the
# requirements files only.
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
