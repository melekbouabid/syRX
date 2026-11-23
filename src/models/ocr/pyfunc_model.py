"""MLflow pyfunc of an OCR engine and its configuration: what the Registry versions."""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.pyfunc import PythonModel


class OcrModel(PythonModel):
    def load_context(self, context: Any) -> None:
        from src.models.ocr.engines import make_engine

        self.engine = make_engine(dict(context.model_config or {}))

    def predict(self, context: Any, model_input: Any, params: dict[str, Any] | None = None) -> Any:
        import cv2

        paths = list(model_input.iloc[:, 0]) if hasattr(model_input, "iloc") else list(model_input)
        return [[line.text for line in self.engine.read(cv2.imread(str(p))).lines] for p in paths]


mlflow.models.set_model(OcrModel())
