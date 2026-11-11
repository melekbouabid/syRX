"""OCR engines compared on the medicine box photos of step 2.7 (src/models/ocr/evaluate.py).

- `rapidocr`: RapidOCR 3 (PP-OCRv4 mobile, ONNX Runtime CPU), the engine the API can serve
  (app/boxes.py, no PyTorch in the backend image);
- `easyocr`: EasyOCR 1.7 (CRAFT detection + CRNN recognition, PyTorch), training venv only,
  Italian + French + English recognition model.

Tesseract is not compared: its Windows installer needs administrator rights on the
project's PC (decision D109). Both engines return app.boxes.OcrResult.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from app.boxes import OcrEngine, OcrLine, OcrResult, RapidOcrEngine

ENGINES = ("rapidocr", "easyocr")


class EasyOcrEngine:
    name = "easyocr"

    def __init__(self, languages: list[str] | None = None, gpu: bool = True) -> None:
        import easyocr

        self.languages = languages or ["it", "fr", "en"]
        self.reader = easyocr.Reader(self.languages, gpu=gpu, verbose=False)

    def read(self, image: np.ndarray) -> OcrResult:
        start = time.perf_counter()
        found = self.reader.readtext(image)  # [(4 points, text, confidence)], BGR accepted
        elapsed = (time.perf_counter() - start) * 1000
        lines = []
        for points, text, score in found:
            xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
            lines.append(OcrLine(str(text), float(score), (min(xs), min(ys), max(xs), max(ys))))
        return OcrResult(lines, elapsed, self.name)


def make_engine(config: dict[str, Any]) -> OcrEngine:
    """Engine from a config of config/ocr/<engine>.yaml (`engine` + its options)."""
    name = config.get("engine")
    if name == "rapidocr":
        return RapidOcrEngine(threads=int(config.get("threads", 4)))
    if name == "easyocr":
        return EasyOcrEngine(config.get("languages"), gpu=bool(config.get("gpu", True)))
    raise ValueError(f"unknown OCR engine {name!r} (one of {', '.join(ENGINES)})")
