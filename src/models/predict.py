"""Prediction with the API's own pipeline: image -> detection -> classification -> result.

    python -m src.models.predict photo.jpg [other.jpg ...]   # one JSON per image
    python -m src.models.predict --split val                 # end-to-end check on C3PI photos

One image: the ONNX models of models/ (TUJYA_MODELS_DIR) run through app.inference.PillScanner
(detector, square crops, classifier), and the result is the JSON of the API's POST /predict
(app.routers.vision.to_predict_out): boxes, top-5 classes with probabilities, timings.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from app.config import PROJECT_DIR, Settings
from app.inference import PillScanner, decode_image
from app.routers.vision import to_predict_out

# --- End-to-end check on the photos of a split ----------------------------------------------
# End-to-end check of the API's own scan pipeline (scanner.py) on C3PI photos.
#
# From the repository root, with the project venv (.venv) active:
#
#     python -m src.models.predict --split val    # choose the threshold
#     python -m src.models.predict --split test   # final numbers
#
# Same protocol as `src/models/evaluate.py` (all consumer photos of the split, the
# most confident box is identified, no box counts as wrong), but with the code the API runs.
# Prints top-1 / top-5 and, for candidate thresholds, the share of photos accepted without
# confirmation and the top-1 accuracy among them. The models are read from TUJYA_MODELS_DIR
# (default: models/ at the repository root).
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def photos(split: str) -> list[tuple[Path, int]]:
    images = PROJECT_DIR / "data" / "raw" / "c3pi" / "images"
    with (PROJECT_DIR / "config" / "splits" / "selected_images.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    out = []
    for r in rows:
        if r["image_type"] != "C3PI_Test":
            continue
        disc, _, name = r["path"].split("/")
        out.append((images / disc / name, int(r["class_id"])))
    return out


def predict_image(scanner: PillScanner, path: Path) -> dict[str, Any]:
    """The API's POST /predict answer for one image file."""
    started = time.perf_counter()
    output = scanner.scan(decode_image(path.read_bytes()))
    return to_predict_out(output, started, scanner.manifest).model_dump()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.predict")
    parser.add_argument("images", nargs="*", type=Path, help="photos to predict (else --split)")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--out", type=Path, default=None, help="per-photo CSV")
    args = parser.parse_args(argv)
    settings = Settings()
    scanner = PillScanner.from_dir(
        settings.models_dir,
        threads=settings.onnx_threads,
        conf=settings.scan_detection_confidence,
    )
    if args.images:
        for path in args.images:
            print(json.dumps({"image": str(path), **predict_image(scanner, path)}, indent=2))
        return
    rows = []
    started = time.perf_counter()
    for path, truth in photos(args.split):
        result = scanner.scan(decode_image(path.read_bytes()))
        if not result.detections:
            rows.append({"truth": truth, "found": 0, "top1": -1, "p1": 0.0, "in_top5": 0})
            continue
        best = max(result.detections, key=lambda d: d.confidence)
        rows.append(
            {
                "truth": truth,
                "found": 1,
                "top1": best.classes[0],
                "p1": best.probabilities[0],
                "in_top5": int(truth in best.classes),
                "ms": result.detection_ms + result.classification_ms,
            }
        )
    seconds = time.perf_counter() - started
    n = len(rows)
    correct = [r["top1"] == r["truth"] for r in rows]
    summary = {
        "split": args.split,
        "photos": n,
        "detected": sum(r["found"] for r in rows) / n,
        "top1": sum(correct) / n,
        "top5": sum(r["in_top5"] for r in rows) / n,
        "mean_inference_ms": sum(r.get("ms", 0.0) for r in rows) / n,
        "wall_seconds": round(seconds, 1),
        "thresholds": [],
    }
    for t in THRESHOLDS:
        accepted = [ok for r, ok in zip(rows, correct, strict=True) if r["p1"] >= t]
        summary["thresholds"].append(
            {
                "threshold": t,
                "accepted_share": len(accepted) / n,
                "top1_when_accepted": sum(accepted) / len(accepted) if accepted else None,
                "wrong_and_accepted_share": (len(accepted) - sum(accepted)) / n,
            }
        )
    print(json.dumps(summary, indent=2))
    if args.out:
        with args.out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["truth", "found", "top1", "p1", "in_top5", "ms"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
