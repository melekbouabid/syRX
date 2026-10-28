"""End-to-end evaluation: detector boxes -> crops -> classifier, on every test photo.

    python -m tujya_vision.evaluate --detector runs/<detect-run>/weights/best.onnx \
        --classifier runs/<classify-run>/weights/best.onnx

Unlike the per-model test scores, this uses all consumer photos of the test split (also the
ones the auto-labeller found doubtful), the detector's own boxes and the ONNX exports, i.e.
what the app will run. Per photo, the most confident box is classified; a photo where the
detector finds nothing counts as wrong. Results are logged to MLflow as an `evaluate` run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import mlflow
from ultralytics import YOLO

from tujya_vision.autolabel import read_image, square_crop
from tujya_vision.prepare import CROP_SIZE, SELECTED, Item
from tujya_vision.train import EXPERIMENT, ROOT


def evaluate(detector: Path, classifier: Path, *, conf: float = 0.25) -> dict[str, float]:
    det = YOLO(str(detector), task="detect")
    cls = YOLO(str(classifier), task="classify")
    index = {name: i for i, name in cls.names.items()}
    with SELECTED.open(newline="", encoding="utf-8") as handle:
        items = [
            Item(r["path"], int(r["class_id"]), r["image_type"], r["split"])
            for r in csv.DictReader(handle)
            if r["split"] == "test" and r["image_type"] == "C3PI_Test"
        ]
    found = top1 = top5 = 0
    for item in items:
        image = read_image(item.file)
        boxes = det.predict(image, conf=conf, imgsz=640, verbose=False)[0].boxes
        if not len(boxes):
            continue
        found += 1
        best = int(boxes.conf.argmax())
        x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[best].tolist())
        crop = square_crop(image, (x1, y1, x2, y2), size=CROP_SIZE)
        probs = cls.predict(crop, imgsz=CROP_SIZE, verbose=False)[0].probs
        truth = index[f"{item.class_id:03d}"]
        top1 += int(probs.top1 == truth)
        top5 += int(truth in probs.top5)
    n = max(1, len(items))
    return {
        "e2e_photos": float(len(items)),
        "e2e_detected": found / n,
        "e2e_top1": top1 / n,
        "e2e_top5": top5 / n,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tujya_vision.evaluate")
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args(argv)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="evaluate-end-to-end"):
        mlflow.log_params(
            {
                "detector": args.detector.resolve().relative_to(ROOT).as_posix(),
                "classifier": args.classifier.resolve().relative_to(ROOT).as_posix(),
                "conf": args.conf,
            }
        )
        metrics = evaluate(args.detector, args.classifier, conf=args.conf)
        mlflow.log_metrics(metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
