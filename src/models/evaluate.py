"""End-to-end evaluation: detector boxes -> crops -> classifier, on every test photo.

    python -m src.models.evaluate        # DVC stage `evaluate`: the ONNX models of models/onnx
    python -m src.models.evaluate --detector runs/<detect-run>/weights/best.onnx \
        --classifier runs/<classify-run>/weights/best.onnx --no-out

Unlike the per-model test scores, this uses all consumer photos of the test split (also the
ones the auto-labeller found doubtful), the detector's own boxes and the ONNX exports, i.e.
what the app will run. Per photo, the most confident box is classified; a photo where the
detector finds nothing counts as wrong. Results are logged to MLflow as a tracked
`evaluate-end-to-end` run (src/utils/mlflow_utils.py) and written to models/evaluate.json
(`dvc metrics show`). Confidence threshold: params.yaml evaluate.conf.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mlflow
from ultralytics import YOLO

from src.data.autolabel import read_image, square_crop
from src.data.prepare import CROP_SIZE, SELECTED, Item
from src.models.train import EXPERIMENT, ROOT, load_params
from src.utils import mlflow_utils as mu

ONNX_DIR = ROOT / "models" / "onnx"
EXPORT_REPORT = ROOT / "models" / "export.json"
OUT = ROOT / "models" / "evaluate.json"


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


def _relative(path: Path) -> str:
    resolved = path.resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.evaluate")
    parser.add_argument("--detector", type=Path, default=ONNX_DIR / "detector.onnx")
    parser.add_argument("--classifier", type=Path, default=ONNX_DIR / "classifier.onnx")
    parser.add_argument("--conf", type=float, help="default: params.yaml evaluate.conf")
    parser.add_argument("--no-out", action="store_true", help="do not write models/evaluate.json")
    args = parser.parse_args(argv)
    params = load_params()
    conf = args.conf if args.conf is not None else float(params["evaluate"]["conf"])
    # The registered versions being evaluated, when the models come from the export stage.
    tags = {"stage": "evaluate"}
    if EXPORT_REPORT.is_file() and args.detector.resolve().parent == ONNX_DIR.resolve():
        report = json.loads(EXPORT_REPORT.read_text(encoding="utf-8"))
        for model, entry in report.items():
            tags[f"{model}_version"] = f"{entry['registered_model']}/{entry['version']}"
            tags[f"{model}_run_id"] = entry["run_id"]
    mu.configure()
    with mu.start_tracked_run(
        EXPERIMENT,
        "evaluate-end-to-end",
        seed=int(params["train"]["seed"]),
        dataset=f"{params['train']['dataset']} test split, consumer photos",
        params={
            "detector": _relative(args.detector),
            "classifier": _relative(args.classifier),
            "conf": conf,
        },
        tags=tags,
    ) as run:
        metrics = evaluate(args.detector, args.classifier, conf=conf)
        mlflow.log_metrics(metrics)
    if not args.no_out:
        OUT.write_text(
            json.dumps({**metrics, "run_id": run.info.run_id}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
