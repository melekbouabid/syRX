"""End-to-end check of the API's own scan pipeline (app/scanner.py) on C3PI photos.

    python scripts/evaluate_scan.py --split val      # choose the confirmation threshold
    python scripts/evaluate_scan.py --split test     # final numbers at that threshold

Same protocol as `vision/tujya_vision/evaluate.py` (all consumer photos of the split, the
most confident box is identified, no box counts as wrong), but with the code the API runs.
Prints top-1 / top-5 and, for candidate thresholds, the share of photos accepted without
confirmation and the top-1 accuracy among them. Run from backend/ with the models published
in SYNERGYRX_MODELS_DIR (default ../models).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import PROJECT_DIR, Settings
from app.scanner import PillScanner, decode_image

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def photos(split: str) -> list[tuple[Path, int]]:
    images = PROJECT_DIR / "data" / "vision" / "c3pi" / "images"
    with (PROJECT_DIR / "config" / "splits" / "selected_images.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    out = []
    for r in rows:
        if r["image_type"] != "C3PI_Test":
            continue
        disc, _, name = r["path"].split("/")
        out.append((images / disc / name, int(r["class_id"])))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--out", type=Path, default=None, help="per-photo CSV")
    args = parser.parse_args()
    settings = Settings()
    scanner = PillScanner.from_dir(
        settings.models_dir,
        threads=settings.onnx_threads,
        conf=settings.scan_detection_confidence,
    )
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
