"""Reproducible demonstration of the data drift monitor on five photo scenarios.

Scenarios: test-v2 (no drift), train-v2, dark, blurred and out-of-distribution photos.
Run: `PYTHONPATH="backend;." python -m src.data.drift_demo [--api URL] [--mlflow]`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from app.drift import DriftMonitor, Reference  # noqa: E402
from app.inference import PillScanner  # noqa: E402
from src.data.drift_reference import (  # noqa: E402
    EXPERIMENT,
    MODELS,
    OUT,
    photo_stats,
    photos,
    read_bgr,
)

OFF_TOPIC = ROOT / "data" / "raw" / "off_topic"
BOXES = ROOT / "data" / "raw" / "medicine_boxes" / "italian_otc"
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def darken(image: np.ndarray, factor: float = 0.35) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def blur(image: np.ndarray, share: float = 0.02) -> np.ndarray:
    sigma = max(1.0, share * max(image.shape[:2]))
    return cv2.GaussianBlur(image, (0, 0), sigma)


def out_of_distribution(n: int) -> list[Path]:
    """Off-topic pictures first, then medicine boxes, `n` at most (sorted: reproducible)."""
    files = sorted(p for p in OFF_TOPIC.rglob("*") if p.suffix.lower() in SUFFIXES)
    files += sorted(p for p in BOXES.rglob("*") if p.suffix.lower() in SUFFIXES)
    return files[:n]


def scenarios(n: int) -> dict[str, tuple[list[Path], Callable[[np.ndarray], np.ndarray]]]:
    test = photos("test", n)
    return {
        "test-v2": (test, lambda image: image),
        "train-v2": (photos("train", n), lambda image: image),
        "dark": (test, darken),
        "blurred": (test, blur),
        "out-of-distribution": (out_of_distribution(n), lambda image: image),
    }


def images(files: list[Path], change: Callable[[np.ndarray], np.ndarray]) -> Iterator[np.ndarray]:
    for path in files:
        try:
            yield change(read_bgr(path))
        except ValueError:  # an unreadable file (some downloads are HTML pages): skipped
            continue


def run_local(
    name: str, files: list[Path], change: Callable[[np.ndarray], np.ndarray], scanner: PillScanner
) -> dict[str, Any]:
    monitor = DriftMonitor(Reference.load(OUT / "reference.json"), window=max(1, len(files)))
    for image in images(files, change):
        monitor.record(photo_stats(scanner, image))
    return monitor.report()


def run_api(
    name: str, files: list[Path], change: Callable[[np.ndarray], np.ndarray], url: str
) -> dict[str, Any]:
    import httpx

    with httpx.Client(base_url=url, timeout=120) as client:
        for image in images(files, change):
            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                continue
            sent = client.post(
                "/predict", files={"image": (f"{name}.jpg", encoded.tobytes(), "image/jpeg")}
            )
            sent.raise_for_status()
        response = client.get("/monitoring/drift")
        response.raise_for_status()
        report: dict[str, Any] = response.json()
        return report


def line(name: str, report: dict[str, Any]) -> str:
    drifted = ", ".join(report.get("drifted") or []) or "-"
    return (
        f"{name:<20} {report['status']:<12} window {report['window_size']:>4}  drifted: {drifted}"
    )


def log_mlflow(name: str, report: dict[str, Any]) -> None:
    import mlflow

    from src.utils import mlflow_utils as mu

    with mu.start_tracked_run(
        EXPERIMENT,
        f"drift-demo-{name}",
        seed=0,
        dataset="c3pi-v2-detect-test",
        params={"scenario": name, "photos": report["window_size"]},
        tags={"stage": "drift-demo", "plan_step": "2.8", "drift_status": report["status"]},
    ):
        for feature in report["features"]:
            for key in ("ks_statistic", "ks_pvalue", "psi"):
                if feature[key] is not None:
                    mlflow.log_metric(f"{feature['feature']}_{key}", feature[key])
        mlflow.log_metric("drifted_features", float(len(report["drifted"])))
        mlflow.log_dict(report, "drift_report.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.data.drift_demo")
    parser.add_argument("--photos", type=int, default=0, help="photos per scenario")
    parser.add_argument("--api", default="", help="send to this running API instead")
    parser.add_argument("--only", default="", help="comma-separated scenarios")
    parser.add_argument("--mlflow", action="store_true", help="log one run per scenario")
    parser.add_argument("--out", type=Path, default=None, help="write the reports as JSON")
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    reference = Reference.load(OUT / "reference.json")
    n = args.photos or int(reference.thresholds["window"])
    chosen = [s for s in args.only.split(",") if s]
    scanner = None if args.api else PillScanner.from_dir(MODELS)
    if args.mlflow:
        from src.utils import mlflow_utils as mu

        mu.configure()
    reports: dict[str, Any] = {}
    for name, (files, change) in scenarios(n).items():
        if chosen and name not in chosen:
            continue
        if args.api:
            report = run_api(name, files, change, args.api)
        else:
            assert scanner is not None
            report = run_local(name, files, change, scanner)
        reports[name] = report
        print(line(name, report), flush=True)
        if args.mlflow:
            log_mlflow(name, report)
    if args.out:
        args.out.write_text(json.dumps(reports, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
