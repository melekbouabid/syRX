"""Grad-CAM pictures of the demo gallery, precomputed with the served classifier (D104).

The API runs ONNX models only, without PyTorch (D97), so it cannot compute Grad-CAM, which
needs the gradients of the PyTorch model. This script does it offline, in the training venv
(PyTorch, Ultralytics), for the photos of the app's demo gallery (app/gallery.py):

    PYTHONPATH="backend;." .venv-train/Scripts/python -m src.models.gradcam_gallery

or through DVC (stage `gradcam_gallery` of dvc.yaml, output data/processed/c3pi/gradcam,
never in git). For each photo it runs the served ONNX scan (app/inference.py: same boxes and
same 224 px square crops as the API), then Grad-CAM (src/utils/gradcam.py) of the PyTorch
weights of the same classifier version (models/train/classifier, checked against the run id
of models/models.json) on the last convolution before the pooling of the head, for the top-1
class the API gives. The heat map of each pill is blended over its crop square in the photo,
so the picture keeps the proportions of the photo and the scan boxes fit on it.

Outputs: one JPEG per photo (name from `app.gallery.gradcam_name`) and gradcam.json (model,
layer, per-pill class and probability from ONNX and from PyTorch, which must agree).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "backend") not in sys.path:  # the training venv has no editable root package
    sys.path.insert(0, str(ROOT / "backend"))

from app.gallery import DemoPhoto, load_gallery  # noqa: E402
from app.inference import (  # noqa: E402
    CROP_MARGIN,
    PillScanner,
    decode_image,
    square_crop,
    to_tensor,
)

DEFAULTS = {
    "gallery": ROOT / "config" / "demo_gallery.csv",
    "split": ROOT / "config" / "splits" / "selected_images.csv",
    "holdout": ROOT / "config" / "splits" / "c3pi_v2" / "test.csv",
    "images": ROOT / "data" / "raw" / "c3pi" / "images",
    "models": ROOT / "models",
    "weights": ROOT / "models" / "train" / "classifier",
    "out": ROOT / "data" / "processed" / "c3pi" / "gradcam",
}
MAX_SIDE = 1280  # longest side of the saved pictures (the photos are up to 4,000 px)
JPEG_QUALITY = 85
MAX_PROB_DIFF = 1e-3  # PyTorch and ONNX Runtime must give the same probabilities


@dataclass(frozen=True)
class Square:
    """The square the classifier sees, in photo pixels (may extend past the photo edges)."""

    left: int
    top: int
    side: int


def crop_square(box: tuple[float, ...]) -> Square:
    """Same square as app.inference.square_crop (centred, 20 % margin each side)."""
    x1, y1, x2, y2 = box
    side = max(x2 - x1, y2 - y1) * (1 + 2 * CROP_MARGIN)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left, top = round(cx - side / 2), round(cy - side / 2)
    return Square(left, top, round(cx + side / 2) - left)


def paste_heatmap(target: np.ndarray, heatmap: np.ndarray, square: Square) -> None:
    """Keep, inside `square`, the larger of `target` and the heat map resized to the square."""
    resized = cv2.resize(heatmap, (square.side, square.side), interpolation=cv2.INTER_LINEAR)
    height, width = target.shape
    x0, y0 = max(0, square.left), max(0, square.top)
    x1, y1 = min(width, square.left + square.side), min(height, square.top + square.side)
    if x0 >= x1 or y0 >= y1:
        return
    part = resized[y0 - square.top : y1 - square.top, x0 - square.left : x1 - square.left]
    np.maximum(target[y0:y1, x0:x1], part, out=target[y0:y1, x0:x1])


def compose(image: np.ndarray, heatmaps: list[tuple[np.ndarray, Square]]) -> np.ndarray:
    """The photo, slightly darkened, with each pill's heat map (JET colours) over its square."""
    from src.utils.gradcam import overlay

    heat = np.zeros(image.shape[:2], np.float32)
    inside = np.zeros(image.shape[:2], np.uint8)
    for heatmap, square in heatmaps:
        paste_heatmap(heat, heatmap, square)
        cv2.rectangle(
            inside,
            (square.left, square.top),
            (square.left + square.side, square.top + square.side),
            1,
            -1,
        )
    blended = overlay(image, heat, alpha=0.5)
    darker = (image * 0.6).astype(np.uint8)
    picture = np.where(inside[..., None] == 1, blended, darker)
    for _, square in heatmaps:
        cv2.rectangle(
            picture,
            (square.left, square.top),
            (square.left + square.side, square.top + square.side),
            (255, 255, 255),
            max(2, round(max(image.shape[:2]) / 600)),
        )
    return picture


def shrink(picture: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    longest = max(picture.shape[:2])
    if longest <= max_side:
        return picture
    ratio = max_side / longest
    size = (round(picture.shape[1] * ratio), round(picture.shape[0] * ratio))
    return cv2.resize(picture, size, interpolation=cv2.INTER_AREA)


def served_classifier(models_dir: Path, weights_dir: Path) -> dict[str, Any]:
    """The served classifier's Registry entry; fails if the PyTorch weights are another run."""
    manifest = json.loads((models_dir / "models.json").read_text("utf-8"))
    served = manifest["classify"]
    trained = json.loads((weights_dir.parent / f"{weights_dir.name}.json").read_text("utf-8"))
    if trained.get("run_id") != served.get("run_id"):
        raise SystemExit(
            f"{weights_dir} holds run {trained.get('run_id')}, but the API serves run "
            f"{served.get('run_id')} ({served.get('registered_model')} v{served.get('version')})"
        )
    return dict(served)


def run(args: argparse.Namespace) -> dict[str, Any]:
    photos: list[DemoPhoto] = load_gallery(
        args.gallery, args.split, args.holdout, args.images, args.out
    )
    if not photos:
        raise SystemExit("no gallery photo available (config files or C3PI images missing)")
    served = served_classifier(args.models, args.weights)

    import torch
    from ultralytics import YOLO

    from src.utils.gradcam import GradCAM

    weights = args.weights / "weights" / "best.pt"
    net = YOLO(str(weights)).model.float().eval()
    head = len(net.model) - 1
    layer = net.model[head].conv  # 1 x 1 convolution of the Classify head, 7 x 7 at 224 px
    scanner = PillScanner.from_dir(args.models)

    args.out.mkdir(parents=True, exist_ok=True)
    for stale in args.out.glob("*.jpg"):
        stale.unlink()

    entries = []
    with GradCAM(net, layer) as cam:
        for photo in photos:
            image = decode_image(photo.file.read_bytes())
            output = scanner.scan(image)
            heatmaps, pills = [], []
            for detection in output.detections:
                crop = square_crop(image, detection.box, scanner.cls_size)
                tensor = torch.from_numpy(to_tensor(crop))
                index = scanner.class_ids.index(detection.classes[0])
                with torch.no_grad():
                    probabilities = net(tensor)[0][0].numpy()
                onnx_probability = detection.probabilities[0]
                difference = abs(float(probabilities[index]) - onnx_probability)
                if difference > MAX_PROB_DIFF or int(probabilities.argmax()) != index:
                    raise SystemExit(f"PyTorch and ONNX disagree on {photo.path}")
                square = crop_square(detection.box)
                heatmaps.append((cam(tensor, index), square))
                pills.append(
                    {
                        "box": [round(v, 1) for v in detection.box],
                        "square": [square.left, square.top, square.side],
                        "class_id": detection.classes[0],
                        "probability_onnx": round(onnx_probability, 5),
                        "probability_pytorch": round(float(probabilities[index]), 5),
                    }
                )
            picture = shrink(compose(image, heatmaps))
            ok, encoded = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                raise SystemExit(f"cannot encode the Grad-CAM of {photo.path}")
            photo.gradcam.write_bytes(encoded.tobytes())
            entries.append(
                {
                    "path": photo.path,
                    "class_id": photo.class_id,
                    "file": photo.gradcam.name,
                    "size": [picture.shape[1], picture.shape[0]],
                    "pills": pills,
                }
            )
            print(f"{photo.gradcam.name}: {len(pills)} pill(s)")

    report = {
        "method": "Grad-CAM (Selvaraju et al., 2017), src/utils/gradcam.py",
        "target": "top-1 class of the served ONNX scan, for each detected pill",
        "layer": f"model.{head}.conv",
        "classifier": {
            key: served.get(key)
            for key in ("registered_model", "version", "aliases", "run_id", "arch", "precision")
        },
        "weights": weights.relative_to(ROOT).as_posix(),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "images": entries,
    }
    (args.out / "gradcam.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for name, default in DEFAULTS.items():
        parser.add_argument(f"--{name}", type=Path, default=default)
    args = parser.parse_args(argv)
    report = run(args)
    print(f"{len(report['images'])} Grad-CAM picture(s) in {args.out}")


if __name__ == "__main__":
    main()
