"""Export the trained models to ONNX, register them and promote the best (DVC stage export_onnx).

    python -m src.models.export_onnx                  # both models
    python -m src.models.export_onnx --task detect    # one of them

For each model described by models/train/<model>.json (written by src/models/train.py, or by
src/models/register_baseline.py for the baseline):
1. exports weights/best.pt to ONNX (static input, simplified) as models/onnx/<model>.onnx;
2. checks ONNX Runtime against PyTorch on test images (same top-1, same boxes);
3. logs the ONNX file in the training run as an MLflow 3 LoggedModel (`name=<model>`) with the
   test and parity metrics, and registers it in the Model Registry
   (params.yaml registry.<model>.name: tujya-pill-detector / tujya-pill-classifier);
4. calls promote_if_better (src/utils/mlflow_utils.py): the new version gets the alias
   `champion` only if it beats the current champion by params.yaml registry.<model>.min_gain
   on registry.<model>.metric; otherwise `challenger` or archived.

Writes models/export.json (`dvc metrics show`): parity metrics, registered version, alias.
The API serves the champions: `python -m src.models.publish` copies them into models/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import mlflow
import numpy as np
import onnx
import onnxruntime as ort
from ultralytics import YOLO

from src.data.autolabel import read_image
from src.data.prepare import CLASSIFY, DETECT
from src.models.train import MODELS, OUT, ROOT, load_params
from src.utils import mlflow_utils as mu

ONNX_DIR = ROOT / "models" / "onnx"
REPORT = ROOT / "models" / "export.json"


def onnx_check_classify(
    model: YOLO, onnx_path: Path, images: list[Path], imgsz: int
) -> dict[str, float]:
    """Top-1 agreement and largest probability gap between PyTorch and ONNX Runtime."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    agree, gaps = 0, []
    for path in images:
        torch_probs = model.predict(str(path), imgsz=imgsz, verbose=False, device="cpu")[0]
        torch_probs = torch_probs.probs.data.cpu().numpy()
        image = read_image(path)
        rgb = image[:, :, ::-1]
        # Same preprocessing as Ultralytics classification at inference: resize the short
        # side, centre crop, scale to [0, 1], CHW (the crops are already square).
        h, w = rgb.shape[:2]
        scale = imgsz / min(h, w)
        resized = cv2.resize(
            rgb, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR
        )
        top = (resized.shape[0] - imgsz) // 2
        left = (resized.shape[1] - imgsz) // 2
        crop = resized[top : top + imgsz, left : left + imgsz]
        tensor = np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.float32) / 255.0
        onnx_probs = session.run(None, {name: tensor})[0][0]
        agree += int(onnx_probs.argmax() == torch_probs.argmax())
        gaps.append(float(np.abs(onnx_probs - torch_probs).max()))
    return {
        "onnx_top1_agreement": agree / max(1, len(images)),
        "onnx_max_prob_diff": max(gaps) if gaps else 0.0,
    }


def onnx_check_detect(
    model: YOLO, onnx_path: Path, images: list[Path], imgsz: int
) -> dict[str, float]:
    """Same number of boxes and matching boxes between PyTorch and the ONNX export."""
    onnx_model = YOLO(str(onnx_path), task="detect")
    same_count, ious = 0, []
    for path in images:
        a = model.predict(str(path), imgsz=imgsz, verbose=False, device="cpu")[0].boxes
        b = onnx_model.predict(str(path), imgsz=imgsz, verbose=False, device="cpu")[0].boxes
        same_count += int(len(a) == len(b))
        for box in a.xyxy.cpu().numpy():
            if len(b):
                other = b.xyxy.cpu().numpy()
                x1 = np.maximum(box[0], other[:, 0])
                y1 = np.maximum(box[1], other[:, 1])
                x2 = np.minimum(box[2], other[:, 2])
                y2 = np.minimum(box[3], other[:, 3])
                inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
                area = (box[2] - box[0]) * (box[3] - box[1])
                areas = (other[:, 2] - other[:, 0]) * (other[:, 3] - other[:, 1])
                ious.append(float((inter / (area + areas - inter)).max()))
    return {
        "onnx_same_box_count": same_count / max(1, len(images)),
        "onnx_mean_box_iou": float(np.mean(ious)) if ious else 0.0,
    }


def sample_images(task: str, n: int) -> list[Path]:
    """`n` test images spread over the classes (classification) or the first `n` photos."""
    if task == "classify":
        files = sorted((CLASSIFY / "test").glob("*/*.jpg"))
        return files[:: max(1, len(files) // n)][:n]
    return sorted((DETECT / "images" / "test").glob("*.jpg"))[:n]


def export(weights: Path, imgsz: int, target: Path, *, simplify: bool) -> Path:
    """ONNX export of `weights` written to `target` (the export runs in a temporary copy, so
    the train stage's outputs are never modified)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "best.pt"
        shutil.copy2(weights, copy)
        produced = YOLO(str(copy)).export(
            format="onnx", imgsz=imgsz, simplify=simplify, dynamic=False
        )
        shutil.move(str(produced), target)
    return target


def export_and_register(task: str, params: dict[str, Any]) -> dict[str, Any]:
    model_name = MODELS[task]
    summary = json.loads((OUT / f"{model_name}.json").read_text(encoding="utf-8"))
    weights = OUT / model_name / "weights" / "best.pt"
    imgsz = int(summary["imgsz"])
    rules = params["registry"][model_name]
    settings = params["export"]

    onnx_path = export(
        weights, imgsz, ONNX_DIR / f"{model_name}.onnx", simplify=settings["simplify"]
    )
    check = (onnx_check_classify if task == "classify" else onnx_check_detect)(
        YOLO(str(weights)), onnx_path, sample_images(task, settings["parity_images"]), imgsz
    )
    test = {k: float(v) for k, v in summary.items() if k.startswith("test_")}
    arch = Path(summary["weights"]).stem

    # The ONNX model belongs to the run that trained (or re-evaluated) the weights.
    with mlflow.start_run(run_id=summary["run_id"]) as run:
        mlflow.set_tags({"export_git_commit": mu.git_commit(), "onnx_exported": "true"})
        mlflow.log_metrics(check)
        info = mu.log_model(
            onnx.load(str(onnx_path)),
            name=model_name,
            metrics={**test, **check},
            params={"arch": arch, "imgsz": imgsz, "task": task},
        )
        tags = run.data.tags
    version = mu.register(
        info.model_uri,
        rules["name"],
        tags={
            "task": task,
            "arch": arch,
            "dataset": summary["dataset"],
            "git_commit": tags.get("git_commit", "unknown"),
            "baseline": tags.get("baseline", "false"),
        },
        description=(
            f"{arch} ({task}), ONNX {imgsz}px, dataset {summary['dataset']}, "
            f"{rules['metric']}={test[rules['metric']]:.4f}, run {summary['run_id']}"
        ),
    )
    promotion = mu.promote_if_better(
        rules["name"], version.version, rules["metric"], rules["min_gain"]
    )
    print(promotion.reason)
    return {
        "registered_model": rules["name"],
        "version": int(version.version),
        "run_id": summary["run_id"],
        "model_id": info.model_id,
        "alias": promotion.status,
        "promoted": promotion.promoted,
        "metric": rules["metric"],
        "value": promotion.value,
        **check,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.export_onnx")
    parser.add_argument("--task", choices=["detect", "classify"], action="append")
    args = parser.parse_args(argv)
    params = load_params()
    mu.configure()
    report = {MODELS[task]: export_and_register(task, params) for task in args.task or MODELS}
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
