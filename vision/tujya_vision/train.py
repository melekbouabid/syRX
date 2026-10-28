"""Train, evaluate, export and register the two Tujya vision models.

    python -m tujya_vision.train classify --epochs 10            # pill identification
    python -m tujya_vision.train detect --epochs 5 --imgsz 416   # pill detection

Each run:
1. trains an Ultralytics YOLO model from pretrained weights (transfer learning);
2. evaluates it on the test split (top-1 / top-5 for identification, mAP for detection);
3. exports it to ONNX and checks that ONNX Runtime gives the same predictions as PyTorch;
4. logs params, per-epoch metrics, test metrics and artefacts to MLflow, and registers the
   ONNX model in the Model Registry ("tujya-pill-classifier" / "tujya-pill-detector").

MLflow server: MLFLOW_TRACKING_URI, default http://localhost:5000 (docker compose up -d).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import mlflow
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ultralytics import YOLO, settings

from tujya_vision.autolabel import read_image
from tujya_vision.prepare import CLASSIFY, DETECT

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "vision" / "runs"
CLASSES = ROOT / "config" / "pill_classes.csv"
REGISTERED = {"classify": "tujya-pill-classifier", "detect": "tujya-pill-detector"}
EXPERIMENT = "tujya-vision"


def device() -> str:
    return "0" if torch.cuda.is_available() else "cpu"


def hardware() -> dict[str, str]:
    info = {
        "torch": torch.__version__,
        "cuda": str(torch.cuda.is_available()),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": str(os.cpu_count()),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def _log_epochs(results_csv: Path) -> None:
    """Per-epoch metrics from Ultralytics' results.csv, as MLflow metric histories."""
    with results_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            row = {k.strip(): v for k, v in row.items()}
            step = int(float(row.pop("epoch")))
            metrics = {}
            for key, value in row.items():
                try:
                    metrics[key.replace("(B)", "").replace(":", "_")] = float(value)
                except (TypeError, ValueError):
                    continue
            mlflow.log_metrics(metrics, step=step)


def _onnx_check_classify(
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


def _onnx_check_detect(
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


def classify_test(model: YOLO, folder: Path, imgsz: int, batch: int) -> dict[str, float]:
    """Top-1 / top-5 on the test crops, matching folder names to the model's class names.

    Done here rather than with `model.val(split="test")` because a class can have no test
    crop (its test photos were all doubtful), and a per-folder dataset would then number
    the classes differently from training.
    """
    index = {name: i for i, name in model.names.items()}
    files = sorted(folder.glob("*/*.jpg"))
    top1 = top5 = 0
    for start in range(0, len(files), batch):
        chunk = files[start : start + batch]
        results = model.predict(
            [str(f) for f in chunk], imgsz=imgsz, verbose=False, device=device()
        )
        for path, result in zip(chunk, results, strict=True):
            truth = index[path.parent.name]
            top1 += int(result.probs.top1 == truth)
            top5 += int(truth in result.probs.top5)
    n = max(1, len(files))
    classes = len({f.parent.name for f in files})
    return {
        "test_top1": top1 / n,
        "test_top5": top5 / n,
        "test_images": float(len(files)),
        "test_classes": float(classes),
    }


def _sample_images(task: str, n: int = 50) -> list[Path]:
    if task == "classify":
        return sorted((CLASSIFY / "test").glob("*/*.jpg"))[:: max(1, 600 // n)][:n]
    return sorted((DETECT / "images" / "test").glob("*.jpg"))[:n]


def run(args: argparse.Namespace) -> None:
    task = args.task
    settings.update({"mlflow": False})  # MLflow logging is done here, explicitly
    weights = args.weights or ("yolo11n-cls.pt" if task == "classify" else "yolo11n.pt")
    data = str(CLASSIFY) if task == "classify" else str(DETECT / "data.yaml")
    imgsz = args.imgsz or (224 if task == "classify" else 640)
    name = f"{task}-{time.strftime('%Y%m%d-%H%M%S')}"
    params: dict[str, Any] = {
        "task": task,
        "weights": weights,
        "epochs": args.epochs,
        "imgsz": imgsz,
        "batch": args.batch,
        "device": device(),
        "seed": args.seed,
        "data": Path(data).relative_to(ROOT).as_posix(),
        "fraction": args.fraction,
        "patience": args.patience,
    }

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=name) as active:
        mlflow.log_params(params)
        mlflow.set_tags({**hardware(), "dataset": "NLM C3PI subset (150 classes)"})
        for split in ("train", "val", "test"):
            folder = CLASSIFY / split if task == "classify" else DETECT / "images" / split
            pattern = "*/*.jpg" if task == "classify" else "*.jpg"
            mlflow.log_param(f"n_{split}_images", len(list(folder.glob(pattern))))
        if CLASSES.exists():
            mlflow.log_artifact(str(CLASSES), "config")

        model = YOLO(weights)
        started = time.monotonic()
        model.train(
            data=data,
            epochs=args.epochs,
            imgsz=imgsz,
            batch=args.batch,
            device=device(),
            workers=args.workers,
            project=str(RUNS),
            name=name,
            seed=args.seed,
            deterministic=True,
            fraction=args.fraction,
            patience=args.patience,
            plots=True,
            verbose=False,
        )
        mlflow.log_metric("train_minutes", (time.monotonic() - started) / 60)
        run_dir = RUNS / name
        _log_epochs(run_dir / "results.csv")
        best = run_dir / "weights" / "best.pt"
        model = YOLO(str(best))

        if task == "classify":
            metrics = classify_test(model, CLASSIFY / "test", imgsz, args.batch)
        else:
            test = model.val(
                data=data,
                split="test",
                imgsz=imgsz,
                batch=args.batch,
                device=device(),
                project=str(RUNS),
                name=f"{name}-test",
                plots=True,
                verbose=False,
            )
            metrics = {
                "test_mAP50": float(test.box.map50),
                "test_mAP50_95": float(test.box.map),
                "test_precision": float(test.box.mp),
                "test_recall": float(test.box.mr),
            }
        mlflow.log_metrics(metrics)

        onnx_path = Path(model.export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False))
        sample = _sample_images(task)
        check = (_onnx_check_classify if task == "classify" else _onnx_check_detect)(
            model, onnx_path, sample, imgsz
        )
        mlflow.log_metrics(check)
        metrics.update(check)

        mlflow.log_artifact(str(best), "weights")
        for extra in run_dir.glob("*.png"):
            mlflow.log_artifact(str(extra), "plots")
        mlflow.log_artifact(str(run_dir / "results.csv"), "training")
        mlflow.log_artifact(str(run_dir / "args.yaml"), "training")
        mlflow.onnx.log_model(
            onnx.load(str(onnx_path)),
            name="model",
            registered_model_name=REGISTERED[task],
            input_example=None,
        )
        summary = {"run_id": active.info.run_id, "run_dir": str(run_dir), **params, **metrics}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m tujya_vision.train")
    parser.add_argument("task", choices=["classify", "detect"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=0, help="default 224 (classify), 640 (detect)")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--weights", default="")
    parser.add_argument("--fraction", type=float, default=1.0, help="share of train set used")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
