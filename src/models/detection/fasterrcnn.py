"""Faster R-CNN (torchvision, two-stage detector) with the same interface as yolo_backend.

The two-stage detector the module guide cites next to YOLO: a ResNet-50 FPN backbone
pretrained on COCO (transfer learning), its box predictor replaced by one for 2 classes
(background, pill). Same three functions as src/models/detection/yolo_backend.py, so
src/models/detection/runner.py runs the Optuna search, the retraining and the test exactly
as for the YOLO models:

- `train(hp, data_root, work, ...)`: SGD or AdamW (hyperparameter `optimizer`), learning rate
  `lr0` with a linear warm-up then a cosine decay, batch size `batch`, input size `imgsz`,
  mixed precision on GPU, best epoch kept on the validation mAP@[.5:.95] (early stopping
  with `patience`); augmentation on the train split only: horizontal / vertical flips
  (`fliplr`, `flipud`) and HSV jitter (`hsv_h`, `hsv_s`, `hsv_v`), as in Ultralytics. Mosaic
  and MixUp are YOLO training tricks with no torchvision counterpart: not searched here;
- `predict(weights, images, imgsz)`: raw boxes and scores (score threshold 0.001);
- `export_onnx(weights, imgsz, target)`: the API's ONNX contract, input 1x3xSxS (letterboxed
  RGB in [0, 1]) and output 1x5xN (cx, cy, w, h, score in input pixels), the layout of a
  one-class Ultralytics export, so backend/app/inference.py serves it unchanged.

Images are letterboxed to `imgsz` x `imgsz` (grey padding, as Ultralytics and the API) for
training, prediction and export alike, and the boxes mapped back to the original pixels.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.models.detection.labels import Sample, read_bgr, samples, to_torchvision
from src.models.detection.latency import letterbox
from src.models.detection.metrics import ImagePrediction, evaluate
from src.models.detection.yolo_backend import TrainResult

NUM_CLASSES = 2  # background + pill
SCORE_THRESH = 0.001
DETECTIONS_PER_IMAGE = 100
WARMUP_ITERATIONS = 300
PRETRAINED = "fasterrcnn_resnet50_fpn"  # value of `weights` in config/hpo/detector_fasterrcnn.yaml


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    imgsz: int = 640, *, pretrained: bool = True, num_classes: int = NUM_CLASSES
) -> nn.Module:
    """Faster R-CNN ResNet-50 FPN, COCO weights (when `pretrained`), new 2-class predictor.

    The internal transform resizes to exactly `imgsz` (images arrive letterboxed square).
    """
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
        weights_backbone=None,  # the COCO weights include the backbone
        min_size=imgsz,
        max_size=imgsz,
        box_score_thresh=SCORE_THRESH,
        box_detections_per_img=DETECTIONS_PER_IMAGE,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# --- data ---------------------------------------------------------------------------------


def letterbox_geometry(height: int, width: int, size: int) -> tuple[float, int, int]:
    """Scale and left / top padding of latency.letterbox for an image of this size."""
    ratio = min(size / height, size / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    left = round((size - new_w) / 2 - 0.1)
    top = round((size - new_h) / 2 - 0.1)
    return ratio, left, top


def to_letterbox(boxes: np.ndarray, ratio: float, left: int, top: int) -> np.ndarray:
    return boxes * ratio + np.array([left, top, left, top], dtype=np.float32)


def from_letterbox(
    boxes: np.ndarray, ratio: float, left: int, top: int, height: int, width: int
) -> np.ndarray:
    out = (boxes - np.array([left, top, left, top], dtype=np.float32)) / ratio
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out


def hsv_jitter(image: np.ndarray, h: float, s: float, v: float, rng: random.Random) -> np.ndarray:
    """Random hue / saturation / value gains (Ultralytics' augment_hsv)."""
    if not (h or s or v):
        return image
    gains = [1 + rng.uniform(-1, 1) * g for g in (h, s, v)]
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * gains[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * gains[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * gains[2], 0, 255).astype(np.uint8)
    merged = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


def augment(
    image: np.ndarray, boxes: np.ndarray, hp: dict[str, Any], rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """Flips and HSV jitter of a letterboxed image and its x1 y1 x2 y2 boxes (train only)."""
    height, width = image.shape[:2]
    boxes = boxes.copy()
    if rng.random() < float(hp.get("fliplr", 0.0)):
        image = image[:, ::-1]
        boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
    if rng.random() < float(hp.get("flipud", 0.0)):
        image = image[::-1]
        boxes[:, [1, 3]] = height - boxes[:, [3, 1]]
    image = hsv_jitter(
        np.ascontiguousarray(image),
        float(hp.get("hsv_h", 0.0)),
        float(hp.get("hsv_s", 0.0)),
        float(hp.get("hsv_v", 0.0)),
        rng,
    )
    return image, boxes


def image_tensor(image: np.ndarray) -> torch.Tensor:
    """BGR uint8 HxWx3 -> RGB float 3xHxW in [0, 1]."""
    rgb = np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))
    return torch.from_numpy(rgb).float().div_(255.0)


class YoloBoxDataset(Dataset):
    """A YOLO split as torchvision detection samples (letterboxed image, target dict)."""

    def __init__(
        self,
        items: Sequence[Sample],
        imgsz: int,
        hp: dict[str, Any] | None = None,
        *,
        train: bool = False,
        seed: int = 0,
    ) -> None:
        self.items, self.imgsz, self.hp, self.train = list(items), imgsz, hp or {}, train
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.items[index]
        image = read_bgr(sample.image)
        height, width = image.shape[:2]
        text = sample.label.read_text(encoding="utf-8") if sample.label.exists() else ""
        boxes, labels = to_torchvision(text, width, height)
        ratio, left, top = letterbox_geometry(height, width, self.imgsz)
        image = letterbox(image, self.imgsz)
        boxes = to_letterbox(boxes, ratio, left, top)
        if self.train:
            image, boxes = augment(image, boxes, self.hp, self.rng)
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
        }
        return image_tensor(image), target


def collate(batch: list[Any]) -> tuple[list[torch.Tensor], list[dict[str, torch.Tensor]]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


# --- training -----------------------------------------------------------------------------


def make_optimizer(model: nn.Module, hp: dict[str, Any]) -> torch.optim.Optimizer:
    parameters = [p for p in model.parameters() if p.requires_grad]
    lr = float(hp.get("lr0", 0.01))
    if str(hp.get("optimizer", "SGD")).lower() == "adamw":
        return torch.optim.AdamW(
            parameters, lr=lr, weight_decay=float(hp.get("weight_decay", 1e-4))
        )
    return torch.optim.SGD(
        parameters,
        lr=lr,
        momentum=float(hp.get("momentum", 0.9)),
        weight_decay=float(hp.get("weight_decay", 1e-4)),
    )


def learning_rate_factor(iteration: int, total: int, warmup: int) -> float:
    """Linear warm-up, then cosine decay to 1 % of the initial learning rate."""
    if iteration < warmup:
        return 0.001 + 0.999 * iteration / max(1, warmup)
    progress = (iteration - warmup) / max(1, total - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def save_checkpoint(model: nn.Module, path: Path, imgsz: int, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "imgsz": imgsz, "arch": PRETRAINED, **extra}, path
    )


def load_checkpoint(path: Path | str) -> tuple[nn.Module, int]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    model = build_model(int(checkpoint["imgsz"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model, int(checkpoint["imgsz"])


@torch.no_grad()
def _predict_model(
    model: nn.Module, items: Sequence[Path], imgsz: int, batch: int = 8
) -> list[ImagePrediction]:
    model.eval()
    target = device()
    model.to(target)
    out: list[ImagePrediction] = []
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        images, geometry = [], []
        for path in chunk:
            image = read_bgr(path)
            height, width = image.shape[:2]
            geometry.append((*letterbox_geometry(height, width, imgsz), height, width))
            images.append(image_tensor(letterbox(image, imgsz)).to(target))
        with torch.autocast(target.type, enabled=target.type == "cuda"):
            results = model(images)
        for result, (ratio, left, top, height, width) in zip(results, geometry, strict=True):
            boxes = result["boxes"].float().cpu().numpy()
            out.append(
                ImagePrediction(
                    from_letterbox(boxes, ratio, left, top, height, width).astype(np.float32),
                    result["scores"].float().cpu().numpy().astype(np.float32),
                )
            )
    return out


def train(
    hp: dict[str, Any],
    data_root: Path,
    work: Path,
    *,
    weights: str,
    epochs: int,
    name: str,
    seed: int,
    fraction: float = 1.0,
    patience: int = 100,
    plots: bool = False,
) -> TrainResult:
    """Fine-tune the COCO model on the train split, keep the best validation epoch."""
    imgsz, batch = int(hp.get("imgsz", 640)), int(hp.get("batch", 4))
    run_dir = work / name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    items = samples(data_root, "train")
    if fraction < 1.0:
        items = sorted(
            random.Random(seed).sample(items, max(1, round(len(items) * fraction))),
            key=lambda s: s.image.name,
        )
    val = samples(data_root, "val")
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        YoloBoxDataset(items, imgsz, hp, train=True, seed=seed),
        batch_size=batch,
        shuffle=True,
        num_workers=int(hp.get("workers", 2)),
        collate_fn=collate,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=len(items) > batch,
    )
    model = build_model(imgsz, pretrained=weights == PRETRAINED).to(device())
    optimizer = make_optimizer(model, hp)
    base_lr = [group["lr"] for group in optimizer.param_groups]
    scaler = torch.amp.GradScaler(enabled=device().type == "cuda")
    total = epochs * max(1, len(loader))
    warmup = min(WARMUP_ITERATIONS, max(1, len(loader)))
    best_value, best_epoch, iteration = -1.0, 0, 0
    best = run_dir / "weights" / "best.pt"
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        started, losses = time.monotonic(), []
        for images, targets in loader:
            factor = learning_rate_factor(iteration, total, warmup)
            for group, lr in zip(optimizer.param_groups, base_lr, strict=True):
                group["lr"] = lr * factor
            images = [image.to(device()) for image in images]
            targets = [{k: v.to(device()) for k, v in t.items()} for t in targets]
            with torch.autocast(device().type, enabled=device().type == "cuda"):
                loss = sum(model(images, targets).values())
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            iteration += 1
        predictions = _predict_model(model, [s.image for s in val], imgsz)
        scores = evaluate(predictions, [s.truth() for s in val])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(losses)) if losses else float("nan"),
                "val_mAP50_95": scores["mAP50_95"],
                "val_mAP50": scores["mAP50"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_minutes": (time.monotonic() - started) / 60,
            }
        )
        if scores["mAP50_95"] > best_value:
            best_value, best_epoch = scores["mAP50_95"], epoch
            save_checkpoint(model, best, imgsz, {"epoch": epoch, "hp": dict(hp)})
        elif epoch - best_epoch >= patience:
            break
    _write_results(history, run_dir / "results.csv")
    return TrainResult(best, run_dir, history)


def _write_results(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    keys = list(history[0])
    lines = [",".join(keys)] + [",".join(str(row[k]) for k in keys) for row in history]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def predict(
    weights: Path | str, images: Sequence[Path], imgsz: int, *, batch: int = 8
) -> list[ImagePrediction]:
    model, trained_size = load_checkpoint(weights)
    return _predict_model(model, list(images), imgsz or trained_size, batch)


# --- export -------------------------------------------------------------------------------


class OnnxWrapper(nn.Module):
    """1x3xSxS letterboxed image -> 1x5xN (cx, cy, w, h, score), the API's detector layout."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        result = self.model([images[0]])[0]
        boxes, scores = result["boxes"], result["scores"]
        centre = (boxes[:, :2] + boxes[:, 2:]) / 2
        size = boxes[:, 2:] - boxes[:, :2]
        return torch.cat([centre, size, scores[:, None]], dim=1).transpose(0, 1)[None]


def export_onnx(weights: Path, imgsz: int, target: Path, *, opset: int = 17) -> Path:
    """ONNX export with the API's input and output layout, plus Ultralytics-like metadata."""
    import onnx

    model, _ = load_checkpoint(weights)
    model.eval()
    wrapper = OnnxWrapper(model).eval()
    dummy = torch.rand(1, 3, imgsz, imgsz)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(target),
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes={"output0": {2: "detections"}},
        opset_version=opset,
        dynamo=False,
    )
    exported = onnx.load(str(target))
    for key, value in {
        "task": "detect",
        "names": "{0: 'pill'}",
        "imgsz": f"[{imgsz}, {imgsz}]",
        "arch": PRETRAINED,
        "description": "Faster R-CNN ResNet-50 FPN (torchvision), output 1x5xN",
    }.items():
        entry = exported.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(exported, str(target))
    return target


def parameter_count(weights: Path) -> int:
    model, _ = load_checkpoint(weights)
    return int(sum(p.numel() for p in model.parameters()))
