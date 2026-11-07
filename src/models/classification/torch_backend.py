"""Training, prediction, ONNX export and Grad-CAM of the classifiers (PyTorch).

One training loop for the six architectures of src/models/architecture.py, driven by
src/models/classification/runner.py (the same functions as the detectors' backends):

- `train(hp, data_root, work, ...)` -> `TrainResult` (best weights on the validation top-1,
  per-epoch history);
- `predict(weights, files, imgsz)` -> class probabilities, one row per crop, with the API's
  preprocessing (src/models/classification/data.py `crop_tensor`);
- `export_onnx(weights, imgsz, target, metadata)` -> ONNX file with the API's input
  (1x3xSxS RGB in [0, 1]) and output (1xN probabilities), class list in the metadata;
- `gradcam_pictures(...)` -> PNG pictures of Grad-CAM (src/utils/gradcam.py) on test crops.

Training recipe (hyperparameters in config/hpo/classifier_<arch>.yaml, searched by Optuna):

1. transfer learning: `head_epochs` epochs with the pretrained backbone frozen, only the new
   classification layer learns (skipped by the baseline, trained from scratch);
2. selective fine-tuning: the last `unfreeze` backbone blocks are unfrozen and trained with
   the head, at `lr * backbone_lr_mult` (cosine decay, AdamW);
3. class imbalance: `imbalance` = `class_weights` (weights (n_c / mean)^-power from the TRAIN
   counts) or `focal` (focal loss, gamma `focal_gamma`, same weights), or `none`;
4. augmentation of the TRAIN split only: random rotation (pills lie in any orientation),
   random resized crop, RandAugment (`randaugment` magnitude, 0 = off), and MixUp or CutMix
   on a share `mix_prob` of the batches; the validation and test crops only get the API's
   resize (no leak of augmented or test data into the choice of the model);
5. early stopping on the validation top-1 (`patience`), time limit `max_minutes`.

Mirror flips are not used: a mirrored imprint never occurs on a real pill.
"""

from __future__ import annotations

import csv
import math
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models import architecture as A
from src.models.classification import data as D
from src.models.classification.backend import TrainResult
from src.models.detection.labels import read_bgr

ROTATION_FILL = 114  # grey corners after a rotation (the letterbox grey of the detector)
GRAD_CLIP = 5.0
ARCH_OPTIONS = ("width", "dropout")  # architecture options read from the hyperparameters


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- losses -------------------------------------------------------------------------------


def class_weights(labels: Sequence[int], num_classes: int, power: float = 0.5) -> np.ndarray:
    """(n_c / mean n)^-power per class, from the TRAIN labels; mean 1 over present classes.

    A class absent from train gets weight 1 (it never appears in a target anyway).
    """
    count = D.counts(list(labels), num_classes).astype(np.float64)
    present = count > 0
    weights = np.ones(num_classes)
    if present.any():
        weights[present] = (count[present] / count[present].mean()) ** (-power)
        weights[present] /= weights[present].mean()
    return weights.astype(np.float32)


def soft_target_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy with soft targets (MixUp, CutMix, label smoothing), class weights and
    the focal factor (1 - p)^gamma; gamma 0 and no weights is the plain cross-entropy."""
    log_p = torch.log_softmax(logits.float(), dim=1)
    terms = -targets * log_p
    if gamma > 0:
        terms = terms * (1.0 - log_p.exp()).pow(gamma)
    if weights is not None:
        terms = terms * weights[None, :]
    return terms.sum(dim=1).mean()


def smooth(targets: torch.Tensor, epsilon: float) -> torch.Tensor:
    if epsilon <= 0:
        return targets
    return targets * (1.0 - epsilon) + epsilon / targets.shape[1]


# --- data ---------------------------------------------------------------------------------


def random_view(image: np.ndarray, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """A random rotation (any angle, grey corners) then a random resized crop (70 to 100 % of
    the area, aspect ratio 0.85 to 1.18) of a BGR crop, size x size, in one cv2.warpAffine.

    The same augmentation as torchvision RandomRotation(180) + RandomResizedCrop, about 25
    times cheaper on the CPU (8.7 ms -> 0.3 ms per crop, D102): the Kaggle notebooks have 4
    CPU cores to feed the GPU. A large downscale goes through INTER_AREA first (no aliasing).
    """
    draw = rng.random if rng is not None else np.random.random
    h, w = image.shape[:2]
    angle = float(draw()) * 360.0 - 180.0
    area = 0.7 + 0.3 * float(draw())
    ratio = math.exp(math.log(0.85) + float(draw()) * (math.log(1.18) - math.log(0.85)))
    cw, ch = min(w, w * math.sqrt(area * ratio)), min(h, h * math.sqrt(area / ratio))
    x0, y0 = float(draw()) * (w - cw), float(draw()) * (h - ch)
    rotation = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0), [0.0, 0.0, 1.0]])
    crop = np.array([[size / cw, 0, -x0 * size / cw], [0, size / ch, -y0 * size / ch], [0, 0, 1]])
    matrix = crop @ rotation  # original pixel -> output pixel
    factor = size / max(cw, ch)
    if factor < 0.75:
        image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
        matrix = matrix @ np.diag([1 / factor, 1 / factor, 1.0])
    return cv2.warpAffine(
        image,
        matrix[:2],
        (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(ROTATION_FILL,) * 3,
    )


class TrainView:
    """Augmentation of the train crops: BGR uint8 image in, RGB float CHW in [0, 1] out."""

    def __init__(self, imgsz: int, randaugment: int = 0, ops: int = 2) -> None:
        from torchvision.transforms import v2

        self.imgsz = imgsz
        self.augment = (
            v2.RandAugment(num_ops=ops, magnitude=int(randaugment)) if randaugment > 0 else None
        )

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        view = random_view(image, self.imgsz)
        tensor = torch.from_numpy(np.ascontiguousarray(view[:, :, ::-1].transpose(2, 0, 1)))
        if self.augment is not None:
            tensor = self.augment(tensor)  # on the uint8 crop, as torchvision expects
        return tensor.float().div_(255.0)


def train_transform(imgsz: int, randaugment: int = 0, ops: int = 2) -> TrainView:
    """Augmentation of the train crops (rotation, resized crop, RandAugment)."""
    return TrainView(imgsz, randaugment, ops)


class TrainCrops(Dataset):
    def __init__(self, items: Sequence[D.Crop], transform: Callable[..., Any]) -> None:
        self.items, self.transform = list(items), transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        crop = self.items[index]
        return self.transform(read_bgr(crop.path)), crop.label


class EvalCrops(Dataset):
    """Crops with the API's preprocessing only (validation, test)."""

    def __init__(self, files: Sequence[Path], imgsz: int) -> None:
        self.files, self.imgsz = list(files), imgsz

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(D.load_tensor(self.files[index], self.imgsz)[0])


def _loader(
    dataset: Dataset, batch: int, workers: int, *, shuffle: bool, seed: int = 0
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle and len(dataset) > batch,  # type: ignore[arg-type]
        persistent_workers=workers > 0,
        generator=generator,
        worker_init_fn=_seed_worker,
    )


def _seed_worker(worker: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def mixer(num_classes: int, mixup_alpha: float, cutmix_alpha: float) -> Callable[..., Any]:
    """MixUp or CutMix (one of the two, at random) on a batch: images and soft targets."""
    from torchvision.transforms import v2

    return v2.RandomChoice(
        [
            v2.MixUp(alpha=mixup_alpha, num_classes=num_classes),
            v2.CutMix(alpha=cutmix_alpha, num_classes=num_classes),
        ]
    )


# --- checkpoints --------------------------------------------------------------------------


def model_options(hp: dict[str, Any]) -> dict[str, Any]:
    return {k: hp[k] for k in ARCH_OPTIONS if k in hp}


def save_checkpoint(net: A.ClassifierNet, path: Path, imgsz: int, options: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save(
        {
            "arch": net.arch,
            "num_classes": net.num_classes,
            "imgsz": imgsz,
            "options": options,
            "state_dict": state,
        },
        path,
    )


def load(weights: Path | str, map_location: str | torch.device = "cpu") -> A.ClassifierNet:
    """The network of a checkpoint, in evaluation mode (no download: weights=None)."""
    checkpoint = torch.load(str(weights), map_location=map_location, weights_only=True)
    net = A.build(
        checkpoint["arch"],
        int(checkpoint["num_classes"]),
        pretrained=False,
        **(checkpoint.get("options") or {}),
    )
    net.load_state_dict(checkpoint["state_dict"])
    return net.eval()


def checkpoint_info(weights: Path | str) -> dict[str, Any]:
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
    return {k: v for k, v in checkpoint.items() if k != "state_dict"}


# --- prediction ---------------------------------------------------------------------------


@torch.no_grad()
def _probabilities(net: nn.Module, loader: DataLoader, where: torch.device) -> np.ndarray:
    net.eval()
    out = []
    for images in loader:
        logits = net(images.to(where, non_blocking=True))
        out.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), np.float32)


def predict(
    weights: Path | str, files: Sequence[Path], imgsz: int, *, batch: int = 64, workers: int = 0
) -> np.ndarray:
    """Class probabilities (crops x classes) with the API's preprocessing, float32."""
    where = device()
    net = load(weights).to(where)
    return _probabilities(
        net, _loader(EvalCrops(files, imgsz), batch, workers, shuffle=False), where
    )


def parameter_count(weights: Path | str) -> int:
    return A.count_parameters(load(weights))


# --- training -----------------------------------------------------------------------------


def _phase_epochs(hp: dict[str, Any], epochs: int, pretrained: bool) -> int:
    """Head-only epochs (transfer learning) before the selective fine-tuning."""
    if not pretrained:
        return 0
    return max(0, min(int(hp.get("head_epochs", 1)), epochs))


def lr_factor(steps: int) -> Callable[[int], float]:
    """Learning-rate factor per step: constant (head-only phase, `steps` 0), else a cosine
    decay from 1 to 0.01 over `steps` (selective fine-tuning)."""
    if steps <= 0:
        return lambda _: 1.0
    return lambda step: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps))


def _write_history(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    keys = list(history[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def train(
    hp: dict[str, Any],
    data_root: Path,
    work: Path,
    *,
    model: str,
    pretrained: bool,
    epochs: int,
    name: str,
    seed: int,
    fraction: float = 1.0,
    patience: int = 100,
    max_minutes: float | None = None,
    log: Callable[[str], None] = print,
) -> TrainResult:
    """Train `model` on the crops of `data_root` (recipe in the module docstring)."""
    started = time.monotonic()
    seed_everything(seed)
    where = device()
    classes = D.read_classes(data_root)
    num_classes = len(classes)
    imgsz = int(hp["imgsz"])
    A.check_size(model, imgsz)
    train_items = D.stratified_fraction(D.crops(data_root, "train"), fraction, seed)
    val_items = D.crops(data_root, "val")
    if not train_items or not val_items:
        raise ValueError(f"no train or validation crops in {data_root}")
    batch, workers = int(hp.get("batch", 64)), int(hp.get("workers", 4))
    options = model_options(hp)
    # Default memory format: channels_last made the backward pass of the batch-norm networks
    # 10 to 13 times slower on the development GPU (torch 2.14, measured in D102).
    net = A.build(model, num_classes, pretrained=pretrained, **options).to(where)

    transform = train_transform(
        imgsz, int(hp.get("randaugment", 0)), int(hp.get("randaugment_ops", 2))
    )
    train_loader = _loader(
        TrainCrops(train_items, transform), batch, workers, shuffle=True, seed=seed
    )
    val_loader = _loader(
        EvalCrops([c.path for c in val_items], imgsz), batch, workers, shuffle=False
    )
    val_truth = np.asarray([c.label for c in val_items])

    imbalance = str(hp.get("imbalance", "none"))
    weights = None
    if imbalance in ("class_weights", "focal"):
        values = class_weights(
            [c.label for c in train_items], num_classes, float(hp.get("class_weight_power", 0.5))
        )
        weights = torch.tensor(values, device=where)
    gamma = float(hp.get("focal_gamma", 2.0)) if imbalance == "focal" else 0.0
    mix_prob = float(hp.get("mix_prob", 0.0))
    mix = mixer(num_classes, float(hp.get("mixup_alpha", 0.2)), float(hp.get("cutmix_alpha", 1.0)))
    epsilon = float(hp.get("label_smoothing", 0.0))
    lr, weight_decay = float(hp["lr"]), float(hp.get("weight_decay", 0.05))
    backbone_mult = float(hp.get("backbone_lr_mult", 0.1))
    unfreeze = int(hp.get("unfreeze", 10**6))
    head_epochs = _phase_epochs(hp, epochs, pretrained)
    use_amp = where.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = random.Random(seed)

    run_dir = work / name
    best_path = run_dir / "weights" / "best.pt"
    best_score, best_epoch, waited = -1.0, 0, 0
    history: list[dict[str, float]] = []
    optimizer: torch.optim.Optimizer | None = None
    schedule: torch.optim.lr_scheduler.LambdaLR | None = None
    steps_per_epoch = max(1, len(train_loader))

    for epoch in range(1, epochs + 1):
        phase = 0 if epoch <= head_epochs else 1
        if epoch == 1 or epoch == head_epochs + 1:
            # Phase change: head only (transfer learning), then the selective fine-tuning.
            trainable = A.set_trainable(net, 0 if phase == 0 else unfreeze)
            groups = A.parameter_groups(net, lr, backbone_mult, weight_decay)
            optimizer = torch.optim.AdamW(groups)
            phase_steps = steps_per_epoch * (head_epochs if phase == 0 else epochs - head_epochs)
            schedule = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_factor(phase_steps if phase == 1 else 0)
            )
            log(f"epoch {epoch}: phase {phase}, {trainable / 1e6:.2f} M trainable parameters")
        assert optimizer is not None and schedule is not None
        A.train_mode(net)
        epoch_started, total, seen = time.monotonic(), 0.0, 0
        for images, labels in train_loader:
            images = images.to(where, non_blocking=True)
            labels = labels.to(where, non_blocking=True)
            if mix_prob > 0 and rng.random() < mix_prob and len(labels) > 1:
                images, targets = mix(images, labels)
            else:
                targets = nn.functional.one_hot(labels, num_classes).float()
            targets = smooth(targets, epsilon)
            with torch.autocast(where.type, dtype=torch.float16, enabled=use_amp):
                logits = net(images)
            loss = soft_target_loss(logits, targets, weights, gamma)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total += float(loss.detach()) * len(labels)
            seen += len(labels)
        probs = _probabilities(net, val_loader, where)
        order = np.argsort(-probs, axis=1)
        top1 = float(np.mean(order[:, 0] == val_truth))
        top5 = float(np.mean((order[:, :5] == val_truth[:, None]).any(axis=1)))
        val_loss = float(
            -np.mean(np.log(np.clip(probs[np.arange(len(val_truth)), val_truth], 1e-12, 1)))
        )
        row = {
            "epoch": float(epoch),
            "phase": float(phase),
            "train_loss": total / max(1, seen),
            "val_loss": val_loss,
            "val_top1": top1,
            "val_top5": top5,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(row)
        log(
            f"epoch {epoch}/{epochs}: loss {row['train_loss']:.3f} val_top1 {top1:.4f} "
            f"({row['epoch_seconds']:.0f} s)"
        )
        if top1 > best_score:
            best_score, best_epoch, waited = top1, epoch, 0
            save_checkpoint(net, best_path, imgsz, options)
        else:
            waited += 1
        if waited >= patience:
            log(f"early stop at epoch {epoch} (best {best_epoch})")
            break
        if max_minutes is not None and (time.monotonic() - started) / 60 >= max_minutes:
            log(f"time limit of {max_minutes:.0f} min reached at epoch {epoch}")
            break
    _write_history(history, run_dir / "results.csv")
    return TrainResult(best_path, run_dir, history)


# --- ONNX export --------------------------------------------------------------------------


def export_onnx(
    weights: Path | str, imgsz: int, target: Path, metadata: dict[str, str] | None = None
) -> Path:
    """Static 1x3xSxS ONNX model returning probabilities; `metadata` added to the model
    properties (`names`, `imgsz`, `task`, the class list: what the API and import read)."""
    import onnx

    net = load(weights)
    module = A.export_module(net)
    target.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    torch.onnx.export(
        module,
        (dummy,),
        str(target),
        dynamo=False,  # the TorchScript exporter, as Ultralytics (no onnxscript needed)
        opset_version=17,
        input_names=["images"],
        output_names=["output0"],
        do_constant_folding=True,
    )
    model = onnx.load(str(target))
    props = {"imgsz": f"[{imgsz}, {imgsz}]", "task": "classify", "batch": "1", **(metadata or {})}
    onnx.helper.set_model_props(model, props)
    onnx.checker.check_model(model)
    onnx.save(model, str(target))
    return target


# --- Grad-CAM -----------------------------------------------------------------------------


def gradcam_pictures(
    weights: Path | str,
    items: Sequence[tuple[Path, int]],
    imgsz: int,
    out_dir: Path,
    names: dict[int, str] | None = None,
) -> list[Path]:
    """One PNG per crop: the crop and its Grad-CAM for the predicted class, side by side,
    with the true and predicted classes written underneath (src/utils/gradcam.py)."""
    from src.utils.gradcam import GradCAM, overlay

    net = load(weights)
    parts = net.parts()
    names = names or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with GradCAM(net, parts.cam_layer, tokens=parts.cam_tokens) as cam:
        for index, (path, label) in enumerate(items):
            tensor = torch.from_numpy(D.load_tensor(path, imgsz))
            with torch.no_grad():
                probs = torch.softmax(net(tensor), dim=1)[0]
            predicted = int(probs.argmax())
            heatmap = cam(tensor, predicted)
            image = cv2.resize(read_bgr(path), (imgsz, imgsz), interpolation=cv2.INTER_AREA)
            panel = np.hstack([image, overlay(image, heatmap)])
            caption = np.full((40, panel.shape[1], 3), 255, np.uint8)
            ok = "ok" if predicted == label else "error"
            text = (
                f"true {names.get(label, label)}  pred {names.get(predicted, predicted)} "
                f"{float(probs[predicted]):.2f} ({ok})"
            )
            cv2.putText(caption, text, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            target = out_dir / f"gradcam_{index:02d}_{ok}.png"
            cv2.imwrite(str(target), np.vstack([panel, caption]))
            written.append(target)
    return written
