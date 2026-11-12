"""Classifier architectures compared at step 2.3 (docs/module1/kaggle_classifieurs.md).

One wrapper, `ClassifierNet`, for the six candidates, so the training loop, the selective
fine-tuning, the ONNX export and Grad-CAM (src/utils/gradcam.py) are the same code for all:

| name             | network                                    | pretrained weights      |
|------------------|--------------------------------------------|-------------------------|
| baseline-cnn     | `SmallCNN`, four convolution blocks (here) | none: trained from zero |
| resnet50         | torchvision ResNet-50                      | ImageNet-1k (V2)        |
| efficientnetv2s  | torchvision EfficientNetV2-S               | ImageNet-1k             |
| convnext-tiny    | torchvision ConvNeXt-Tiny                  | ImageNet-1k             |
| vit-b16          | torchvision ViT-B/16 (224 px)              | ImageNet-1k             |
| yolo11s-cls      | Ultralytics YOLO11s-cls                    | ImageNet-1k             |

The network receives RGB images in [0, 1] (what the API feeds, backend/app/inference.py
`to_tensor`) and normalises them itself (ImageNet mean and standard deviation for the
torchvision models, none for YOLO, which is trained on [0, 1]), so the exported ONNX model has
the API's input. `export_module` adds the softmax: the ONNX output is the probabilities, like
the Ultralytics export the API serves today.

Transfer learning then selective fine-tuning: each network is described as an ordered list
of backbone `blocks` (stem first) and a `head`. `set_trainable(net, k)` freezes everything but
the head and the last `k` blocks (`k` >= number of blocks: the whole network); frozen blocks
keep their batch-norm statistics (`train_mode`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NO_NORMALISATION = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))


class SmallCNN(nn.Module):
    """The baseline: a small convolutional network trained from scratch (no transfer).

    Four blocks of two 3x3 convolutions (batch norm, ReLU) followed by a 2x2 max pooling,
    channels `width`, 2x, 4x, 8x; global average pooling, dropout, linear classifier.
    About 1.2 M parameters with width 32.
    """

    def __init__(self, num_classes: int, width: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        blocks = []
        channels = 3
        for i in range(4):
            out = width * 2**i
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels, out, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out, out, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            channels = out
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class YoloLogits(nn.Module):
    """An Ultralytics ClassificationModel that always returns logits.

    Its Classify head returns logits in training mode but (probabilities, logits) in
    evaluation mode; this wrapper keeps the logits, so the loss, the softmax of the export and
    Grad-CAM see the same output in both modes.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out[1] if isinstance(out, tuple | list) else out


@dataclass
class Parts:
    """How a network splits into backbone blocks and head, and where Grad-CAM looks."""

    blocks: list[nn.Module]
    head: list[nn.Module]
    cam_layer: nn.Module
    cam_tokens: bool = False  # the layer outputs tokens (ViT): reshaped to a grid for Grad-CAM


class ClassifierNet(nn.Module):
    """Normalisation + network; input RGB in [0, 1], output logits (one per class)."""

    def __init__(
        self,
        arch: str,
        body: nn.Module,
        parts: Callable[[nn.Module], Parts],
        num_classes: int,
        normalisation: tuple[tuple[float, ...], tuple[float, ...]],
    ) -> None:
        super().__init__()
        self.arch = arch
        self.num_classes = num_classes
        self.body = body
        self._parts = parts
        mean, std = normalisation
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body((x - self.mean) / self.std)

    def parts(self) -> Parts:
        return self._parts(self.body)


class Probabilities(nn.Module):
    """What is exported to ONNX: softmax of the logits (the API reads probabilities)."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=1)


def export_module(net: ClassifierNet) -> nn.Module:
    return Probabilities(net).eval()


# --- the six architectures ----------------------------------------------------------------


def _baseline(num_classes: int, pretrained: bool, **kwargs: Any) -> nn.Module:
    return SmallCNN(
        num_classes, width=int(kwargs.get("width", 32)), dropout=float(kwargs.get("dropout", 0.2))
    )


def _baseline_parts(body: nn.Module) -> Parts:
    features = list(body.features)
    return Parts(features, [body.classifier], features[-1])


def _resnet50(num_classes: int, pretrained: bool, **_: Any) -> nn.Module:
    from torchvision.models import ResNet50_Weights, resnet50

    body = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    body.fc = nn.Linear(body.fc.in_features, num_classes)
    return body


def _resnet50_parts(body: Any) -> Parts:
    stem = nn.ModuleList([body.conv1, body.bn1])
    blocks = [stem, body.layer1, body.layer2, body.layer3, body.layer4]
    return Parts(blocks, [body.fc], body.layer4)


def _efficientnetv2s(num_classes: int, pretrained: bool, **_: Any) -> nn.Module:
    from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

    body = efficientnet_v2_s(
        weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
    )
    last = body.classifier[-1]
    body.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return body


def _features_parts(body: Any) -> Parts:
    """torchvision EfficientNet and ConvNeXt: `features` stages, then `classifier`."""
    blocks = list(body.features)
    return Parts(blocks, [body.classifier], blocks[-1])


def _convnext_tiny(num_classes: int, pretrained: bool, **_: Any) -> nn.Module:
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

    body = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
    last = body.classifier[-1]
    body.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return body


def _vit_b16(num_classes: int, pretrained: bool, **_: Any) -> nn.Module:
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    body = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
    body.heads.head = nn.Linear(body.heads.head.in_features, num_classes)
    return body


def _vit_parts(body: Any) -> Parts:
    layers = list(body.encoder.layers)
    # Grad-CAM on the input norm of the last block: the class token reads the patch tokens
    # there (after the last block only the class token reaches the head, so the patch tokens
    # of the last output get no gradient).
    return Parts(
        [body.conv_proj, *layers], [body.heads, body.encoder.ln], layers[-1].ln_1, cam_tokens=True
    )


def _yolo11s_cls(num_classes: int, pretrained: bool, **_: Any) -> nn.Module:
    from ultralytics import YOLO
    from ultralytics.nn.tasks import ClassificationModel

    # The .yaml builds the same network without downloading weights (tests, CI).
    model = YOLO("yolo11s-cls.pt" if pretrained else "yolo11s-cls.yaml").model
    ClassificationModel.reshape_outputs(model, num_classes)
    model = model.float()
    for parameter in model.parameters():
        parameter.requires_grad = True
    return YoloLogits(model)


def _yolo_parts(body: Any) -> Parts:
    layers = list(body.model.model)
    head = layers[-1]
    # The 1x1 convolution of the Classify head is the last layer with a spatial map.
    return Parts(layers[:-1], [head], head.conv)


@dataclass(frozen=True)
class Spec:
    builder: Callable[..., nn.Module]
    parts: Callable[[Any], Parts]
    normalisation: tuple[tuple[float, ...], tuple[float, ...]]
    fixed_size: int | None = None  # input side the network requires (ViT position embeddings)
    pretrained: bool = True  # has pretrained weights (transfer learning)


ARCHITECTURES: dict[str, Spec] = {
    "baseline-cnn": Spec(
        _baseline, _baseline_parts, (IMAGENET_MEAN, IMAGENET_STD), pretrained=False
    ),
    "resnet50": Spec(_resnet50, _resnet50_parts, (IMAGENET_MEAN, IMAGENET_STD)),
    "efficientnetv2s": Spec(_efficientnetv2s, _features_parts, (IMAGENET_MEAN, IMAGENET_STD)),
    "convnext-tiny": Spec(_convnext_tiny, _features_parts, (IMAGENET_MEAN, IMAGENET_STD)),
    "vit-b16": Spec(_vit_b16, _vit_parts, (IMAGENET_MEAN, IMAGENET_STD), fixed_size=224),
    "yolo11s-cls": Spec(_yolo11s_cls, _yolo_parts, NO_NORMALISATION),
}


def build(arch: str, num_classes: int, *, pretrained: bool = True, **kwargs: Any) -> ClassifierNet:
    """The network `arch` with a new classification layer of `num_classes` outputs.

    `pretrained` loads the ImageNet weights (downloaded once by torchvision / Ultralytics);
    the baseline has none. `kwargs` are architecture options (baseline: width, dropout).
    """
    if arch not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {arch!r}; one of {sorted(ARCHITECTURES)}")
    spec = ARCHITECTURES[arch]
    body = spec.builder(num_classes, pretrained and spec.pretrained, **kwargs)
    return ClassifierNet(arch, body, spec.parts, num_classes, spec.normalisation)


def check_size(arch: str, imgsz: int) -> None:
    fixed = ARCHITECTURES[arch].fixed_size
    if fixed is not None and imgsz != fixed:
        raise ValueError(f"{arch} needs {fixed} px images, got {imgsz}")


# --- transfer learning then selective fine-tuning ------------------------------------------


def set_trainable(net: ClassifierNet, unfreeze: int) -> int:
    """Train the head and the last `unfreeze` backbone blocks; returns trainable parameters.

    `unfreeze` = 0: the head only (transfer learning, the pretrained features are kept);
    `unfreeze` >= the number of blocks: the whole network (every parameter, ViT's class token
    and position embeddings included).
    """
    parts = net.parts()
    if unfreeze >= len(parts.blocks):
        for parameter in net.parameters():
            parameter.requires_grad = True
    else:
        for parameter in net.parameters():
            parameter.requires_grad = False
        trainable = [*parts.head, *parts.blocks[len(parts.blocks) - unfreeze :]]
        for module in trainable:
            for parameter in module.parameters():
                parameter.requires_grad = True
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


def train_mode(net: ClassifierNet) -> None:
    """net.train(), but fully frozen blocks stay in eval mode (batch-norm statistics kept)."""
    net.train()
    for module in net.parts().blocks:
        if not any(p.requires_grad for p in module.parameters()):
            module.eval()


def parameter_groups(
    net: ClassifierNet, lr: float, backbone_lr_mult: float, weight_decay: float
) -> list[dict[str, Any]]:
    """AdamW groups: the head at `lr`, the unfrozen backbone at `lr * backbone_lr_mult`;
    no weight decay on biases and normalisation weights (1-d parameters)."""
    head_ids = {id(p) for module in net.parts().head for p in module.parameters()}
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for parameter in net.parameters():
        if not parameter.requires_grad:
            continue
        key = (id(parameter) in head_ids, parameter.ndim <= 1)
        groups.setdefault(key, []).append(parameter)
    out = []
    for (is_head, no_decay), params in groups.items():
        out.append(
            {
                "params": params,
                "lr": lr if is_head else lr * backbone_lr_mult,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        )
    return out


def count_parameters(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))
