"""Grad-CAM: where a classifier looks when it predicts a class (Selvaraju et al., 2017).

    from src.utils.gradcam import GradCAM, overlay
    with GradCAM(model, layer) as cam:
        heatmap = cam(image_tensor, class_index)      # H x W in [0, 1]
    picture = overlay(bgr_image, heatmap)

For a convolutional layer with activations A (K maps of h x w) and the score y_c of class c
before the softmax, Grad-CAM weights every map by the mean of its gradient,
alpha_k = mean_ij dy_c / dA_kij, and keeps the positive part of the weighted sum,
ReLU(sum_k alpha_k A_k), upsampled to the input size and scaled to [0, 1].

Transformers (ViT) output tokens, not maps: `tokens=True` drops the class token and folds the
patch tokens back into their grid (14 x 14 for ViT-B/16 at 224 px) before the same formula.

It needs the PyTorch model and its gradients, so it runs in the training venv, never in the
API, which serves ONNX models without PyTorch (D97). The pictures of the app's demo gallery
are precomputed by `src/models/gradcam_gallery.py` with the weights of the served classifier
and served as files next to the gallery (D104).
"""

from __future__ import annotations

import math
from types import TracebackType
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn


def tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
    """(N, 1 + h*w, C) tokens of a ViT -> (N, C, h, w) maps, the class token dropped."""
    patches = tokens[:, 1:, :]
    side = math.isqrt(patches.shape[1])
    if side * side != patches.shape[1]:
        raise ValueError(f"{patches.shape[1]} patch tokens do not form a square grid")
    return patches.reshape(tokens.shape[0], side, side, tokens.shape[2]).permute(0, 3, 1, 2)


class GradCAM:
    """Grad-CAM of `layer` inside `model` (hooks removed on `close` or at the end of `with`)."""

    def __init__(self, model: nn.Module, layer: nn.Module, *, tokens: bool = False) -> None:
        self.model, self.layer, self.tokens = model, layer, tokens
        self._activation: torch.Tensor | None = None
        self._gradient: torch.Tensor | None = None
        self._handle = layer.register_forward_hook(self._keep)

    def _keep(self, module: nn.Module, inputs: Any, output: Any) -> None:
        output = output[0] if isinstance(output, tuple | list) else output
        self._activation = output
        if output.requires_grad:
            output.register_hook(self._keep_gradient)

    def _keep_gradient(self, gradient: torch.Tensor) -> None:
        self._gradient = gradient

    def __call__(self, image: torch.Tensor, target: int | None = None) -> np.ndarray:
        """Heat map (H x W floats in [0, 1]) of one image (1 x 3 x H x W) for class `target`
        (default: the predicted class)."""
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("one image at a time: a 1 x C x H x W tensor")
        was_training = self.model.training
        self.model.eval()
        image = image.detach().clone().requires_grad_(True)  # gradients even if frozen
        try:
            with torch.enable_grad():
                scores = self.model(image)
                if isinstance(scores, tuple | list):
                    scores = scores[-1]
                index = int(scores[0].argmax()) if target is None else int(target)
                self.model.zero_grad(set_to_none=True)
                scores[0, index].backward()
        finally:
            self.model.train(was_training)
        if self._activation is None or self._gradient is None:
            raise RuntimeError("the layer was not reached by the forward pass")
        activation, gradient = self._activation.detach(), self._gradient.detach()
        if self.tokens:
            activation, gradient = tokens_to_map(activation), tokens_to_map(gradient)
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = nn.functional.interpolate(
            cam, size=image.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        cam = cam - cam.min()
        peak = float(cam.max())
        heatmap = (cam / peak) if peak > 0 else cam
        return heatmap.float().cpu().numpy()

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> GradCAM:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        self.close()


def overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """BGR uint8 image with the heat map (JET colours) blended over it."""
    if heatmap.shape != image.shape[:2]:
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    colours = cv2.applyColorMap(np.uint8(np.clip(heatmap, 0, 1) * 255), cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, colours, alpha, 0)
