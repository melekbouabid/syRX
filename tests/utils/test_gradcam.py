"""Grad-CAM (src/utils/gradcam.py) on small models whose answer is known.

Needs PyTorch (CI job `vision`; skipped elsewhere).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from src.utils.gradcam import GradCAM, overlay, tokens_to_map  # noqa: E402


class Brightness(nn.Module):
    """Class 0 score = mean brightness of the feature map, class 1 = its opposite."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Conv2d(3, 4, 3, padding=1)
        with torch.no_grad():
            self.features.weight.fill_(0.0)
            self.features.bias.fill_(0.0)
            for k in range(4):
                self.features.weight[k, :, 1, 1] = 1.0  # each map copies the brightness
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.pool(torch.relu(self.features(x))).flatten(1).mean(1, keepdim=True)
        return torch.cat([score, -score], dim=1)


def _image() -> torch.Tensor:
    image = torch.zeros(1, 3, 32, 32)
    image[:, :, 2:12, 20:30] = 1.0  # a bright patch in the top right corner
    return image


def test_the_heat_map_points_at_what_drives_the_class() -> None:
    model = Brightness()
    with GradCAM(model, model.features) as cam:
        heatmap = cam(_image(), 0)
    assert heatmap.shape == (32, 32)
    assert heatmap.min() >= 0.0 and heatmap.max() == pytest.approx(1.0)
    row, col = np.unravel_index(int(heatmap.argmax()), heatmap.shape)
    assert 2 <= row < 12 and 20 <= col < 30
    assert heatmap[25:, :10].max() < 0.05  # the dark corner gets nothing


def test_class_against_the_evidence_gives_an_empty_map_and_hooks_are_removed() -> None:
    model = Brightness()
    cam = GradCAM(model, model.features)
    assert float(cam(_image(), 1).max()) == 0.0  # ReLU keeps positive evidence only
    assert cam(_image()).max() == pytest.approx(1.0)  # default: the predicted class (0)
    cam.close()
    assert not model.features._forward_hooks
    assert model.training  # the mode of the model is restored


def test_frozen_weights_still_give_a_map() -> None:
    model = Brightness()
    for parameter in model.parameters():
        parameter.requires_grad = False
    with GradCAM(model, model.features) as cam:
        assert cam(_image(), 0).max() == pytest.approx(1.0)


def test_tokens_are_folded_into_their_grid() -> None:
    tokens = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)  # class + 2x2
    grid = tokens_to_map(tokens)
    assert grid.shape == (2, 3, 2, 2)
    assert grid[0, :, 0, 0].tolist() == tokens[0, 1].tolist()  # first patch, class dropped
    with pytest.raises(ValueError, match="square"):
        tokens_to_map(torch.zeros(1, 4, 3))


def test_inputs_are_checked_and_overlay_keeps_the_image_size() -> None:
    model = Brightness()
    with GradCAM(model, model.features) as cam, pytest.raises(ValueError, match="one image"):
        cam(torch.zeros(2, 3, 8, 8))
    picture = overlay(np.zeros((40, 50, 3), np.uint8), np.ones((10, 10), np.float32))
    assert picture.shape == (40, 50, 3) and picture.dtype == np.uint8
