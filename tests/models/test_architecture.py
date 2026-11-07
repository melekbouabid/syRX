"""The six classifier architectures (src/models/architecture.py): outputs, selective
fine-tuning, frozen blocks.

Needs PyTorch and torchvision (CI job `vision`, CPU wheels; the job `src` skips this file).
Networks are built without pretrained weights (no download); YOLO11s-cls also needs
Ultralytics and is skipped without it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from src.models import architecture as A  # noqa: E402

TORCHVISION = ["baseline-cnn", "resnet50", "efficientnetv2s", "convnext-tiny", "vit-b16"]


def _archs() -> list[str]:
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return TORCHVISION
    return [*TORCHVISION, "yolo11s-cls"]


@pytest.mark.parametrize("arch", _archs())
def test_every_architecture_outputs_one_logit_per_class(arch: str) -> None:
    net = A.build(arch, 7, pretrained=False).eval()
    size = A.ARCHITECTURES[arch].fixed_size or 64
    with torch.no_grad():
        logits = net(torch.rand(2, 3, size, size))
    assert logits.shape == (2, 7)
    parts = net.parts()
    assert parts.blocks and parts.head
    # Selective fine-tuning: the head alone, the last block, everything.
    total = A.count_parameters(net)
    head = sum(p.numel() for m in parts.head for p in m.parameters())
    assert A.set_trainable(net, 0) == head
    last = sum(p.numel() for p in parts.blocks[-1].parameters())
    assert A.set_trainable(net, 1) == head + last
    assert A.set_trainable(net, len(parts.blocks)) == total


def test_frozen_blocks_keep_their_batch_norm_statistics() -> None:
    net = A.build("baseline-cnn", 3, pretrained=False)
    A.set_trainable(net, 1)
    A.train_mode(net)
    blocks = net.parts().blocks
    assert not blocks[0].training and blocks[-1].training
    groups = A.parameter_groups(net, 1e-3, 0.1, 0.05)
    rates = sorted({g["lr"] for g in groups})
    assert rates == pytest.approx([1e-4, 1e-3])  # backbone at lr * 0.1, head at lr
    assert all(g["weight_decay"] == 0.0 for g in groups if g["params"][0].ndim <= 1)
    with pytest.raises(ValueError, match="224"):
        A.check_size("vit-b16", 192)
    with pytest.raises(ValueError, match="unknown"):
        A.build("alexnet", 3)


def test_frozen_backbone_is_not_updated_during_the_head_epochs() -> None:
    torch.manual_seed(0)
    net = A.build("resnet50", 6, pretrained=False)
    before = {k: v.clone() for k, v in net.state_dict().items()}
    A.set_trainable(net, 0)  # transfer learning: the head only
    optimizer = torch.optim.SGD([p for p in net.parameters() if p.requires_grad], lr=0.1)
    A.train_mode(net)
    net(torch.rand(4, 3, 32, 32)).sum().backward()
    optimizer.step()
    after = net.state_dict()
    assert torch.equal(after["body.layer1.0.conv1.weight"], before["body.layer1.0.conv1.weight"])
    # Batch-norm statistics of the frozen blocks did not move either (eval mode).
    assert torch.equal(after["body.bn1.running_mean"], before["body.bn1.running_mean"])
    assert not torch.equal(after["body.fc.weight"], before["body.fc.weight"])
