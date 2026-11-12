"""Comparison table and figure of the classifiers (src/models/classification/compare.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.classification import compare


def test_comparison_table_and_markers() -> None:
    rows = [
        compare.Row(
            "resnet50",
            "a" * 32,
            0.95,
            0.94,
            0.99,
            0.93,
            0.96,
            0.93,
            12.0,
            90.0,
            23.5e6,
            "champion",
            "8",
        ),
        compare.Row(
            "yolo11s-cls (champion v1, 150 classes)",
            "b" * 32,
            None,
            0.41,
            0.44,
            0.2,
            0.9,
            0.4,
            5.0,
            22.6,
            None,
            "challenger",
            "1",
        ),
    ]
    table = compare.markdown_table(rows)
    lines = table.strip().splitlines()
    assert len(lines) == 4
    assert (
        "| resnet50 | 0.9500 | 0.9400 | 0.9900 | 0.9300 | 0.9600 | 0.9300 | 12.0 | 90.0 | 23.5 "
        "| v8 | champion | `aaaaaaaa` |" in table
    )
    text = f"intro\n{compare.BEGIN}\nold\n{compare.END}\nend\n"
    assert "old" not in compare.replace_between_markers(text, table)


def test_comparison_figure(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    rows = [
        compare.Row("a", "a" * 32, None, 0.9, None, None, None, None, 10.0, 20.0, None, "champion"),
        compare.Row("b", "b" * 32, None, 0.8, None, None, None, None, None, 5.0, None, ""),
    ]
    assert compare.plot(rows, tmp_path / "figure.png").stat().st_size > 0


def test_neighbour_labels_are_stacked() -> None:
    # Latency panel of step 2.3: EfficientNetV2-S, ResNet-50, ConvNeXt-Tiny close together.
    points = [(17.6, 0.9089), (18.6, 0.9027), (32.7, 0.8966), (4.5, 0.8594), (84.9, 0.8515)]
    offsets = compare.label_offsets(points)
    assert offsets[:3] == [4.0, -30.0, -52.0]  # below, in the empty part of the plot
    assert offsets[3:] == [4.0, 4.0]  # far on the log axis: label above the point
