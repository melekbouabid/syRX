"""Kaggle notebooks of the classifiers (src/models/classification/notebook.py)."""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

from src.models.classification import notebook

COMMITTED = sorted(notebook.FOLDER.glob("classifier_*.ipynb"))


PLAN_ARCHS = {
    "baseline-cnn",
    "resnet50",
    "efficientnetv2s",
    "convnext-tiny",
    "vit-b16",
    "yolo11s-cls",
}


def test_the_six_architectures_of_the_plan_have_a_notebook() -> None:
    assert set(notebook.ARCHS) == PLAN_ARCHS


@pytest.mark.parametrize("arch", sorted(notebook.ARCHS))
def test_generated_notebooks_are_valid(arch: str) -> None:
    nb = notebook.build(arch)
    nbformat.validate(nb)
    source = "\n".join(cell.source for cell in nb.cells)
    assert f'ARCH = "{arch}"' in source
    assert f'BRANCH = "experiment/classifier-{arch}"' in source
    assert "run_experiment" in source and "TUJYA_SMOKE" in source
    assert f"classifier_{arch}_export.zip" in source
    assert "token" not in source.lower().replace("never put a token", "")
    code = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert all(cell.outputs == [] and cell.execution_count is None for cell in code)
    for cell in code:
        compile(cell.source, f"{arch}-{cell.id}", "exec")  # valid Python


@pytest.mark.parametrize("path", COMMITTED, ids=[p.stem for p in COMMITTED])
def test_committed_notebooks_are_the_generated_ones(path: Path) -> None:
    arch = path.stem.removeprefix("classifier_")
    committed = nbformat.read(str(path), as_version=4)
    generated = notebook.build(arch)
    assert [c.source for c in committed.cells] == [c.source for c in generated.cells]
    assert all(c.get("outputs", []) == [] for c in committed.cells)  # nbstripout
    assert (notebook.ROOT / "config" / "hpo" / f"classifier_{arch}.yaml").is_file()
