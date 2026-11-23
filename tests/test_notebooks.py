"""Committed notebooks: summary notebooks keep their outputs, Kaggle notebooks have none.

Nothing is executed here (the summary notebooks need MLflow, Qdrant and Ollama).
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
SUMMARY = sorted(NOTEBOOKS.glob("module[12]/[0-9][0-9]_*.ipynb"))
KAGGLE = sorted(NOTEBOOKS.glob("module1/kaggle/*.ipynb"))
MAX_BYTES = 2 * 1024 * 1024


def test_the_summary_notebooks_are_found() -> None:
    assert len(SUMMARY) >= 11
    assert {p.parent.name for p in SUMMARY} == {"module1", "module2"}


@pytest.mark.parametrize("path", SUMMARY, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_summary_notebook_is_valid_executed_and_small(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code, "no code cell"
    assert notebook.cells[0].cell_type == "markdown"  # title and goal first
    assert all(cell.execution_count for cell in code), "not executed from start to end"
    outputs = [output for cell in code for output in cell.outputs]
    assert outputs, "no output"
    assert not [o for o in outputs if o.output_type == "error"], "an output is an error"
    assert path.stat().st_size < MAX_BYTES


@pytest.mark.parametrize("path", KAGGLE, ids=lambda p: p.stem)
def test_kaggle_notebook_has_no_output(path: Path) -> None:
    notebook = nbformat.read(path, as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert all(not cell.outputs and cell.execution_count is None for cell in code)


def test_the_index_lists_every_summary_notebook() -> None:
    index = (NOTEBOOKS / "README.md").read_text(encoding="utf-8")
    missing = [p.name for p in SUMMARY if f"{p.parent.name}/{p.name}" not in index]
    assert not missing
