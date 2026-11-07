"""Kaggle packaging, notebook generator and comparison table (src/models/detection)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import nbformat
import pytest

from src.models.detection import compare, notebook, package_kaggle
from tests.models.detection.helpers import make_dataset

COMMITTED = sorted(notebook.FOLDER.glob("detector_*.ipynb"))


def test_package_for_kaggle(tmp_path: Path) -> None:
    data = make_dataset(tmp_path / "data")
    (data / "summary.json").write_text("{}", encoding="utf-8")
    (data / "data.yaml").write_text("path: C:/somewhere/local\n", encoding="utf-8")
    result = package_kaggle.package(data, tmp_path / "out", owner="someone")
    folder = Path(result["folder"])
    manifest = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["images"] == {"train": 6, "val": 3, "test": 4, "holdout": 2}
    assert manifest["name"] == "c3pi-v2-detect"
    meta = json.loads((folder / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert meta["id"] == "someone/tujya-c3pi-detect-v2"
    assert meta["licenses"] == [{"name": "ODbL-1.0"}]
    assert "ODbL" in (folder / "README.md").read_text(encoding="utf-8")
    # The local data.yaml (absolute path of this PC) is not packed: the runner writes its own.
    assert not (folder / "data.yaml").exists()
    with zipfile.ZipFile(result["zip"]) as archive:
        names = set(archive.namelist())
    assert "images/train/train_000.jpg" in names
    assert "labels/test/test_003.txt" in names
    assert {"test_holdout.txt", "MANIFEST.json", "summary.json"} <= names
    assert "dataset-metadata.json" not in names


def test_package_refuses_a_broken_dataset(tmp_path: Path) -> None:
    data = make_dataset(tmp_path / "data")
    (data / "labels" / "val" / "val_000.txt").unlink()
    with pytest.raises(ValueError, match="without labels"):
        package_kaggle.check_dataset(data)
    data = make_dataset(tmp_path / "data2")
    (data / "test_holdout.txt").write_text("train_000.jpg\n", encoding="utf-8")
    with pytest.raises(ValueError, match="holdout"):
        package_kaggle.check_dataset(data)


@pytest.mark.parametrize("arch", sorted(notebook.ARCHS))
def test_generated_notebooks_are_valid(arch: str) -> None:
    nb = notebook.build(arch)
    nbformat.validate(nb)
    source = "\n".join(cell.source for cell in nb.cells)
    assert f'ARCH = "{arch}"' in source
    assert f'BRANCH = "experiment/detector-{arch}"' in source
    assert "run_experiment" in source
    assert "TUJYA_SMOKE" in source
    assert "token" not in source.lower().replace("never put a token", "")
    code = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert all(cell.outputs == [] and cell.execution_count is None for cell in code)
    for cell in code:
        compile(cell.source, f"{arch}-{cell.id}", "exec")  # valid Python


@pytest.mark.parametrize("path", COMMITTED, ids=[p.stem for p in COMMITTED])
def test_committed_notebooks_are_the_generated_ones(path: Path) -> None:
    arch = path.stem.removeprefix("detector_")
    committed = nbformat.read(str(path), as_version=4)
    generated = notebook.build(arch)
    assert [c.source for c in committed.cells] == [c.source for c in generated.cells]
    assert all(c.get("outputs", []) == [] for c in committed.cells)  # nbstripout
    assert (notebook.ROOT / "config" / "hpo" / f"detector_{arch}.yaml").is_file()


def test_comparison_table_and_markers() -> None:
    rows = [
        compare.Row("yolov8s", "a" * 32, 0.9, 0.88, 0.97, 0.95, 0.94, 80.0, 11.1e6, "champion"),
        compare.Row(
            "yolo11n (baseline v1)", "b" * 32, None, 0.85, 0.96, 0.9, 0.9, 40.0, None, "challenger"
        ),
    ]
    table = compare.markdown_table(rows)
    lines = table.strip().splitlines()
    assert len(lines) == 4
    assert (
        "| yolov8s | 0.9000 | 0.8800 | 0.9700 | 0.950 | 0.940 |  | 80.0 |  |  | 11.1 |  "
        "| champion |" in table
    )
    assert "`bbbbbbbb`" in lines[3]
    text = f"intro\n{compare.BEGIN}\nold\n{compare.END}\nend\n"
    updated = compare.replace_between_markers(text, table)
    assert "old" not in updated
    assert updated.startswith("intro\n") and updated.endswith("end\n")
    with pytest.raises(ValueError, match="markers"):
        compare.replace_between_markers("no markers", table)


def test_comparison_table_has_size_p95_whole_test_and_version() -> None:
    row = compare.Row(
        "yolo11s",
        "c" * 32,
        0.87,
        0.8885,
        0.974,
        0.952,
        0.936,
        68.2,
        9.4e6,
        "champion",
        test_all_map=0.8722,
        latency_p95_ms=75.0,
        size_mb=37.9,
        version="10",
    )
    table = compare.markdown_table([row])
    assert "Taille ONNX (Mo)" in table and "p95 (ms)" in table
    assert (
        "| yolo11s | 0.8700 | 0.8885 | 0.9740 | 0.952 | 0.936 | 0.8722 | 68.2 | 75.0 | 37.9 "
        "| 9.4 | v10 | champion | `cccccccc` |" in table
    )


def test_comparison_figure_has_two_panels(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    rows = [
        compare.Row("a", "a" * 32, None, 0.88, None, None, None, 20.0, None, "champion", size_mb=9),
        compare.Row("b", "b" * 32, None, 0.82, None, None, None, 900.0, None, "", size_mb=160),
        compare.Row("c", "c" * 32, None, 0.80, None, None, None, None, None, "", size_mb=None),
    ]
    path = compare.plot(rows, tmp_path / "figure.png")
    assert path.is_file() and path.stat().st_size > 0
