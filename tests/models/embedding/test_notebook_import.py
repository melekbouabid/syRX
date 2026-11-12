"""Kaggle notebook (src/models/embedding/notebook.py) and import of its export (step 3.5)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import nbformat
import pytest

from src.models.embedding import import_kaggle, notebook


def test_generated_notebook_is_valid_and_committed() -> None:
    nb = notebook.build()
    nbformat.validate(nb)
    source = "\n".join(cell.source for cell in nb.cells)
    assert f'BRANCH = "{notebook.BRANCH}"' in source and "TUJYA_SMOKE" in source
    assert "requirements-finetune.txt" in source and "finetune_bge-m3_export.zip" in source
    assert "token" not in source.lower().replace("never put a token", "")
    code = [cell for cell in nb.cells if cell.cell_type == "code"]
    assert all(cell.outputs == [] and cell.execution_count is None for cell in code)
    committed = notebook.PATH.read_bytes().replace(b"\r\n", b"\n")
    assert committed == (nbformat.writes(nb) + "\n").encode("utf-8")


def export_zip(tmp_path: Path, *, smoke: bool) -> Path:
    folder = tmp_path / "export" / "finetune-bge-m3"
    (folder / "model").mkdir(parents=True)
    (folder / "model" / "config.json").write_text("{}", encoding="utf-8")
    run = {
        "smoke": smoke,
        "config": {"model": "BAAI/bge-m3", "seed": 0, "epochs": 2, "smoke": {"x": 1}},
        "environment": {"gpu": "Tesla T4", "platform": "kaggle"},
        "code": {"git_commit": "abc", "git_branch": notebook.BRANCH},
        "training": {
            "frozen_layers": 0,
            "train_minutes": 30.0,
            "history": [{"step": 10, "loss": 0.5}],
        },
        "metrics": {"base_recall_at_5": 0.8, "tuned_recall_at_5": 0.9},
    }
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    archive = tmp_path / "finetune_bge-m3_export.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for path in folder.rglob("*"):
            zipped.write(path, path.relative_to(folder.parent))
    return archive


def test_extract_refuses_unsafe_paths_and_finds_the_export(tmp_path: Path) -> None:
    target = import_kaggle.extract(export_zip(tmp_path, smoke=False), tmp_path / "models")
    assert target.name == "finetune-bge-m3" and (target / "model" / "config.json").is_file()
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zipped:
        zipped.writestr("../evil/run.json", "{}")
    with pytest.raises(ValueError, match="unsafe"):
        import_kaggle.extract(bad, tmp_path / "models")


def test_params_keep_scalars_only() -> None:
    params = import_kaggle.flat_params({"model": "m", "epochs": 2, "smoke": {"a": 1}})
    assert params == {"model": "m", "epochs": "2"}


def test_import_logs_one_run_with_metrics_before_and_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow
    from mlflow import MlflowClient

    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(import_kaggle, "TARGET", tmp_path / "models")
    monkeypatch.setattr(import_kaggle, "SUMMARIES", tmp_path / "compare")
    monkeypatch.setattr(import_kaggle, "EXPERIMENT", "test-finetuning")
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("test-finetuning", artifact_location=(tmp_path / "art").as_uri())
    archive = export_zip(tmp_path, smoke=False)
    with pytest.raises(SystemExit, match="smoke"):  # a real export is never imported as smoke
        import_kaggle.import_export(archive, smoke=True)
    summary = import_kaggle.import_export(archive)
    run = MlflowClient().get_run(summary["run_id"])
    assert run.data.metrics["base_recall_at_5"] == 0.8
    assert run.data.metrics["tuned_recall_at_5"] == 0.9
    assert run.data.params["model"] == "BAAI/bge-m3" and "smoke" not in run.data.params
    assert run.data.tags["smoke"] == "false" and run.data.tags["plan_step"] == "3.5"
    assert run.data.tags["kaggle_git_branch"] == notebook.BRANCH
    assert run.data.tags["frozen_layers"] == "0"
    history = MlflowClient().get_metric_history(summary["run_id"], "train_loss")
    assert [(m.step, m.value) for m in history] == [(10, 0.5)]
    saved = json.loads((tmp_path / "compare" / "embedding_finetune-bge-m3.json").read_text())
    assert (
        saved["run_id"] == summary["run_id"] and (tmp_path / "models" / "finetune-bge-m3").is_dir()
    )
    while mlflow.active_run():
        mlflow.end_run()
