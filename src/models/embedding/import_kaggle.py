"""Import the export of the fine-tuning notebook into MLflow and DVC (step 3.5).

    # training venv, MLflow server running (docker compose up -d mlflow)
    python -m src.models.embedding.import_kaggle runs/kaggle/finetune_bge-m3_export.zip
    python -m src.models.embedding.import_kaggle <export.zip> --smoke     # a local smoke run

1. unzips the export into data/models/embeddings/<name> (git-ignored; `dvc add` it on the
   branch experiment/fine-tuning-embedding, never git);
2. logs one run in the experiment `tujya-rag-finetuning`: the training configuration as
   params, the retrieval metrics on the validation pairs before (`base_*`) and after
   (`tuned_*`) training, the loss of every logged step, the provenance tags (git commit and
   branch of the Kaggle code, `dvc_rev`, GPU, seed, `smoke`), run.json and the loss curve;
3. writes models/compare/embedding_<name>.json (summary used by docs and the step 3.3 grid).

The model is not registered here: whether it replaces bge-m3 is decided by the retrieval
benchmark of the app (step 3.3, `--embedder local:data/models/embeddings/<name>/model`),
whose best configuration gets the alias `champion`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / "data" / "models" / "embeddings"
SUMMARIES = ROOT / "models" / "compare"
EXPERIMENT = "tujya-rag-finetuning"


def extract(archive: Path, target_root: Path = TARGET) -> Path:
    """Unzip the export (one top folder with run.json) into target_root/<folder>."""
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(archive) as zipped:
        for member in zipped.namelist():  # no absolute path or ".." in the archive
            if member.startswith(("/", "\\")) or ".." in Path(member).parts:
                raise ValueError(f"unsafe path in the archive: {member}")
        zipped.extractall(tmp)
        found = sorted(Path(tmp).rglob("run.json"))
        if not found:
            raise ValueError(f"{archive} has no run.json")
        folder = found[0].parent
        target = target_root / folder.name
        if target.exists():
            shutil.rmtree(target)
        target_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(folder, target)
    return target


def flat_params(config: dict[str, Any]) -> dict[str, str]:
    return {k: str(v) for k, v in config.items() if not isinstance(v, (dict, list))}


def loss_curve(history: list[dict[str, Any]], path: Path) -> Path | None:
    points = [(h["step"], h["loss"]) for h in history if "loss" in h and "step" in h]
    if not points:
        return None
    try:
        import matplotlib
    except ImportError:  # project venv: the curve is optional, the metric history is logged
        return None

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=120)
    ax.plot([p[0] for p in points], [p[1] for p in points], color="#2a78d6")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (CachedMNRL)")
    ax.grid(True, color="#e1e0d9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _shown(path: Path) -> str:
    """Path relative to the repository when it is inside it (the usual case), else absolute."""
    with contextlib.suppress(ValueError):
        path = path.resolve().relative_to(ROOT)
    return path.as_posix()


def import_export(archive: Path, *, smoke: bool = False) -> dict[str, Any]:
    import mlflow

    from src.utils import mlflow_utils as mu

    folder = extract(archive, TARGET)
    run_info = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    if bool(run_info.get("smoke")) != smoke:
        raise SystemExit(
            f"{archive}: smoke={run_info.get('smoke')} but --smoke={smoke} (a smoke export is "
            "imported with --smoke, a real one without)"
        )
    config, env = run_info["config"], run_info.get("environment", {})
    code = run_info.get("code", {})
    name = ("smoke-" if smoke else "") + folder.name  # e.g. finetune-bge-m3
    mu.configure()
    with mu.start_tracked_run(
        EXPERIMENT,
        name,
        seed=int(config.get("seed", 0)),
        dataset="tujya-rag-finetune-pairs",
        dataset_path="data/processed/rag/finetune",
        params=flat_params(config),
        tags={
            "stage": "finetune",
            "plan_step": "3.5",
            "task": "embedding",
            "smoke": str(smoke).lower(),
            "base_model": str(config.get("model", "")),
            "frozen_layers": str(run_info.get("training", {}).get("frozen_layers", "")),
            "gpu": str(env.get("gpu", "")),
            "platform": str(env.get("platform", "")),
            "kaggle_git_commit": str(code.get("git_commit", "")),
            "kaggle_git_branch": str(code.get("git_branch", "")),
        },
    ) as run:
        mlflow.log_metrics({k: float(v) for k, v in run_info["metrics"].items()})
        mlflow.log_metric("train_minutes", float(run_info["training"]["train_minutes"]))
        for h in run_info["training"].get("history", []):
            if "loss" in h and "step" in h:
                mlflow.log_metric("train_loss", float(h["loss"]), step=int(h["step"]))
        mlflow.log_artifact(str(folder / "run.json"))
        with tempfile.TemporaryDirectory() as tmp:
            curve = loss_curve(run_info["training"].get("history", []), Path(tmp) / "loss.png")
            if curve:
                mlflow.log_artifact(str(curve))
        run_id = run.info.run_id
    summary = {
        "name": folder.name,
        "run_id": run_id,
        "smoke": smoke,
        "model_path": _shown(folder / "model"),
        "base_model": config.get("model"),
        "frozen_layers": run_info["training"].get("frozen_layers"),
        "metrics": run_info["metrics"],
    }
    SUMMARIES.mkdir(parents=True, exist_ok=True)
    (SUMMARIES / f"embedding_{name}.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.models.embedding.import_kaggle")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    print(json.dumps(import_export(args.archive, smoke=args.smoke), indent=1))


if __name__ == "__main__":
    main()
