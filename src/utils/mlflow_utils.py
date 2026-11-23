"""MLflow helpers shared by every training, export and evaluation script.

`start_tracked_run` tags each run with git_commit, dvc_rev, dataset and seed, and logs
requirements, dvc.lock and params.yaml, so a run can be traced to its code and data.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow.entities import Run
from mlflow.models.model import ModelInfo

ROOT = Path(__file__).resolve().parents[2]  # repository root
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"  # docker compose up -d mlflow


# Files of the repository logged with every run (when they exist).
REPRODUCIBILITY_FILES = ("dvc.lock", "params.yaml")


def configure(tracking_uri: str | None = None) -> str:
    """Point MLflow at MLFLOW_TRACKING_URI (default: the local server of docker-compose.yml)."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    return uri


# --- provenance ---------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str | None:
    try:
        # Fixed program and arguments, `git` from PATH as everywhere else in the tooling.
        done = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def git_commit(root: Path = ROOT) -> str:
    """Commit checked out in `root`, or "unknown" outside a git repository."""
    return _git(root, "rev-parse", "HEAD") or "unknown"


def git_dirty(root: Path = ROOT) -> bool:
    """True when tracked files differ from the commit (untracked files are ignored)."""
    status = _git(root, "status", "--porcelain", "--untracked-files=no")
    return bool(status)


def dvc_rev(root: Path = ROOT) -> str:
    """Git object id of `dvc.lock` on disk, i.e. the version of every DVC input.

    Find the matching commits with `git log --find-object=<dvc_rev>`; "none" without a dvc.lock.
    """
    lock = root / "dvc.lock"
    if not lock.is_file():
        return "none"
    blob = _git(root, "hash-object", "dvc.lock")
    if blob:
        return blob
    content = lock.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(content) + content).hexdigest()


def dataset_md5(path: str, root: Path = ROOT) -> str | None:
    """DVC hash recorded in dvc.lock for an output folder or file (None when not found)."""
    lock = root / "dvc.lock"
    if not lock.is_file():
        return None
    data = yaml.safe_load(lock.read_text(encoding="utf-8")) or {}
    wanted = path.strip("/").replace("\\", "/")
    for stage in (data.get("stages") or {}).values():
        for entry in stage.get("outs") or []:
            if entry.get("path", "").strip("/") == wanted:
                return entry.get("md5")
    return None


def frozen_requirements() -> str:
    """`name==version` of every package installed in the running environment, sorted.

    Read from the installed distributions, so it works in uv venvs that have no pip.
    """
    seen: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            seen.setdefault(name.lower().replace("_", "-"), f"{name}=={dist.version}")
    return "\n".join(seen[key] for key in sorted(seen)) + "\n"


def provenance_tags(dataset: str, seed: int, root: Path = ROOT) -> dict[str, str]:
    tags = {
        "git_commit": git_commit(root),
        "dvc_rev": dvc_rev(root),
        "dataset": dataset,
        "seed": str(seed),
    }
    if git_dirty(root):
        tags["git_dirty"] = "true"
    return tags


def log_reproducibility(root: Path = ROOT, files: tuple[str, ...] = REPRODUCIBILITY_FILES) -> None:
    """requirements.txt (frozen now), dvc.lock and params.yaml under `reproducibility/`."""
    with tempfile.TemporaryDirectory() as tmp:
        requirements = Path(tmp) / "requirements.txt"
        requirements.write_text(frozen_requirements(), encoding="utf-8")
        mlflow.log_artifact(str(requirements), "reproducibility")
    for name in files:
        path = root / name
        if path.is_file():
            mlflow.log_artifact(str(path), "reproducibility")


@contextmanager
def start_tracked_run(
    experiment: str,
    run_name: str | None = None,
    *,
    seed: int,
    dataset: str,
    dataset_path: str | None = None,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
    root: Path = ROOT,
) -> Iterator[Run]:
    """Start a run with the provenance tags and the reproducibility artefacts.

    `dataset_path` is the DVC output the data comes from (e.g.
    "data/processed/c3pi/datasets"); its hash in dvc.lock becomes the tag `dataset_md5`.
    """
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        all_tags = provenance_tags(dataset, seed, root)
        if dataset_path:
            md5 = dataset_md5(dataset_path, root)
            if md5:
                all_tags["dataset_md5"] = md5
        all_tags.update(tags or {})
        mlflow.set_tags(all_tags)
        mlflow.log_params({"seed": seed, **(params or {})})
        log_reproducibility(root)
        yield run


# --- models and Model Registry ------------------------------------------------------------


def _onnx_requirements() -> list[str]:
    """Pinned runtime requirements of an ONNX model (what the API needs to serve it)."""
    pins = []
    for package in ("onnx", "onnxruntime"):
        try:
            pins.append(f"{package}=={metadata.version(package)}")
        except metadata.PackageNotFoundError:
            pins.append(package)
    return pins


def log_model(
    model: Any,
    *,
    name: str,
    flavor: str = "onnx",
    metrics: Mapping[str, float] | None = None,
    params: Mapping[str, Any] | None = None,
    pip_requirements: list[str] | None = None,
    **kwargs: Any,
) -> ModelInfo:
    """Log a model in the active run as an MLflow 3 LoggedModel called `name`.

    `metrics` are linked to the model; ONNX requirements are given, not inferred.
    """
    if pip_requirements is None and flavor == "onnx":
        pip_requirements = _onnx_requirements()
    module = getattr(mlflow, flavor)
    info: ModelInfo = module.log_model(
        model,
        name=name,
        params=dict(params) if params else None,
        pip_requirements=pip_requirements,
        **kwargs,
    )
    if metrics:
        mlflow.log_metrics(dict(metrics), model_id=info.model_id)
    return info
