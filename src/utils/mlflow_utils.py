"""MLflow helpers shared by every training, export and evaluation script (lab MLflow).

Every run started with `start_tracked_run` can be traced back to the exact code, data and
environment that produced it:

- tags `git_commit` (and `git_dirty` when the working tree had uncommitted changes),
  `dvc_rev` (git object id of `dvc.lock`, which pins every DVC-tracked input), `dataset`
  (plus `dataset_md5`, the DVC hash of the dataset folder when it is a pipeline output) and
  `seed` (also logged as a parameter);
- artefacts under `reproducibility/`: `requirements.txt` frozen from the running
  environment, `dvc.lock` and `params.yaml`.

`git checkout <git_commit> && dvc checkout` then restores the code and the data of the run.

Models are logged with the MLflow 3 API (`name=`, one LoggedModel per call, metrics linked to
it), registered in the Model Registry, and promoted with aliases (the stages Staging /
Production are deprecated since MLflow 2.9): `champion` is the version the API serves,
`challenger` the runner-up. `promote_if_better` is the guard rail of the lab: a new version
replaces the champion only if it beats it by at least `min_gain` on a given metric.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import mlflow
import yaml
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.models.model import ModelInfo

ROOT = Path(__file__).resolve().parents[2]  # repository root
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"  # docker compose up -d mlflow

CHAMPION = "champion"
CHALLENGER = "challenger"
# Tag set on registered versions by promote_if_better: champion / challenger / archived.
STATUS_TAG = "status"
ARCHIVED = "archived"

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
    """Git object id of `dvc.lock` as it is on disk, i.e. the version of every DVC input.

    It equals the blob id git stores once the file is committed, so
    `git log --find-object=<dvc_rev>` finds the commits with this exact data version. Falls back
    to the SHA-1 of the content outside a git repository, "none" without a dvc.lock.
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
    """Log a model in the active run with the MLflow 3 API (a LoggedModel called `name`).

    `metrics` are logged to the run and linked to the LoggedModel, so the Models tab shows
    them next to the model. The requirements are given (not inferred) for ONNX: inference
    starts a subprocess and is slow; the ONNX runtime is all a served model needs.
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


def register(
    model_uri: str,
    name: str,
    *,
    alias: str | None = None,
    tags: Mapping[str, str] | None = None,
    description: str | None = None,
    client: MlflowClient | None = None,
) -> ModelVersion:
    """Register `model_uri` as a new version of `name`, optionally under an alias."""
    client = client or MlflowClient()
    version = mlflow.register_model(model_uri, name, tags=dict(tags) if tags else None)
    if description:
        client.update_model_version(name, version.version, description=description)
    if alias:
        client.set_registered_model_alias(name, alias, version.version)
    return client.get_model_version(name, version.version)


def alias_version(name: str, alias: str, client: MlflowClient | None = None) -> str | None:
    """Version number holding `alias`, or None."""
    client = client or MlflowClient()
    try:
        return str(client.get_model_version_by_alias(name, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def version_metric(
    name: str, version: str | int, metric: str, client: MlflowClient | None = None
) -> float | None:
    """Value of `metric` for a registered version: its run's metric, else its LoggedModel's."""
    client = client or MlflowClient()
    mv = client.get_model_version(name, str(version))
    if mv.run_id:
        value = client.get_run(mv.run_id).data.metrics.get(metric)
        if value is not None:
            return float(value)
    model_id = getattr(mv, "model_id", None)
    if model_id:
        for item in client.get_logged_model(model_id).metrics or []:
            if item.key == metric:
                return float(item.value)
    return None


@dataclass(frozen=True)
class Promotion:
    """Outcome of promote_if_better."""

    name: str
    version: str
    metric: str
    value: float
    promoted: bool  # the candidate is now the champion
    status: str  # champion / challenger / archived: alias given to the candidate
    champion_before: str | None = None
    champion_value: float | None = None
    reason: str = ""


def promote_if_better(
    name: str,
    version: str | int,
    metric: str,
    min_gain: float = 0.0,
    *,
    greater_is_better: bool = True,
    eligible: bool = True,
    ineligible_reason: str = "",
    client: MlflowClient | None = None,
) -> Promotion:
    """Give the alias `champion` to `version` only if it beats the champion by `min_gain`.

    - no champion yet: the candidate becomes champion;
    - candidate better by at least `min_gain` (and strictly better): it becomes champion and
      the former champion becomes challenger;
    - otherwise the champion stays; the candidate becomes challenger if there is none or if
      it beats the current challenger, else it is archived.

    The versions involved get the tag `status` (champion / challenger / archived) and the
    candidate a tag `promotion` explaining the decision, so the Registry shows the history.

    `eligible=False` (a guard the caller checked failed, `ineligible_reason` says which) keeps
    the candidate away from both aliases: it is archived whatever its score (D100).
    """
    client = client or MlflowClient()
    version = str(version)
    value = version_metric(name, version, metric, client)
    if value is None:
        raise ValueError(f"{name} version {version} has no metric {metric!r}")
    sign = 1.0 if greater_is_better else -1.0

    def gain_over(other: float) -> float:
        return sign * (value - other)

    champion = alias_version(name, CHAMPION, client)
    if not eligible:
        reason = (
            f"not eligible: {ineligible_reason or 'guard failed'} ({metric}={value:.4f}); archived"
        )
        client.set_model_version_tag(name, version, STATUS_TAG, ARCHIVED)
        client.set_model_version_tag(name, version, "promotion", reason)
        return Promotion(name, version, metric, value, False, ARCHIVED, champion, None, reason)
    if champion == version:
        return Promotion(
            name, version, metric, value, True, CHAMPION, champion, value, "already champion"
        )
    champion_value = (
        version_metric(name, champion, metric, client) if champion is not None else None
    )

    if champion is None or champion_value is None:
        reason = f"first champion ({metric}={value:.4f})"
        _set_status(client, name, version, CHAMPION)
        client.set_model_version_tag(name, version, "promotion", reason)
        return Promotion(name, version, metric, value, True, CHAMPION, champion, None, reason)

    gain = gain_over(champion_value)
    if gain > 0 and gain >= min_gain:
        reason = (
            f"promoted: {metric}={value:.4f} vs {champion_value:.4f} (v{champion}), "
            f"gain {gain:+.4f} >= {min_gain}"
        )
        _set_status(client, name, champion, CHALLENGER)
        _set_status(client, name, version, CHAMPION)
        client.set_model_version_tag(name, version, "promotion", reason)
        client.set_model_version_tag(name, version, "promoted_from", champion)
        return Promotion(
            name, version, metric, value, True, CHAMPION, champion, champion_value, reason
        )

    challenger = alias_version(name, CHALLENGER, client)
    challenger_value = (
        version_metric(name, challenger, metric, client) if challenger is not None else None
    )
    if challenger_value is None or gain_over(challenger_value) > 0:
        status = CHALLENGER
        _set_status(client, name, version, CHALLENGER)
    else:
        status = ARCHIVED
        client.set_model_version_tag(name, version, STATUS_TAG, ARCHIVED)
    reason = (
        f"not promoted: {metric}={value:.4f} vs champion {champion_value:.4f} (v{champion}), "
        f"gain {gain:+.4f} < {min_gain}; {status}"
    )
    client.set_model_version_tag(name, version, "promotion", reason)
    return Promotion(name, version, metric, value, False, status, champion, champion_value, reason)


def _set_status(client: MlflowClient, name: str, version: str, alias: str) -> None:
    """Move `alias` to `version`, archive the version that loses it, tag the new holder."""
    previous = alias_version(name, alias, client)
    client.set_registered_model_alias(name, alias, version)
    client.set_model_version_tag(name, version, STATUS_TAG, alias)
    if previous is not None and previous != version:
        holder = client.get_model_version(name, previous)
        # Only champion / challenger are statuses: `served` (step 2.4) does not keep one.
        if not set(holder.aliases or []) & {CHAMPION, CHALLENGER}:
            client.set_model_version_tag(name, previous, STATUS_TAG, ARCHIVED)
